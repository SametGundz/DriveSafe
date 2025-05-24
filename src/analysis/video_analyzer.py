#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video analyzer module for the driver drowsiness detection application.

This module provides the VideoAnalyzer class that handles:
- Loading and processing video files
- Frame-by-frame analysis for drowsiness detection
- Multithreaded processing for improved performance
- Progress reporting and callbacks
"""

import os
import cv2
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal, QObject, QMutex

from src.utils.mediapipe_helper import MediaPipeHelper, get_mediapipe_helper
from src.analysis.gaze_processor import GazeProcessor, GazeData
from src.detection.drowsiness_detector import DrowsinessDetector

# Get module-specific logger
logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """Data class for storing frame analysis results."""
    timestamp: float
    frame_number: int
    ear: Optional[float] = None
    mar: Optional[float] = None
    perclos: Optional[float] = None
    face_detected: bool = False
    landmarks: Optional[List[List[float]]] = None
    gaze_data: Optional[GazeData] = None
    gaze_zone_id: Optional[int] = None  # Bakılan bölgenin zone ID'si
    drowsiness_level: float = 0.0
    is_drowsy: bool = False
    is_distracted: bool = False

class AnalysisWorker(QThread):
    """Worker thread for video analysis."""
    
    # Define signals
    frame_processed = pyqtSignal(FrameData, np.ndarray)
    progress_updated = pyqtSignal(int)
    analysis_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path: str, config: Dict[str, Any] = None):
        """
        Initialize the analysis worker.
        
        Args:
            video_path: Path to the video file
            config: Configuration dictionary
        """
        super().__init__()
        
        self.video_path = video_path
        self.config = config or {}
        
        # Initialize flags
        self.is_running = False
        self.is_paused = False
        self.mutex = QMutex()
        
        # Initialize components
        self.mediapipe_helper = None
        self.gaze_processor = None
        self.drowsiness_detector = None
        
        # Initialize results storage
        self.frame_data = []
        self.eye_closure_history = []
        
        # Initialize video properties
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        
        logger.debug(f"AnalysisWorker initialized for {video_path}")
    
    def run(self):
        """Run the analysis thread."""
        try:
            self.is_running = True
            
            # Open video file
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error_occurred.emit(f"Could not open video file: {self.video_path}")
                return
            
            # Get video properties
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / self.fps
            
            logger.info(f"Video properties: {self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames, {self.duration:.2f}s")
            
            # Initialize components
            self.mediapipe_helper = get_mediapipe_helper()
            self.gaze_processor = GazeProcessor(self.config)
            self.drowsiness_detector = DrowsinessDetector()
            
            # Initialize PERCLOS calculation
            max_history_frames = int(self.config.get('detection', {}).get('perclos', {}).get('window_size', 150))
            ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
            
            # Process frames
            frame_number = 0
            processed_frames = 0
            
            while self.is_running and frame_number < self.frame_count:
                # Check if paused
                self.mutex.lock()
                is_paused = self.is_paused
                self.mutex.unlock()
                
                if is_paused:
                    time.sleep(0.1)
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_number / self.fps
                
                # Process frame
                landmarks, face_detected = self.mediapipe_helper.detect_face_landmarks(frame)
                
                # Initialize frame data
                frame_data = FrameData(
                    timestamp=timestamp,
                    frame_number=frame_number,
                    face_detected=face_detected
                )
                
                # Create a copy for visualization
                processed_frame = frame.copy()
                
                if face_detected:
                    # Get landmarks
                    frame_data.landmarks = landmarks
                    
                    # Get eye landmarks
                    left_eye_landmarks = self.mediapipe_helper.get_eye_landmarks(landmarks, left_eye=True)
                    right_eye_landmarks = self.mediapipe_helper.get_eye_landmarks(landmarks, left_eye=False)
                    
                    # Calculate EAR
                    left_ear = self.mediapipe_helper.get_eye_aspect_ratio(left_eye_landmarks)
                    right_ear = self.mediapipe_helper.get_eye_aspect_ratio(right_eye_landmarks)
                    ear = (left_ear + right_ear) / 2.0
                    frame_data.ear = ear
                    
                    # Calculate MAR
                    mar = self.mediapipe_helper.get_mouth_aspect_ratio(landmarks)
                    frame_data.mar = mar
                    
                    # Update eye closure history for PERCLOS
                    is_eye_closed = ear < ear_threshold
                    self.eye_closure_history.append(1 if is_eye_closed else 0)
                    
                    # Keep history within window size
                    if len(self.eye_closure_history) > max_history_frames:
                        self.eye_closure_history = self.eye_closure_history[-max_history_frames:]
                    
                    # Calculate PERCLOS
                    if self.eye_closure_history:
                        perclos = (sum(self.eye_closure_history) / len(self.eye_closure_history)) * 100.0
                        frame_data.perclos = perclos
                    
                    # Process gaze
                    gaze_data = self.gaze_processor.process_frame(frame, landmarks, timestamp)
                    frame_data.gaze_data = gaze_data
                    frame_data.is_distracted = gaze_data.is_distracted if gaze_data else False
                    
                    # Gaze Zone ID'sini kaydet
                    if gaze_data and gaze_data.zone_id is not None:
                        frame_data.gaze_zone_id = gaze_data.zone_id
                    
                    # Update drowsiness detection
                    drowsiness_result = self.drowsiness_detector.update(
                        ear_value=ear,
                        head_pose=None,
                        gaze_direction=None
                    )
                    frame_data.drowsiness_level = drowsiness_result['drowsiness_level']
                    frame_data.is_drowsy = frame_data.drowsiness_level > 0.5
                    
                    # Visualize results
                    # Draw landmarks
                    if self.config.get('visualization', {}).get('show_landmarks', False):
                        processed_frame = self.mediapipe_helper.draw_landmarks(processed_frame, landmarks)
                    
                    # Visualize drowsiness
                    processed_frame = self.drowsiness_detector.visualize(
                        processed_frame,
                        ear_left=left_ear,
                        ear_right=right_ear,
                        show_metrics=True
                    )
                    
                    # Visualize gaze
                    if gaze_data and gaze_data.gaze_vector is not None:
                        processed_frame = self.gaze_processor.visualize_gaze(processed_frame, landmarks, gaze_data)
                
                # Store frame data
                self.frame_data.append(frame_data)
                
                # Emit frame processed signal
                self.frame_processed.emit(frame_data, processed_frame)
                
                # Update progress
                processed_frames += 1
                progress = int((processed_frames / self.frame_count) * 100)
                self.progress_updated.emit(progress)
                
                # Increment frame number
                frame_number += 1
            
            # Calculate final statistics
            statistics = self._calculate_statistics()
            
            # Emit analysis completed signal
            self.analysis_completed.emit(statistics)
            
            # Clean up
            self.cap.release()
            
        except Exception as e:
            logger.error(f"Error in analysis worker: {str(e)}")
            self.error_occurred.emit(f"Analysis error: {str(e)}")
        
        finally:
            self.is_running = False
            
            # Clean up resources
            if self.cap and self.cap.isOpened():
                self.cap.release()
            
            if self.mediapipe_helper:
                self.mediapipe_helper.release()
            
            if self.gaze_processor:
                self.gaze_processor.release()
    
    def pause(self):
        """Pause the analysis."""
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
    
    def resume(self):
        """Resume the analysis."""
        self.mutex.lock()
        self.is_paused = False
        self.mutex.unlock()
    
    def stop(self):
        """Stop the analysis."""
        self.is_running = False
        self.wait()
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate analysis statistics.
        
        Returns:
            Dict: Statistics dictionary
        """
        if not self.frame_data:
            return {
                "total_frames": 0,
                "total_duration": 0,
                "face_detection_rate": 0,
                "avg_ear": 0,
                "min_ear": 0,
                "avg_mar": 0,
                "max_mar": 0,
                "avg_perclos": 0,
                "max_perclos": 0,
                "drowsiness_percentage": 0,
                "distraction_percentage": 0
            }
        
        # Calculate basic statistics
        total_frames = len(self.frame_data)
        face_detected_frames = sum(1 for data in self.frame_data if data.face_detected)
        face_detection_rate = (face_detected_frames / total_frames) * 100 if total_frames > 0 else 0
        
        # Filter frames with detected faces
        valid_frames = [data for data in self.frame_data if data.face_detected]
        if not valid_frames:
            return {
                "total_frames": total_frames,
                "total_duration": self.duration,
                "face_detection_rate": face_detection_rate,
                "avg_ear": 0,
                "min_ear": 0,
                "avg_mar": 0,
                "max_mar": 0,
                "avg_perclos": 0,
                "max_perclos": 0,
                "drowsiness_percentage": 0,
                "distraction_percentage": 0
            }
        
        # Calculate EAR statistics
        ear_values = [data.ear for data in valid_frames if data.ear is not None]
        avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0
        min_ear = min(ear_values) if ear_values else 0
        
        # Calculate MAR statistics
        mar_values = [data.mar for data in valid_frames if data.mar is not None]
        avg_mar = sum(mar_values) / len(mar_values) if mar_values else 0
        max_mar = max(mar_values) if mar_values else 0
        
        # Calculate PERCLOS statistics
        perclos_values = [data.perclos for data in valid_frames if data.perclos is not None]
        avg_perclos = sum(perclos_values) / len(perclos_values) if perclos_values else 0
        max_perclos = max(perclos_values) if perclos_values else 0
        
        # Calculate drowsiness percentage
        drowsy_frames = sum(1 for data in valid_frames if data.is_drowsy)
        drowsiness_percentage = (drowsy_frames / len(valid_frames)) * 100 if valid_frames else 0
        
        # Calculate distraction percentage
        distracted_frames = sum(1 for data in valid_frames if data.is_distracted)
        distraction_percentage = (distracted_frames / len(valid_frames)) * 100 if valid_frames else 0
        
        # Get gaze statistics
        gaze_stats = self.gaze_processor.calculate_statistics(self.fps) if self.gaze_processor else {}
        
        # Combine statistics
        statistics = {
            "total_frames": total_frames,
            "total_duration": self.duration,
            "face_detection_rate": face_detection_rate,
            "avg_ear": avg_ear,
            "min_ear": min_ear,
            "avg_mar": avg_mar,
            "max_mar": max_mar,
            "avg_perclos": avg_perclos,
            "max_perclos": max_perclos,
            "drowsiness_percentage": drowsiness_percentage,
            "distraction_percentage": distraction_percentage
        }
        
        # Add gaze statistics
        statistics.update(gaze_stats)
        
        return statistics

class VideoAnalyzer(QObject):
    """
    Class for analyzing videos for driver drowsiness.
    
    This class provides methods to:
    - Load and process video files
    - Analyze frames for drowsiness detection
    - Generate statistics and reports
    """
    
    # Define signals
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(FrameData, np.ndarray)
    analysis_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the VideoAnalyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config or {}
        self.worker = None
        self.video_path = None
        self.is_analyzing = False
        
        logger.debug("VideoAnalyzer initialized")
    
    def analyze_video(self, video_path: str) -> bool:
        """
        Start analyzing a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            bool: True if analysis started successfully, False otherwise
        """
        if self.is_analyzing:
            logger.warning("Analysis already in progress")
            return False
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            self.error_occurred.emit(f"Video file not found: {video_path}")
            return False
        
        try:
            # Store video path
            self.video_path = video_path
            
            # Create worker thread
            self.worker = AnalysisWorker(video_path, self.config)
            
            # Connect signals
            self.worker.progress_updated.connect(self.progress_updated)
            self.worker.frame_processed.connect(self.frame_processed)
            self.worker.analysis_completed.connect(self._on_analysis_completed)
            self.worker.error_occurred.connect(self.error_occurred)
            
            # Start worker thread
            self.worker.start()
            self.is_analyzing = True
            
            logger.info(f"Started analysis of video: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting analysis: {str(e)}")
            self.error_occurred.emit(f"Error starting analysis: {str(e)}")
            return False
    
    def pause_analysis(self):
        """Pause the analysis."""
        if self.worker and self.is_analyzing:
            self.worker.pause()
            logger.debug("Analysis paused")
    
    def resume_analysis(self):
        """Resume the analysis."""
        if self.worker and self.is_analyzing:
            self.worker.resume()
            logger.debug("Analysis resumed")
    
    def stop_analysis(self):
        """Stop the analysis."""
        if self.worker and self.is_analyzing:
            self.worker.stop()
            self.is_analyzing = False
            logger.debug("Analysis stopped")
    
    def _on_analysis_completed(self, statistics: Dict[str, Any]):
        """
        Handle analysis completion.
        
        Args:
            statistics: Analysis statistics
        """
        self.is_analyzing = False
        
        # Frame-Zone eşlemesi dosyasını oluştur
        if hasattr(self, 'worker') and self.worker and hasattr(self.worker, 'frame_data'):
            self._save_frame_zone_data()
        
        self.analysis_completed.emit(statistics)
        logger.info("Analysis completed")
    
    def _save_frame_zone_data(self) -> Optional[str]:
        """
        Her frame için gaze zone bilgilerini JSON olarak kaydet.
        
        Returns:
            str: Kaydedilen dosyanın yolu veya None
        """
        if not self.video_path or not hasattr(self, 'worker') or not self.worker:
            logger.warning("Cannot save frame zone data: No valid worker or video path")
            return None
        
        try:
            # Frame data'yı al
            frame_data_list = self.worker.frame_data
            
            if not frame_data_list:
                logger.warning("No frame data available to save")
                return None
            
            # Frame → Zone dictionary oluştur
            frame_to_zone = {}
            for data in frame_data_list:
                frame_to_zone[str(data.frame_number)] = data.gaze_zone_id
            
            # Dosya yolunu oluştur
            video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_dir = os.path.join(os.path.dirname(self.video_path), "analysis_results")
            os.makedirs(output_dir, exist_ok=True)
            
            prediction_path = os.path.join(output_dir, f"{video_name}_gaze_zones.json")
            
            # JSON olarak kaydet
            with open(prediction_path, 'w', encoding='utf-8') as f:
                json.dump(frame_to_zone, f, indent=2)
            
            logger.info(f"Frame-zone mapping saved to {prediction_path}")
            return prediction_path
            
        except Exception as e:
            logger.error(f"Error saving frame zone data: {str(e)}")
            return None
    
    def generate_report(self, statistics: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a report from analysis statistics.
        
        Args:
            statistics: Analysis statistics
            output_path: Path to save the report (optional)
            
        Returns:
            str: Report text or file path if saved
        """
        # Create report
        report = f"Video Analysis Report\n"
        report += f"==================\n\n"
        
        # Add video information
        report += f"Video Information:\n"
        report += f"- File: {os.path.basename(self.video_path)}\n"
        report += f"- Duration: {statistics.get('total_duration', 0):.2f} seconds\n"
        report += f"- Total frames: {statistics.get('total_frames', 0)}\n"
        report += f"- Face detection rate: {statistics.get('face_detection_rate', 0):.2f}%\n\n"
        
        # Add drowsiness statistics
        report += f"Drowsiness Analysis:\n"
        report += f"- Average EAR: {statistics.get('avg_ear', 0):.4f}\n"
        report += f"- Minimum EAR: {statistics.get('min_ear', 0):.4f}\n"
        report += f"- Average MAR: {statistics.get('avg_mar', 0):.4f}\n"
        report += f"- Maximum MAR: {statistics.get('max_mar', 0):.4f}\n"
        report += f"- Average PERCLOS: {statistics.get('avg_perclos', 0):.2f}%\n"
        report += f"- Maximum PERCLOS: {statistics.get('max_perclos', 0):.2f}%\n"
        report += f"- Drowsiness percentage: {statistics.get('drowsiness_percentage', 0):.2f}%\n\n"
        
        # Add distraction statistics
        report += f"Distraction Analysis:\n"
        report += f"- Distraction percentage: {statistics.get('distraction_percentage', 0):.2f}%\n"
        
        # Add zone statistics if available
        if 'zone_percentages' in statistics:
            report += f"- Gaze zone percentages:\n"
            zone_percentages = statistics.get('zone_percentages', {})
            for zone_id, percentage in zone_percentages.items():
                zone_name = self._get_zone_name(zone_id)
                report += f"  - {zone_name} ({zone_id}): {percentage:.2f}%\n"
        
        # Add distraction reasons if available
        if 'distraction_reasons' in statistics:
            report += f"- Distraction reasons:\n"
            distraction_reasons = statistics.get('distraction_reasons', {})
            for reason, percentage in distraction_reasons.items():
                report += f"  - {reason}: {percentage:.2f}%\n"
        
        # Save report if output path is provided
        if output_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Write report to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                logger.info(f"Report saved to {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
        
        return report
    
    def _get_zone_name(self, zone_id: int) -> str:
        """
        Get zone name from zone ID.
        
        Args:
            zone_id: Zone ID
            
        Returns:
            str: Zone name
        """
        from src.detection.gaze_zone_detector import get_gaze_zone_detector
        zone_detector = get_gaze_zone_detector()
        return zone_detector.get_zone_name(zone_id) 