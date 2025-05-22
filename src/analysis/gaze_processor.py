#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaze processor module for video analysis.

This module provides the GazeProcessor class that handles:
- Processing gaze data from video frames
- Calculating gaze metrics over time
- Detecting attention and distraction patterns
"""

import os
import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass

from src.utils.gaze_detector import GazeDetector, get_gaze_detector
from src.detection.gaze_zone_detector import get_gaze_zone_detector

# Get module-specific logger
logger = logging.getLogger(__name__)

@dataclass
class GazeData:
    """Data class for storing gaze analysis results."""
    timestamp: float
    gaze_vector: Optional[np.ndarray] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    zone_id: Optional[int] = None
    zone_name: Optional[str] = None
    is_distracted: bool = False
    distraction_reason: Optional[str] = None

class GazeProcessor:
    """
    Class for processing gaze data from video frames.
    
    This class provides methods to:
    - Process gaze data from video frames
    - Calculate gaze metrics over time
    - Detect attention and distraction patterns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the GazeProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.gaze_detector = get_gaze_detector()
        self.zone_detector = get_gaze_zone_detector()
        
        # Initialize results storage
        self.gaze_data = []
        self.zone_durations = {}
        self.zone_percentages = {}
        
        # Initialize thresholds
        self.max_pitch_deviation = self.config.get('max_pitch_deviation', 30.0)
        self.max_yaw_deviation = self.config.get('max_yaw_deviation', 40.0)
        
        logger.debug("GazeProcessor initialized")
    
    def process_frame(self, frame: np.ndarray, landmarks: List[List[float]], timestamp: float) -> GazeData:
        """
        Process a single frame to extract gaze data.
        
        Args:
            frame: Video frame
            landmarks: Facial landmarks
            timestamp: Frame timestamp in seconds
            
        Returns:
            GazeData: Processed gaze data
        """
        try:
            # Predict gaze
            gaze_vector, _ = self.gaze_detector.predict_gaze(frame, landmarks)
            
            if gaze_vector is not None:
                # Convert to degrees
                pitch, yaw = np.rad2deg(gaze_vector)
                
                # Get gaze zone
                zone_id = self.zone_detector.get_gaze_target_zone(gaze_vector)
                zone_name = self.zone_detector.get_zone_name(zone_id)
                
                # Check if distracted
                is_distracted = not self.gaze_detector.is_looking_forward(
                    gaze_vector, 
                    max_pitch_deg=self.max_pitch_deviation,
                    max_yaw_deg=self.max_yaw_deviation
                )
                
                # Determine distraction reason
                distraction_reason = None
                if is_distracted:
                    if abs(pitch) > self.max_pitch_deviation:
                        if pitch > 0:
                            distraction_reason = "Looking up"
                        else:
                            distraction_reason = "Looking down"
                    elif abs(yaw) > self.max_yaw_deviation:
                        if yaw > 0:
                            distraction_reason = "Looking right"
                        else:
                            distraction_reason = "Looking left"
                
                # Create and store gaze data
                gaze_data = GazeData(
                    timestamp=timestamp,
                    gaze_vector=gaze_vector,
                    pitch=pitch,
                    yaw=yaw,
                    zone_id=zone_id,
                    zone_name=zone_name,
                    is_distracted=is_distracted,
                    distraction_reason=distraction_reason
                )
                
                self.gaze_data.append(gaze_data)
                
                # Update zone statistics
                if zone_id is not None:
                    if zone_id not in self.zone_durations:
                        self.zone_durations[zone_id] = 0
                    self.zone_durations[zone_id] += 1  # Count frames for now, will convert to time later
                
                return gaze_data
            
        except Exception as e:
            logger.error(f"Error processing gaze: {str(e)}")
        
        # Return empty data if processing failed
        return GazeData(timestamp=timestamp)
    
    def visualize_gaze(self, frame: np.ndarray, landmarks: List[List[float]], gaze_data: GazeData) -> np.ndarray:
        """
        Visualize gaze on the frame.
        
        Args:
            frame: Video frame
            landmarks: Facial landmarks
            gaze_data: Gaze data
            
        Returns:
            np.ndarray: Frame with visualized gaze
        """
        if gaze_data.gaze_vector is not None:
            # Get face center for gaze origin
            face_landmarks = np.array(landmarks)
            face_center = np.mean(face_landmarks, axis=0).astype(int)
            
            # Draw gaze vector
            processed_frame = self.gaze_detector.draw_gaze(
                frame.copy(),
                gaze_data.gaze_vector,
                (face_center[0], face_center[1]),
                length=100,
                thickness=2,
                color=(0, 255, 255) if gaze_data.is_distracted else (0, 255, 0)
            )
            
            # Add gaze information text
            cv2.putText(
                processed_frame,
                f"Pitch: {gaze_data.pitch:.1f}° Yaw: {gaze_data.yaw:.1f}°",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1
            )
            
            # Add zone information
            if gaze_data.zone_id is not None:
                cv2.putText(
                    processed_frame,
                    f"Zone: {gaze_data.zone_name} ({gaze_data.zone_id})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1
                )
            
            # Add distraction warning if distracted
            if gaze_data.is_distracted:
                cv2.putText(
                    processed_frame,
                    f"DISTRACTED: {gaze_data.distraction_reason}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            return processed_frame
        
        return frame
    
    def calculate_statistics(self, fps: float = 30.0) -> Dict[str, Any]:
        """
        Calculate gaze statistics.
        
        Args:
            fps: Video frames per second
            
        Returns:
            Dict: Statistics dictionary
        """
        if not self.gaze_data:
            return {
                "total_frames": 0,
                "total_duration": 0,
                "distraction_percentage": 0,
                "zone_durations": {},
                "zone_percentages": {}
            }
        
        # Calculate total frames and duration
        total_frames = len(self.gaze_data)
        total_duration = total_frames / fps
        
        # Calculate distraction percentage
        distracted_frames = sum(1 for data in self.gaze_data if data.is_distracted)
        distraction_percentage = (distracted_frames / total_frames) * 100 if total_frames > 0 else 0
        
        # Calculate zone durations and percentages
        zone_durations = {zone_id: frames / fps for zone_id, frames in self.zone_durations.items()}
        zone_percentages = {zone_id: (frames / total_frames) * 100 for zone_id, frames in self.zone_durations.items()}
        
        # Store zone statistics
        self.zone_durations = zone_durations
        self.zone_percentages = zone_percentages
        
        # Get distraction reasons
        distraction_reasons = {}
        for data in self.gaze_data:
            if data.is_distracted and data.distraction_reason:
                reason = data.distraction_reason
                if reason not in distraction_reasons:
                    distraction_reasons[reason] = 0
                distraction_reasons[reason] += 1
        
        # Convert to percentages
        distraction_reasons = {reason: (count / total_frames) * 100 for reason, count in distraction_reasons.items()}
        
        return {
            "total_frames": total_frames,
            "total_duration": total_duration,
            "distraction_percentage": distraction_percentage,
            "zone_durations": zone_durations,
            "zone_percentages": zone_percentages,
            "distraction_reasons": distraction_reasons
        }
    
    def reset(self):
        """Reset the processor state."""
        self.gaze_data = []
        self.zone_durations = {}
        self.zone_percentages = {}
    
    def release(self):
        """Release resources."""
        if self.gaze_detector:
            self.gaze_detector.release() 