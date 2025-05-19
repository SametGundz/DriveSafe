#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera handler for the driver drowsiness detection system.

This module handles camera operations, video processing, and facial landmark detection.
"""

import cv2
import time
import logging
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from src.utils import MediaPipeHelper, get_mediapipe_helper, get_ear_calibrator
from src.detection.drowsiness_detector import DrowsinessDetector


class CameraHandler(QObject):
    """
    Handler for camera operations and video processing.
    
    This class handles camera initialization, frame processing,
    facial landmark detection, and metrics calculation.
    """
    
    # Define signals
    status_updated = pyqtSignal(str)
    capture_started = pyqtSignal(bool)
    capture_stopped = pyqtSignal()
    metrics_updated = pyqtSignal(float, float, float)
    head_pose_updated = pyqtSignal(float, float, float)  # pitch, yaw, roll
    alert_triggered = pyqtSignal(str)
    frame_processed = pyqtSignal(dict)
    
    def __init__(self, video_panel, metrics_panel, config):
        """
        Initialize the camera handler.
        
        Args:
            video_panel: The video panel widget
            metrics_panel: The metrics panel widget
            config: Configuration dictionary
        """
        super().__init__()
        
        # Store references to UI components
        self.video_panel = video_panel
        self.metrics_panel = metrics_panel
        self.config = config
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Camera variables
        self.cap = None
        self.is_capturing = False
        self.camera_id = self.config.get('camera', {}).get('device_id', 0)
        
        # MediaPipe helper
        self.mediapipe_helper = None
        
        # Display options
        self.show_landmarks = False
        self.show_head_pose = False
        self.show_gaze = False
        
        # EAR calibrator
        self.ear_calibrator = get_ear_calibrator()
        
        # Drowsiness detector
        self.drowsiness_detector = DrowsinessDetector()
        
        # Frame counter for debugging
        self._frame_counter = 0
        
        # Eye closure history for PERCLOS
        self.eye_closure_history = []
        self.max_history_frames = int(self.config.get('detection', {}).get('perclos_window_sec', 60) * 30)
        
        # Camera timer
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.update_interval_ms = 33  # ~30 FPS
        
        # Head pose values
        self.head_pitch = 0.0
        self.head_yaw = 0.0
        self.head_roll = 0.0

        # Kamera başlangıçta kapalı olduğunu belirt
        self.video_panel.setText("Kamera kapalı. Başlatmak için 'Başlat' düğmesine tıklayın.")
        self.logger.info("Camera handler initialized with camera inactive")
    
    def start_capture(self):
        """Start camera capture."""
        if self.is_capturing:
            return
        
        # Try to open the camera
        try:
            self.logger.info(f"Opening camera ID: {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties from config
            camera_width = self.config.get('camera', {}).get('width', 640)
            camera_height = self.config.get('camera', {}).get('height', 480)
            camera_fps = self.config.get('camera', {}).get('fps', 30)
            
            self.logger.info(f"Setting camera properties: width={camera_width}, height={camera_height}, fps={camera_fps}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, camera_fps)
            
            if not self.cap.isOpened():
                error_msg = "Kamera açılamadı! Kamera bağlantısını kontrol edin."
                self.logger.error(error_msg)
                self.status_updated.emit(error_msg)
                return
            
            # Initialize MediaPipe
            self.mediapipe_helper = get_mediapipe_helper()
            
            # Get actual camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Actual camera properties: width={actual_width}, height={actual_height}, fps={actual_fps}")
            
            # Start the camera timer
            self.is_capturing = True
            self.camera_timer.start(self.update_interval_ms)
            
            # Reset PERCLOS calculation
            self.eye_closure_history = []
            # Initialize with open eyes
            for _ in range(min(20, self.max_history_frames)):
                self.eye_closure_history.append(0)  # 0 = open eye
            
            self.status_updated.emit("Algılama başlatıldı.")
            self.capture_started.emit(True)
            self.logger.info("Camera capture started successfully")
            
        except Exception as e:
            error_msg = f"Kamera başlatılamadı: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.status_updated.emit(error_msg)
    
    def stop_capture(self):
        """Stop camera capture."""
        # Stop the camera timer
        self.is_capturing = False
        self.camera_timer.stop()
        
        # Release the camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Release MediaPipe resources
        if self.mediapipe_helper is not None:
            self.mediapipe_helper.release()
            self.mediapipe_helper = None
        
        # Clear the video frame
        self.video_panel.setText("Kamera görüntüsü burada gösterilecek")
        
        self.status_updated.emit("Algılama durduruldu.")
        self.capture_stopped.emit()
        self.logger.info("Camera capture stopped")
    
    def toggle_landmarks(self, checked):
        """
        Toggle visibility of facial landmarks.
        
        Args:
            checked: Whether landmarks should be shown
        """
        self.show_landmarks = checked
        status = "gösteriliyor" if self.show_landmarks else "gizleniyor"
        self.status_updated.emit(f"Yüz işaretleri {status}")
    
    def toggle_head_pose(self, checked):
        """
        Toggle the display of head pose.
        
        Args:
            checked: Whether head pose should be shown
        """
        old_value = self.show_head_pose
        self.show_head_pose = checked
        self.logger.info(f"Head pose visualization toggled from {old_value} to {self.show_head_pose}")
        
        if self.show_head_pose:
            status_msg = "Baş duruşu gösteriliyor"
            if self.mediapipe_helper and not self.mediapipe_helper.can_detect_head_pose():
                status_msg += " (UYARI: Head pose estimator hazır değil!)"
        else:
            status_msg = "Baş duruşu gizleniyor"
            
        self.status_updated.emit(status_msg)
    
    def toggle_gaze(self, checked):
        """
        Toggle the display of gaze direction.
        
        Args:
            checked: Whether gaze direction should be shown
        """
        self.show_gaze = checked
        status = "gösteriliyor" if self.show_gaze else "gizleniyor"
        self.status_updated.emit(f"Bakış yönü {status}")
    
    def _update_camera_frame(self):
        """Update the video frame with current camera image."""
        if not self.cap or not self.is_capturing:
            return
        
        # Start timing for FPS calculation
        frame_start_time = time.time()
        
        # Read a frame from the camera
        ret, frame = self.cap.read()
        
        if not ret:
            error_msg = "Kameradan görüntü alınamadı!"
            self.logger.error(error_msg)
            self.status_updated.emit(error_msg)
            self.stop_capture()
            return
        
        # Print frame info occasionally
        if self._frame_counter % 30 == 0:
            self.logger.debug(f"Frame received - shape: {frame.shape}")
        self._frame_counter += 1
        
        # Mirror the frame horizontally (selfie view)
        frame = cv2.flip(frame, 1)
        
        # Process the frame with MediaPipe
        landmarks, face_detected = self.mediapipe_helper.detect_face_landmarks(frame)
        
        # Initialize metrics with default values
        ear = 0.0
        normalized_ear = 0.0
        mar = 0.0
        perclos = 0.0
        
        # Variables for drowsiness detection
        left_ear = 0.0
        right_ear = 0.0
        
        if face_detected:
            # Get eye landmarks
            left_eye_landmarks = self.mediapipe_helper.get_eye_landmarks(landmarks, left_eye=True)
            right_eye_landmarks = self.mediapipe_helper.get_eye_landmarks(landmarks, left_eye=False)
            
            # Get mouth landmarks - only inner lip
            inner_lip_landmarks = self.mediapipe_helper.get_inner_lip_landmarks(landmarks)
            
            # Calculate metrics
            left_ear = self.mediapipe_helper.get_eye_aspect_ratio(left_eye_landmarks)
            right_ear = self.mediapipe_helper.get_eye_aspect_ratio(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2.0  # Average EAR
            
            # Calculate normalized EAR
            normalized_ear = self.ear_calibrator.update(ear)
            
            # Calculate MAR
            mar = self.mediapipe_helper.get_mouth_aspect_ratio(landmarks)
            
            # Determine eye closure state
            ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
            if self.ear_calibrator.is_calibrated:
                # Use normalized value with threshold 0.3
                is_eye_closed = normalized_ear < 0.3
            else:
                # Use absolute EAR value
                is_eye_closed = ear < ear_threshold
            
            # Update eye closure history
            self.eye_closure_history.append(1 if is_eye_closed else 0)
            
            # Keep history within window size
            if len(self.eye_closure_history) > self.max_history_frames:
                self.eye_closure_history.pop(0)
            
            # Calculate PERCLOS
            perclos = 0.0
            if self.eye_closure_history:
                # Minimum required history check
                min_history_frames = min(10, self.max_history_frames // 10)
                if len(self.eye_closure_history) >= min_history_frames:
                    perclos = (sum(self.eye_closure_history) / len(self.eye_closure_history)) * 100.0
            
            # Create a copy of the frame for processing
            processed_frame = frame.copy()
            
            # Detect head pose
            if self.mediapipe_helper.can_detect_head_pose():
                try:
                    # Get head pose from landmarks
                    pitch, yaw, roll = self.mediapipe_helper.get_head_pose(landmarks, frame.shape)
                    
                    # Store head pose values
                    self.head_pitch = pitch
                    self.head_yaw = yaw
                    self.head_roll = roll
                    
                    # DEBUG: Log head pose values
                    if self._frame_counter % 30 == 0:
                        self.logger.debug(f"Head pose: pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f}")
                    
                    # Emit head pose signal for 3D model
                    self.head_pose_updated.emit(pitch, yaw, roll)
                    
                    # Draw head pose on frame if enabled
                    if self.show_head_pose:
                        # Uyarı olarak loglayalım
                        if self._frame_counter % 100 == 0:
                            self.logger.info(f"Drawing head pose: show_head_pose={self.show_head_pose}")
                        
                        # Orijinal ve işlenmiş karelerin her ikisine de çizim yap
                        frame = self.mediapipe_helper.draw_head_pose(frame, landmarks, pitch, yaw, roll)
                        processed_frame = self.mediapipe_helper.draw_head_pose(processed_frame, landmarks, pitch, yaw, roll)
                except Exception as e:
                    self.logger.error(f"Error processing head pose: {str(e)}", exc_info=True)
            else:
                # No head pose estimator available - still emit default values
                self.head_pose_updated.emit(0.0, 0.0, 0.0)
                if self._frame_counter % 30 == 0:
                    self.logger.warning("Head pose estimation not available")
            
            # Visualize gaze direction if enabled
            if self.show_gaze:
                try:
                    ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
                    frame_skip = self.config.get('detection', {}).get('gaze', {}).get('frame_skip', 3)
                    processed_frame, normalized_face = self.mediapipe_helper.visualize_gaze(
                        processed_frame, 
                        landmarks,
                        ear_value=ear,
                        ear_threshold=ear_threshold,
                        frame_skip=frame_skip
                    )
                    
                    # Show normalized face in small window if available
                    if normalized_face is not None:
                        norm_face_display = cv2.resize(normalized_face, (112, 112))
                        h, w = norm_face_display.shape[:2]
                        processed_frame[10:10+h, processed_frame.shape[1]-w-10:processed_frame.shape[1]-10] = norm_face_display
                except Exception as e:
                    self.logger.error(f"Error visualizing gaze: {str(e)}")
            
            # Draw landmarks if enabled
            if self.show_landmarks:
                # Create connections for eyes (to form a polygon)
                left_eye_connections = [(i, i+1) for i in range(len(left_eye_landmarks)-1)]
                left_eye_connections.append((len(left_eye_landmarks)-1, 0))  # Close the loop
                
                right_eye_connections = [(i, i+1) for i in range(len(right_eye_landmarks)-1)]
                right_eye_connections.append((len(right_eye_landmarks)-1, 0))  # Close the loop
                
                # Create connections for inner lip (to form a polygon)
                inner_lip_connections = [(i, i+1) for i in range(len(inner_lip_landmarks)-1)]
                inner_lip_connections.append((len(inner_lip_landmarks)-1, 0))  # Close the loop
                
                # Draw eye landmarks and connections
                processed_frame = self.mediapipe_helper.draw_facial_landmarks(
                    processed_frame, left_eye_landmarks, 
                    connections=left_eye_connections,
                    landmark_color=(0, 255, 0), 
                    connection_color=(0, 255, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
                processed_frame = self.mediapipe_helper.draw_facial_landmarks(
                    processed_frame, right_eye_landmarks, 
                    connections=right_eye_connections,
                    landmark_color=(0, 255, 0), 
                    connection_color=(0, 255, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
                
                # Draw only inner lip landmarks and connections
                processed_frame = self.mediapipe_helper.draw_facial_landmarks(
                    processed_frame, inner_lip_landmarks, 
                    connections=inner_lip_connections,
                    landmark_color=(255, 0, 0), 
                    connection_color=(255, 0, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
            
            # Update drowsiness detector
            drowsiness_result = self.drowsiness_detector.update(
                ear_left=left_ear,
                ear_right=right_ear
            )
            
            # Visualize drowsiness detection
            processed_frame = self.drowsiness_detector.visualize(
                processed_frame, 
                ear_left=left_ear, 
                ear_right=right_ear,
                show_metrics=True
            )
            
            # Use processed frame
            frame = processed_frame
        
        # Calculate FPS
        frame_processing_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
        
        # Show FPS if enabled
        if self.config.get('visualization', {}).get('show_fps', True):
            # Add FPS text (bottom left corner)
            cv2.putText(
                frame, 
                f"FPS: {current_fps:.1f}", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
        
        # Display the frame
        self.video_panel.update_frame(frame)
        
        # Update metrics UI
        if self.ear_calibrator.is_calibrated:
            # Show normalized EAR value
            self.metrics_panel.update_metrics(normalized_ear, mar, perclos, is_normalized=True)
        else:
            # Show raw EAR value
            self.metrics_panel.update_metrics(ear, mar, perclos, is_normalized=False)
        
        # Emit metrics for chart updates
        self.metrics_updated.emit(
            normalized_ear if self.ear_calibrator.is_calibrated else ear,
            mar,
            perclos
        )
        
        # Emit frame processed signal
        self.frame_processed.emit({
            'ear': normalized_ear if self.ear_calibrator.is_calibrated else ear,
            'mar': mar,
            'perclos': perclos,
            'face_detected': face_detected,
            'pitch': self.head_pitch,
            'yaw': self.head_yaw,
            'roll': self.head_roll
        })
    
    def get_calibration_overlay(self, frame, status):
        """
        Add calibration overlay to the frame.
        
        Args:
            frame: The frame to add overlay to
            status: Calibration status dictionary
            
        Returns:
            frame: Frame with calibration overlay
        """
        progress = int(status["progress"] * 100)
        
        # Add calibration information
        cv2.putText(
            frame,
            f"Kalibrasyon: %{progress}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        
        # Add progress bar
        bar_width = 300
        bar_height = 20
        filled_width = int(bar_width * status["progress"])
        
        # Progress bar background
        cv2.rectangle(
            frame,
            (20, 60),
            (20 + bar_width, 60 + bar_height),
            (100, 100, 100),
            -1
        )
        
        # Progress bar fill
        cv2.rectangle(
            frame,
            (20, 60),
            (20 + filled_width, 60 + bar_height),
            (0, 255, 0),
            -1
        )
        
        # Show EAR values - current, min and max
        ear_value = self.ear_calibrator.ear_values[-1] if self.ear_calibrator.ear_values else 0.0
        
        cv2.putText(
            frame,
            f"Anlık EAR: {ear_value:.3f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Show raw min/max values
        if hasattr(self.ear_calibrator, 'ear_values') and self.ear_calibrator.ear_values:
            raw_values = self.ear_calibrator.ear_values + self.ear_calibrator._temp_buffer
            if raw_values:
                min_raw = min(raw_values)
                max_raw = max(raw_values)
                
                cv2.putText(
                    frame,
                    f"Min EAR (Ham): {min_raw:.3f}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                cv2.putText(
                    frame,
                    f"Max EAR (Ham): {max_raw:.3f}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
        
        # Show filtered values
        cv2.putText(
            frame,
            f"Min EAR (Filtreli): {status['min_ear']:.3f}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            2
        )
        
        cv2.putText(
            frame,
            f"Max EAR (Filtreli): {status['max_ear']:.3f}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            2
        )
        
        # Show sample count
        cv2.putText(
            frame,
            f"Örnek sayısı: {status['sample_count']}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )
        
        return frame
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_capture()