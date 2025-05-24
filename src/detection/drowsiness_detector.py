#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drowsiness detector module.

This module provides functionality to detect drowsiness based on various metrics:
- Eye closure duration (using Eye Aspect Ratio)
- PERCLOS (percentage of eye closure time)
- Head pose (indicating nodding)
- Gaze direction
"""

import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2

from src.utils.constants import (
    EAR_THRESHOLD,
    MAR_THRESHOLD,
    HEAD_YAW_THRESHOLD,
    HEAD_PITCH_THRESHOLD,
    GAZE_YAW_THRESHOLD,
    GAZE_PITCH_THRESHOLD,
    PERCLOS_WARNING_THRESHOLD,
    PERCLOS_CRITICAL_THRESHOLD
)
from src.utils.facial_metrics import calculate_distance, get_eye_aspect_ratio, get_perclos

# Get module-specific logger
logger = logging.getLogger(__name__)

@dataclass
class DrowsinessState:
    """Data class to track drowsiness detection state."""
    
    # Eye closure tracking
    ear_values: List[float] = field(default_factory=list)
    eyes_closed_start_time: Optional[float] = None
    eyes_closed_duration: float = 0.0
    
    # PERCLOS tracking
    is_eyes_closed: bool = False
    closed_time: float = 0.0
    total_time: float = 0.0
    perclos: float = 0.0
    
    # Various detection flags
    is_drowsy_by_ear: bool = False
    is_drowsy_by_perclos: bool = False
    is_drowsy_by_head_pose: bool = False
    is_drowsy_by_gaze: bool = False
    
    # Alert status
    alert_active: bool = False
    alert_start_time: Optional[float] = None
    

class DrowsinessDetector:
    """
    Detects driver drowsiness using multiple indicators.
    
    This class combines various drowsiness indicators:
    - Eye Aspect Ratio (EAR) for eye closure detection
    - PERCLOS (percentage of eye closure over time)
    - Head pose for detecting nodding
    - Gaze direction for detecting attention loss
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the drowsiness detector.
        
        Args:
            config: Optional configuration parameters
        """
        # Load config or use defaults
        self.config = config or {}
        
        # Parameters for EAR-based detection
        self.ear_threshold = self.config.get('ear_threshold', EAR_THRESHOLD)
        self.ear_time_threshold = self.config.get('ear_time_threshold', 2.0)  # seconds
        
        # Parameters for PERCLOS-based detection
        self.perclos_window = self.config.get('perclos_window', 60.0)  # seconds
        self.perclos_warning_threshold = self.config.get('perclos_warning_threshold', PERCLOS_WARNING_THRESHOLD)
        self.perclos_critical_threshold = self.config.get('perclos_critical_threshold', PERCLOS_CRITICAL_THRESHOLD)
        
        # Parameters for head pose detection
        self.head_yaw_threshold = self.config.get('head_yaw_threshold', HEAD_YAW_THRESHOLD)
        self.head_pitch_threshold = self.config.get('head_pitch_threshold', HEAD_PITCH_THRESHOLD)
        
        # Parameters for gaze detection
        self.gaze_yaw_threshold = self.config.get('gaze_yaw_threshold', GAZE_YAW_THRESHOLD)
        self.gaze_pitch_threshold = self.config.get('gaze_pitch_threshold', GAZE_PITCH_THRESHOLD)
        
        # Alert management
        self.alert_cooldown = self.config.get('alert_cooldown', 5.0)  # seconds
        
        # State tracking
        self.state = DrowsinessState()
        
        logger.info(f"DrowsinessDetector initialized with thresholds: EAR={self.ear_threshold}, "
                   f"PERCLOS warning={self.perclos_warning_threshold}, critical={self.perclos_critical_threshold}")
    
    def update(self, ear_value: Optional[float] = None,
              head_pose: Optional[Tuple[float, float, float]] = None,
              gaze_direction: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Update the drowsiness detection state with new data.
        
        Args:
            ear_value: Current Eye Aspect Ratio value
            head_pose: Current head pose as (roll, pitch, yaw)
            gaze_direction: Current gaze direction as (yaw, pitch)
            
        Returns:
            dict: Current drowsiness state
        """
        current_time = time.time()
        
        # Process EAR if provided
        if ear_value is not None:
            # Add to history
            self.state.ear_values.append(ear_value)
            
            # Update closed eyes state
            self._update_eye_closure_state(ear_value, current_time)
            
            # Update PERCLOS
            self._update_perclos(current_time)
            
            # Check if drowsy by EAR
            self._update_ear_drowsiness(current_time)
            
            # Check if drowsy by PERCLOS
            self._update_perclos_drowsiness()
            
            logger.debug(f"Updated drowsiness metrics: EAR={ear_value:.3f}, PERCLOS={self.state.perclos:.3f}, "
                        f"Eyes closed duration={self.state.eyes_closed_duration:.2f}s")
        
        # Process head pose if provided
        if head_pose is not None:
            roll, pitch, yaw = head_pose
            self._update_head_pose_drowsiness(pitch, yaw)
            logger.debug(f"Head pose: roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")
        
        # Process gaze direction if provided
        if gaze_direction is not None:
            gaze_yaw, gaze_pitch = gaze_direction
            self._update_gaze_drowsiness(gaze_pitch, gaze_yaw)
            logger.debug(f"Gaze direction: yaw={gaze_yaw:.1f}, pitch={gaze_pitch:.1f}")
        
        # Calculate overall drowsiness level
        drowsiness_level = self._calculate_drowsiness_level()
        
        # Determine drowsiness state based on level
        if drowsiness_level < 0.3:
            drowsiness_state = "Alert"
            drowsiness_color = (0, 255, 0)  # Green
        elif drowsiness_level < 0.6:
            drowsiness_state = "Tired"
            drowsiness_color = (0, 165, 255)  # Orange
        elif drowsiness_level < 0.8:
            drowsiness_state = "Drowsy"
            drowsiness_color = (0, 0, 255)  # Red
        else:
            drowsiness_state = "Danger"
            drowsiness_color = (0, 0, 255)  # Red
        
        # Update alert status
        self._update_alert_status(drowsiness_level, current_time)
        
        # Log drowsiness state if alert is active
        if self.state.alert_active:
            logger.warning(f"DROWSINESS ALERT! Level: {drowsiness_level:.2f}, State: {drowsiness_state}, EAR: {ear_value:.2f}, "
                          f"PERCLOS: {self.state.perclos:.2f}")
        
        # Return current state
        return {
            'drowsiness_level': drowsiness_level,
            'drowsiness_state': drowsiness_state,
            'ear_value': ear_value,
            'perclos': self.state.perclos,
            'eyes_closed_duration': self.state.eyes_closed_duration,
            'alert_active': self.state.alert_active,
            'is_drowsy_by_ear': self.state.is_drowsy_by_ear,
            'is_drowsy_by_perclos': self.state.is_drowsy_by_perclos,
            'is_drowsy_by_head_pose': self.state.is_drowsy_by_head_pose,
            'is_drowsy_by_gaze': self.state.is_drowsy_by_gaze
        }
    
    def _update_eye_closure_state(self, ear_value: float, current_time: float) -> None:
        """
        Update eye closure state based on current EAR value.
        
        Args:
            ear_value: Current Eye Aspect Ratio value
            current_time: Current timestamp
        """
        # Consider eyes closed if EAR is below threshold
        is_eyes_closed_now = ear_value < self.ear_threshold
        
        # Track eyes closed duration
        if is_eyes_closed_now:
            if not self.state.is_eyes_closed:
                # Eyes just closed
                self.state.eyes_closed_start_time = current_time
                logger.debug("Eyes closed detected")
            
            if self.state.eyes_closed_start_time is not None:
                self.state.eyes_closed_duration = current_time - self.state.eyes_closed_start_time
        else:
            # Eyes are open
            self.state.eyes_closed_start_time = None
            self.state.eyes_closed_duration = 0.0
            
            if self.state.is_eyes_closed:
                logger.debug("Eyes opened detected")
        
        self.state.is_eyes_closed = is_eyes_closed_now
    
    def _update_perclos(self, current_time: float) -> None:
        """
        Update PERCLOS (percentage of eye closure over time).
        
        Args:
            current_time: Current timestamp
        """
        # Update closed/total time counters
        time_step = 1.0 / 30.0  # Assume 30 FPS if not calculable
        
        # If multiple frames available in history, calculate actual time step
        if len(self.state.ear_values) >= 2:
            time_step = 1.0 / 30.0  # Default to 30 FPS
        
        if self.state.is_eyes_closed:
            self.state.closed_time += time_step
        
        self.state.total_time += time_step
        
        # Calculate PERCLOS over the configured window
        if self.state.total_time > self.perclos_window:
            # Remove excess time from both counters
            excess_time = self.state.total_time - self.perclos_window
            self.state.total_time = self.perclos_window
            
            # Assume ratio of closed time in excess time is same as overall
            closed_time_to_remove = (excess_time / self.state.total_time) * self.state.closed_time
            self.state.closed_time = max(0, self.state.closed_time - closed_time_to_remove)
        
        # Calculate PERCLOS as percentage
        if self.state.total_time > 0:
            self.state.perclos = (self.state.closed_time / self.state.total_time) * 100.0
    
    def _update_ear_drowsiness(self, current_time: float) -> None:
        """
        Update drowsiness detection based on EAR values.
        
        Args:
            current_time: Current timestamp
        """
        # Detect drowsiness based on eyes closed duration
        if self.state.eyes_closed_duration >= self.ear_time_threshold:
            if not self.state.is_drowsy_by_ear:
                self.state.is_drowsy_by_ear = True
                logger.warning(f"Drowsiness detected by EAR: {self.state.eyes_closed_duration:.2f}s eyes closed")
        else:
            self.state.is_drowsy_by_ear = False
    
    def _update_perclos_drowsiness(self) -> None:
        """Update drowsiness detection based on PERCLOS value."""
        previous_state = self.state.is_drowsy_by_perclos
        
        # Check if PERCLOS exceeds threshold
        if self.state.perclos >= self.perclos_critical_threshold:
            self.state.is_drowsy_by_perclos = True
            if not previous_state:
                logger.warning(f"Drowsiness detected by PERCLOS: {self.state.perclos:.2f}% > {self.perclos_critical_threshold}%")
        elif self.state.perclos < self.perclos_warning_threshold:
            self.state.is_drowsy_by_perclos = False
            if previous_state:
                logger.info(f"PERCLOS returned to normal: {self.state.perclos:.2f}%")
    
    def _update_head_pose_drowsiness(self, pitch: float, yaw: float) -> None:
        """
        Update drowsiness detection based on head pose.
        
        Args:
            pitch: Head pitch angle in degrees
            yaw: Head yaw angle in degrees
        """
        previous_state = self.state.is_drowsy_by_head_pose
        
        # Check for head nodding (looking down too much)
        if pitch > self.head_pitch_threshold:
            self.state.is_drowsy_by_head_pose = True
            if not previous_state:
                logger.warning(f"Drowsiness detected by head pose: pitch={pitch:.1f}° > {self.head_pitch_threshold}°")
        else:
            self.state.is_drowsy_by_head_pose = False
    
    def _update_gaze_drowsiness(self, gaze_pitch: float, gaze_yaw: float) -> None:
        """
        Update drowsiness detection based on gaze direction.
        
        Args:
            gaze_pitch: Gaze pitch angle in degrees
            gaze_yaw: Gaze yaw angle in degrees
        """
        previous_state = self.state.is_drowsy_by_gaze
        
        # Check for gaze indicating drowsiness (looking down too much)
        if abs(gaze_pitch) > self.gaze_pitch_threshold or abs(gaze_yaw) > self.gaze_yaw_threshold:
            self.state.is_drowsy_by_gaze = True
            if not previous_state:
                logger.warning(f"Drowsiness detected by gaze: pitch={gaze_pitch:.1f}°, yaw={gaze_yaw:.1f}°")
        else:
            self.state.is_drowsy_by_gaze = False
    
    def _calculate_drowsiness_level(self) -> float:
        """
        Calculate overall drowsiness level from various indicators.
        
        Returns:
            float: Drowsiness level between 0.0 and 1.0
        """
        # Weights for each drowsiness factor
        ear_weight = 0.4
        perclos_weight = 0.4
        head_pose_weight = 0.1
        gaze_weight = 0.1
        
        # Calculate normalized PERCLOS factor (0.0 to 1.0)
        perclos_factor = min(1.0, self.state.perclos / self.perclos_critical_threshold)
        
        # Calculate normalized EAR factor (0.0 to 1.0)
        ear_factor = min(1.0, self.state.eyes_closed_duration / self.ear_time_threshold)
        
        # Calculate combined drowsiness level
        drowsiness_level = (
            ear_weight * ear_factor +
            perclos_weight * perclos_factor +
            head_pose_weight * (1.0 if self.state.is_drowsy_by_head_pose else 0.0) +
            gaze_weight * (1.0 if self.state.is_drowsy_by_gaze else 0.0)
        )
        
        return drowsiness_level
    
    def _update_alert_status(self, drowsiness_level: float, current_time: float) -> None:
        """
        Update drowsiness alert status.
        
        Args:
            drowsiness_level: Current drowsiness level (0.0 to 1.0)
            current_time: Current timestamp
        """
        # Drowsiness level threshold for alert
        alert_threshold = 0.5
        
        previous_alert_state = self.state.alert_active
        
        # Activate alert if drowsiness level exceeds threshold
        if drowsiness_level >= alert_threshold:
            if not self.state.alert_active:
                self.state.alert_active = True
                self.state.alert_start_time = current_time
                logger.warning(f"DROWSINESS ALERT ACTIVATED: Level={drowsiness_level:.2f}")
        else:
            # Check if alert cooldown has passed
            if (self.state.alert_active and 
                self.state.alert_start_time is not None and
                current_time - self.state.alert_start_time > self.alert_cooldown):
                self.state.alert_active = False
                self.state.alert_start_time = None
                logger.info(f"Drowsiness alert deactivated: Level={drowsiness_level:.2f}")
    
    def compute_ear_from_landmarks(self, left_eye_landmarks: List[List[float]], 
                                   right_eye_landmarks: List[List[float]]) -> float:
        """
        Compute EAR from eye landmarks.
        
        Args:
            left_eye_landmarks: Left eye landmarks
            right_eye_landmarks: Right eye landmarks
            
        Returns:
            float: Average EAR value
        """
        # Calculate EAR for each eye
        left_ear = get_eye_aspect_ratio(left_eye_landmarks)
        right_ear = get_eye_aspect_ratio(right_eye_landmarks)
        
        # Return average EAR
        return (left_ear + right_ear) / 2.0
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.state = DrowsinessState()
        logger.info("DrowsinessDetector state reset")
        
    def visualize(self, 
                 frame: np.ndarray, 
                 ear_left: Optional[float] = None,
                 ear_right: Optional[float] = None,
                 show_metrics: bool = True) -> np.ndarray:
        """
        Visualize drowsiness detection results on the frame.
        
        Args:
            frame: Input image frame
            ear_left: Left eye EAR value
            ear_right: Right eye EAR value
            show_metrics: Whether to show metrics on the frame
            
        Returns:
            np.ndarray: Frame with visualized drowsiness detection
        """
        # Get a copy of the frame
        vis_frame = frame.copy()
        
        if not show_metrics:
            return vis_frame
            
        # Calculate average EAR
        avg_ear = None
        if ear_left is not None and ear_right is not None:
            avg_ear = (ear_left + ear_right) / 2.0
        elif ear_left is not None:
            avg_ear = ear_left
        elif ear_right is not None:
            avg_ear = ear_right
            
        # Define font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6  # Slightly smaller text
        thickness = 1  # Thinner lines for better readability
        padding = 5  # Padding for background rectangle
        
        # Starting position for metrics in top-left corner
        start_x = 10
        start_y = 30
        line_height = 25  # Height between lines
        
        # Add EAR value
        if avg_ear is not None:
            ear_text = f"EAR: {avg_ear:.3f}"
            color = (0, 255, 0)  # Green (normal)
            if avg_ear < self.ear_threshold:
                color = (0, 0, 255)  # Red (eyes closed)
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(ear_text, font, font_scale, thickness)
            
            # Draw semi-transparent background
            overlay = vis_frame.copy()
            cv2.rectangle(
                overlay,
                (start_x - padding, start_y - text_height - padding),
                (start_x + text_width + padding, start_y + padding),
                (0, 0, 0),  # Black background
                -1
            )
            # Apply transparency
            cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
            
            # Draw text
            cv2.putText(vis_frame, ear_text, (start_x, start_y), font, font_scale, color, thickness)
            
        # Add PERCLOS value
        start_y += line_height
        perclos_text = f"PERCLOS: {self.state.perclos:.2f}%"
        perclos_color = (0, 255, 0)  # Green (normal)
        if self.state.perclos > self.perclos_critical_threshold:
            perclos_color = (0, 0, 255)  # Red (critical)
        elif self.state.perclos > self.perclos_warning_threshold:
            perclos_color = (0, 165, 255)  # Orange (warning)
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(perclos_text, font, font_scale, thickness)
        
        # Draw semi-transparent background
        overlay = vis_frame.copy()
        cv2.rectangle(
            overlay,
            (start_x - padding, start_y - text_height - padding),
            (start_x + text_width + padding, start_y + padding),
            (0, 0, 0),  # Black background
            -1
        )
        # Apply transparency
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
        
        # Draw text
        cv2.putText(vis_frame, perclos_text, (start_x, start_y), font, font_scale, perclos_color, thickness)
        
        # Calculate drowsiness level
        drowsiness_level = self._calculate_drowsiness_level()
        
        # Determine drowsiness state based on level
        if drowsiness_level < 0.3:
            drowsiness_state = "Alert"
            drowsiness_color = (0, 255, 0)  # Green
        elif drowsiness_level < 0.6:
            drowsiness_state = "Tired"
            drowsiness_color = (0, 165, 255)  # Orange
        elif drowsiness_level < 0.8:
            drowsiness_state = "Drowsy"
            drowsiness_color = (0, 0, 255)  # Red
        else:
            drowsiness_state = "Danger"
            drowsiness_color = (0, 0, 255)  # Red
        
        # Add drowsiness level
        start_y += line_height
        drowsiness_text = f"Drowsiness: {drowsiness_level:.2f} - {drowsiness_state}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(drowsiness_text, font, font_scale, thickness)
        
        # Draw semi-transparent background
        overlay = vis_frame.copy()
        cv2.rectangle(
            overlay,
            (start_x - padding, start_y - text_height - padding),
            (start_x + text_width + padding, start_y + padding),
            (0, 0, 0),  # Black background
            -1
        )
        # Apply transparency
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
        
        # Draw text
        cv2.putText(vis_frame, drowsiness_text, (start_x, start_y), font, font_scale, drowsiness_color, thickness)
        
        # Add eyes closed duration if eyes are closed
        if self.state.is_eyes_closed and self.state.eyes_closed_start_time is not None:
            closed_duration = time.time() - self.state.eyes_closed_start_time
            if closed_duration > 1.0:  # Only show if closed for more than 1 second
                start_y += line_height
                closed_text = f"Eyes Closed: {closed_duration:.1f}s"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(closed_text, font, font_scale, thickness)
                
                # Draw semi-transparent background
                overlay = vis_frame.copy()
                cv2.rectangle(
                    overlay,
                    (start_x - padding, start_y - text_height - padding),
                    (start_x + text_width + padding, start_y + padding),
                    (0, 0, 0),  # Black background
                    -1
                )
                # Apply transparency
                cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
                
                # Draw text
                cv2.putText(vis_frame, closed_text, (start_x, start_y), font, font_scale, (0, 0, 255), thickness)
                
        return vis_frame 