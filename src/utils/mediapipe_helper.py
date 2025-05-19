#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified MediaPipe helper class that integrates all modular components.

This module provides a unified interface to all the MediaPipe-based functionality
in the driver drowsiness detection system, including:
- Face landmark detection
- Facial metrics calculation
- Head pose estimation
- Gaze direction detection and visualization
"""

import os
import cv2
import yaml
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from src.utils.face_landmark_detector import FaceLandmarkDetector, get_face_landmark_detector
from src.utils.facial_metrics import get_eye_aspect_ratio, get_mouth_aspect_ratio
from src.utils.head_pose_estimator import HeadPoseEstimator
from src.utils.gaze_detector import GazeDetector


class MediaPipeHelper:
    """
    Unified helper class that integrates all MediaPipe-based modular components.
    
    This class provides a simplified interface to use all the functionality offered by
    the individual modular components in a unified manner, similar to the legacy
    MediaPipeUtils class but with better internal organization.
    """
    
    def __init__(self, face_detector=None, head_pose_estimator=None, gaze_detector=None):
        """
        Initialize the MediaPipeHelper with all required components.
        
        Args:
            face_detector: Optional FaceLandmarkDetector instance
            head_pose_estimator: Optional HeadPoseEstimator instance
            gaze_detector: Optional GazeDetector instance
        """
        # Initialize or use provided components
        self.face_detector = face_detector or get_face_landmark_detector()
        self.head_pose_estimator = head_pose_estimator or HeadPoseEstimator()
        self.gaze_detector = gaze_detector or GazeDetector()
    
    def detect_face_landmarks(self, frame):
        """
        Detect facial landmarks in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (landmarks, face_detected) - landmarks list and boolean face detection flag
        """
        return self.face_detector.detect_face_landmarks(frame)
    
    def get_face_rect(self, landmarks, padding=0.1):
        """
        Get the rectangle containing the face.
        
        Args:
            landmarks: Facial landmarks
            padding: Padding around the face rect (percentage of size)
            
        Returns:
            tuple: (x, y, w, h) - face rectangle coordinates
        """
        return self.face_detector.get_face_rect(landmarks, padding)
    
    def get_eye_landmarks(self, landmarks, left_eye=True):
        """
        Get eye landmarks.
        
        Args:
            landmarks: Facial landmarks
            left_eye: Flag to get left eye (True) or right eye (False)
            
        Returns:
            list: Eye landmarks
        """
        return self.face_detector.get_eye_landmarks(landmarks, left_eye)
    
    def get_mouth_landmarks(self, landmarks):
        """
        Get mouth landmarks.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            list: Mouth landmarks
        """
        return self.face_detector.get_mouth_landmarks(landmarks)
    
    def get_inner_lip_landmarks(self, landmarks):
        """
        Get inner lip landmarks.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            list: Inner lip landmarks
        """
        return self.face_detector.get_inner_lip_landmarks(landmarks)
    
    def get_outer_lip_landmarks(self, landmarks):
        """
        Get outer lip landmarks.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            list: Outer lip landmarks
        """
        return self.face_detector.get_outer_lip_landmarks(landmarks)
    
    def draw_facial_landmarks(self, frame, landmarks, connections=None,
                           landmark_color=(0, 255, 0), connection_color=(255, 0, 0),
                           landmark_radius=1, connection_thickness=1):
        """
        Draw facial landmarks on the frame.
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks
            connections: Optional list of connections between landmarks
            landmark_color: Color for landmarks
            connection_color: Color for connections
            landmark_radius: Radius of landmark circles
            connection_thickness: Thickness of connection lines
            
        Returns:
            ndarray: Frame with drawn landmarks
        """
        return self.face_detector.draw_facial_landmarks(
            frame, landmarks, connections, landmark_color, connection_color,
            landmark_radius, connection_thickness
        )
    
    def get_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate the eye aspect ratio.
        
        Args:
            eye_landmarks: Eye landmarks
            
        Returns:
            float: EAR value
        """
        return get_eye_aspect_ratio(eye_landmarks)
    
    def get_mouth_aspect_ratio(self, landmarks):
        """
        Calculate the mouth aspect ratio.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            float: MAR value
        """
        mouth_landmarks = self.get_mouth_landmarks(landmarks)
        return get_mouth_aspect_ratio(mouth_landmarks)
    
    def calculate_head_pose(self, landmarks, frame):
        """
        Calculate head pose from facial landmarks.
        
        Args:
            landmarks: Facial landmarks
            frame: Input image frame
            
        Returns:
            tuple: Head pose (rotation matrix, (roll, pitch, yaw))
        """
        return self.head_pose_estimator.calculate_head_pose(landmarks, frame)
    
    def get_head_pose(self, landmarks, frame):
        """
        Get head pose angles.
        
        Args:
            landmarks: Facial landmarks
            frame: Input image frame
            
        Returns:
            tuple: Head pose angles (roll, pitch, yaw)
        """
        _, angles = self.head_pose_estimator.calculate_head_pose(landmarks, frame)
        return angles
    
    def visualize_head_pose(self, frame, landmarks, show_axes=True, show_angles=True,
                           visualization_type='axes'):
        """
        Visualize head pose on the frame.
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks
            show_axes: Flag to show axes
            show_angles: Flag to show angles
            visualization_type: Type of visualization ('axes' or 'cube')
            
        Returns:
            ndarray: Frame with visualized head pose
        """
        return self.head_pose_estimator.visualize_head_pose(
            frame, landmarks, show_axes, show_angles, visualization_type
        )
    
    def get_eye_gaze_direction(self, landmarks):
        """
        Get eye gaze direction.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            tuple: Gaze direction vector (x, y, z)
        """
        return self.gaze_detector.get_eye_gaze_direction(landmarks)
    
    def predict_gaze(self, frame, landmarks):
        """
        Predict gaze direction using the model.
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks
            
        Returns:
            tuple: (gaze_vector, normalized_image)
        """
        return self.gaze_detector.predict_gaze(frame, landmarks)
    
    def visualize_gaze(self, frame, landmarks, ear_value=None, ear_threshold=0.2,
                    frame_skip=3):
        """
        Visualize gaze direction on the frame.
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks
            ear_value: Optional EAR value
            ear_threshold: EAR threshold for closed eyes
            frame_skip: Number of frames to skip between gaze predictions
            
        Returns:
            tuple: (frame with visualized gaze, normalized face image)
        """
        return self.gaze_detector.visualize_gaze(
            frame, landmarks, ear_value, ear_threshold, frame_skip
        )
    
    def release(self):
        """
        Release resources.
        """
        self.face_detector.release()


def get_mediapipe_helper():
    """
    Get or create a MediaPipeHelper instance.
    
    Returns:
        MediaPipeHelper: A MediaPipeHelper instance
    """
    return MediaPipeHelper()


def load_ui_config():
    """
    Load UI configuration from YAML file.
    
    Returns:
        dict: UI configuration
    """
    config_path = os.path.join("config", "ui_config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    
    return config 