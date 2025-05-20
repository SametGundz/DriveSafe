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
import logging
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence

import numpy as np

from src.utils.face_landmark_detector import FaceLandmarkDetector
from src.utils.facial_metrics import get_eye_aspect_ratio, get_mouth_aspect_ratio
from src.utils.head_pose_estimator import HeadPoseEstimator
from src.utils.gaze_detector import GazeDetector

# Get the module logger
logger = logging.getLogger(__name__)

class MediaPipeHelper(FaceLandmarkDetector):
    """
    Unified helper class that integrates all MediaPipe-based modular components.
    
    This class extends FaceLandmarkDetector and integrates other components
    like HeadPoseEstimator and GazeDetector to provide a comprehensive
    interface for face analysis.
    """
    
    def __init__(self, 
                static_image_mode: bool = False, 
                max_num_faces: int = 1,
                refine_landmarks: bool = True,
                min_detection_confidence: float = 0.5,
                min_tracking_confidence: float = 0.5,
                head_pose_estimator: Optional[HeadPoseEstimator] = None,
                gaze_detector: Optional[GazeDetector] = None):
        """
        Initialize the MediaPipeHelper with all required components.
        
        Args:
            static_image_mode: Whether to treat the input images as a batch of static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine the landmark coordinates
            min_detection_confidence: Minimum confidence value for face detection
            min_tracking_confidence: Minimum confidence value for face tracking
            head_pose_estimator: Optional HeadPoseEstimator instance
            gaze_detector: Optional GazeDetector instance
        """
        # Initialize parent class (FaceLandmarkDetector)
        super().__init__(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize other components
        self.head_pose_estimator = head_pose_estimator or HeadPoseEstimator()
        self.gaze_detector = gaze_detector or GazeDetector()
        
        logger.info("MediaPipeHelper initialized with all components")
    
    def get_eye_aspect_ratio(self, eye_landmarks: List[List[float]]) -> float:
        """
        Calculate the eye aspect ratio.
        
        Args:
            eye_landmarks: Eye landmarks
            
        Returns:
            float: EAR value
        """
        return get_eye_aspect_ratio(eye_landmarks)
    
    def get_mouth_aspect_ratio(self, landmarks: List[List[float]]) -> float:
        """
        Calculate the mouth aspect ratio.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            float: MAR value
        """
        mouth_landmarks = self.get_mouth_landmarks(landmarks)
        return get_mouth_aspect_ratio(mouth_landmarks)
    
    def calculate_head_pose(self, landmarks: List[List[float]], frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Calculate head pose from facial landmarks.
        
        Args:
            landmarks: Facial landmarks
            frame: Input image frame
            
        Returns:
            tuple: Head pose (rotation matrix, (roll, pitch, yaw))
        """
        return self.head_pose_estimator.calculate_head_pose(landmarks, frame)
    
    def get_head_pose(self, landmarks: List[List[float]], frame: np.ndarray) -> Tuple[float, float, float]:
        """
        Get head pose angles.
        
        Args:
            landmarks: Facial landmarks
            frame: Input image frame
            
        Returns:
            tuple: Head pose angles (pitch, yaw, roll)
        """
        try:
            # Frame boyutlarını kontrol et
            if isinstance(frame, np.ndarray):
                frame_shape = frame.shape
            else:
                # Frame bir ndarray değilse (örneğin tuple olarak boyutlar verilmişse)
                frame_shape = frame
                # Dummy frame oluştur
                if len(frame_shape) >= 2:
                    h, w = frame_shape[:2]
                    dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
                    _, angles = self.head_pose_estimator.calculate_head_pose(landmarks, dummy_frame)
                    # HeadPoseEstimator açıları (pitch, yaw, roll) sırasında döndürür
                    return angles
            
            # Head pose hesaplama
            _, angles = self.head_pose_estimator.calculate_head_pose(landmarks, frame)
            
            # Açıları döndür - HeadPoseEstimator (pitch, yaw, roll) sırasında döndürür
            # Tutarlılık için bu sırayı koruyalım
            return angles
        except Exception as e:
            logger.error(f"Error in get_head_pose: {str(e)}")
            # Hata durumunda varsayılan değerleri döndür
            return (0.0, 0.0, 0.0)
    
    def visualize_head_pose(self, frame: np.ndarray, landmarks: List[List[float]], 
                           show_axes: bool = True, show_angles: bool = True,
                           visualization_type: str = 'axes') -> np.ndarray:
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
    
    def get_eye_gaze_direction(self, landmarks: List[List[float]]) -> np.ndarray:
        """
        Get eye gaze direction.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            tuple: Gaze direction vector (x, y, z)
        """
        return self.gaze_detector.get_eye_gaze_direction(landmarks)
    
    def predict_gaze(self, frame: np.ndarray, landmarks: List[List[float]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict gaze direction using the model.
        
        Args:
            frame: Input image frame
            landmarks: Facial landmarks
            
        Returns:
            tuple: (gaze_vector, normalized_image)
        """
        return self.gaze_detector.predict_gaze(frame, landmarks)
    
    def visualize_gaze(self, frame: np.ndarray, landmarks: List[List[float]], 
                      ear_value: Optional[float] = None, ear_threshold: float = 0.2,
                      frame_skip: int = 3) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
    
    def can_detect_head_pose(self) -> bool:
        """
        Head pose tespiti yapabilip yapamayacağını kontrol eder.
        
        Returns:
            bool: Head pose tespiti yapılabilirse True, aksi halde False
        """
        # HeadPoseEstimator mevcut ve hazır mı kontrol et
        return hasattr(self, 'head_pose_estimator') and self.head_pose_estimator is not None
    
    def release(self) -> None:
        """
        Release all resources.
        """
        # Release parent class resources
        super().release()
        
        # Release other components
        if hasattr(self.head_pose_estimator, 'release'):
            self.head_pose_estimator.release()
        
        if hasattr(self.gaze_detector, 'release'):
            self.gaze_detector.release()
        
        logger.info("MediaPipeHelper resources released")


def get_mediapipe_helper() -> MediaPipeHelper:
    """
    Get or create a MediaPipeHelper instance.
    
    Returns:
        MediaPipeHelper: A MediaPipeHelper instance
    """
    logger.debug("Creating new MediaPipeHelper instance")
    return MediaPipeHelper()


def load_ui_config() -> Dict[str, Any]:
    """
    Load UI configuration from main config file.
    
    Returns:
        Dict[str, Any]: UI configuration dictionary
    """
    config_path = os.path.join("config", "config.yaml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
            
        # Tüm konfigürasyonu döndür, sadece UI kısmını değil
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {} 
