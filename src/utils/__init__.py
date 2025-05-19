"""
Utility functions and classes for the driver drowsiness detection application.

This package provides various utilities for face detection, landmark detection,
metrics calculation, head pose estimation, and gaze detection.
"""

# Import and expose the modular components
from src.utils.face_landmark_detector import FaceLandmarkDetector, get_face_landmark_detector
from src.utils.facial_metrics import get_eye_aspect_ratio, get_mouth_aspect_ratio
from src.utils.head_pose_estimator import HeadPoseEstimator
from src.utils.gaze_detector import GazeDetector
from src.utils.mediapipe_helper import MediaPipeHelper, get_mediapipe_helper, load_ui_config

# Make key functions and classes available at package level
__all__ = [
    # New modular architecture
    'FaceLandmarkDetector',
    'get_face_landmark_detector',
    'get_eye_aspect_ratio',
    'get_mouth_aspect_ratio',
    'HeadPoseEstimator',
    'GazeDetector',
    'MediaPipeHelper',
    'get_mediapipe_helper',
    'load_ui_config'
] 