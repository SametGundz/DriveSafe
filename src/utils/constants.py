#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constants and common configurations for the driver drowsiness detection system.

This module defines centralized constants and configuration values that are used
across different modules in the project, avoiding duplication and ensuring consistency.
"""

# MediaPipe FaceMesh landmark indices
# -------------------------------------------------------------------------

# Eye landmarks
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Simplified 6-point eye model for EAR calculation
LEFT_EYE_INDICES_6POINT = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES_6POINT = [33, 160, 158, 133, 153, 144]

# Mouth landmarks
OUTER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
MOUTH_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES

# 6-point simplified mouth model
MOUTH_INDICES_6POINT = [61, 291, 0, 17, 269, 405]

# Face contour landmarks
FACE_CONTOUR_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]

# Nose landmarks
NOSE_INDICES = [168, 6, 197, 195, 5, 4, 19, 94, 2]

# Head pose estimation landmarks
HEAD_POSE_LANDMARKS = [1, 9, 57, 130, 287, 359]

# Gaze estimation landmarks
GAZE_LANDMARK_INDICES = [33, 133, 362, 263, 1, 199]

# Detection thresholds
# -------------------------------------------------------------------------

# Eye Aspect Ratio (EAR) thresholds
EAR_THRESHOLD = 0.21  # Below this value, eyes are considered closed
EAR_WARNING_THRESHOLD = 0.25  # Warning level
EAR_ALERT_FRAMES = 48  # Number of consecutive frames with closed eyes to trigger alert (at 30fps)

# Mouth Aspect Ratio (MAR) thresholds
MAR_THRESHOLD = 0.6  # Above this value, mouth is considered open
MAR_WARNING_THRESHOLD = 0.7  # Warning level
MAR_ALERT_FRAMES = 60  # Number of consecutive frames with open mouth to trigger alert (at 30fps)

# PERCLOS thresholds
PERCLOS_WARNING_THRESHOLD = 15.0  # Percentage
PERCLOS_CRITICAL_THRESHOLD = 20.0  # Percentage

# Head pose thresholds (in degrees)
HEAD_YAW_THRESHOLD = 30.0  # Max head rotation left/right
HEAD_PITCH_THRESHOLD = 20.0  # Max head rotation up/down
HEAD_ROLL_THRESHOLD = 20.0  # Max head tilt

# Gaze deviation thresholds (in degrees)
GAZE_YAW_THRESHOLD = 25.0  # Max eye gaze left/right
GAZE_PITCH_THRESHOLD = 15.0  # Max eye gaze up/down

# Time windows
# -------------------------------------------------------------------------
BLINK_DETECTION_WINDOW = 0.15  # Time window in seconds to detect a blink
PERCLOS_WINDOW_SECONDS = 30  # Time window in seconds for PERCLOS calculation
ATTENTION_WINDOW_SECONDS = 5  # Time window for attention monitoring

# Camera & System Settings
# -------------------------------------------------------------------------
DEFAULT_CAMERA_ID = 0
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

# UI Settings
# -------------------------------------------------------------------------
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800
MIN_DISPLAY_WIDTH = 800
MIN_DISPLAY_HEIGHT = 600
GRAPH_HISTORY_SECONDS = 30 