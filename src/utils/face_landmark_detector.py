#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face landmark detection module for the driver drowsiness detection system.

This module provides the FaceLandmarkDetector class that handles:
- Face detection with MediaPipe
- Extraction of facial landmarks
- Access to specific landmark groups (eyes, mouth, etc.)
- Calculation of face bounding rectangles
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Tuple, Dict, Optional, Union, Sequence

# Import constants
from src.utils.constants import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
    OUTER_LIP_INDICES, INNER_LIP_INDICES, MOUTH_INDICES,
    FACE_CONTOUR_INDICES, NOSE_INDICES
)

# Get module-specific logger
logger = logging.getLogger(__name__)

class FaceLandmarkDetector:
    """
    A class for detecting face landmarks using MediaPipe.
    
    This class handles the detection of facial landmarks and provides
    methods to access specific groups of landmarks like eyes, mouth, etc.
    """
    
    def __init__(self, 
                static_image_mode: bool = False, 
                max_num_faces: int = 1,
                refine_landmarks: bool = True,
                min_detection_confidence: float = 0.5,
                min_tracking_confidence: float = 0.5):
        """
        Initialize the Face Landmark Detector.
        
        Args:
            static_image_mode: Whether to treat the input images as a batch of static
                and possibly unrelated images, or a video stream.
            max_num_faces: Maximum number of faces to detect.
            refine_landmarks: Whether to refine the landmark coordinates around the
                eyes and lips, and output additional landmarks around the irises.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                face landmarks to be considered tracked successfully.
        """
        # MediaPipe FaceMesh solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize MediaPipe FaceMesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"FaceLandmarkDetector initialized with settings: static_mode={static_image_mode}, "
                  f"max_faces={max_num_faces}, min_detection_conf={min_detection_confidence}")
    
    def detect_face_landmarks(self, frame: np.ndarray) -> Tuple[List[List[float]], bool]:
        """
        Detect facial landmarks in a frame.
        
        Args:
            frame: Input frame/image (BGR format)
            
        Returns:
            Tuple containing:
            - List of landmarks as [x, y, z] coordinates
            - Boolean indicating if a face was detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize empty list for landmarks
        landmarks: List[List[float]] = []
        face_detected = False
        
        # Extract landmarks if a face is detected
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert normalized coordinates to pixel coordinates
            for landmark in face_landmarks.landmark:
                x, y, z = landmark.x * w, landmark.y * h, landmark.z
                landmarks.append([x, y, z])
            
            logger.debug(f"Face detected with {len(landmarks)} landmarks")
        else:
            logger.debug("No face detected in frame")
        
        return landmarks, face_detected
    
    def get_face_rect(self, landmarks: List[List[float]], padding: float = 0.1) -> Tuple[int, int, int, int]:
        """
        Get the face bounding rectangle from landmarks.
        
        Args:
            landmarks: List of facial landmarks
            padding: Padding factor to add around the face
            
        Returns:
            Tuple containing (x, y, width, height) of the face bounding rectangle
        """
        if not landmarks:
            logger.warning("Attempted to get face rect but no landmarks provided")
            return (0, 0, 0, 0)
        
        # Extract x, y coordinates
        x_coords = [landmark[0] for landmark in landmarks]
        y_coords = [landmark[1] for landmark in landmarks]
        
        # Find bounding box
        left = int(min(x_coords))
        top = int(min(y_coords))
        right = int(max(x_coords))
        bottom = int(max(y_coords))
        
        # Add padding
        width = right - left
        height = bottom - top
        padding_x = int(width * padding)
        padding_y = int(height * padding)
        
        left = max(0, left - padding_x)
        top = max(0, top - padding_y)
        right = right + padding_x
        bottom = bottom + padding_y
        
        logger.debug(f"Face rectangle calculated: x={left}, y={top}, width={right-left}, height={bottom-top}")
        return (left, top, right - left, bottom - top)
    
    def get_specific_landmarks(self, landmarks: List[List[float]], indices: List[int]) -> List[List[float]]:
        """
        Extract specific landmarks by their indices.
        
        Args:
            landmarks: List of all facial landmarks
            indices: List of landmark indices to extract
            
        Returns:
            List of selected landmarks
        """
        if not landmarks:
            logger.warning(f"Attempted to get specific landmarks {indices} but no landmarks provided")
            return []
        
        # Extract requested landmarks if available
        selected_landmarks: List[List[float]] = []
        for idx in indices:
            if idx < len(landmarks):
                selected_landmarks.append(landmarks[idx])
        
        return selected_landmarks
    
    def get_eye_landmarks(self, landmarks: List[List[float]], left_eye: bool = True) -> List[List[float]]:
        """
        Get landmarks for a specific eye.
        
        Args:
            landmarks: List of all facial landmarks
            left_eye: Whether to get left eye (True) or right eye (False) landmarks
            
        Returns:
            List of eye landmarks
        """
        eye_type = "left" if left_eye else "right"
        indices = LEFT_EYE_INDICES if left_eye else RIGHT_EYE_INDICES
        
        logger.debug(f"Getting {eye_type} eye landmarks")
        return self.get_specific_landmarks(landmarks, indices)
    
    def get_mouth_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """
        Get landmarks for the mouth.
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            List of mouth landmarks
        """
        logger.debug("Getting mouth landmarks")
        return self.get_specific_landmarks(landmarks, MOUTH_INDICES)
    
    def get_outer_lip_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """
        Get landmarks for the outer lip contour.
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            List of outer lip landmarks
        """
        logger.debug("Getting outer lip landmarks")
        return self.get_specific_landmarks(landmarks, OUTER_LIP_INDICES)
    
    def get_inner_lip_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """
        Get landmarks for the inner lip contour.
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            List of inner lip landmarks
        """
        logger.debug("Getting inner lip landmarks")
        return self.get_specific_landmarks(landmarks, INNER_LIP_INDICES)
    
    def draw_facial_landmarks(self, frame: np.ndarray, landmarks: List[List[float]], 
                           connections: Optional[List[Tuple[int, int]]] = None,
                           landmark_color: Tuple[int, int, int] = (0, 255, 0),
                           connection_color: Tuple[int, int, int] = (255, 0, 0),
                           landmark_radius: int = 1,
                           connection_thickness: int = 1) -> np.ndarray:
        """
        Draw facial landmarks and connections on the frame.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            connections: Optional list of tuples defining connections between landmarks
            landmark_color: Color for landmarks (BGR)
            connection_color: Color for connections (BGR)
            landmark_radius: Radius of landmark points
            connection_thickness: Thickness of connection lines
            
        Returns:
            np.ndarray: Frame with visualized landmarks
        """
        if not landmarks:
            logger.warning("Attempted to draw landmarks but no landmarks provided")
            return frame.copy()
        
        vis_frame = frame.copy()
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark[0]), int(landmark[1])
            cv2.circle(vis_frame, (x, y), landmark_radius, landmark_color, -1)
        
        # Draw connections
        if connections:
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                    end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                    cv2.line(vis_frame, start_point, end_point, connection_color, connection_thickness)
        
        logger.debug(f"Drew {len(landmarks)} landmarks on frame")
        return vis_frame
    
    def release(self) -> None:
        """Release MediaPipe resources."""
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh:
                self.face_mesh.close()
                logger.info("FaceLandmarkDetector resources released")
        except Exception as e:
            logger.error(f"Error releasing FaceLandmarkDetector resources: {str(e)}")


def get_face_landmark_detector() -> FaceLandmarkDetector:
    """
    Get a new instance of FaceLandmarkDetector.
    
    Returns:
        FaceLandmarkDetector: A new detector instance
    """
    logger.debug("Creating new FaceLandmarkDetector instance")
    return FaceLandmarkDetector() 