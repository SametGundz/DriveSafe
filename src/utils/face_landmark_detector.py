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
from typing import List, Tuple, Dict, Optional, Union

class FaceLandmarkDetector:
    """
    A class for detecting face landmarks using MediaPipe.
    
    This class handles the detection of facial landmarks and provides
    methods to access specific groups of landmarks like eyes, mouth, etc.
    """
    
    # Key facial landmark indices
    # Left eye landmarks
    LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Right eye landmarks
    RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Mouth landmarks - with both outer and inner lip contours
    # Outer lip landmarks clockwise from left corner
    OUTER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    # Inner lip landmarks clockwise from left corner
    INNER_LIP_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    # Combined mouth landmarks for visualization (outer + inner contour)
    MOUTH_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES
    
    # Face contour landmarks
    FACE_CONTOUR_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
    # Nose landmarks
    NOSE_INDICES = [168, 6, 197, 195, 5, 4, 19, 94, 2]
    
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
        landmarks = []
        face_detected = False
        
        # Extract landmarks if a face is detected
        if results.multi_face_landmarks:
            face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert normalized coordinates to pixel coordinates
            for landmark in face_landmarks.landmark:
                x, y, z = landmark.x * w, landmark.y * h, landmark.z
                landmarks.append([x, y, z])
        
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
            return []
        
        # Extract requested landmarks if available
        selected_landmarks = []
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
        indices = self.LEFT_EYE_INDICES if left_eye else self.RIGHT_EYE_INDICES
        return self.get_specific_landmarks(landmarks, indices)
    
    def get_mouth_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """
        Get landmarks for the mouth.
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            List of mouth landmarks
        """
        return self.get_specific_landmarks(landmarks, self.MOUTH_INDICES)
    
    def get_outer_lip_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """
        Get landmarks for the outer lip contour.
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            List of outer lip landmarks
        """
        return self.get_specific_landmarks(landmarks, self.OUTER_LIP_INDICES)
    
    def get_inner_lip_landmarks(self, landmarks: List[List[float]]) -> List[List[float]]:
        """
        Get landmarks for the inner lip contour.
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            List of inner lip landmarks
        """
        return self.get_specific_landmarks(landmarks, self.INNER_LIP_INDICES)
    
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
        
        return vis_frame
    
    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()


# Convenience function to get a preconfigured FaceLandmarkDetector instance
def get_face_landmark_detector() -> FaceLandmarkDetector:
    """
    Factory function to create and return a FaceLandmarkDetector instance.
    
    Returns:
        FaceLandmarkDetector: Initialized FaceLandmarkDetector instance
    """
    return FaceLandmarkDetector() 