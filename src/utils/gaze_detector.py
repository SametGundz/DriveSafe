#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaze detection module for the driver drowsiness detection system.

This module provides the GazeDetector class that handles:
- Basic gaze direction estimation using geometric methods
- Advanced gaze direction estimation using the ETH-XGaze model
- Visualization of gaze vectors
- Face normalization for gaze estimation
"""

import os
import cv2
import numpy as np
import onnxruntime
from typing import List, Tuple, Optional, Union

class GazeDetector:
    """
    A class for detecting and visualizing gaze direction.
    
    This class provides methods to:
    - Estimate gaze direction using simple geometric methods
    - Estimate gaze direction using the ETH-XGaze model
    - Visualize gaze direction on the frame
    - Normalize face for gaze estimation
    """
    
    # ETH-XGaze face model points (3D landmark points)
    GAZE_FACE_MODEL = np.array([
        [-34.04, 39.55, 57.49],  # Right eye outer corner
        [-16.1, 42.4, 67.53],    # Right eye inner corner
        [16.1, 42.4, 67.53],     # Left eye inner corner
        [34.04, 39.55, 57.49],   # Left eye outer corner
        [0.0, 0.0, 8.0],         # Nose tip
        [0.0, -48.0, 21.0],      # Chin tip
    ])
    
    # MediaPipe landmark indices for the 6 points used in ETH-XGaze
    # Right eye outer corner, Right eye inner corner, Left eye inner corner, Left eye outer corner, Nose tip, Chin tip
    GAZE_LANDMARK_INDICES = [33, 133, 362, 263, 1, 199]
    
    def __init__(self, model_path: str = None):
        """
        Initialize the GazeDetector.
        
        Args:
            model_path: Path to the ONNX model file (if None, uses default path)
        """
        # Initialize variables for frame skipping optimization
        self._last_gaze_vector = None
        self._last_normalized_image = None
        self._frame_counter = 0
        
        # Try to load the ONNX model
        if model_path is None:
            self.model_path = os.path.join('models', 'eth_xgaze_model.onnx')
        else:
            self.model_path = model_path
            
        if os.path.exists(self.model_path):
            try:
                self.onnx_session = onnxruntime.InferenceSession(self.model_path)
                print(f"ONNX model loaded: {self.model_path}")
            except Exception as e:
                print(f"Failed to load ONNX model: {str(e)}")
                self.onnx_session = None
        else:
            print(f"WARNING: ONNX model not found: {self.model_path}")
            self.onnx_session = None
    
    def get_eye_gaze_direction(self, landmarks: List[List[float]]) -> Tuple[float, float, float]:
        """
        Calculate approximate eye gaze direction based on eye landmarks.
        
        This method uses a simple geometric approach based on the relative positions of 
        eye landmarks, particularly the iris position relative to the eye corners.
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            Tuple[float, float, float]: (x, y, z) direction vector, where:
                x: -1.0 (far left) to 1.0 (far right)
                y: -1.0 (far up) to 1.0 (far down)
                z: depth estimate (larger values mean looking more forward/focused)
        """
        if landmarks is None or not landmarks:
            return (0.0, 0.0, 0.0)
            
        # Get iris landmarks - MediaPipe provides iris landmarks indices
        # For left eye: 468, 469, 470, 471, 472
        # For right eye: 473, 474, 475, 476, 477
        left_iris_center_idx = 468  # Center of left iris
        right_iris_center_idx = 473  # Center of right iris
        
        # Get eye corner landmarks
        left_eye_corners = [landmarks[362], landmarks[263]]  # Outer and inner corners of left eye
        right_eye_corners = [landmarks[33], landmarks[133]]  # Outer and inner corners of right eye
        
        # Check if we have all the necessary landmarks
        if (len(landmarks) <= max(left_iris_center_idx, right_iris_center_idx) or
            not all(left_eye_corners) or not all(right_eye_corners)):
            return (0.0, 0.0, 0.0)
            
        # Get iris centers
        left_iris_center = landmarks[left_iris_center_idx]
        right_iris_center = landmarks[right_iris_center_idx]
        
        # Calculate horizontal gaze direction by comparing iris position to eye width
        # For left eye
        left_eye_width = self._calculate_distance(left_eye_corners[0], left_eye_corners[1])
        left_iris_to_corner = self._calculate_distance(left_iris_center, left_eye_corners[0])
        left_gaze_x = 2.0 * (left_iris_to_corner / left_eye_width) - 1.0
        
        # For right eye
        right_eye_width = self._calculate_distance(right_eye_corners[0], right_eye_corners[1])
        right_iris_to_corner = self._calculate_distance(right_iris_center, right_eye_corners[0])
        right_gaze_x = 1.0 - 2.0 * (right_iris_to_corner / right_eye_width)
        
        # Average the horizontal gaze from both eyes
        gaze_x = (left_gaze_x + right_gaze_x) / 2.0
        
        # Calculate vertical gaze direction
        # Get top and bottom landmarks of each eye
        left_eye_top_idx, left_eye_bottom_idx = 386, 374
        right_eye_top_idx, right_eye_bottom_idx = 159, 145
        
        if (len(landmarks) <= max(left_eye_top_idx, left_eye_bottom_idx, right_eye_top_idx, right_eye_bottom_idx)):
            gaze_y = 0.0
        else:
            # For left eye
            left_eye_height = self._calculate_distance(landmarks[left_eye_top_idx], landmarks[left_eye_bottom_idx])
            left_iris_to_top = self._calculate_distance(left_iris_center, landmarks[left_eye_top_idx])
            left_gaze_y = 2.0 * (left_iris_to_top / left_eye_height) - 1.0
            
            # For right eye
            right_eye_height = self._calculate_distance(landmarks[right_eye_top_idx], landmarks[right_eye_bottom_idx])
            right_iris_to_top = self._calculate_distance(right_iris_center, landmarks[right_eye_top_idx])
            right_gaze_y = 2.0 * (right_iris_to_top / right_eye_height) - 1.0
            
            # Average the vertical gaze from both eyes
            gaze_y = (left_gaze_y + right_gaze_y) / 2.0
        
        # Estimate z-component (depth/focus)
        # More negative when looking sideways, more positive when looking forward
        gaze_z = 1.0 - (abs(gaze_x) + abs(gaze_y)) / 2.0
        
        return (gaze_x, gaze_y, gaze_z)
    
    @staticmethod
    def _calculate_distance(point1: List[float], point2: List[float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: (x, y) coordinates of first point
            point2: (x, y) coordinates of second point
            
        Returns:
            float: Euclidean distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def draw_gaze_direction(self, frame: np.ndarray, landmarks: List[List[float]], 
                           gaze_dir: Tuple[float, float, float],
                           arrow_color: Tuple[int, int, int] = (0, 0, 255),
                           arrow_length: int = 100,
                           arrow_thickness: int = 2) -> np.ndarray:
        """
        Draw gaze direction arrow on the frame.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            gaze_dir: Tuple of (x, y, z) gaze direction vector
            arrow_color: Color for the arrow (BGR)
            arrow_length: Length of the arrow
            arrow_thickness: Thickness of the arrow
            
        Returns:
            np.ndarray: Frame with visualized gaze direction
        """
        if landmarks is None or not landmarks or len(landmarks) < 470:
            return frame
            
        vis_frame = frame.copy()
        
        # Get iris centers
        left_iris_idx = 468  # Center of left iris
        right_iris_idx = 473  # Center of right iris
        
        # Get eye corners
        left_eye_corners = [landmarks[362], landmarks[263]]  # Outer and inner corners of left eye
        right_eye_corners = [landmarks[33], landmarks[133]]  # Outer and inner corners of right eye
        
        # Calculate eye centers
        left_eye_center = (int(landmarks[left_iris_idx][0]), int(landmarks[left_iris_idx][1]))
        right_eye_center = (int(landmarks[right_iris_idx][0]), int(landmarks[right_iris_idx][1]))
        
        # Convert gaze_dir (x,y,z) to angles
        # gaze_x: -1.0 (far left) to 1.0 (far right)
        # gaze_y: -1.0 (far up) to 1.0 (far down)
        gaze_x, gaze_y, gaze_z = gaze_dir
        
        # Convert these normalized directions to angles (in radians)
        # Yaw angle (horizontal rotation, left-right)
        yaw = np.arcsin(np.clip(gaze_x, -1.0, 1.0))
        # Pitch angle (vertical rotation, up-down)
        pitch = np.arcsin(np.clip(gaze_y, -1.0, 1.0))
        
        # Draw gaze direction for each eye
        for eye_center in [left_eye_center, right_eye_center]:
            # Calculate gaze vector endpoint using angles
            x = -arrow_length * np.sin(yaw) * np.cos(pitch)
            y = -arrow_length * np.sin(pitch)
            z = -arrow_length * np.cos(yaw) * np.cos(pitch)
            
            # Convert 3D coordinates to 2D screen coordinates
            # Simple projection since we don't have camera parameters
            scale_factor = 0.5  # Scale to make the arrow more visible
            dx = scale_factor * x
            dy = scale_factor * y
            
            # Adjust length based on gaze_z (confidence/focus)
            length_adjustment = 0.5 + gaze_z / 2.0  # Maps 0-1 to 0.5-1.0
            dx *= length_adjustment
            dy *= length_adjustment
            
            # Calculate endpoint
            gaze_end = (int(eye_center[0] + dx), int(eye_center[1] + dy))
            
            # Draw arrow
            cv2.arrowedLine(
                vis_frame,
                eye_center,
                gaze_end,
                arrow_color,
                arrow_thickness,
                cv2.LINE_AA,
                tipLength=0.2
            )
            
        # Add a visual center point between eyes for the general gaze direction
        center_point = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                        (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Calculate endpoint for the central gaze direction
        x = -arrow_length * 1.5 * np.sin(yaw) * np.cos(pitch)
        y = -arrow_length * 1.5 * np.sin(pitch)
        scale_factor = 0.5
        dx = scale_factor * x
        dy = scale_factor * y
        
        # Adjust length based on gaze_z (confidence/focus)
        length_adjustment = 0.5 + gaze_z / 2.0
        dx *= length_adjustment
        dy *= length_adjustment
        
        # Calculate endpoint
        gaze_end = (int(center_point[0] + dx), int(center_point[1] + dy))
        
        # Draw central arrow
        cv2.arrowedLine(
            vis_frame,
            center_point,
            gaze_end,
            (0, 165, 255),  # Orange color for the central arrow
            arrow_thickness + 1,
            cv2.LINE_AA,
            tipLength=0.2
        )
        
        return vis_frame
    
    def estimate_head_pose(self, landmarks: List[List[float]], frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate head pose for gaze normalization.
        
        Args:
            landmarks: List of facial landmarks
            frame: Input frame for calculating image dimensions
            
        Returns:
            rvec, tvec: Rotation and translation vectors
        """
        # Calculate camera matrix from frame dimensions
        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        distortion = np.zeros((4, 1), dtype=np.float64)
        
        # Get the 6 points used for gaze estimation
        landmarks_2d = []
        for idx in self.GAZE_LANDMARK_INDICES:
            if idx < len(landmarks):
                landmarks_2d.append([landmarks[idx][0], landmarks[idx][1]])
        
        landmarks_2d = np.array(landmarks_2d, dtype=np.float64)
        
        # Solve PnP to get head pose
        if len(landmarks_2d) == 6:  # All points are found
            # Initial estimate
            ret, rvec, tvec = cv2.solvePnP(
                self.GAZE_FACE_MODEL, 
                landmarks_2d, 
                camera_matrix, 
                distortion, 
                flags=cv2.SOLVEPNP_EPNP
            )
            
            # Refine estimate
            ret, rvec, tvec = cv2.solvePnP(
                self.GAZE_FACE_MODEL, 
                landmarks_2d, 
                camera_matrix, 
                distortion, 
                rvec, tvec, 
                True
            )
            
            return rvec, tvec
        
        # Not enough points
        return np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
    
    def normalize_face(self, img: np.ndarray, landmarks: List[List[float]], frame: np.ndarray) -> np.ndarray:
        """
        Normalize face image for ETH-XGaze model.
        
        Args:
            img: Input frame
            landmarks: List of facial landmarks
            frame: Input frame for calculating image dimensions
            
        Returns:
            img_normalized: Normalized face image
        """
        # Estimate head pose
        hr, ht = self.estimate_head_pose(landmarks, frame)
        
        # Calculate camera matrix from frame dimensions
        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Normalized camera parameters (from ETH-XGaze)
        focal_norm = 960  # Normalized focal length
        distance_norm = 600  # Normalized distance between eye and camera
        roi_size = (224, 224)  # Cropped eye image size
        
        # Calculate 3D positions of landmarks
        ht = ht.reshape((3, 1))
        hR = cv2.Rodrigues(hr)[0]  # Rotation matrix
        Fc = np.dot(hR, self.GAZE_FACE_MODEL.T) + ht  # Rotate and translate face model
        
        # Find face center
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
        
        # Normalize image
        distance = np.linalg.norm(face_center)  # Actual distance between eye and camera
        
        z_scale = distance_norm / distance
        cam_norm = np.array([  # Virtual camera intrinsics
            [focal_norm, 0, roi_size[0] / 2],
            [0, focal_norm, roi_size[1] / 2],
            [0, 0, 1.0],
        ])
        
        # Calculate transformation matrix
        S = np.array([  # Scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        
        R = np.c_[right, down, forward].T  # Rotation matrix
        
        # Calculate transformation matrix for normalized image
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))
        
        # Transform image
        img_normalized = cv2.warpPerspective(img, W, roi_size)
        
        return img_normalized
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ETH-XGaze model.
        
        Args:
            image: Normalized face image (224x224)
            
        Returns:
            processed_img: Processed image (model input)
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (224, 224))
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Rearrange channels (HWC -> CHW)
        image = image.transpose(2, 0, 1)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_gaze(self, frame: np.ndarray, landmarks: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict gaze direction using ETH-XGaze model.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            
        Returns:
            gaze_vector: Gaze direction vector (pitch, yaw)
            normalized_image: Normalized face image
        """
        if self.onnx_session is None:
            # Return empty vector if model is not loaded
            return np.zeros(2), None
        
        # Normalize face
        normalized_image = self.normalize_face(frame, landmarks, frame)
        
        # Preprocess image
        processed_img = self.preprocess_image(normalized_image)
        
        # Run inference
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        gaze = self.onnx_session.run([output_name], {input_name: processed_img})[0]
        
        # Get gaze vector (pitch, yaw)
        gaze_vector = gaze[0]
        
        return gaze_vector, normalized_image
    
    def draw_gaze(self, image: np.ndarray, pitchyaw: np.ndarray, origin: Tuple[int, int], 
                 length: int = 50, thickness: int = 2, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Visualize gaze direction.
        
        Args:
            image: Input image
            pitchyaw: Gaze direction vector (pitch, yaw)
            origin: Origin point (x, y) of gaze vector
            length: Arrow length
            thickness: Arrow thickness
            color: Arrow color
            
        Returns:
            image: Image with gaze direction visualized
        """
        pitch, yaw = pitchyaw
        
        # Convert pitch and yaw to radians if they're not already
        # For the ETH-XGaze model, they are already in radians
        pitch = pitch
        yaw = yaw
        
        # Calculate gaze vector
        x = -length * np.sin(yaw) * np.cos(pitch)
        y = -length * np.sin(pitch)
        z = -length * np.cos(yaw) * np.cos(pitch)
        
        # Project 3D vector to 2D
        point_2d = (int(origin[0] + x), int(origin[1] + y))
        
        # Draw arrow
        cv2.arrowedLine(image, origin, point_2d, color, thickness)
        
        return image
    
    def visualize_gaze(self, frame: np.ndarray, landmarks: List[List[float]], 
                        ear_value: float = None, ear_threshold: float = 0.2,
                        frame_skip: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict and visualize gaze direction.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            ear_value: Eye Aspect Ratio value, used to check if eyes are open
            ear_threshold: EAR threshold, below which eyes are considered closed
            frame_skip: Number of frames to skip between predictions (for optimization)
            
        Returns:
            frame: Frame with visualized gaze direction
            normalized_image: Normalized face image (if available)
        """
        if not landmarks or self.onnx_session is None:
            return frame, None
        
        # If EAR value is provided and below threshold, don't visualize gaze (eyes are closed)
        if ear_value is not None and ear_value < ear_threshold:
            return frame, None
        
        # Predict gaze every frame_skip frames, use previous prediction in between
        self._frame_counter += 1
        if self._frame_counter >= frame_skip:
            # Predict gaze
            self._last_gaze_vector, self._last_normalized_image = self.predict_gaze(frame, landmarks)
            self._frame_counter = 0
        
        # If no previous prediction is available, make initial prediction
        if self._last_gaze_vector is None or self._last_normalized_image is None:
            self._last_gaze_vector, self._last_normalized_image = self.predict_gaze(frame, landmarks)
        
        gaze_vector = self._last_gaze_vector
        normalized_image = self._last_normalized_image
        
        if gaze_vector is None:
            return frame, None
        
        # Find eye centers
        # Use outer and inner corners of each eye
        left_eye_outer = 263  # Left eye outer corner
        left_eye_inner = 362  # Left eye inner corner
        right_eye_outer = 33  # Right eye outer corner
        right_eye_inner = 133  # Right eye inner corner
        
        # Check if eye points are available
        eye_points_valid = len(landmarks) > max(left_eye_outer, left_eye_inner, right_eye_outer, right_eye_inner)
        
        if eye_points_valid:
            # Calculate center of each eye
            left_eye_center_x = (landmarks[left_eye_outer][0] + landmarks[left_eye_inner][0]) / 2
            left_eye_center_y = (landmarks[left_eye_outer][1] + landmarks[left_eye_inner][1]) / 2
            
            right_eye_center_x = (landmarks[right_eye_outer][0] + landmarks[right_eye_inner][0]) / 2
            right_eye_center_y = (landmarks[right_eye_outer][1] + landmarks[right_eye_inner][1]) / 2
            
            # Calculate midpoint between eyes (origin of gaze vector)
            gaze_origin_x = int((left_eye_center_x + right_eye_center_x) / 2)
            gaze_origin_y = int((left_eye_center_y + right_eye_center_y) / 2)
            
            gaze_origin = (gaze_origin_x, gaze_origin_y)
            
            # Visualize origin point (small blue circle)
            cv2.circle(frame, gaze_origin, 3, (255, 0, 0), -1)
        else:
            # If eye points are not valid, use center of face rectangle
            face_rect = self._get_face_rect(landmarks)
            gaze_origin = (face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2)
        
        # Draw gaze direction
        frame = self.draw_gaze(frame, gaze_vector, gaze_origin)
        
        # Show gaze angles on screen
        pitch, yaw = np.rad2deg(gaze_vector)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, normalized_image
    
    def _get_face_rect(self, landmarks: List[List[float]], padding: float = 0.1) -> Tuple[int, int, int, int]:
        """
        Get face bounding rectangle from landmarks.
        
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


# Create a singleton instance
gaze_detector = GazeDetector() 