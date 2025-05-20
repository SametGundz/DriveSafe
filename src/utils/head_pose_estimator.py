#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Head pose estimation module for the driver drowsiness detection system.

This module provides the HeadPoseEstimator class that handles:
- Head pose estimation using facial landmarks
- Calculation of pitch, yaw, and roll angles
- Visualization of head pose with 3D axes or a 3D cube
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Optional

class HeadPoseEstimator:
    """
    A class for estimating head pose from facial landmarks.
    
    This class provides methods to:
    - Calculate head pose (pitch, yaw, roll) from facial landmarks
    - Visualize head pose with 3D axes or a cube
    """
    
    # Default 3D model points for standard 6-point head pose estimation
    # These points correspond to specific facial landmarks:
    # - Nose tip
    # - Chin
    # - Left eye left corner
    # - Left mouth corner
    # - Right eye right corner
    # - Right mouth corner
    MODEL_POINTS = np.array([
        [0.0, 0.0, 0.0],           # Nose tip (origin)
        [0.0, -330.0, -65.0],      # Chin
        [-225.0, 170.0, -135.0],   # Left eye left corner
        [-150.0, -150.0, -125.0],  # Left mouth corner
        [225.0, 170.0, -135.0],    # Right eye right corner
        [150.0, -150.0, -125.0]    # Right mouth corner
    ], dtype=np.float64)
    
    # Head pose estimation landmark indices
    # These indices map MediaPipe landmarks to the 6 points used in head pose estimation
    HEAD_POSE_LANDMARKS = [1, 9, 57, 130, 287, 359]
    
    def __init__(self):
        """Initialize the HeadPoseEstimator."""
        pass
    
    def calculate_head_pose(self, landmarks: List[List[float]], frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Calculate head pose from facial landmarks.
        
        Args:
            landmarks: List of facial landmarks
            frame: Input frame for calculating image dimensions
            
        Returns:
            frame: Input frame (unchanged)
            angles: Tuple of (pitch, yaw, roll) angles in degrees
        """
        # If no landmarks are detected, return default values
        if not landmarks or len(landmarks) < max(self.HEAD_POSE_LANDMARKS) + 1:
            return frame, (0.0, 0.0, 0.0)
        
        try:    
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
            
            # Extract 2D positions of the landmarks used for head pose
            face_coordinates = []
            invalid_landmarks = False
            
            for idx in self.HEAD_POSE_LANDMARKS:
                if idx < len(landmarks):
                    if isinstance(landmarks[idx], list) and len(landmarks[idx]) >= 2:
                        x, y = landmarks[idx][0], landmarks[idx][1]
                        # Validate coordinates are within frame bounds and not NaN
                        if (0 <= x < w and 0 <= y < h and 
                            not np.isnan(x) and not np.isnan(y)):
                            face_coordinates.append([x, y])
                        else:
                            invalid_landmarks = True
                            break
                else:
                    invalid_landmarks = True
                    break
            
            # If we don't have all the required landmarks or they are invalid, return default values
            if invalid_landmarks or len(face_coordinates) != len(self.HEAD_POSE_LANDMARKS):
                return frame, (0.0, 0.0, 0.0)
            
            face_coordinates = np.array(face_coordinates, dtype=np.float64)
            
            # Solve the PnP problem to get rotation and translation vectors
            ret, rvec, tvec = cv2.solvePnP(
                self.MODEL_POINTS, 
                face_coordinates, 
                camera_matrix, 
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not ret:
                return frame, (0.0, 0.0, 0.0)
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Calculate Euler angles from rotation matrix
            angles = self.rotation_matrix_to_angles(rotation_matrix)
            
            return frame, angles
            
        except Exception as e:
            # Log error and return default values
            print(f"Error in head pose calculation: {str(e)}")
            return frame, (0.0, 0.0, 0.0)
    
    def rotation_matrix_to_angles(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a rotation matrix to Euler angles (pitch, yaw, roll) using a more robust method.
        
        This implementation handles singularities (gimbal lock) better than the basic method.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Tuple of (pitch, yaw, roll) angles in degrees
        """
        try:
            # Check if valid rotation matrix
            if rotation_matrix.shape != (3, 3):
                return (0.0, 0.0, 0.0)
            
            # Handle singularity case (gimbal lock)
            # Check for gimbal lock: when rotation_matrix[2,0] is close to +/-1
            threshold = 0.998
            if abs(rotation_matrix[2, 0]) > threshold:
                # Gimbal lock detected
                # Set yaw (y) to ±90 degrees based on the sign of rotation_matrix[2,0]
                y = -math.copysign(math.pi/2, rotation_matrix[2, 0])
                # In gimbal lock, roll and pitch are coupled
                z = 0.0  # Arbitrary choice for roll
                # Compute pitch given our roll choice
                x = math.atan2(rotation_matrix[0, 1], rotation_matrix[1, 1])
            else:
                # Normal case - no gimbal lock
                # Y rotation (yaw)
                y = math.asin(-rotation_matrix[2, 0])
                cos_y = math.cos(y)
                
                # X rotation (pitch)
                x = math.atan2(rotation_matrix[2, 1] / cos_y, rotation_matrix[2, 2] / cos_y)
                
                # Z rotation (roll)
                z = math.atan2(rotation_matrix[1, 0] / cos_y, rotation_matrix[0, 0] / cos_y)
            
            # Create debug log with angles in degrees 
            angles_degrees = (x * 180.0 / math.pi, y * 180.0 / math.pi, z * 180.0 / math.pi)
            
            # Filter out NaN values
            if any(math.isnan(angle) for angle in angles_degrees):
                return (0.0, 0.0, 0.0)
            
            return angles_degrees
            
        except Exception as e:
            # Handle any errors in calculation
            print(f"Error calculating angles from rotation matrix: {str(e)}")
            return (0.0, 0.0, 0.0)
    
    def draw_head_pose_axes(self, frame: np.ndarray, landmarks: List[List[float]], 
                           length: int = 50) -> np.ndarray:
        """
        Draw 3D axes showing head pose direction.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            length: Length of the axes arrows
            
        Returns:
            np.ndarray: Frame with visualized head pose axes
        """
        if not landmarks:
            return frame
            
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
        
        # Extract 2D positions of the landmarks used for head pose
        face_coordinates = []
        for idx in self.HEAD_POSE_LANDMARKS:
            if idx < len(landmarks):
                x, y = landmarks[idx][0], landmarks[idx][1]
                face_coordinates.append([x, y])
                
        # If we don't have all the required landmarks, return original frame
        if len(face_coordinates) != len(self.HEAD_POSE_LANDMARKS):
            return frame
            
        face_coordinates = np.array(face_coordinates, dtype=np.float64)
        
        # Solve the PnP problem to get rotation and translation vectors
        ret, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS, 
            face_coordinates, 
            camera_matrix, 
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not ret:
            return frame
            
        # Define the origin point and axis endpoints
        axis_points = np.float64([
            [0, 0, 0],           # Origin
            [length, 0, 0],      # X-axis (red)
            [0, length, 0],      # Y-axis (green)
            [0, 0, length]       # Z-axis (blue)
        ])
        
        # Project 3D points to the image plane
        axis_points_2d, _ = cv2.projectPoints(
            axis_points, rvec, tvec, camera_matrix, distortion
        )
        
        # Convert to integer points
        axis_points_2d = np.int32(axis_points_2d.reshape(-1, 2))
        
        # Draw the axes
        # X-axis (red)
        cv2.line(frame, tuple(axis_points_2d[0]), tuple(axis_points_2d[1]), (0, 0, 255), 1)
        # Y-axis (green)
        cv2.line(frame, tuple(axis_points_2d[0]), tuple(axis_points_2d[2]), (0, 255, 0), 1)
        # Z-axis (blue)
        cv2.line(frame, tuple(axis_points_2d[0]), tuple(axis_points_2d[3]), (255, 0, 0), 1)
        
        return frame
    
    def draw_head_pose_cube(self, frame: np.ndarray, landmarks: List[List[float]], 
                           cube_size: Optional[int] = None) -> np.ndarray:
        """
        Draw a 3D cube showing head orientation.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            cube_size: Size of the cube (if None, calculated based on frame width)
            
        Returns:
            np.ndarray: Frame with visualized head pose cube
        """
        if not landmarks:
            return frame
            
        # Set default cube size
        if cube_size is None:
            h, w = frame.shape[:2]
            cube_size = w // 5
        
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
        
        # Extract 2D positions of the landmarks used for head pose
        face_coordinates = []
        for idx in self.HEAD_POSE_LANDMARKS:
            if idx < len(landmarks):
                x, y = landmarks[idx][0], landmarks[idx][1]
                face_coordinates.append([x, y])
                
        # If we don't have all the required landmarks, return original frame
        if len(face_coordinates) != len(self.HEAD_POSE_LANDMARKS):
            return frame
            
        face_coordinates = np.array(face_coordinates, dtype=np.float64)
        
        # Solve the PnP problem to get rotation and translation vectors
        ret, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS, 
            face_coordinates, 
            camera_matrix, 
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not ret:
            return frame
            
        # Define the cube points
        half_size = cube_size / 2
        cube_points = np.float64([
            [-half_size, -half_size, -half_size],  # Bottom back left corner
            [half_size, -half_size, -half_size],   # Bottom back right corner
            [half_size, half_size, -half_size],    # Top back right corner
            [-half_size, half_size, -half_size],   # Top back left corner
            [-half_size, -half_size, half_size],   # Bottom front left corner
            [half_size, -half_size, half_size],    # Bottom front right corner
            [half_size, half_size, half_size],     # Top front right corner
            [-half_size, half_size, half_size]     # Top front left corner
        ])
        
        # Project 3D points to the image plane
        cube_points_2d, _ = cv2.projectPoints(
            cube_points, rvec, tvec, camera_matrix, distortion
        )
        
        # Convert to integer points
        cube_points_2d = np.int32(cube_points_2d.reshape(-1, 2))
        
        # Define colors - BGR format
        RED = (0, 0, 255)       # Red (bottom face)
        BLUE = (255, 0, 0)      # Blue (top face)
        GREEN = (0, 255, 0)     # Green (vertical edges)
        
        # Draw the cube
        # Bottom face - Red
        cv2.line(frame, tuple(cube_points_2d[0]), tuple(cube_points_2d[1]), RED, 2)
        cv2.line(frame, tuple(cube_points_2d[1]), tuple(cube_points_2d[2]), RED, 2)
        cv2.line(frame, tuple(cube_points_2d[2]), tuple(cube_points_2d[3]), RED, 2)
        cv2.line(frame, tuple(cube_points_2d[3]), tuple(cube_points_2d[0]), RED, 2)
        
        # Top face - Blue
        cv2.line(frame, tuple(cube_points_2d[4]), tuple(cube_points_2d[5]), BLUE, 2)
        cv2.line(frame, tuple(cube_points_2d[5]), tuple(cube_points_2d[6]), BLUE, 2)
        cv2.line(frame, tuple(cube_points_2d[6]), tuple(cube_points_2d[7]), BLUE, 2)
        cv2.line(frame, tuple(cube_points_2d[7]), tuple(cube_points_2d[4]), BLUE, 2)
        
        # Vertical edges - Green
        cv2.line(frame, tuple(cube_points_2d[0]), tuple(cube_points_2d[4]), GREEN, 2)
        cv2.line(frame, tuple(cube_points_2d[1]), tuple(cube_points_2d[5]), GREEN, 2)
        cv2.line(frame, tuple(cube_points_2d[2]), tuple(cube_points_2d[6]), GREEN, 2)
        cv2.line(frame, tuple(cube_points_2d[3]), tuple(cube_points_2d[7]), GREEN, 2)
        
        return frame
    
    def visualize_head_pose(self, frame: np.ndarray, landmarks: List[List[float]], 
                           show_axes: bool = True, show_angles: bool = True,
                           visualization_type: str = 'axes') -> np.ndarray:
        """
        Visualize head pose on the frame.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            show_axes: Whether to show the axes/cube
            show_angles: Whether to show angle values on the frame
            visualization_type: Type of visualization ('axes' or 'cube')
            
        Returns:
            np.ndarray: Frame with visualized head pose
        """
        if not landmarks:
            return frame
            
        # Calculate head pose
        vis_frame, angles = self.calculate_head_pose(landmarks, frame.copy())
        pitch, yaw, roll = angles
        
        if show_axes:
            if visualization_type == 'cube':
                vis_frame = self.draw_head_pose_cube(vis_frame, landmarks)
            else:  # Default to axes
                vis_frame = self.draw_head_pose_axes(vis_frame, landmarks)
        
        if show_angles:
            # Show angle values
            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), angles)):
                k, v = info
                text = f"{k}: {int(v)}"
                cv2.putText(vis_frame, text, (20, i*30 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        
        return vis_frame


# Create a singleton instance
head_pose_estimator = HeadPoseEstimator() 