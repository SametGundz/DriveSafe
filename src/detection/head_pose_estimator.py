import numpy as np
import cv2
import mediapipe as mp
import os
from typing import Tuple, Optional, List, Dict, Union
import yaml

class HeadPoseEstimator:
    """
    A class to estimate head pose (pitch, yaw, roll) using MediaPipe face landmarks and a 3D reference model.
    
    Attributes:
        face_mesh: MediaPipe FaceMesh object
        model_3d: 3D reference model points for head pose estimation
        camera_matrix: Camera matrix for perspective projection
        dist_coeffs: Distortion coefficients
        config: Configuration dictionary loaded from config.yaml
        use_degrees: Whether to return angles in degrees (True) or radians (False)
    """
    
    # Key facial landmarks for pose estimation
    # Selected landmarks around the face outline, eyes, nose, and mouth for stable pose estimation
    POSE_LANDMARKS = [33, 263, 1, 61, 291, 199, 6, 4, 
                      19, 94, 2, 164, 0, 11, 12, 13, 14, 
                      15, 16, 17, 18, 200, 199, 175]

    def __init__(self, use_degrees: bool = True):
        """
        Initialize the HeadPoseEstimator with MediaPipe and 3D model.
        
        Args:
            use_degrees: Whether to return angles in degrees (True) or radians (False)
        """
        # Initialize MediaPipe FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load 3D reference model
        self.model_3d = self._load_3d_model()
        
        # Load configuration from YAML file
        self.config = self._load_config()
        
        # Set to return angles in degrees or radians
        self.use_degrees = use_degrees
        
        # Initialize camera matrix and distortion coefficients with default values
        # These will be updated with frame dimensions when processing
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def _load_3d_model(self) -> np.ndarray:
        """
        Load 3D reference model for head pose estimation.
        
        Returns:
            np.ndarray: 3D reference model points
        """
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                 'models', 'headpose_3d_model.npy')
        try:
            model_3d = np.load(model_path)
            return model_3d
        except FileNotFoundError:
            print(f"Warning: 3D reference model not found at {model_path}. Using default model.")
            # Create a generic 3D model of a face if the file is not found
            return self._create_default_3d_model()

    def _create_default_3d_model(self) -> np.ndarray:
        """
        Create a default 3D face model if the reference model file is not available.
        
        Returns:
            np.ndarray: Default 3D model points
        """
        # Generic 3D model of a face (simplified)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ])
        return model_points

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                  'config', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except (FileNotFoundError, yaml.YAMLError):
            print(f"Warning: Could not load config file at {config_path}. Using default values.")
            return {}

    def _get_camera_matrix(self, frame_size: Tuple[int, int]) -> np.ndarray:
        """
        Get camera matrix based on frame size.
        
        Args:
            frame_size: Tuple containing frame width and height
            
        Returns:
            np.ndarray: Camera matrix
        """
        width, height = frame_size
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float32
        )
        return camera_matrix

    def detect(self, frame) -> Tuple[Optional[float], Optional[float], Optional[float], np.ndarray]:
        """
        Detect head pose from frame.
        
        Args:
            frame: Input frame/image
            
        Returns:
            Tuple containing:
            - Pitch angle (rotation around X-axis)
            - Yaw angle (rotation around Y-axis)
            - Roll angle (rotation around Z-axis)
            - Rotation vector (used for visualization)
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Update camera matrix based on frame size
        self.camera_matrix = self._get_camera_matrix((w, h))
        
        # Process the frame with MediaPipe FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        # Default return values if no face is detected
        pitch = None
        yaw = None
        roll = None
        rotation_vector = np.zeros((3, 1), dtype=np.float32)
        
        if results.multi_face_landmarks:
            # Get the first face detected
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract the selected landmarks for pose estimation
            image_points = []
            for idx in self.POSE_LANDMARKS:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    image_points.append((x, y))
            
            if len(image_points) >= 4:  # Need at least 4 points for solvePnP
                image_points = np.array(image_points, dtype=np.float32)
                
                # Use only the first corresponding points from the 3D model 
                # if we have more landmarks than model points
                model_points = self.model_3d[:len(image_points)] if len(image_points) <= len(self.model_3d) else self.model_3d
                
                # Solve for pose
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, 
                    image_points, 
                    self.camera_matrix, 
                    self.dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    
                    # Get Euler angles from rotation matrix
                    angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
                    pitch, yaw, roll = angles
                    
                    # Convert to degrees if required
                    if self.use_degrees:
                        pitch = np.degrees(pitch)
                        yaw = np.degrees(yaw)
                        roll = np.degrees(roll)
        
        return pitch, yaw, roll, rotation_vector

    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll).
        
        Args:
            R: Rotation matrix
            
        Returns:
            Tuple containing pitch, yaw, and roll angles in radians
        """
        # Ensure the rotation matrix is valid
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        # Check if singular (gimbal lock)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # pitch
            y = np.arctan2(-R[2, 0], sy)      # yaw
            z = np.arctan2(R[1, 0], R[0, 0])  # roll
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return x, y, z  # pitch, yaw, roll

    def visualize(self, frame, pitch: float, yaw: float, roll: float, rotation_vector: np.ndarray) -> np.ndarray:
        """
        Visualize head pose on the frame.
        
        Args:
            frame: Input frame/image
            pitch: Pitch angle (rotation around X-axis)
            yaw: Yaw angle (rotation around Y-axis)
            roll: Roll angle (rotation around Z-axis)
            rotation_vector: Rotation vector for pose visualization
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        vis_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Draw pose information as text
        if pitch is not None and yaw is not None and roll is not None:
            # Format angles with 2 decimal places
            pitch_text = f"Pitch: {pitch:.2f}{'°' if self.use_degrees else ' rad'}"
            yaw_text = f"Yaw: {yaw:.2f}{'°' if self.use_degrees else ' rad'}"
            roll_text = f"Roll: {roll:.2f}{'°' if self.use_degrees else ' rad'}"
            
            # Put text on frame
            cv2.putText(vis_frame, pitch_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, yaw_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, roll_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw 3D axes to visualize orientation
            # Define the origin of the coordinate system as the nose tip
            nose_tip = (int(w / 2), int(h / 2))
            
            # Project 3D axes onto the image plane
            axis_length = 100  # length of the axes in 3D space
            axes_3d = np.float32([[0, 0, 0], [axis_length, 0, 0], 
                                  [0, axis_length, 0], [0, 0, axis_length]])
            
            translation_vector = np.array([0, 0, 500], dtype=np.float32).reshape(3, 1)
            axes_2d, _ = cv2.projectPoints(axes_3d, rotation_vector, translation_vector, 
                                         self.camera_matrix, self.dist_coeffs)
            
            # Draw the axes
            origin = (int(axes_2d[0][0][0]), int(axes_2d[0][0][1]))
            
            # X-axis (red)
            cv2.line(vis_frame, origin, (int(axes_2d[1][0][0]), int(axes_2d[1][0][1])), (0, 0, 255), 3)
            
            # Y-axis (green)
            cv2.line(vis_frame, origin, (int(axes_2d[2][0][0]), int(axes_2d[2][0][1])), (0, 255, 0), 3)
            
            # Z-axis (blue)
            cv2.line(vis_frame, origin, (int(axes_2d[3][0][0]), int(axes_2d[3][0][1])), (255, 0, 0), 3)
        
        return vis_frame
