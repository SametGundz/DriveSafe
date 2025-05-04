import cv2
import numpy as np
import mediapipe as mp
import yaml
from typing import Dict, Tuple, List, Optional
import os

class EARMARDetector:
    """
    A class to detect Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) using MediaPipe FaceMesh.
    
    Attributes:
        face_mesh: MediaPipe FaceMesh object
        config: Configuration dictionary loaded from config.yaml
        left_eye_indices: List of landmark indices for the left eye
        right_eye_indices: List of landmark indices for the right eye
        mouth_indices: List of landmark indices for the mouth
        ear_threshold: Threshold for determining if eyes are closed
        mar_threshold: Threshold for determining if mouth is open
    """

    # Landmark indices for face components
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Left eye landmark indices
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Right eye landmark indices
    MOUTH_INDICES = [61, 291, 0, 17, 269, 405]  # Mouth landmark indices

    def __init__(self):
        """Initialize the EARMARDetector with MediaPipe and configurations."""
        # Initialize MediaPipe FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load configuration from YAML file
        self.config = self._load_config()
        
        # Set thresholds from config or use defaults
        self.ear_threshold = self.config.get('ear_threshold', 0.2)
        self.mar_threshold = self.config.get('mar_threshold', 0.6)

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

    def _calculate_ear(self, landmarks, eye_indices) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) for a given eye.
        
        Args:
            landmarks: List of all facial landmarks
            eye_indices: Indices of landmarks forming the eye
            
        Returns:
            float: The calculated EAR value
        """
        # Get the eye landmarks coordinates
        eye_landmarks = [landmarks[idx] for idx in eye_indices]
        
        # Calculate horizontal distance (eye width)
        horizontal_dist = np.linalg.norm(
            np.array([eye_landmarks[0][0], eye_landmarks[0][1]]) - 
            np.array([eye_landmarks[3][0], eye_landmarks[3][1]])
        )
        
        # Calculate vertical distances
        vertical_dist1 = np.linalg.norm(
            np.array([eye_landmarks[1][0], eye_landmarks[1][1]]) - 
            np.array([eye_landmarks[5][0], eye_landmarks[5][1]])
        )
        vertical_dist2 = np.linalg.norm(
            np.array([eye_landmarks[2][0], eye_landmarks[2][1]]) - 
            np.array([eye_landmarks[4][0], eye_landmarks[4][1]])
        )
        
        # Calculate EAR: Average of vertical distances divided by horizontal distance
        if horizontal_dist == 0:
            return 0.0
        
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear

    def _calculate_mar(self, landmarks, mouth_indices) -> float:
        """
        Calculate the Mouth Aspect Ratio (MAR).
        
        Args:
            landmarks: List of all facial landmarks
            mouth_indices: Indices of landmarks forming the mouth
            
        Returns:
            float: The calculated MAR value
        """
        # Get the mouth landmarks coordinates
        mouth_landmarks = [landmarks[idx] for idx in mouth_indices]
        
        # Calculate horizontal distance (mouth width)
        horizontal_dist = np.linalg.norm(
            np.array([mouth_landmarks[0][0], mouth_landmarks[0][1]]) - 
            np.array([mouth_landmarks[1][0], mouth_landmarks[1][1]])
        )
        
        # Calculate vertical distance (mouth height)
        vertical_dist = np.linalg.norm(
            np.array([mouth_landmarks[2][0], mouth_landmarks[2][1]]) - 
            np.array([mouth_landmarks[4][0], mouth_landmarks[4][1]])
        )
        
        # Calculate MAR: Vertical distance divided by horizontal distance
        if horizontal_dist == 0:
            return 0.0
        
        mar = vertical_dist / horizontal_dist
        return mar

    def detect(self, frame) -> Tuple[Optional[float], Optional[float], bool, bool]:
        """
        Detect EAR and MAR values from a frame.
        
        Args:
            frame: Input frame/image
            
        Returns:
            Tuple containing:
            - Average EAR value (None if no face detected)
            - MAR value (None if no face detected)
            - Boolean indicating if eyes are closed
            - Boolean indicating if mouth is open
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Process the frame with MediaPipe FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        # Default return values if no face is detected
        avg_ear = None
        mar = None
        eyes_closed = False
        mouth_open = False
        
        if results.multi_face_landmarks:
            # Get the first face detected
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert normalized coordinates to pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x, y, z = landmark.x * w, landmark.y * h, landmark.z
                landmarks.append([x, y, z])
            
            # Calculate left and right EAR
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
            
            # Calculate average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR
            mar = self._calculate_mar(landmarks, self.MOUTH_INDICES)
            
            # Determine if eyes are closed and mouth is open based on thresholds
            eyes_closed = avg_ear < self.ear_threshold
            mouth_open = mar > self.mar_threshold
        
        return avg_ear, mar, eyes_closed, mouth_open

    def visualize(self, frame, ear, mar, eyes_closed, mouth_open) -> np.ndarray:
        """
        Visualize EAR and MAR values on the frame.
        
        Args:
            frame: Input frame/image
            ear: Eye Aspect Ratio value
            mar: Mouth Aspect Ratio value
            eyes_closed: Boolean indicating if eyes are closed
            mouth_open: Boolean indicating if mouth is open
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw EAR and MAR values on the frame
        eye_status = "CLOSED" if eyes_closed else "OPEN"
        mouth_status = "OPEN" if mouth_open else "CLOSED"
        
        ear_text = f"EAR: {ear:.2f} (Eyes {eye_status})" if ear is not None else "EAR: No Face"
        mar_text = f"MAR: {mar:.2f} (Mouth {mouth_status})" if mar is not None else "MAR: No Face"
        
        # Set color based on status
        ear_color = (0, 0, 255) if eyes_closed else (0, 255, 0)  # Red if closed, green if open
        mar_color = (0, 0, 255) if mouth_open else (0, 255, 0)  # Red if open, green if closed
        
        # Put text on frame
        cv2.putText(vis_frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        cv2.putText(vis_frame, mar_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mar_color, 2)
        
        return vis_frame

