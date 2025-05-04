import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
import os
import yaml
from typing import Dict, Tuple, List, Optional, Union
from src.utils.model_loader import ModelLoader

class GazeEstimator:
    """
    A class to estimate gaze direction using the ETH-XGaze model.
    
    This class processes eye images to predict a normalized 3D gaze direction vector
    using the ETH-XGaze model.
    
    Attributes:
        model: PyTorch model for gaze estimation
        device: Device for model inference ('cpu' or 'cuda')
        face_mesh: MediaPipe FaceMesh object
        config: Configuration dictionary loaded from config.yaml
    """
    
    # Indices for left and right eye landmarks in MediaPipe FaceMesh
    LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Default input size for the ETH-XGaze model
    MODEL_INPUT_SIZE = (224, 224)
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the GazeEstimator with the ETH-XGaze model.
        
        Args:
            device: Device for model inference ('cpu' or 'cuda')
        """
        # Set device for inference
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Load the ETH-XGaze model
        model_loader = ModelLoader()
        self.model = model_loader.load_eth_xgaze_model(self.device)
        
        if self.model is None:
            print("Warning: ETH-XGaze model could not be loaded. Gaze estimation will not work.")
        
        # Initialize MediaPipe FaceMesh for facial landmark detection
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load configuration from YAML file
        self.config = self._load_config()
    
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
    
    def _extract_eye_image(self, frame: np.ndarray, landmarks: List, eye_indices: List, padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract an eye image from the frame using facial landmarks.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            eye_indices: List of indices for the eye contour
            padding: Padding factor to add around the eye region
            
        Returns:
            np.ndarray: Extracted eye image or None if extraction fails
        """
        try:
            # Get frame dimensions
            h, w, _ = frame.shape
            
            # Extract eye landmarks
            eye_landmarks = np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in eye_indices])
            
            # Calculate bounding box with padding
            min_x, min_y = np.min(eye_landmarks, axis=0)
            max_x, max_y = np.max(eye_landmarks, axis=0)
            
            # Add padding
            width = max_x - min_x
            height = max_y - min_y
            
            min_x = max(0, min_x - padding * width)
            min_y = max(0, min_y - padding * height)
            max_x = min(w, max_x + padding * width)
            max_y = min(h, max_y + padding * height)
            
            # Extract eye region
            eye_image = frame[int(min_y):int(max_y), int(min_x):int(max_x)]
            
            # Ensure the eye image is not empty
            if eye_image.size == 0:
                return None
            
            return eye_image
            
        except Exception as e:
            print(f"Error extracting eye image: {str(e)}")
            return None
    
    def _preprocess_eye_image(self, eye_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the eye image for the ETH-XGaze model.
        
        Args:
            eye_image: Input eye image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Resize to model input size
        eye_image = cv2.resize(eye_image, self.MODEL_INPUT_SIZE)
        
        # Convert BGR to RGB
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
        
        # Normalize image
        eye_image = eye_image.astype(np.float32) / 255.0
        
        # Standardize using ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        eye_image = (eye_image - mean) / std
        
        # Convert to tensor and add batch dimension
        eye_tensor = torch.tensor(eye_image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        
        return eye_tensor.to(self.device)
    
    def _postprocess_gaze(self, gaze_output: torch.Tensor) -> np.ndarray:
        """
        Postprocess the model output to get the normalized gaze vector.
        
        Args:
            gaze_output: Raw model output tensor
            
        Returns:
            np.ndarray: Normalized 3D gaze direction vector [x, y, z]
        """
        # Convert to numpy
        gaze = gaze_output.detach().cpu().numpy()[0]
        
        # Normalize the gaze vector
        gaze_norm = np.linalg.norm(gaze)
        if gaze_norm > 0:
            gaze = gaze / gaze_norm
        
        return gaze
    
    def detect_from_eye_image(self, eye_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict gaze direction from a pre-extracted eye image.
        
        Args:
            eye_image: Input eye image
            
        Returns:
            np.ndarray: Normalized 3D gaze direction vector [x, y, z] or None if prediction fails
        """
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            preprocessed = self._preprocess_eye_image(eye_image)
            
            # Run inference
            with torch.no_grad():
                output = self.model(preprocessed)
            
            # Postprocess output
            gaze_direction = self._postprocess_gaze(output)
            
            return gaze_direction
            
        except Exception as e:
            print(f"Error in gaze prediction: {str(e)}")
            return None
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect gaze direction from a full frame with face.
        
        Args:
            frame: Input frame/image
            
        Returns:
            Tuple containing:
            - Average gaze direction vector (None if no face detected)
            - Left eye gaze direction vector (None if not available)
            - Right eye gaze direction vector (None if not available)
        """
        if self.model is None:
            return None, None, None
        
        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Process the frame with MediaPipe FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        # Default return values if no face is detected
        avg_gaze = None
        left_gaze = None
        right_gaze = None
        
        if results.multi_face_landmarks:
            # Get the first face detected
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert normalized coordinates to pixel coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                x, y, z = landmark.x * w, landmark.y * h, landmark.z
                landmarks.append([x, y, z])
            
            # Extract eye images
            left_eye_img = self._extract_eye_image(frame, landmarks, self.LEFT_EYE_CONTOUR)
            right_eye_img = self._extract_eye_image(frame, landmarks, self.RIGHT_EYE_CONTOUR)
            
            # Predict gaze for each eye if available
            if left_eye_img is not None:
                left_gaze = self.detect_from_eye_image(left_eye_img)
            
            if right_eye_img is not None:
                right_gaze = self.detect_from_eye_image(right_eye_img)
            
            # Calculate average gaze direction if both eyes are available
            if left_gaze is not None and right_gaze is not None:
                avg_gaze = (left_gaze + right_gaze) / 2.0
                # Normalize average gaze vector
                avg_gaze_norm = np.linalg.norm(avg_gaze)
                if avg_gaze_norm > 0:
                    avg_gaze = avg_gaze / avg_gaze_norm
            elif left_gaze is not None:
                avg_gaze = left_gaze
            elif right_gaze is not None:
                avg_gaze = right_gaze
        
        return avg_gaze, left_gaze, right_gaze
    
    def visualize(self, frame: np.ndarray, gaze_direction: np.ndarray, 
                 origin_point: np.ndarray, length: float = 50.0, 
                 color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Visualize the gaze direction vector on the frame.
        
        Args:
            frame: Input frame/image
            gaze_direction: 3D gaze direction vector [x, y, z]
            origin_point: 2D origin point for the vector [x, y]
            length: Length of the visualization arrow
            color: Color of the visualization arrow (BGR)
            
        Returns:
            np.ndarray: Frame with visualized gaze direction
        """
        if gaze_direction is None or origin_point is None:
            return frame
        
        # Create a copy of the frame
        viz_frame = frame.copy()
        
        # Extract vector components
        x, y, z = gaze_direction
        
        # Calculate endpoint of the vector projection
        # Note: y-axis is inverted in image coordinates
        end_point = (
            int(origin_point[0] + length * x),
            int(origin_point[1] - length * y)  # Invert y for image coordinates
        )
        
        # Draw the line
        cv2.arrowedLine(viz_frame, 
                       (int(origin_point[0]), int(origin_point[1])), 
                       end_point, 
                       color, 
                       2)
        
        return viz_frame
