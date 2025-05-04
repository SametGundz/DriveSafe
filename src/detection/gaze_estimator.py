#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bakış Yönü Tahmini (Gaze Direction Estimation) modülü.

Bu modül, ETH-XGaze modelini kullanarak sürücünün bakış yönünü (yaw, pitch açıları)
tahmin etmek için tasarlanmıştır. Model, yüz görüntüsünü girdi olarak alır ve 
yaw ve pitch açılarını çıktı olarak verir.

Landmark Indeksleri:
Bu modül, MediaPipe Face Mesh tarafından sağlanan yüz işaret noktalarını (landmarks) kullanır.
Gaze tahmini için gerekli olan özel landmark indeksleri şunlardır:
- Sol göz köşeleri: 33, 133
- Sağ göz köşeleri: 362, 263
- Burun: 4, 5

Doğru çalışması için, bu indekslerin MediaPipe Face Mesh çıktısında mevcut olması gerekir.
Tam bir MediaPipe Face Mesh noktaları referansı için: 
https://developers.google.com/mediapipe/solutions/vision/face_landmarker
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger('driver_monitoring')

class GazeResNet(nn.Module):
    """
    Gaze estimation model based on ResNet50 architecture from ETH-XGaze
    """
    def __init__(self):
        super(GazeResNet, self).__init__()
        logger.debug("Creating model architecture (GazeResNet)...")
        
        # Load pretrained ResNet50 model
        self.gaze_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Change last fully connected layer for gaze estimation (2 values: yaw and pitch)
        # Update: The ETH-XGaze model has gaze_fc as a sequential layer
        # with index 0 (matching the state dict keys "gaze_fc.0.weight" and "gaze_fc.0.bias")
        self.gaze_fc = nn.Sequential(nn.Linear(2048, 2))
    
    def forward(self, x):
        # Feature extraction using ResNet
        x = self.gaze_network.conv1(x)
        x = self.gaze_network.bn1(x)
        x = self.gaze_network.relu(x)
        x = self.gaze_network.maxpool(x)
        
        x = self.gaze_network.layer1(x)
        x = self.gaze_network.layer2(x)
        x = self.gaze_network.layer3(x)
        x = self.gaze_network.layer4(x)
        
        x = self.gaze_network.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Gaze direction estimation
        gaze = self.gaze_fc(x)
        
        return gaze

class GazeEstimator:
    """
    Gaze direction estimator using ETH-XGaze model
    """
    def __init__(self, model_path=None):
        """
        Initialize gaze estimator
        
        Args:
            model_path: Path to the model weights file
        """
        # Search for model file in possible locations
        if model_path is None:
            model_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pretrained', 'eth_xgaze.pth.tar'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pretrained', 'eth_xgaze.pth'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'pretrained', 'eth_xgaze.pth.tar'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'pretrained', 'eth_xgaze.pth'),
                os.path.join(os.path.dirname(__file__), 'models', 'eth_xgaze.pth.tar'),
                os.path.join(os.path.dirname(__file__), 'models', 'eth_xgaze.pth'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'eth_xgaze_model.pth.tar'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'eth_xgaze_model.pth'),
            ]
            
            for path in model_paths:
                if os.path.isfile(path):
                    model_path = path
                    logger.info(f"Model dosyası bulundu: {model_path}")
                    break
            
            if model_path is None:
                logger.error("Beklenen hiçbir konumda model dosyası bulunamadı.")
                raise FileNotFoundError("ETH-XGaze model dosyası bulunamadı")
        
        # Set device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Kullanılan cihaz: {self.device}")
        
        # Load face model for head pose estimation
        self._load_face_model()
        
        # Create model and load weights
        self.model = self._load_model(model_path)
        
        # Image normalization for model input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_face_model(self):
        """
        Load 3D face model for head pose estimation
        """
        try:
            # Try to load from the file
            face_model_path = os.path.join(os.path.dirname(__file__), 'face_model.txt')
            
            # Also check the ETH-XGaze model path in the project root
            eth_face_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                             'ETH-XGaze', 'face_model.txt')
            
            if os.path.isfile(face_model_path):
                logger.debug(f"Yüz modeli yükleniyor: {face_model_path}")
                face_model_load = np.loadtxt(face_model_path)
                
                # For ETH-XGaze, use the proper indices as in their original implementation
                # The indices are different based on which face model is used
                if len(face_model_load) >= 51:  # Full ETH-XGaze face model has 51 points
                    logger.debug(f"{len(face_model_load)} noktalı ETH-XGaze yüz modeli kullanılıyor")
                    # Standard ETH-XGaze landmarks: left eye, right eye, nose
                    landmark_use = [36, 39, 42, 45, 30, 31]
                elif len(face_model_load) >= 30:  # Partial model
                    logger.debug(f"{len(face_model_load)} noktalı orta büyüklükte yüz modeli kullanılıyor")
                    landmark_use = [20, 23, 26, 29, 15, 19]
                else:  # Small model
                    logger.debug(f"{len(face_model_load)} noktalı küçük yüz modeli kullanılıyor")
                    landmark_use = list(range(min(6, len(face_model_load))))
                
                # Check if we have valid indices
                if max(landmark_use) < len(face_model_load):
                    self.face_model = face_model_load[landmark_use, :]
                    logger.debug(f"Yüz modeli başarıyla yüklendi, kullanılan işaret noktası indeksleri: {landmark_use}")
                else:
                    logger.warning(f"{len(face_model_load)} boyutlu yüz modeli için geçersiz işaret noktası indeksleri")
                    self.face_model = self._get_simplified_face_model()
            
            # If local file doesn't exist or has issues, try ETH-XGaze path
            elif os.path.isfile(eth_face_model_path):
                logger.info(f"ETH-XGaze yüz modeli yükleniyor: {eth_face_model_path}")
                face_model_load = np.loadtxt(eth_face_model_path)
                
                # Copy the file to our inference directory for future use
                try:
                    import shutil
                    shutil.copy(eth_face_model_path, face_model_path)
                    logger.info(f"ETH-XGaze yüz modeli kopyalandı: {face_model_path}")
                except Exception as e:
                    logger.warning(f"Yüz modeli dosyası kopyalanamadı: {e}")
                
                if len(face_model_load) >= 51:  # Full ETH-XGaze face model
                    # Using standard landmarks from ETH-XGaze implementation
                    # These correspond to eye corners and nose points
                    landmark_use = [36, 39, 42, 45, 30, 31]
                    self.face_model = face_model_load[landmark_use, :]
                    logger.debug(f"ETH-XGaze işaret noktası indeksleri kullanılıyor: {landmark_use}")
                else:
                    # Fallback to first 6 points if not enough landmarks
                    landmark_use = list(range(min(6, len(face_model_load))))
                    self.face_model = face_model_load[landmark_use, :]
                    logger.debug(f"ETH-XGaze modelinden ilk {len(landmark_use)} nokta kullanılıyor")
                
                logger.info("ETH-XGaze yüz modeli başarıyla yüklendi")
            
            else:
                # Fallback to hardcoded model
                logger.warning(f"Yüz modeli dosyası bulunamadı: {face_model_path} veya {eth_face_model_path}")
                logger.warning("Basitleştirilmiş yüz modeli kullanılıyor")
                self.face_model = self._get_simplified_face_model()
            
        except Exception as e:
            logger.error(f"Yüz modeli yükleme hatası: {e}")
            logger.warning("Hata nedeniyle basitleştirilmiş yüz modeli kullanılıyor")
            self.face_model = self._get_simplified_face_model()
            
        # Validate the face model has the correct shape
        if self.face_model.shape[0] != 6 or self.face_model.shape[1] != 3:
            logger.warning(f"Yüz modeli geçersiz şekle sahip: {self.face_model.shape}, beklenen: (6, 3)")
            logger.warning("Basitleştirilmiş yüz modeline sıfırlanıyor")
            self.face_model = self._get_simplified_face_model()
        else:
            logger.debug(f"Yüz modeli başarıyla yüklendi, şekil: {self.face_model.shape}")
    
    def _get_simplified_face_model(self):
        """
        Return a simplified face model for PnP algorithm
        
        Returns:
            3D face model with 6 landmarks matching MediaPipe's face mesh points
        """
        return np.array([
            [-35.0, -50.0, -30.0],  # Left eye left corner
            [-10.0, -50.0, -30.0],  # Left eye right corner
            [10.0, -50.0, -30.0],   # Right eye left corner
            [35.0, -50.0, -30.0],   # Right eye right corner
            [-20.0, 0.0, -50.0],    # Left nose point
            [20.0, 0.0, -50.0],     # Right nose point
        ])
    
    def _load_model(self, model_path):
        """
        Load the gaze estimation model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            PyTorch model with loaded weights
        """
        logger.debug(f"Model ağırlıkları yükleniyor: {model_path}")
        
        try:
            # Create model
            model = GazeResNet()
            model = model.to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            logger.debug(f"Model başarıyla yüklendi. Tip: {type(checkpoint)}")
            
            # Check checkpoint structure
            if isinstance(checkpoint, dict):
                logger.debug(f"Kontrol noktası anahtarları: {checkpoint.keys()}")
                
                # Extract model state from various possible keys
                if 'model_state' in checkpoint:
                    logger.debug("'model_state' anahtarı altında model ağırlıkları bulundu")
                    state_dict = checkpoint['model_state']
                    # Print sample keys to help with debugging
                    logger.debug(f"state_dict'teki örnek anahtarlar: {list(state_dict.keys())[:5]}")
                elif 'state_dict' in checkpoint:
                    logger.debug("'state_dict' anahtarı altında model ağırlıkları bulundu")
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    logger.debug("'model_state_dict' anahtarı altında model ağırlıkları bulundu")
                    state_dict = checkpoint['model_state_dict']
                else:
                    logger.debug("Tanınan anahtar bulunamadı, kontrol noktası doğrudan state_dict olarak kullanılıyor")
                    state_dict = checkpoint
            else:
                logger.debug("Kontrol noktası bir sözlük değil, doğrudan kullanılıyor")
                state_dict = checkpoint
            
            # Try to load weights with strict=True
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.debug("Model strict=True ile yüklendi")
            except Exception as e:
                logger.warning(f"Kesin yükleme başarısız oldu: {e}")
                # If that fails, try to load weights with strict=False
                logger.debug("strict=False ile ağırlıkları yükleme deneniyor")
                model.load_state_dict(state_dict, strict=False)
                logger.debug("Model strict=False ile yüklendi")
            
            # Set model to evaluation mode
            model.eval()
            logger.info("Model başarıyla yüklendi ve değerlendirme moduna alındı")
            
            return model
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return None
    
    def _estimate_head_pose(self, landmarks, camera_matrix=None, distance_coefficients=None):
        """
        Estimate head pose (rotation and translation) from facial landmarks
        
        Args:
            landmarks: Facial landmarks (shape: 6x2)
            camera_matrix: Camera intrinsic parameters
            distance_coefficients: Camera distortion coefficients
            
        Returns:
            rvec, tvec: Rotation and translation vectors
        """
        # Default camera matrix if not provided
        if camera_matrix is None:
            camera_matrix = np.array([
                [1000.0, 0.0, 320.0],  # Assuming 640x480 camera
                [0.0, 1000.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
        
        # Default distortion coefficients if not provided
        if distance_coefficients is None:
            distance_coefficients = np.zeros((4, 1))
        
        # Check the shape of landmarks array and fix if needed
        # The array should have shape (6,2) before reshaping to (6,1,2)
        landmarks_np = np.array(landmarks)
        if landmarks_np.shape != (6, 2):
            # Extract just the x,y coordinates (first 2 columns) if we have more dimensions
            if len(landmarks_np.shape) > 1 and landmarks_np.shape[1] >= 2:
                landmarks_np = landmarks_np[:, :2]
            else:
                # If landmarks have unexpected shape, log and raise exception
                logger.error(f"Geçersiz landmark şekli: {landmarks_np.shape}. Beklenen: (6,2).")
                raise ValueError(f"Geçersiz landmark şekli: {landmarks_np.shape}. Beklenen: (6,2).")
        
        # Reshape face model and landmarks for PnP
        face_model_pts = self.face_model.reshape(6, 1, 3)
        landmarks_reshaped = landmarks_np.reshape(6, 1, 2).astype(np.float32)
        
        # Solve PnP to get rotation and translation
        success, rvec, tvec = cv2.solvePnP(
            face_model_pts, 
            landmarks_reshaped, 
            camera_matrix, 
            distance_coefficients,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        # Refine if successful
        if success:
            success, rvec, tvec = cv2.solvePnP(
                face_model_pts,
                landmarks_reshaped,
                camera_matrix,
                distance_coefficients,
                rvec,
                tvec,
                True
            )
        
        return rvec, tvec
    
    def _normalize_face(self, image, landmarks, rvec, tvec, camera_matrix=None):
        """
        Normalize face image according to ETH-XGaze procedure
        
        Args:
            image: Input image
            landmarks: Facial landmarks (shape: 6x2)
            rvec: Rotation vector
            tvec: Translation vector
            camera_matrix: Camera intrinsic parameters
            
        Returns:
            normalized_image: Normalized face image
        """
        # Default camera matrix if not provided
        if camera_matrix is None:
            camera_matrix = np.array([
                [1000.0, 0.0, 320.0],  # Assuming 640x480 camera
                [0.0, 1000.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
        
        # Normalized camera parameters (from ETH-XGaze)
        focal_norm = 960  # Normalized focal length
        distance_norm = 600  # Normalized distance
        roiSize = (224, 224)  # Output size
        
        # Compute 3D positions of the landmarks
        tvec = tvec.reshape((3, 1))
        rotation_matrix = cv2.Rodrigues(rvec)[0]  # Convert rotation vector to rotation matrix
        face_3d = np.dot(rotation_matrix, self.face_model.T) + tvec
        
        # Compute face center (average of eye centers and nose)
        eye_center = np.mean(face_3d[:, 0:4], axis=1).reshape((3, 1))
        nose_center = np.mean(face_3d[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(np.concatenate((eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
        
        # Scale to normalized distance
        distance = np.linalg.norm(face_center)
        z_scale = distance_norm / distance
        
        # Create normalized camera matrix
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0]
        ])
        
        # Scaling matrix
        S = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale]
        ])
        
        # Compute rotation matrix R
        hRx = rotation_matrix[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T
        
        # Compute transformation matrix
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))
        
        # Warp image
        img_warped = cv2.warpPerspective(image, W, roiSize)
        
        logger.debug("Baş pozisyonu tabanlı yüz normalizasyonu başarıyla tamamlandı")
        
        return img_warped
    
    def get_face_from_landmarks(self, image, landmarks):
        """
        Extract normalized face image from landmarks
        
        Args:
            image: Input image
            landmarks: Dictionary containing facial landmarks from FaceLandmarkDetector
            
        Returns:
            face_img: Normalized face image for gaze estimation
        """
        try:
            if 'all_landmarks' not in landmarks:
                logger.error("Verilen landmarks sözlüğünde yüz işaret noktaları bulunamadı")
                return None
                
            # We need specific landmarks for head pose estimation
            # In MediaPipe Face Mesh, these correspond to:
            # Left eye: 33, 133 (corners)
            # Right eye: 362, 263 (corners)
            # Nose: 4, 5
            landmark_indices = [33, 133, 362, 263, 4, 5]
            
            # Extract the required landmarks
            selected_landmarks = []
            all_landmarks = landmarks['all_landmarks']
            
            for idx in landmark_indices:
                if idx < len(all_landmarks):
                    # Make sure we're just getting the x,y coordinates
                    x, y = all_landmarks[idx][0], all_landmarks[idx][1]
                    selected_landmarks.append([x, y])
                else:
                    logger.error(f"Landmark indeksi {idx}, işaret noktaları dizisinin uzunluğunun ({len(all_landmarks)}) dışında")
                    return None
                    
            selected_landmarks = np.array(selected_landmarks)
            
            # Verify shape
            if selected_landmarks.shape != (6, 2):
                logger.error(f"Seçilen işaret noktaları için geçersiz şekil: {selected_landmarks.shape}, beklenen: (6, 2)")
                return None
                
            logger.debug(f"{len(all_landmarks)} noktalı tüm işaret noktalarından {len(selected_landmarks)} nokta seçildi")
            
            # Estimate head pose
            rvec, tvec = self._estimate_head_pose(selected_landmarks)
            logger.debug("Baş pozisyonu tahmini başarılı")
            
            # Normalize face
            face_img = self._normalize_face(image, selected_landmarks, rvec, tvec)
            logger.debug("Yüz normalizasyonu başarılı")
            
            return face_img
        except Exception as e:
            logger.error(f"İşaret noktalarından yüz çıkarma hatası: {e}")
            return None
    
    def estimate_gaze(self, image, landmarks=None):
        """
        Estimate gaze direction from image and facial landmarks
        
        Args:
            image: Input image
            landmarks: Dictionary containing facial landmarks from FaceLandmarkDetector
            
        Returns:
            success: Whether gaze estimation was successful
            gaze_angles: Tuple of (yaw, pitch) in degrees
        """
        try:
            # Check if we have landmarks or we need to normalize the image
            if landmarks is not None:
                # Get normalized face
                face_img = self.get_face_from_landmarks(image, landmarks)
                if face_img is None:
                    logger.error("İşaret noktalarından yüz çıkarılamadı")
                    return False, (0, 0)
            else:
                # Assume image is already a normalized face
                face_img = image
            
            # Convert to RGB (from BGR)
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Preprocess image for model
            input_img = self.transform(face_img_rgb)
            input_tensor = input_img.unsqueeze(0).to(self.device)
            logger.debug(f"Girdi tensor şekli: {input_tensor.shape}, veri tipi: {input_tensor.dtype}")
            
            # Get gaze prediction
            with torch.no_grad():
                gaze_output = self.model(input_tensor)
            
            # Convert to numpy array
            gaze_pitchyaw = gaze_output[0].cpu().numpy()
            
            # Convert to degrees
            gaze_yaw = float(gaze_pitchyaw[1]) * 180.0 / np.pi
            gaze_pitch = float(gaze_pitchyaw[0]) * 180.0 / np.pi
            
            logger.debug(f"Tahmin edilen bakış açıları - yaw: {gaze_yaw:.2f}°, pitch: {gaze_pitch:.2f}°")
            
            return True, (gaze_yaw, gaze_pitch)
        
        except Exception as e:
            logger.error(f"Bakış tahmini sırasında hata: {e}")
            return False, (0, 0)
    
    def draw_gaze_vector(self, image, landmarks, gaze_angles, length=100.0, thickness=2, color=(0, 0, 255)):
        """
        Draw gaze direction vector on the input image
        
        Args:
            image: Input image
            landmarks: Dictionary containing facial landmarks
            gaze_angles: Tuple of (yaw, pitch) in degrees
            length: Length of the gaze vector
            thickness: Thickness of the arrow
            color: Color of the arrow (BGR)
            
        Returns:
            image: Image with gaze vector drawn
        """
        try:
            output_image = image.copy()
            
            # Convert angles to radians
            yaw_rad = gaze_angles[0] * np.pi / 180.0
            pitch_rad = gaze_angles[1] * np.pi / 180.0
            
            # Get the center of each eye
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                left_eye_center = np.mean(landmarks['left_eye'], axis=0).astype(int)
                right_eye_center = np.mean(landmarks['right_eye'], axis=0).astype(int)
                
                # Draw gaze vector from each eye
                for eye_center in [left_eye_center, right_eye_center]:
                    self._draw_eye_gaze(output_image, eye_center, gaze_angles, color, length)
                
                return output_image
            else:
                logger.error("İşaret noktaları sözlüğünde göz koordinatları bulunamadı")
                return image
        except Exception as e:
            logger.error(f"Bakış vektörü çizilirken hata: {e}")
            return image
    
    def _draw_eye_gaze(self, image, eye_center, gaze_angles, color=(0, 0, 255), length=50):
        """
        Draw gaze direction vector from one eye
        
        Args:
            image: Input image
            eye_center: (x, y) coordinates of the eye center
            gaze_angles: Tuple of (yaw, pitch) in degrees
            color: Color of the arrow (BGR)
            length: Length of the arrow
            
        Returns:
            None (modifies image in-place)
        """
        # Convert angles to radians
        yaw_rad = gaze_angles[0] * np.pi / 180.0
        pitch_rad = gaze_angles[1] * np.pi / 180.0
        
        # Calculate gaze vector endpoint
        x = -length * np.sin(yaw_rad) * np.cos(pitch_rad)
        y = -length * np.sin(pitch_rad)
        z = -length * np.cos(yaw_rad) * np.cos(pitch_rad)
        
        # Project 3D gaze direction onto image plane
        # We use a simple scaling here since we don't have camera parameters
        scale_x = 1.0
        scale_y = 1.0
        
        dx = scale_x * x
        dy = scale_y * y
        
        # Calculate endpoint
        gaze_end = (int(eye_center[0] + dx), int(eye_center[1] + dy))
        
        # Draw arrow
        cv2.arrowedLine(
            image,
            tuple(eye_center),
            gaze_end,
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.2
        )

    def detect(self, frame, landmarks=None):
        """
        Detect gaze direction from frame
        
        Args:
            frame: Input frame
            landmarks: Dictionary containing facial landmarks (optional)
            
        Returns:
            success: Whether gaze estimation was successful
            gaze_angles: Tuple of (yaw, pitch) in degrees
            face_img: Normalized face image (for visualization)
        """
        # This is an adapter method to integrate with the rest of the system
        if landmarks is None:
            return False, (0, 0), None
        
        # Get normalized face
        face_img = self.get_face_from_landmarks(frame, landmarks)
        if face_img is None:
            return False, (0, 0), None
        
        # Estimate gaze
        success, gaze_angles = self.estimate_gaze(face_img)
        
        return success, gaze_angles, face_img

def main():
    """
    Test fonksiyonu. Bakış tahmincisini test eder.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Bakış tahmincisini oluştur
    gaze_estimator = GazeEstimator()
    
    # Test görüntüsü yükle (varsa)
    test_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'example', 'input', 'test_face.jpg')
    
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        
        # Bakış tahminini gerçekleştir
        success, gaze_angles = gaze_estimator.estimate_gaze(image)
        
        if success:
            print(f"Tahmin edilen bakış yönü: yaw={gaze_angles[0]:.2f}°, pitch={gaze_angles[1]:.2f}°")
        else:
            print("Bakış tahmini başarısız oldu.")
    else:
        print(f"Test görüntüsü bulunamadı: {test_image_path}")


if __name__ == "__main__":
    main()
