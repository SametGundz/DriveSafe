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
import logging
import time
from typing import List, Tuple, Optional, Union, Dict, Any

# İlgili dosyanın en üst kısmına yeni import ekleyin
from src.detection.gaze_zone_detector import get_gaze_zone_detector

# Logger oluşturma
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GazeDetector")

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
    ], dtype=np.float32)
    
    # MediaPipe landmark indices for the 6 points used in ETH-XGaze
    # Right eye outer corner, Right eye inner corner, Left eye inner corner, Left eye outer corner, Nose tip, Chin tip
    GAZE_LANDMARK_INDICES = [33, 133, 362, 263, 1, 199]
    
    # Göz landmark indeksleri
    LEFT_IRIS_CENTER_IDX = 468  # Sol iris merkezi
    RIGHT_IRIS_CENTER_IDX = 473  # Sağ iris merkezi
    LEFT_EYE_OUTER_IDX = 263  # Sol göz dış köşesi
    LEFT_EYE_INNER_IDX = 362  # Sol göz iç köşesi
    RIGHT_EYE_OUTER_IDX = 33  # Sağ göz dış köşesi
    RIGHT_EYE_INNER_IDX = 133  # Sağ göz iç köşesi
    
    # Göz üst ve alt landmark indeksleri
    LEFT_EYE_TOP_IDX = 386  # Sol göz üst noktası
    LEFT_EYE_BOTTOM_IDX = 374  # Sol göz alt noktası 
    RIGHT_EYE_TOP_IDX = 159  # Sağ göz üst noktası
    RIGHT_EYE_BOTTOM_IDX = 145  # Sağ göz alt noktası
    
    def __init__(self, model_path: str = None, use_model_loader: bool = True):
        """
        Initialize the GazeDetector.
        
        Args:
            model_path: Path to the ONNX model file (if None, uses default path)
            use_model_loader: Whether to use the ModelLoader class (recommended)
        """
        start_time = time.time()
        
        # Initialize variables for frame skipping optimization
        self._last_gaze_vector = None
        self._last_normalized_image = None
        self._frame_counter = 0
        
        # Kamera parametreleri ve dönüşüm matrisleri önbelleği
        self._camera_matrix_cache = {}
        self._transform_matrix_cache = {}
        
        # Önişleme ve normalizasyon için bellek önbelleği
        self._rgb_buffer = None
        self._resized_buffer = None
        
        # ONNX model parametreleri
        self.input_name = None
        self.output_name = None
        self.onnx_session = None
        
        # Try to load the ONNX model
        self.model_path = self._find_model_path(model_path)
        
        if use_model_loader:
            self._load_model_with_loader()
        else:
            self._load_model_directly()
        
        logger.info(f"GazeDetector initialized in {time.time() - start_time:.2f}s")
    
    def _find_model_path(self, model_path: str = None) -> str:
        """
        Find the correct model path by checking multiple possible locations.
        
        Args:
            model_path: User-provided model path
            
        Returns:
            str: Path to the model file
        """
        # Check user-provided path first
        if model_path and os.path.exists(model_path):
            return model_path
        
        # List of possible paths to search
        possible_paths = [
            # User-provided path
            model_path,
            
            # Default paths
            os.path.join('models', 'eth_xgaze_model.onnx'),
            os.path.join('models', 'eth_xgaze.onnx'),
            
            # Common locations
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'eth_xgaze_model.onnx'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'eth_xgaze.onnx'),
            
            # More relative paths
            os.path.join('..', 'models', 'eth_xgaze_model.onnx'),
            os.path.join('..', 'models', 'eth_xgaze.onnx')
        ]
        
        # Check each path
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        # If no path is found, return the default path (will log a warning later)
        return os.path.join('models', 'eth_xgaze_model.onnx')
    
    def _load_model_with_loader(self):
        """Load model using the ModelLoader class."""
        try:
            # İlk olarak ModelLoader'ı dinamik olarak import etmeyi dene
            from src.utils.model_loader import get_model_loader
            
            loader = get_model_loader()
            model = loader.load_eth_xgaze_model()
            
            if model is not None:
                if hasattr(model, 'session'):  # ONNXGazeModel durumu
                    self.onnx_session = model.session
                    self.input_name = model.input_name
                    self.output_name = model.output_name
                    logger.info(f"Model loaded using ModelLoader: {self.model_path}")
                else:
                    # PyTorch modeli - önce ONNX'e dönüştürmeyi dene
                    logger.warning("ModelLoader returned a PyTorch model, not ONNX. Using direct loading instead.")
                    self._load_model_directly()
            else:
                logger.warning("ModelLoader failed to load model. Using direct loading instead.")
                self._load_model_directly()
                
        except ImportError:
            logger.warning("ModelLoader not found. Using direct loading instead.")
            self._load_model_directly()
        except Exception as e:
            logger.error(f"Error using ModelLoader: {str(e)}. Using direct loading instead.")
            self._load_model_directly()
    
    def _load_model_directly(self):
        """Load ONNX model directly using onnxruntime."""
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                
                # ONNX Runtime optimizasyonları
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.enable_cpu_mem_arena = True
                sess_options.enable_mem_pattern = True
                
                # CPU optimizasyonları
                sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)
                sess_options.inter_op_num_threads = min(2, os.cpu_count() or 1)
                
                # Providers seçimi
                providers = ['CPUExecutionProvider']
                
                # GPU kullanımını kontrol et
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
                self.onnx_session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
                self.input_name = self.onnx_session.get_inputs()[0].name
                self.output_name = self.onnx_session.get_outputs()[0].name
                
                logger.info(f"ONNX model loaded directly: {self.model_path}")
                logger.info(f"Input name: {self.input_name}, Output name: {self.output_name}")
                
            except ImportError:
                logger.error("Failed to import onnxruntime. Please install it: pip install onnxruntime")
                self.onnx_session = None
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {str(e)}")
                self.onnx_session = None
        else:
            logger.warning(f"ONNX model not found: {self.model_path}")
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
            
        # Landmark indekslerinin sınırlar içinde olup olmadığını kontrol et
        max_landmark_idx = max(
            self.LEFT_IRIS_CENTER_IDX, self.RIGHT_IRIS_CENTER_IDX,
            self.LEFT_EYE_OUTER_IDX, self.LEFT_EYE_INNER_IDX,
            self.RIGHT_EYE_OUTER_IDX, self.RIGHT_EYE_INNER_IDX,
            self.LEFT_EYE_TOP_IDX, self.LEFT_EYE_BOTTOM_IDX,
            self.RIGHT_EYE_TOP_IDX, self.RIGHT_EYE_BOTTOM_IDX
        )
        
        if len(landmarks) <= max_landmark_idx:
            return (0.0, 0.0, 0.0)
            
        # NumPy ile hızlandırma
        # Göz köşeleri
        left_eye_corners = np.array([landmarks[self.LEFT_EYE_OUTER_IDX][:2], landmarks[self.LEFT_EYE_INNER_IDX][:2]])
        right_eye_corners = np.array([landmarks[self.RIGHT_EYE_OUTER_IDX][:2], landmarks[self.RIGHT_EYE_INNER_IDX][:2]])
        
        # İris merkezleri
        left_iris_center = np.array(landmarks[self.LEFT_IRIS_CENTER_IDX][:2])
        right_iris_center = np.array(landmarks[self.RIGHT_IRIS_CENTER_IDX][:2])
        
        # Yatay bakış hesaplama
        # Sol göz
        left_eye_width = np.linalg.norm(left_eye_corners[0] - left_eye_corners[1])
        left_iris_to_corner = np.linalg.norm(left_iris_center - left_eye_corners[0])
        left_gaze_x = 2.0 * (left_iris_to_corner / max(left_eye_width, 1e-6)) - 1.0
        
        # Sağ göz
        right_eye_width = np.linalg.norm(right_eye_corners[0] - right_eye_corners[1])
        right_iris_to_corner = np.linalg.norm(right_iris_center - right_eye_corners[0])
        right_gaze_x = 1.0 - 2.0 * (right_iris_to_corner / max(right_eye_width, 1e-6))
        
        # Her iki gözün yatay bakışının ortalaması
        gaze_x = (left_gaze_x + right_gaze_x) / 2.0
        
        # Dikey bakış hesaplama
        # Sol göz
        left_eye_top = np.array(landmarks[self.LEFT_EYE_TOP_IDX][:2])
        left_eye_bottom = np.array(landmarks[self.LEFT_EYE_BOTTOM_IDX][:2])
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_iris_to_top = np.linalg.norm(left_iris_center - left_eye_top)
        left_gaze_y = 2.0 * (left_iris_to_top / max(left_eye_height, 1e-6)) - 1.0
        
        # Sağ göz
        right_eye_top = np.array(landmarks[self.RIGHT_EYE_TOP_IDX][:2])
        right_eye_bottom = np.array(landmarks[self.RIGHT_EYE_BOTTOM_IDX][:2])
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_iris_to_top = np.linalg.norm(right_iris_center - right_eye_top)
        right_gaze_y = 2.0 * (right_iris_to_top / max(right_eye_height, 1e-6)) - 1.0
        
        # Her iki gözün dikey bakışının ortalaması
        gaze_y = (left_gaze_y + right_gaze_y) / 2.0
        
        # Derinlik bileşeni (z) - yanlara/yukarı-aşağı bakma miktarı azaldıkça daha odaklı
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
        # NumPy ile daha hızlı hesaplama
        return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))
    
    def draw_gaze(self, image: np.ndarray, pitchyaw: np.ndarray, origin: Tuple[int, int], 
                length: int = 50, thickness: int = 2, color: Tuple[int, int, int] = (0, 0, 255),
                overlay: bool = True) -> np.ndarray:
        """
        Visualize gaze direction.
        
        Args:
            image: Input image
            pitchyaw: Gaze direction vector (pitch, yaw)
            origin: Origin point (x, y) of gaze vector
            length: Arrow length
            thickness: Arrow thickness
            color: Arrow color
            overlay: If True, draw on a copy of the image, else modify the original
            
        Returns:
            image: Image with gaze direction visualized
        """
        # Kopyalama veya doğrudan değiştirme
        vis_image = image.copy() if overlay else image
        
        try:
            pitch, yaw = pitchyaw
            
            # Bakış vektörü hesapla
            x = -length * np.sin(yaw) * np.cos(pitch)
            y = -length * np.sin(pitch)
            
            # 2D noktaya projeksiyon
            point_2d = (int(origin[0] + x), int(origin[1] + y))
            
            # Ok çiz
            cv2.arrowedLine(vis_image, origin, point_2d, color, thickness, cv2.LINE_AA, tipLength=0.2)
            
            # İsteğe bağlı: Açıları metin olarak göster
            # cv2.putText(vis_image, f"P:{np.rad2deg(pitch):.1f} Y:{np.rad2deg(yaw):.1f}", 
            #           (origin[0]-30, origin[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return vis_image
            
        except Exception as e:
            logger.warning(f"Error drawing gaze: {str(e)}")
            return vis_image
    
    def get_camera_matrix(self, frame: np.ndarray) -> np.ndarray:
        """
        Get camera matrix for the input frame.
        
        Args:
            frame: Input frame
            
        Returns:
            np.ndarray: Camera matrix
        """
        # Frame boyutları
        h, w = frame.shape[:2]
        
        # Önbellekte var mı kontrol et
        cache_key = f"{w}x{h}"
        if cache_key in self._camera_matrix_cache:
            return self._camera_matrix_cache[cache_key]
        
        # Yoksa oluştur
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Önbelleğe ekle
        self._camera_matrix_cache[cache_key] = camera_matrix
        
        return camera_matrix
    
    def estimate_head_pose(self, landmarks: List[List[float]], frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate head pose for gaze normalization.
        
        Args:
            landmarks: List of facial landmarks
            frame: Input frame for calculating image dimensions
            
        Returns:
            rvec, tvec: Rotation and translation vectors
        """
        if not landmarks or len(landmarks) < max(self.GAZE_LANDMARK_INDICES):
            return np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        
        # Get camera matrix
        camera_matrix = self.get_camera_matrix(frame)
        distortion = np.zeros((4, 1), dtype=np.float64)
        
        # Get the 6 points used for gaze estimation
        landmarks_2d = np.zeros((6, 2), dtype=np.float64)
        
        valid_points = True
        for i, idx in enumerate(self.GAZE_LANDMARK_INDICES):
            if idx < len(landmarks):
                landmarks_2d[i] = [landmarks[idx][0], landmarks[idx][1]]
            else:
                valid_points = False
                break
        
        # Solve PnP to get head pose if all points are valid
        if valid_points:
            try:
                # Initial estimate with EPNP (hızlı)
                ret, rvec, tvec = cv2.solvePnP(
                    self.GAZE_FACE_MODEL, 
                    landmarks_2d, 
                    camera_matrix, 
                    distortion, 
                    flags=cv2.SOLVEPNP_EPNP
                )
                
                # Refine with Levenberg-Marquardt (daha doğru)
                ret, rvec, tvec = cv2.solvePnP(
                    self.GAZE_FACE_MODEL, 
                    landmarks_2d, 
                    camera_matrix, 
                    distortion, 
                    rvec, tvec, 
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                return rvec, tvec
            except cv2.error as e:
                logger.error(f"CV2 error in solvePnP: {str(e)}")
        
        # Failed to estimate pose
        return np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
    
    def normalize_face(self, img: np.ndarray, landmarks: List[List[float]], frame: np.ndarray,
                     target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Normalize face image for ETH-XGaze model.
        
        Args:
            img: Input frame
            landmarks: List of facial landmarks
            frame: Input frame for calculating image dimensions
            target_size: Size of the normalized image
            
        Returns:
            img_normalized: Normalized face image
        """
        if not landmarks or len(landmarks) < max(self.GAZE_LANDMARK_INDICES):
            # Normalizasyon yapılamıyorsa boş bir görüntü döndür
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        try:
            # Baş duruşunu tahmin et
            hr, ht = self.estimate_head_pose(landmarks, frame)
            
            # Kamera matrisini al
            camera_matrix = self.get_camera_matrix(frame)
            
            # ETH-XGaze normalizasyon parametreleri
            focal_norm = 960  # Normalize edilmiş odak uzaklığı
            distance_norm = 600  # Normalize edilmiş göz-kamera mesafesi
            
            # 3D landmark konumlarını hesapla
            ht = ht.reshape((3, 1))
            hR = cv2.Rodrigues(hr)[0]  # Rotasyon matrisi
            
            # Yüz modelini döndür ve öteleme uygula
            Fc = np.dot(hR, self.GAZE_FACE_MODEL.T) + ht
            
            # Yüz merkezi bul
            two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
            nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
            face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
            
            # Normalizasyon
            distance = np.linalg.norm(face_center)  # Göz-kamera mesafesi
            
            # Z ekseninde ölçekleme
            z_scale = distance_norm / max(distance, 1e-6)
            
            # Sanal kamera içsel parametreleri
            cam_norm = np.array([
                [focal_norm, 0, target_size[0] / 2],
                [0, focal_norm, target_size[1] / 2],
                [0, 0, 1.0],
            ])
            
            # Dönüşüm matrisi hesapla
            # Ölçekleme matrisi
            S = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, z_scale],
            ])
            
            # Rotasyon hesapla
            hRx = hR[:, 0]
            forward = (face_center / max(distance, 1e-6)).reshape(3)
            down = np.cross(forward, hRx)
            down = down / max(np.linalg.norm(down), 1e-6)
            right = np.cross(down, forward)
            right = right / max(np.linalg.norm(right), 1e-6)
            
            R = np.c_[right, down, forward].T  # Rotasyon matrisi
            
            # Son dönüşüm matrisi
            W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))
            
            # Görüntüyü dönüştür
            img_normalized = cv2.warpPerspective(img, W, target_size)
            
            return img_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing face: {str(e)}")
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for ETH-XGaze model.
        
        Args:
            image: Normalized face image
            target_size: Size of the preprocessed image
            
        Returns:
            processed_img: Processed image (model input)
        """
        try:
            # BGR'den RGB'ye dönüştür
            if self._rgb_buffer is None or self._rgb_buffer.shape != image.shape:
                self._rgb_buffer = np.empty(image.shape, dtype=np.uint8)
            
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
            
            # Boyutlandır
            if image.shape[:2] != target_size:
                if self._resized_buffer is None or self._resized_buffer.shape[:2] != target_size:
                    self._resized_buffer = np.empty((target_size[1], target_size[0], 3), dtype=np.uint8)
                
                cv2.resize(self._rgb_buffer, target_size, dst=self._resized_buffer)
                image_rgb_resized = self._resized_buffer
            else:
                image_rgb_resized = self._rgb_buffer
            
            # [0, 1] aralığına normalize et
            image_float = image_rgb_resized.astype(np.float32) / 255.0
            
            # Ortalama ve standart sapma ile normalize et
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (image_float - mean) / std
            
            # Kanal sıralamasını değiştir (HWC -> CHW)
            transposed = normalized.transpose(2, 0, 1)
            
            # Batch boyutunu ekle
            batched = np.expand_dims(transposed, axis=0)
            
            return batched
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return np.zeros((1, 3, target_size[1], target_size[0]), dtype=np.float32)
    
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
                # Model yüklü değilse boş vektör döndür
                return np.zeros(2, dtype=np.float32), None
            
            try:
                # Yüzü normalize et
                normalized_image = self.normalize_face(frame, landmarks, frame)
                
                # Görüntü geçerli mi kontrol et
                if normalized_image is None or normalized_image.size == 0 or np.all(normalized_image == 0):
                    return np.zeros(2, dtype=np.float32), None
                
                # Görüntüyü ön işle
                processed_img = self.preprocess_image(normalized_image)
                
                # Çıkarım yap
                gaze = self.onnx_session.run([self.output_name], {self.input_name: processed_img})[0]
                
                # Bakış vektörünü al (pitch, yaw)
                gaze_vector = gaze[0].astype(np.float32)
                
                return gaze_vector, normalized_image
                
            except Exception as e:
                logger.error(f"Error in gaze prediction: {str(e)}")
                return np.zeros(2, dtype=np.float32), None
    
    def draw_gaze(self, image: np.ndarray, pitchyaw: np.ndarray, origin: Tuple[int, int], 
                 length: int = 50, thickness: int = 2, color: Tuple[int, int, int] = (0, 0, 255),
                 overlay: bool = True) -> np.ndarray:
        """
        Visualize gaze direction.
        
        Args:
            image: Input image
            pitchyaw: Gaze direction vector (pitch, yaw)
            origin: Origin point (x, y) of gaze vector
            length: Arrow length
            thickness: Arrow thickness
            color: Arrow color
            overlay: If True, draw on a copy of the image, else modify the original
            
        Returns:
            image: Image with gaze direction visualized
        """
        # Kopyalama veya doğrudan değiştirme
        vis_image = image.copy() if overlay else image
        
        try:
            pitch, yaw = pitchyaw
            
            # Bakış vektörü hesapla
            x = -length * np.sin(yaw) * np.cos(pitch)
            y = -length * np.sin(pitch)
            
            # 2D noktaya projeksiyon
            point_2d = (int(origin[0] + x), int(origin[1] + y))
            
            # Ok çiz
            cv2.arrowedLine(vis_image, origin, point_2d, color, thickness, cv2.LINE_AA, tipLength=0.2)
            
            # İsteğe bağlı: Açıları metin olarak göster
            # cv2.putText(vis_image, f"P:{np.rad2deg(pitch):.1f} Y:{np.rad2deg(yaw):.1f}", 
            #           (origin[0]-30, origin[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return vis_image
            
        except Exception as e:
            logger.warning(f"Error drawing gaze: {str(e)}")
            return vis_image
    
    def visualize_gaze(self, frame: np.ndarray, landmarks: List[List[float]], 
                      ear_value: float = None, ear_threshold: float = 0.2,
                      frame_skip: int = 3, overlay: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict and visualize gaze direction.
        
        Args:
            frame: Input frame
            landmarks: List of facial landmarks
            ear_value: Eye Aspect Ratio value, used to check if eyes are open
            ear_threshold: EAR threshold, below which eyes are considered closed
            frame_skip: Number of frames to skip between predictions (for optimization)
            overlay: If True, draw on a copy of the frame, else modify the original
            
        Returns:
            frame: Frame with visualized gaze direction
            normalized_image: Normalized face image (if available)
        """
        # Kopyalama veya doğrudan değiştirme
        vis_frame = frame.copy() if overlay else frame
        
        # Geçerlilik kontrolü
        if not landmarks or self.onnx_session is None:
            return vis_frame, None
        
        # Göz kapalı kontrolü (EAR kullanarak)
        if ear_value is not None and ear_value < ear_threshold:
            return vis_frame, None
        
        # Her frame_skip karede bir tahmin yap, arada kalan karelerde son tahmini kullan
        do_prediction = False
        
        self._frame_counter += 1
        if self._frame_counter >= frame_skip:
            do_prediction = True
            self._frame_counter = 0
        
        # İlk tahmin veya yeni tahmin yapılacaksa
        if do_prediction or self._last_gaze_vector is None or self._last_normalized_image is None:
            self._last_gaze_vector, self._last_normalized_image = self.predict_gaze(frame, landmarks)
        
        # Tahmin sonuçları
        gaze_vector = self._last_gaze_vector
        normalized_image = self._last_normalized_image
        
        # Geçerlilik kontrolü
        if gaze_vector is None:
            return vis_frame, None
        
        # Göz merkezlerini bul
        eye_points_valid = (len(landmarks) > max(self.LEFT_EYE_OUTER_IDX, self.LEFT_EYE_INNER_IDX, 
                                              self.RIGHT_EYE_OUTER_IDX, self.RIGHT_EYE_INNER_IDX))
        
        if eye_points_valid:
            # Her göz için merkez hesapla
            left_eye_center = (
                int((landmarks[self.LEFT_EYE_OUTER_IDX][0] + landmarks[self.LEFT_EYE_INNER_IDX][0]) / 2),
                int((landmarks[self.LEFT_EYE_OUTER_IDX][1] + landmarks[self.LEFT_EYE_INNER_IDX][1]) / 2)
            )
            
            right_eye_center = (
                int((landmarks[self.RIGHT_EYE_OUTER_IDX][0] + landmarks[self.RIGHT_EYE_INNER_IDX][0]) / 2),
                int((landmarks[self.RIGHT_EYE_OUTER_IDX][1] + landmarks[self.RIGHT_EYE_INNER_IDX][1]) / 2)
            )
            
            # Gözler arası orta nokta (bakış vektörünün başlangıç noktası)
            gaze_origin = (
                int((left_eye_center[0] + right_eye_center[0]) / 2),
                int((left_eye_center[1] + right_eye_center[1]) / 2)
            )
            
            # Başlangıç noktasını göster (küçük mavi daire)
            cv2.circle(vis_frame, gaze_origin, 3, (255, 0, 0), -1)
            
        else:
            # Göz noktaları geçerli değilse, yüz dikdörtgeninin merkezini kullan
            face_rect = self._get_face_rect(landmarks)
            gaze_origin = (
                face_rect[0] + face_rect[2] // 2, 
                face_rect[1] + face_rect[3] // 2
            )
        
        # Bakış yönünü çiz
        vis_frame = self.draw_gaze(vis_frame, gaze_vector, gaze_origin, overlay=False)
        
        # Pitch, yaw değerlerini ekranda göstermeyi kaldırdık, bu şimdi main_window'dan yapılıyor
        # Ana pencerede daha iyi biçimlendirilmiş metin için
        
        return vis_frame, normalized_image
    
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
        
        # NumPy ile daha verimli işlem
        landmarks_array = np.array(landmarks)
        
        # Minimum ve maksimum koordinatları bul
        left = int(np.min(landmarks_array[:, 0]))
        top = int(np.min(landmarks_array[:, 1]))
        right = int(np.max(landmarks_array[:, 0]))
        bottom = int(np.max(landmarks_array[:, 1]))
        
        # Dolgu ekle
        width = right - left
        height = bottom - top
        padding_x = int(width * padding)
        padding_y = int(height * padding)
        
        left = max(0, left - padding_x)
        top = max(0, top - padding_y)
        right = right + padding_x
        bottom = bottom + padding_y
        
        return (left, top, right - left, bottom - top)
    
    def check_attention(self, gaze_vector: np.ndarray, 
                       max_pitch_deviation: float = 30.0,
                       max_yaw_deviation: float = 40.0) -> float:
        """
        Calculate driver attention score based on gaze vector.
        
        Args:
            gaze_vector: (pitch, yaw) gaze direction in radians
            max_pitch_deviation: Maximum allowed pitch deviation in degrees
            max_yaw_deviation: Maximum allowed yaw deviation in degrees
            
        Returns:
            float: Attention score (0.0-1.0), where 1.0 is full attention
        """
        if gaze_vector is None or len(gaze_vector) < 2:
            return 1.0  # Varsayılan olarak tam dikkat
        
        # Radyandan dereceye çevir
        pitch_deg = np.rad2deg(gaze_vector[0])
        yaw_deg = np.rad2deg(gaze_vector[1])
        
        # Yatay sapma (yaw - sağa/sola bakma)
        yaw_attention = 1.0 - min(1.0, abs(yaw_deg) / max_yaw_deviation)
        
        # Dikey sapma (pitch - yukarı/aşağı bakma)
        pitch_attention = 1.0 - min(1.0, abs(pitch_deg) / max_pitch_deviation)
        
        # Toplam dikkat skoru (en düşük değeri al)
        attention = min(yaw_attention, pitch_attention)
        
        return max(0.0, min(1.0, attention))
    
    def is_looking_forward(self, gaze_vector: np.ndarray, 
                         max_pitch_deg: float = 15.0,
                         max_yaw_deg: float = 20.0) -> bool:
        """
        Check if driver is looking forward based on gaze vector.
        
        Args:
            gaze_vector: (pitch, yaw) gaze direction in radians
            max_pitch_deg: Maximum allowed pitch deviation in degrees
            max_yaw_deg: Maximum allowed yaw deviation in degrees
            
        Returns:
            bool: True if looking forward, False otherwise
        """
        attention = self.check_attention(gaze_vector, max_pitch_deg, max_yaw_deg)
        return attention > 0.7  # %70'ten fazla dikkat ileri bakıyor sayılır
    
    def get_gaze_target_zone(self, gaze_vector: np.ndarray) -> int:
        """
        Bakış vektörüne göre hedef bölgeyi belirler.
        
        Args:
            gaze_vector: (pitch, yaw) açılarını içeren bakış yönü vektörü (radyan)
                
        Returns:
            int: Hedef bölge ID'si (0-8) veya None
        """
        if gaze_vector is None or len(gaze_vector) < 2:
            return None
        
        # Açıları derece cinsine çevir
        pitch_deg = np.rad2deg(gaze_vector[0])
        yaw_deg = np.rad2deg(gaze_vector[1])
        
        # Debug log eklentisi
        logger.debug(f"Gaze angles for zone detection - Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°")
        
        # GazeZoneDetector kullanarak bölge tespiti
        zone_detector = get_gaze_zone_detector()
        current_time = time.time()
        zone = zone_detector.update(pitch_deg, yaw_deg, current_time)
        
        # Debug log eklentisi
        if zone is None:
            logger.debug(f"No zone detected for Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°")
        else:
            zone_name = zone_detector.get_zone_name(zone)
            logger.debug(f"Detected zone: {zone_name} ({zone}) for Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°")
        
        return zone
    
    def visualize_gaze_zone(self, frame: np.ndarray, landmarks: List[List[float]]) -> Tuple[np.ndarray, Optional[int]]:
        """
        Bakış bölgesini tespit eder.
        
        NOT: Bu metot artık görsel çıktı oluşturmamaktadır. Ana pencere bu işlevi üstlenmiştir.
        Sadece bölge ID'sini döndürmek için kullanılmaktadır.
        
        Args:
            frame: Giriş karesi
            landmarks: Yüz landmarkları listesi
            
        Returns:
            Tuple[np.ndarray, Optional[int]]: 
                - Değiştirilmemiş kare
                - Tespit edilen bölge ID'si
        """
        if not landmarks:
            return frame, None
        
        # Bakış yönünü tahmin et
        gaze_vector, _ = self.predict_gaze(frame, landmarks)
        
        # Bölge tespiti yap
        zone_id = self.get_gaze_target_zone(gaze_vector)
        
        # Artık görsel değişiklik yapmıyoruz
        # Sadece zone_id değerini döndürüyoruz
        return frame, zone_id
    
    def reset(self):
        """Reset frame counter and cached values."""
        self._frame_counter = 0
        self._last_gaze_vector = None
        self._last_normalized_image = None
        self._rgb_buffer = None
        self._resized_buffer = None
        
    def release(self):
        """Release resources."""
        self.onnx_session = None
        self._camera_matrix_cache.clear()
        self._transform_matrix_cache.clear()
        self.reset()


# Create a function to get a pre-configured GazeDetector
def get_gaze_detector(model_path: str = None, frame_skip: int = 3) -> GazeDetector:
    """
    Factory function to create and configure a GazeDetector instance.
    
    Args:
        model_path: Optional path to the ONNX model file
        frame_skip: Number of frames to skip between predictions
    
    Returns:
        GazeDetector: Configured GazeDetector instance
    """
    detector = GazeDetector(model_path)
    detector._frame_skip = frame_skip
    return detector