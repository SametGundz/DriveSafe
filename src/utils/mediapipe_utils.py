#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MediaPipe utility functions for the drowsiness detection system.

This module provides utility functions for working with MediaPipe,
particularly for face mesh processing operations.
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import os
import onnxruntime
from typing import List, Tuple, Dict, Optional, Union

class MediaPipeUtils:
    """
    A utility class for MediaPipe operations, particularly face mesh functionality.
    
    Attributes:
        mp_face_mesh: MediaPipe FaceMesh solution
        face_mesh: MediaPipe FaceMesh object
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
    
    # Baş duruşu hesaplama için gerekli noktalar (estimator.py'den alındı)
    HEAD_POSE_LANDMARKS = [1, 9, 57, 130, 287, 359]
    
    # 3D model noktaları (estimator.py'den alındı)
    MODEL_POINTS = np.array([
        [0.0, 0.0, 0.0],           # Burun ucu (origin)
        [0.0, -330.0, -65.0],      # Çene
        [-225.0, 170.0, -135.0],   # Sol göz sol köşesi
        [-150.0, -150.0, -125.0],  # Sol ağız köşesi
        [225.0, 170.0, -135.0],    # Sağ göz sağ köşesi
        [150.0, -150.0, -125.0]    # Sağ ağız köşesi
    ], dtype=np.float64)
    
    # Detaylı 3D yüz modeli (58 nokta)
    DETAILED_MODEL_POINTS = np.array([
        [-7.308957, 0.913869, 0.000000],  # 0
        [-6.775290, -0.730814, -0.012799],  # 1
        [-5.665918, -3.286078, 1.022951],  # 2
        [-5.011779, -4.876396, 1.047961],  # 3
        [-4.056931, -5.947019, 1.636229],  # 4
        [-1.833492, -7.056977, 4.061275],  # 5
        [0.000000, -7.415691, 4.070434],  # 6
        [1.833492, -7.056977, 4.061275],  # 7
        [4.056931, -5.947019, 1.636229],  # 8
        [5.011779, -4.876396, 1.047961],  # 9
        [5.665918, -3.286078, 1.022951],  # 10
        [6.775290, -0.730814, -0.012799],  # 11
        [7.308957, 0.913869, 0.000000],  # 12
        [5.311432, 5.485328, 3.987654],  # 13
        [4.461908, 6.189018, 5.594410],  # 14
        [3.550622, 6.185143, 5.712299],  # 15
        [2.542231, 5.862829, 4.687939],  # 16
        [1.789930, 5.393625, 4.413414],  # 17
        [2.693583, 5.018237, 5.072837],  # 18
        [3.530191, 4.981603, 4.937805],  # 19
        [4.490323, 5.186498, 4.694397],  # 20
        [-5.311432, 5.485328, 3.987654],  # 21
        [-4.461908, 6.189018, 5.594410],  # 22
        [-3.550622, 6.185143, 5.712299],  # 23
        [-2.542231, 5.862829, 4.687939],  # 24
        [-1.789930, 5.393625, 4.413414],  # 25
        [-2.693583, 5.018237, 5.072837],  # 26
        [-3.530191, 4.981603, 4.937805],  # 27
        [-4.490323, 5.186498, 4.694397],  # 28
        [1.330353, 7.122144, 6.903745],  # 29
        [2.533424, 7.878085, 7.451034],  # 30
        [4.861131, 7.878672, 6.601275],  # 31
        [6.137002, 7.271266, 5.200823],  # 32
        [6.825897, 6.760612, 4.402142],  # 33
        [-1.330353, 7.122144, 6.903745],  # 34
        [-2.533424, 7.878085, 7.451034],  # 35
        [-4.861131, 7.878672, 6.601275],  # 36
        [-6.137002, 7.271266, 5.200823],  # 37
        [-6.825897, 6.760612, 4.402142],  # 38
        [-2.774015, -2.080775, 5.048531],  # 39
        [-0.509714, -1.571179, 6.566167],  # 40
        [0.000000, -1.646444, 6.704956],  # 41
        [0.509714, -1.571179, 6.566167],  # 42
        [2.774015, -2.080775, 5.048531],  # 43
        [0.589441, -2.958597, 6.109526],  # 44
        [0.000000, -3.116408, 6.097667],  # 45
        [-0.589441, -2.958597, 6.109526],  # 46
        [-0.981972, 4.554081, 6.301271],  # 47
        [-0.973987, 1.916389, 7.654050],  # 48
        [-2.005628, 1.409845, 6.165652],  # 49
        [-1.930245, 0.424351, 5.914376],  # 50
        [-0.746313, 0.348381, 6.263227],  # 51
        [0.000000, 0.000000, 6.763430],  # 52 BURUN UCU
        [0.746313, 0.348381, 6.263227],  # 53
        [1.930245, 0.424351, 5.914376],  # 54
        [2.005628, 1.409845, 6.165652],  # 55
        [0.973987, 1.916389, 7.654050],  # 56
        [0.981972, 4.554081, 6.301271]   # 57
    ], dtype=np.float64)
    
    # Detaylı model için MediaPipe yüz işaretleri eşleştirmesi
    # Bu indeksler, MediaPipe'ın 468 yüz işareti ile 58 noktalı modeli eşleştirir
    # Şu anda sadece yaklaşık bir eşleştirme, geliştirilebilir
    DETAILED_MODEL_MEDIAPIPE_MAPPING = [
        10,    # 0 - yüz konturu sol üst
        152,   # 1 - yüz konturu sol alt
        162,   # 2 - çene sol
        169,   # 3 - çene sol orta
        186,   # 4 - çene sol iç
        200,   # 5 - çene orta sol
        199,   # 6 - çene orta
        175,   # 7 - çene orta sağ
        201,   # 8 - çene sağ iç
        208,   # 9 - çene sağ orta
        21,    # 10 - çene sağ
        127,   # 11 - yüz konturu sağ alt
        338,   # 12 - yüz konturu sağ üst
        297,   # 13 - sağ kaş dış
        299,   # 14 - sağ kaş dış orta
        296,   # 15 - sağ kaş orta dış
        336,   # 16 - sağ kaş orta
        334,   # 17 - sağ kaş orta iç
        293,   # 18 - sağ kaş iç orta
        276,   # 19 - sağ kaş iç
        283,   # 20 - sağ kaş iç üst
        70,    # 21 - sol kaş dış
        63,    # 22 - sol kaş dış orta
        105,   # 23 - sol kaş orta dış
        66,    # 24 - sol kaş orta
        107,   # 25 - sol kaş orta iç
        55,    # 26 - sol kaş iç orta
        65,    # 27 - sol kaş iç
        52,    # 28 - sol kaş iç üst
        336,   # 29 - sağ göz üst orta
        296,   # 30 - sağ göz üst iç
        334,   # 31 - sağ göz üst dış
        293,   # 32 - sağ göz alt dış
        276,   # 33 - sağ göz alt orta
        107,   # 34 - sol göz üst orta
        66,    # 35 - sol göz üst iç
        105,   # 36 - sol göz üst dış
        55,    # 37 - sol göz alt dış
        65,    # 38 - sol göz alt orta
        329,   # 39 - burun kökü sağ
        98,    # 40 - burun kökü sağ orta
        19,    # 41 - burun kökü orta
        97,    # 42 - burun kökü sol orta
        100,   # 43 - burun kökü sol
        209,   # 44 - burun orta sağ
        94,    # 45 - burun orta
        205,   # 46 - burun orta sol
        360,   # 47 - burun üst sağ
        344,   # 48 - burun üst sağ orta
        370,   # 49 - burun üst orta sağ
        4,     # 50 - burun üst orta
        438,   # 51 - burun üst orta sol
        1,     # 52 - burun ucu
        19,    # 53 - burun alt orta sağ
        2,     # 54 - burun alt orta
        94,    # 55 - burun alt orta sol
        141,   # 56 - burun alt sağ
        218    # 57 - burun alt sol
    ]
    
    # ETH-XGaze için yüz modeli (3D landmark noktaları)
    GAZE_FACE_MODEL = np.array([
        [-34.04, 39.55, 57.49],  # Sağ göz dış köşesi
        [-16.1, 42.4, 67.53],    # Sağ göz iç köşesi
        [16.1, 42.4, 67.53],     # Sol göz iç köşesi
        [34.04, 39.55, 57.49],   # Sol göz dış köşesi
        [0.0, 0.0, 8.0],         # Burun ucu
        [0.0, -48.0, 21.0],      # Çene ucu
    ])
    
    # ETH-XGaze için önemli yüz noktaları
    # Sırasıyla: Sağ göz dış köşesi, Sağ göz iç köşesi, Sol göz iç köşesi, Sol göz dış köşesi, Burun ucu, Çene ucu
    GAZE_LANDMARK_INDICES = [33, 133, 362, 263, 1, 199]
    
    def __init__(self, 
                static_image_mode: bool = False, 
                max_num_faces: int = 1,
                refine_landmarks: bool = True,
                min_detection_confidence: float = 0.5,
                min_tracking_confidence: float = 0.5):
        """
        Initialize the MediaPipeUtils.
        
        Args:
            static_image_mode: Whether to treat images as static (not video)
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # ONNX model yükleme
        self.model_path = os.path.join('models', 'eth_xgaze_model.onnx')
        if os.path.exists(self.model_path):
            self.onnx_session = onnxruntime.InferenceSession(self.model_path)
            print(f"ONNX model yüklendi: {self.model_path}")
        else:
            print(f"UYARI: ONNX model bulunamadı: {self.model_path}")
            self.onnx_session = None
            
        # Varsayılan olarak basit modeli kullan
        self.use_detailed_model = False
    
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
    
    def get_eye_aspect_ratio(self, eye_landmarks: List[List[float]]) -> float:
        """
        Calculate the Eye Aspect Ratio (EAR) for eye landmarks.
        
        The EAR is calculated using 6 landmarks that outline the eye.
        
        Args:
            eye_landmarks: List of eye landmarks
            
        Returns:
            float: EAR value
        """
        # MediaPipe face mesh gives 16 landmarks per eye
        # But we need specific points for EAR calculation
        if len(eye_landmarks) < 16:
            return 0.0
        
        # For the left eye:
        # Vertical landmarks: (top to bottom pairs)
        # 386 & 374 (upper and lower eyelid)
        # 385 & 380 (upper and lower eyelid)
        # 387 & 373 (upper and lower eyelid)
        # Horizontal landmarks: 263 & 362 (eye corners)
        
        # For the right eye:
        # Vertical landmarks: (top to bottom pairs)
        # 159 & 145 (upper and lower eyelid)
        # 158 & 153 (upper and lower eyelid)
        # 160 & 144 (upper and lower eyelid)
        # Horizontal landmarks: 33 & 133 (eye corners)
        
        # Check if this is left or right eye based on landmark coordinates
        is_left_eye = eye_landmarks[0][0] > 300  # A rough check if x > 300 pixels
        
        if is_left_eye:
            # Extract main landmarks for left eye EAR calculation
            # Using indices in LEFT_EYE_INDICES
            # 362=0, 385=13, 386=12, 387=11, 373=5, 374=4, 380=3, 263=8
            corner1 = eye_landmarks[0]  # 362 (outer corner)
            corner2 = eye_landmarks[8]  # 263 (inner corner)
            upper1 = eye_landmarks[12]  # 386 (upper eyelid)
            lower1 = eye_landmarks[4]   # 374 (lower eyelid)
            upper2 = eye_landmarks[13]  # 385 (upper eyelid)
            lower2 = eye_landmarks[3]   # 380 (lower eyelid)
            upper3 = eye_landmarks[11]  # 387 (upper eyelid)
            lower3 = eye_landmarks[5]   # 373 (lower eyelid)
        else:
            # Extract main landmarks for right eye EAR calculation
            # Using indices in RIGHT_EYE_INDICES
            # 33=0, 158=11, 159=10, 160=12, 144=3, 145=4, 153=6, 133=8
            corner1 = eye_landmarks[0]  # 33 (outer corner)
            corner2 = eye_landmarks[8]  # 133 (inner corner)
            upper1 = eye_landmarks[10]  # 159 (upper eyelid)
            lower1 = eye_landmarks[4]   # 145 (lower eyelid)
            upper2 = eye_landmarks[11]  # 158 (upper eyelid)
            lower2 = eye_landmarks[6]   # 153 (lower eyelid)
            upper3 = eye_landmarks[12]  # 160 (upper eyelid)
            lower3 = eye_landmarks[3]   # 144 (lower eyelid)
            
        # Calculate horizontal distance (eye width)
        horizontal_dist = np.linalg.norm(np.array([corner1[0], corner1[1]]) - np.array([corner2[0], corner2[1]]))
        
        # Calculate vertical distances (3 points along the eye)
        vertical_dist1 = np.linalg.norm(np.array([upper1[0], upper1[1]]) - np.array([lower1[0], lower1[1]]))
        vertical_dist2 = np.linalg.norm(np.array([upper2[0], upper2[1]]) - np.array([lower2[0], lower2[1]]))
        vertical_dist3 = np.linalg.norm(np.array([upper3[0], upper3[1]]) - np.array([lower3[0], lower3[1]]))
        
        # Calculate EAR using all three vertical measurements
        if horizontal_dist == 0:
            return 0.0
        
        ear = (vertical_dist1 + vertical_dist2 + vertical_dist3) / (3.0 * horizontal_dist)
        return ear
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points.
        
        Args:
            point1: (x, y) coordinates of first point
            point2: (x, y) coordinates of second point
            
        Returns:
            float: Euclidean distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_mouth_aspect_ratio(self, landmarks: List[List[float]]) -> float:
        """
        Calculate the Mouth Aspect Ratio (MAR) using only inner lip landmarks.
        
        Formula:
        MAR = (vertical distance between lips) / (horizontal distance between lips)
        
        Args:
            landmarks: List of all facial landmarks
            
        Returns:
            float: MAR value
        """
        if landmarks is None or not landmarks:
            return 0.0
            
        # Get only inner lip landmarks
        inner_lip = self.get_inner_lip_landmarks(landmarks)
        
        if not inner_lip or len(inner_lip) < 8:
            return 0.0
            
        # Inner lip landmarks key points:
        # Left corner: inner_lip[0] (corresponds to INNER_LIP_INDICES[0] = 78)
        # Right corner: inner_lip[10] (corresponds to INNER_LIP_INDICES[10] = 308)
        # Top middle: inner_lip[5] (corresponds to INNER_LIP_INDICES[5] = 13)
        # Bottom middle: inner_lip[15] (corresponds to INNER_LIP_INDICES[15] = 14)
        
        # Find horizontal distance (width) between corners of the inner lips
        horizontal_distance = self._calculate_distance(
            inner_lip[0],  # left corner of inner lip
            inner_lip[10]   # right corner of inner lip
        )
        
        # Find vertical distance (height) between top and bottom of inner lips
        vertical_distance = self._calculate_distance(
            inner_lip[5],  # top middle of inner lip
            inner_lip[15]   # bottom middle of inner lip
        )
        
        # Calculate the average of multiple vertical distances for better accuracy
        # These are different positions along the inner lip
        vertical_distances = []
        
        # Add the middle vertical distance
        vertical_distances.append(vertical_distance)
        
        # Add left-side vertical distance 
        left_vertical = self._calculate_distance(
            inner_lip[4],  # left-side of top inner lip
            inner_lip[16]  # left-side of bottom inner lip
        )
        vertical_distances.append(left_vertical)
        
        # Add right-side vertical distance
        right_vertical = self._calculate_distance(
            inner_lip[6],  # right-side of top inner lip
            inner_lip[14]  # right-side of bottom inner lip
        )
        vertical_distances.append(right_vertical)
        
        # Calculate average vertical distance
        avg_vertical_distance = sum(vertical_distances) / len(vertical_distances)
        
        # Calculate MAR
        if horizontal_distance == 0:  # Prevent division by zero
            return 0.0
            
        mar = avg_vertical_distance / horizontal_distance
        return mar
    
    def get_eye_gaze_direction(self, landmarks: List[List[float]]) -> Tuple[float, float, float]:
        """
        Calculate approximate eye gaze direction based on eye landmarks.
        
        This method estimates gaze direction using the relative positions of eye landmarks,
        particularly the iris position relative to the eye corners.
        
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
        
        # Convert our gaze_dir (x,y,z) to angles
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
            # We'll use a simple projection since we don't have camera parameters
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
    
    def draw_gaze_direction_v2(self, frame: np.ndarray, landmarks: List[List[float]], 
                           gaze_dir: Tuple[float, float, float],
                           arrow_color: Tuple[int, int, int] = (0, 0, 255),
                           arrow_length: int = 100,
                           arrow_thickness: int = 2) -> np.ndarray:
        """
        Draw gaze direction arrow on the frame using ETH-XGaze approach.
        
        This method draws gaze direction vectors in a more accurate way,
        simulating the ETH-XGaze model approach with yaw and pitch angles.
        
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
        
        # Get iris center and eye corner landmarks
        left_iris_idx = 468  # Center of left iris
        right_iris_idx = 473  # Center of right iris
        left_eye_outer_idx = 362  # Left eye outer corner
        left_eye_inner_idx = 263  # Left eye inner corner
        right_eye_outer_idx = 33  # Right eye outer corner
        right_eye_inner_idx = 133  # Right eye inner corner
        
        # Calculate face width for scaling
        left_eye_outer = landmarks[left_eye_outer_idx]
        right_eye_outer = landmarks[right_eye_outer_idx]
        face_width = np.linalg.norm(np.array([left_eye_outer[0], left_eye_outer[1]]) - 
                                    np.array([right_eye_outer[0], right_eye_outer[1]]))
        
        # Get normalized scale based on face size
        scale_factor = face_width / 300.0  # Adjust divisor to calibrate arrow length
        
        # Get eye centers
        left_eye_center = (int(landmarks[left_iris_idx][0]), int(landmarks[left_iris_idx][1]))
        right_eye_center = (int(landmarks[right_iris_idx][0]), int(landmarks[right_iris_idx][1]))
        
        # Convert normalized gaze direction to yaw and pitch angles
        gaze_x, gaze_y, gaze_z = gaze_dir
        
        # Use arctan2 for yaw to get the correct sign
        # Use arcsin for pitch (vertical angle)
        yaw = np.arctan2(gaze_x, -gaze_z)  # Horizontal angle (left-right)
        pitch = np.arcsin(np.clip(-gaze_y, -1.0, 1.0))  # Vertical angle (up-down)
        
        # Calculate 3D gaze vectors for each eye
        for eye_center in [left_eye_center, right_eye_center]:
            # Calculate 3D vector components using spherical coordinates
            x = arrow_length * np.sin(yaw) * np.cos(pitch)
            y = arrow_length * np.sin(pitch)
            z = arrow_length * np.cos(yaw) * np.cos(pitch)
            
            # Project to 2D plane
            dx = scale_factor * x
            dy = scale_factor * y
            
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
        
        # Draw combined central gaze vector
        middle_point = ((left_eye_center[0] + right_eye_center[0]) // 2,
                        (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Calculate longer central vector
        x = 1.5 * arrow_length * np.sin(yaw) * np.cos(pitch)
        y = 1.5 * arrow_length * np.sin(pitch)
        z = 1.5 * arrow_length * np.cos(yaw) * np.cos(pitch)
        
        # Project to 2D, apply scaling
        dx = scale_factor * x
        dy = scale_factor * y
        
        # Calculate endpoint
        central_end = (int(middle_point[0] + dx), int(middle_point[1] + dy))
        
        # Draw central arrow
        cv2.arrowedLine(
            vis_frame,
            middle_point,
            central_end,
            (0, 165, 255),  # Orange for central arrow
            arrow_thickness + 1,
            cv2.LINE_AA,
            tipLength=0.2
        )
        
        # Draw angle values on frame
        yaw_deg = yaw * 180.0 / np.pi
        pitch_deg = pitch * 180.0 / np.pi
        cv2.putText(
            vis_frame,
            f"Yaw: {yaw_deg:.1f}°, Pitch: {pitch_deg:.1f}°",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return vis_frame
    
    def calculate_head_pose(self, landmarks: List[List[float]], frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Baş duruşunu hesapla (yaw, pitch, roll) - estimator.py yaklaşımıyla
        
        Args:
            landmarks: MediaPipe yüz işaretleri
            frame: Görüntü karesi
            
        Returns:
            Tuple containing:
            - Frame with head pose visualization
            - Angles (pitch, yaw, roll) in degrees
        """
        if not landmarks:
            return frame, (0, 0, 0)
            
        h, w, c = frame.shape
        
        if self.use_detailed_model:
            # Detaylı model için 2D landmark noktalarını al
            face_coordination_in_image = []
            for idx in self.DETAILED_MODEL_MEDIAPIPE_MAPPING:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    face_coordination_in_image.append([x, y])
                    # Baş duruşu hesaplama için kullanılan noktaları vurgula
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
            # Nokta sayısı yetersizse basit modele geri dön
            if len(face_coordination_in_image) != len(self.DETAILED_MODEL_MEDIAPIPE_MAPPING):
                print(f"Detaylı model için yeterli nokta bulunamadı ({len(face_coordination_in_image)}/{len(self.DETAILED_MODEL_MEDIAPIPE_MAPPING)}). Basit modele dönülüyor.")
                self.use_detailed_model = False
                return self.calculate_head_pose(landmarks, frame)
                
            face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
            model_points = self.DETAILED_MODEL_POINTS
        else:
            # Basit model için 2D landmark noktalarını al
            face_coordination_in_image = []
            for idx in self.HEAD_POSE_LANDMARKS:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    face_coordination_in_image.append([x, y])
                    # Baş duruşu hesaplama için kullanılan noktaları vurgula
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            
            # Nokta sayısı yetersizse işleme devam etme
            if len(face_coordination_in_image) != len(self.HEAD_POSE_LANDMARKS):
                return frame, (0, 0, 0)
                
            face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
            model_points = self.MODEL_POINTS
        
        # Kamera matrisi güncelle
        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distorsiyon katsayıları
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # SolvePnP ile rotasyon vektörünü ve translasyon vektörünü hesapla
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, face_coordination_in_image, cam_matrix, dist_matrix)
        
        if not success:
            return frame, (0, 0, 0)
        
        # Rotasyon matrisini hesapla
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        
        # Euler açılarını hesapla
        angles = self.rotation_matrix_to_angles(rotation_matrix)
        
        return frame, angles
    
    def rotation_matrix_to_angles(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Rotasyon matrisinden Euler açılarını hesapla (estimator.py'den alındı)
        
        Args:
            rotation_matrix: Rotasyon matrisi
            
        Returns:
            Tuple[float, float, float]: (pitch, yaw, roll) açıları (derece cinsinden)
        """
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                        rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # Derece cinsine çevir
        return (x * 180.0 / math.pi, y * 180.0 / math.pi, z * 180.0 / math.pi)
    
    def draw_head_pose_cube(self, frame: np.ndarray, landmarks: List[List[float]], 
                           cube_size: int = None) -> np.ndarray:
        """
        Baş duruşunu temsil eden bir küp çiz
        
        Args:
            frame: Görüntü karesi
            landmarks: Yüz işaretleri
            cube_size: Küp boyutu (None ise yüz boyutuna göre otomatik ayarlanır)
            
        Returns:
            np.ndarray: Küp eklenmiş görüntü
        """
        if not landmarks:
            return frame
            
        h, w, c = frame.shape
        
        # Kullanılacak model ve indeksleri belirle
        if self.use_detailed_model:
            landmark_indices = self.DETAILED_MODEL_MEDIAPIPE_MAPPING
            model_points = self.DETAILED_MODEL_POINTS
        else:
            landmark_indices = self.HEAD_POSE_LANDMARKS
            model_points = self.MODEL_POINTS
        
        # 2D landmark noktalarını al
        face_coordination_in_image = []
        for idx in landmark_indices:
            if idx < len(landmarks):
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                face_coordination_in_image.append([x, y])
        
        # Nokta sayısı yetersizse işleme devam etme
        if len(face_coordination_in_image) != len(landmark_indices):
            return frame
            
        face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
        
        # Kamera matrisi güncelle
        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distorsiyon katsayıları
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # SolvePnP ile rotasyon vektörünü ve translasyon vektörünü hesapla
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, face_coordination_in_image, cam_matrix, dist_matrix)
        
        if not success:
            return frame
        
        # Burun ucunu bul (indeks 1) ve küpün merkezi olarak kullan
        nose_tip = None
        if 1 < len(landmarks):
            nose_tip = (int(landmarks[1][0]), int(landmarks[1][1]))
        else:
            # Burun ucu bulunamadıysa, yüzün merkezini kullan
            face_rect = self.get_face_rect(landmarks)
            nose_tip = (face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2)
        
        # Yüz boyutunu hesapla
        if cube_size is None:
            # Yüzün genişliğini ve yüksekliğini hesapla
            # Yüz konturu noktalarını kullanarak daha doğru bir yüz boyutu hesapla
            face_contour_points = []
            for idx in self.FACE_CONTOUR_INDICES:
                if idx < len(landmarks):
                    face_contour_points.append([landmarks[idx][0], landmarks[idx][1]])
            
            if face_contour_points:
                # Kontur noktalarından min/max değerleri bul
                x_coords = [p[0] for p in face_contour_points]
                y_coords = [p[1] for p in face_contour_points]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                face_width = max_x - min_x
                face_height = max_y - min_y
            else:
                # Kontur noktaları bulunamazsa get_face_rect kullan
                face_rect = self.get_face_rect(landmarks)
                face_width = face_rect[2]
                face_height = face_rect[3]
            
            # Küp boyutunu yüz boyutuna göre ayarla (yüz genişliğinin %120'si)
            cube_size = int(max(face_width, face_height) * 1.2)
            
            # Minimum ve maksimum değerler belirle
            cube_size = max(150, min(cube_size, 400))  # 150 ile 400 arasında sınırla
        
        # Yüzün etrafında bir küp oluştur
        half_size = cube_size / 2
        
        # Küp köşe noktaları (3D) - burun ucunu merkez alarak
        cube_points = np.array([
            # Ön yüz (z = -half_size)
            [-half_size, -half_size, -half_size],  # Sol üst ön
            [half_size, -half_size, -half_size],   # Sağ üst ön
            [half_size, half_size, -half_size],    # Sağ alt ön
            [-half_size, half_size, -half_size],   # Sol alt ön
            
            # Arka yüz (z = half_size)
            [-half_size, -half_size, half_size],   # Sol üst arka
            [half_size, -half_size, half_size],    # Sağ üst arka
            [half_size, half_size, half_size],     # Sağ alt arka
            [-half_size, half_size, half_size]     # Sol alt arka
        ], dtype=np.float32)
        
        # Küp köşe noktalarını 2D'ye projeksiyon yap
        cube_2d, jac = cv2.projectPoints(cube_points, rotation_vec, translation_vec, cam_matrix, dist_matrix)
        cube_2d = cube_2d.astype(int)
        
        # Küp kenarlarını çiz
        # Ön yüz
        frame = cv2.line(frame, tuple(cube_2d[0][0]), tuple(cube_2d[1][0]), (0, 255, 0), 2)  # Üst
        frame = cv2.line(frame, tuple(cube_2d[1][0]), tuple(cube_2d[2][0]), (0, 255, 0), 2)  # Sağ
        frame = cv2.line(frame, tuple(cube_2d[2][0]), tuple(cube_2d[3][0]), (0, 255, 0), 2)  # Alt
        frame = cv2.line(frame, tuple(cube_2d[3][0]), tuple(cube_2d[0][0]), (0, 255, 0), 2)  # Sol
        
        # Arka yüz
        frame = cv2.line(frame, tuple(cube_2d[4][0]), tuple(cube_2d[5][0]), (0, 0, 255), 2)  # Üst
        frame = cv2.line(frame, tuple(cube_2d[5][0]), tuple(cube_2d[6][0]), (0, 0, 255), 2)  # Sağ
        frame = cv2.line(frame, tuple(cube_2d[6][0]), tuple(cube_2d[7][0]), (0, 0, 255), 2)  # Alt
        frame = cv2.line(frame, tuple(cube_2d[7][0]), tuple(cube_2d[4][0]), (0, 0, 255), 2)  # Sol
        
        # Bağlantı kenarları
        frame = cv2.line(frame, tuple(cube_2d[0][0]), tuple(cube_2d[4][0]), (255, 0, 0), 2)  # Sol üst
        frame = cv2.line(frame, tuple(cube_2d[1][0]), tuple(cube_2d[5][0]), (255, 0, 0), 2)  # Sağ üst
        frame = cv2.line(frame, tuple(cube_2d[2][0]), tuple(cube_2d[6][0]), (255, 0, 0), 2)  # Sağ alt
        frame = cv2.line(frame, tuple(cube_2d[3][0]), tuple(cube_2d[7][0]), (255, 0, 0), 2)  # Sol alt
        
        return frame
    
    def draw_head_pose_axes(self, frame: np.ndarray, landmarks: List[List[float]], 
                           length: int = 50) -> np.ndarray:
        """
        Baş duruşu eksenlerini çiz
        
        Args:
            frame: Görüntü karesi
            landmarks: Yüz işaretleri
            length: Eksen uzunluğu
            
        Returns:
            np.ndarray: Eksenler eklenmiş görüntü
        """
        if not landmarks:
            return frame
            
        h, w, c = frame.shape
        
        # Kullanılacak model ve indeksleri belirle
        if self.use_detailed_model:
            landmark_indices = self.DETAILED_MODEL_MEDIAPIPE_MAPPING
            model_points = self.DETAILED_MODEL_POINTS
        else:
            landmark_indices = self.HEAD_POSE_LANDMARKS
            model_points = self.MODEL_POINTS
        
        # 2D landmark noktalarını al
        face_coordination_in_image = []
        for idx in landmark_indices:
            if idx < len(landmarks):
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                face_coordination_in_image.append([x, y])
        
        # Nokta sayısı yetersizse işleme devam etme
        if len(face_coordination_in_image) != len(landmark_indices):
            return frame
            
        face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)
        
        # Kamera matrisi güncelle
        focal_length = 1 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distorsiyon katsayıları
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # SolvePnP ile rotasyon vektörünü ve translasyon vektörünü hesapla
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, face_coordination_in_image, cam_matrix, dist_matrix)
        
        if not success:
            return frame
        
        # Burun ucunu bul (indeks 1) ve eksenlerin orijini olarak kullan
        nose_tip = None
        if 1 < len(landmarks):
            nose_tip = (int(landmarks[1][0]), int(landmarks[1][1]))
        else:
            # Burun ucu bulunamadıysa, yüzün merkezini kullan
            face_rect = self.get_face_rect(landmarks)
            nose_tip = (face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2)
            
        # Eksen noktalarını oluştur
        axis_points = np.array([
            [0, 0, 0],       # Orijin
            [length, 0, 0],  # X ekseni
            [0, length, 0],  # Y ekseni
            [0, 0, length]   # Z ekseni
        ], dtype=np.float32)
        
        # 3D noktaları 2D'ye projeksiyon yap
        imgpts, jac = cv2.projectPoints(axis_points, rotation_vec, translation_vec, cam_matrix, dist_matrix)
        imgpts = imgpts.astype(int)
        
        # Eksenlerin yönlerini hesapla
        origin = nose_tip
        x_end = (origin[0] + int(imgpts[1][0][0] - imgpts[0][0][0]), 
                origin[1] + int(imgpts[1][0][1] - imgpts[0][0][1]))
        
        y_end = (origin[0] + int(imgpts[2][0][0] - imgpts[0][0][0]), 
                origin[1] + int(imgpts[2][0][1] - imgpts[0][0][1]))
        
        z_end = (origin[0] + int(imgpts[3][0][0] - imgpts[0][0][0]), 
                origin[1] + int(imgpts[3][0][1] - imgpts[0][0][1]))
        
        # Eksenleri çiz
        frame = cv2.line(frame, origin, x_end, (0, 0, 255), 3)  # X ekseni - Kırmızı
        frame = cv2.line(frame, origin, y_end, (0, 255, 0), 3)  # Y ekseni - Yeşil
        frame = cv2.line(frame, origin, z_end, (255, 0, 0), 3)  # Z ekseni - Mavi
        
        return frame
    
    def visualize_head_pose(self, frame: np.ndarray, landmarks: List[List[float]], 
                           show_axes: bool = True, show_angles: bool = True,
                           visualization_type: str = 'cube',
                           use_detailed_model: bool = False) -> np.ndarray:
        """
        Baş duruşunu görselleştir
        
        Args:
            frame: Görüntü karesi
            landmarks: Yüz işaretleri
            show_axes: Görselleştirmeyi göster
            show_angles: Açıları göster
            visualization_type: Görselleştirme tipi ('cube' veya 'axes')
            use_detailed_model: Detaylı 3D modeli kullan
            
        Returns:
            np.ndarray: Görselleştirilmiş görüntü
        """
        if not landmarks:
            return frame
        
        # Detaylı model kullanımını ayarla
        self.use_detailed_model = use_detailed_model
            
        # Baş duruşunu hesapla
        vis_frame, angles = self.calculate_head_pose(landmarks, frame.copy())
        pitch, yaw, roll = angles
        
        if show_axes:
            # Görselleştirme tipine göre çiz
            if visualization_type.lower() == 'cube':
                vis_frame = self.draw_head_pose_cube(vis_frame, landmarks)
            else:
                vis_frame = self.draw_head_pose_axes(vis_frame, landmarks)
        
        if show_angles:
            # Açıları göster
            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), angles)):
                k, v = info
                text = f"{k}: {int(v)}"
                cv2.putText(vis_frame, text, (20, i*30 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        
        return vis_frame
    
    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
    
    def estimate_head_pose(self, landmarks: List[List[float]], frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baş duruşunu tahmin et
        
        Args:
            landmarks: 2D yüz işaret noktaları
            frame: Giriş görüntüsü
            
        Returns:
            rvec, tvec: Rotasyon ve translasyon vektörleri
        """
        # Kamera matrisini ve distorsiyon katsayılarını hesapla
        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        distortion = np.zeros((4, 1), dtype=np.float64)
        
        # Gaze için kullanılan 6 noktayı al
        landmarks_2d = []
        for idx in self.GAZE_LANDMARK_INDICES:
            if idx < len(landmarks):
                landmarks_2d.append([landmarks[idx][0], landmarks[idx][1]])
        
        landmarks_2d = np.array(landmarks_2d, dtype=np.float64)
        
        # SolvePnP ile baş duruşunu hesapla
        if len(landmarks_2d) == 6:  # Tüm noktalar bulundu
            ret, rvec, tvec = cv2.solvePnP(
                self.GAZE_FACE_MODEL, 
                landmarks_2d, 
                camera_matrix, 
                distortion, 
                flags=cv2.SOLVEPNP_EPNP
            )
            
            # Daha fazla optimize et
            ret, rvec, tvec = cv2.solvePnP(
                self.GAZE_FACE_MODEL, 
                landmarks_2d, 
                camera_matrix, 
                distortion, 
                rvec, tvec, 
                True
            )
            
            return rvec, tvec
        
        # Yeterli nokta bulunamadı
        return np.zeros((3, 1), dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
    
    def normalize_face(self, img: np.ndarray, landmarks: List[List[float]], frame: np.ndarray) -> np.ndarray:
        """
        Yüz görüntüsünü normalize et (ETH-XGaze'den uyarlandı)
        
        Args:
            img: Giriş görüntüsü
            landmarks: 2D yüz işaret noktaları
            frame: Orijinal kare
            
        Returns:
            img_normalized: Normalize edilmiş yüz görüntüsü
        """
        # Baş duruşunu tahmin et
        hr, ht = self.estimate_head_pose(landmarks, frame)
        
        # Kamera matrisini hesapla
        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Normalize edilmiş kamera parametreleri
        focal_norm = 960  # Normalize edilmiş kameranın odak uzaklığı
        distance_norm = 600  # Göz ve kamera arasındaki normalize edilmiş mesafe
        roi_size = (224, 224)  # Kırpılmış göz görüntüsünün boyutu
        
        # İşaret noktalarının 3D pozisyonlarını hesapla
        ht = ht.reshape((3, 1))
        hR = cv2.Rodrigues(hr)[0]  # Rotasyon matrisi
        Fc = np.dot(hR, self.GAZE_FACE_MODEL.T) + ht  # Yüz modelini döndür ve taşı
        
        # Yüz merkezini bul (göz ve burun merkezlerinin ortalaması)
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))
        
        # Görüntüyü normalize et
        distance = np.linalg.norm(face_center)  # Göz ve orijinal kamera arasındaki gerçek mesafe
        
        z_scale = distance_norm / distance
        cam_norm = np.array([  # Sanal kameranın iç parametreleri
            [focal_norm, 0, roi_size[0] / 2],
            [0, focal_norm, roi_size[1] / 2],
            [0, 0, 1.0],
        ])
        
        # Dönüşüm matrisini hesapla
        S = np.array([  # Ölçekleme matrisi
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
        
        R = np.c_[right, down, forward].T  # Rotasyon matrisi
        
        # Normalize edilmiş görüntü için dönüşüm matrisini hesapla
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix)))
        
        # Görüntüyü dönüştür
        img_normalized = cv2.warpPerspective(img, W, roi_size)
        
        return img_normalized
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        ETH-XGaze modeli için görüntüyü ön işleme tabi tut
        
        Args:
            image: Normalize edilmiş yüz görüntüsü (224x224)
            
        Returns:
            processed_img: İşlenmiş görüntü (model girdisi için)
        """
        # BGR'dan RGB'ye dönüştür
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Görüntüyü yeniden boyutlandır
        image = cv2.resize(image, (224, 224))
        
        # Görüntüyü normalize et ([0,1] aralığına)
        image = image.astype(np.float32) / 255.0
        
        # Ortalama ve standart sapma ile normalize et
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Kanalları düzenle (HWC -> CHW)
        image = image.transpose(2, 0, 1)
        
        # Batch boyutu ekle
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_gaze(self, frame: np.ndarray, landmarks: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bakış yönünü tahmin et
        
        Args:
            frame: Giriş görüntüsü
            landmarks: 2D yüz işaret noktaları
            
        Returns:
            gaze_vector: Bakış yönü vektörü (pitch, yaw)
            normalized_image: Normalize edilmiş yüz görüntüsü
        """
        if self.onnx_session is None:
            # Model yüklü değilse boş vektör döndür
            return np.zeros(2), None
        
        # Yüzü normalize et
        normalized_image = self.normalize_face(frame, landmarks, frame)
        
        # Görüntüyü ön işleme tabi tut
        processed_img = self.preprocess_image(normalized_image)
        
        # Model çıkarımı yap
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        gaze = self.onnx_session.run([output_name], {input_name: processed_img})[0]
        
        # Bakış yönü vektörünü al (pitch, yaw)
        gaze_vector = gaze[0]  # [pitch, yaw]
        
        return gaze_vector, normalized_image
    
    def draw_gaze(self, image: np.ndarray, pitchyaw: np.ndarray, origin: Tuple[int, int], 
                 length: int = 50, thickness: int = 2, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Bakış yönünü görselleştir
        
        Args:
            image: Giriş görüntüsü
            pitchyaw: Bakış yönü vektörü (pitch, yaw)
            origin: Bakış yönü orijin noktası (x, y)
            length: Ok uzunluğu
            thickness: Ok kalınlığı
            color: Ok rengi
            
        Returns:
            image: Bakış yönü çizilmiş görüntü
        """
        pitch, yaw = pitchyaw
        
        # Pitch ve yaw'ı radyana dönüştür
        pitch = pitch
        yaw = yaw
        
        # Bakış yönü vektörünü hesapla
        x = -length * np.sin(yaw) * np.cos(pitch)
        y = -length * np.sin(pitch)
        z = -length * np.cos(yaw) * np.cos(pitch)
        
        # 3D vektörü 2D'ye projeksiyon
        point_2d = (int(origin[0] + x), int(origin[1] + y))
        
        # Ok çiz
        cv2.arrowedLine(image, origin, point_2d, color, thickness)
        
        return image
    
    def visualize_gaze(self, frame: np.ndarray, landmarks: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bakış yönünü tahmin et ve görselleştir
        
        Args:
            frame: Giriş görüntüsü
            landmarks: 2D yüz işaret noktaları
            
        Returns:
            frame: Bakış yönü çizilmiş görüntü
            normalized_image: Normalize edilmiş yüz görüntüsü (eğer varsa)
        """
        if not landmarks or self.onnx_session is None:
            return frame, None
        
        # Bakış yönünü tahmin et
        gaze_vector, normalized_image = self.predict_gaze(frame, landmarks)
        
        if gaze_vector is None:
            return frame, None
        
        # Burun ucunu bul (bakış yönü orijini olarak)
        nose_tip = None
        if 1 < len(landmarks):  # 1 indeksi burun ucudur
            nose_tip = (int(landmarks[1][0]), int(landmarks[1][1]))
        
        # Burun ucu bulunamadıysa, yüzün merkezini kullan
        if nose_tip is None:
            face_rect = self.get_face_rect(landmarks)
            nose_tip = (face_rect[0] + face_rect[2] // 2, face_rect[1] + face_rect[3] // 2)
        
        # Bakış yönünü çiz
        frame = self.draw_gaze(frame, gaze_vector, nose_tip)
        
        # Bakış açılarını ekranda göster
        pitch, yaw = np.rad2deg(gaze_vector)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, normalized_image

# Convenience function to get a preconfigured MediaPipeUtils instance
def get_mediapipe_face_mesh() -> MediaPipeUtils:
    """
    Factory function to create and return a MediaPipeUtils instance.
    
    Returns:
        MediaPipeUtils: Initialized MediaPipeUtils instance
    """
    return MediaPipeUtils()

# Load UI configuration from YAML file
def load_ui_config() -> Dict:
    """
    Load UI configuration from YAML file.
    
    Returns:
        Dict: UI configuration dictionary
    """
    import yaml
    import os
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config', 'ui_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading UI config: {e}")
        # Return default configuration
        return {
            'window': {
                'title': 'Sürücü Uykululuk Tespiti',
                'width': 1200,
                'height': 800,
                'min_width': 800,
                'min_height': 600
            },
            'video_frame': {
                'width': 640,
                'height': 480
            },
            'layout': {
                'margin': 10,
                'padding': 5,
                'spacing': 10
            },
            'fonts': {
                'family': 'Arial',
                'title_size': 14,
                'label_size': 12,
                'value_size': 16
            },
            'controls': {
                'button_width': 120,
                'button_height': 40
            },
            'indicators': {
                'ear': {
                    'min': 0.0,
                    'max': 0.4,
                    'warning_threshold': 0.25,
                    'critical_threshold': 0.21
                },
                'mar': {
                    'min': 0.0,
                    'max': 1.0,
                    'warning_threshold': 0.7,
                    'critical_threshold': 0.8
                },
                'perclos': {
                    'min': 0.0,
                    'max': 100.0,
                    'warning_threshold': 15.0,
                    'critical_threshold': 20.0
                }
            },
            'chart': {
                'history_duration': 30,
                'line_width': 2,
                'y_range_ear': [0.0, 0.4],
                'y_range_mar': [0.0, 1.0],
                'y_range_perclos': [0.0, 100.0]
            },
            'camera': {
                'device': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            }
        }
