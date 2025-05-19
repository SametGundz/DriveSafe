#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for calculating facial metrics like EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio).

This module provides functions to calculate various metrics from facial landmarks
detected using MediaPipe Face Mesh.
"""

import math
import os
import cv2
import numpy as np
import mediapipe as mp
import yaml
from typing import List, Optional, Tuple, Dict, Any, Union, Sequence

# Import constants
from src.utils.constants import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, 
    LEFT_EYE_INDICES_6POINT, RIGHT_EYE_INDICES_6POINT,
    INNER_LIP_INDICES, OUTER_LIP_INDICES, MOUTH_INDICES, MOUTH_INDICES_6POINT,
    EAR_THRESHOLD, MAR_THRESHOLD
)


def calculate_distance(point1: Sequence[float], point2: Sequence[float]) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        point1: First point coordinates [x, y, z]
        point2: Second point coordinates [x, y, z]
        
    Returns:
        Euclidean distance between the points
    """
    # Numpy kullanarak daha hızlı hesaplama
    return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))


def get_eye_aspect_ratio(eye_landmarks: List[List[float]]) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for the given eye landmarks.
    
    MediaPipe Face Mesh'in göz landmarkları için özel olarak tasarlanmıştır.
    
    Args:
        eye_landmarks: List of landmark coordinates [x, y, z] for an eye
        
    Returns:
        EAR value (between 0-0.5 typically)
    """
    if not eye_landmarks or len(eye_landmarks) < 4:
        return 0.25  # Default value if insufficient landmarks
    
    try:
        # MediaPipe göz noktaları için özel EAR hesaplaması
        
        # Göz noktalarını NumPy dizisine dönüştürerek hızlandırma
        points = np.array(eye_landmarks)
        
        # MediaPipe'ın tam göz landmarkları için (16 nokta)
        if len(eye_landmarks) >= 16:
            # Kontur sırasına göre sırala (x koordinatlarına göre)
            sorted_x = np.argsort(points[:, 0])
            left_most = sorted_x[0]  # En soldaki nokta
            right_most = sorted_x[-1]  # En sağdaki nokta
            
            # Üst ve alt noktaları bul (y koordinatlarına göre)
            sorted_y = np.argsort(points[:, 1])
            top_points = sorted_y[:5]  # En üstteki 5 nokta
            bottom_points = sorted_y[-5:]  # En alttaki 5 nokta
            
            # Göz genişliği
            eye_width = calculate_distance(eye_landmarks[left_most], eye_landmarks[right_most])
            
            # Göz yüksekliği için üst ve alt noktaların ortalamasını al
            top_mean = np.mean(points[top_points][:, 1])
            bottom_mean = np.mean(points[bottom_points][:, 1])
            
            # Dikey mesafe
            vertical_distance = bottom_mean - top_mean
            
            # EAR hesaplama
            ear = vertical_distance / max(eye_width, 1e-6)  # 0'a bölmeyi önle
            
            # EAR'ı normalize et - MediaPipe, dlib'den farklı ölçekte değerler verebilir
            # Tipik açık göz EAR değeri 0.2-0.3 aralığında
            ear = min(max(ear * 1.5, 0.0), 0.5)
            
            return float(ear)
            
        # Az sayıda landmark varsa (örn. 6-point model)
        elif len(eye_landmarks) >= 6:
            # 6-noktalı standart EAR hesaplaması
            # Dlib 6-noktalı göz modeline göre indeksler:
            # 0=sol köşe, 1=üst-sol, 2=üst-sağ, 3=sağ köşe, 4=alt-sağ, 5=alt-sol
            
            # Dikey mesafeler
            A = calculate_distance(eye_landmarks[1], eye_landmarks[5])  # Üst-sol ile alt-sol
            B = calculate_distance(eye_landmarks[2], eye_landmarks[4])  # Üst-sağ ile alt-sağ
            
            # Yatay mesafe
            C = calculate_distance(eye_landmarks[0], eye_landmarks[3])  # Sol köşe ile sağ köşe
            
            # EAR formülü
            ear = (A + B) / (2.0 * max(C, 1e-6))  # 0'a bölmeyi önle
            
            # Normalize et
            ear = min(ear, 0.5)
            
            return float(ear)
            
        # Daha az nokta varsa (minimum 4)
        else:
            # Basit bir dikdörtgen yaklaşımı
            # En uç noktaları bul
            x_sorted = np.argsort(points[:, 0])
            y_sorted = np.argsort(points[:, 1])
            
            left = points[x_sorted[0]]
            right = points[x_sorted[-1]]
            top = points[y_sorted[0]]
            bottom = points[y_sorted[-1]]
            
            width = calculate_distance(left, right)
            height = calculate_distance(top, bottom)
            
            ear = height / max(width, 1e-6)  # 0'a bölmeyi önle
            ear = min(ear, 0.5)
            
            return float(ear)
    
    except Exception as e:
        print(f"EAR hesaplanırken hata oluştu: {str(e)}")
        return 0.25  # Hata durumunda varsayılan değer


def get_mouth_aspect_ratio(mouth_landmarks: List[List[float]]) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR) for the given mouth landmarks.
    
    MAR is the ratio of the height of the mouth to the width of the mouth.
    Kapalı ağız için tipik değerler 0.05-0.15, açık ağız için 0.3-0.7 aralığındadır.
    
    Args:
        mouth_landmarks: List of landmark coordinates for the mouth
            
    Returns:
        MAR value (typically between 0.05-0.7)
    """
    if not mouth_landmarks or len(mouth_landmarks) < 4:
        return 0.1  # Default value if insufficient landmarks
    
    try:
        # Noktaları NumPy dizisine dönüştürerek hızlandırma
        points = np.array(mouth_landmarks)
        
        # X ve Y koordinatlarına göre sıralama
        x_sorted = np.argsort(points[:, 0])
        y_sorted = np.argsort(points[:, 1])
        
        # Dış dudak landmarkları için (20+ nokta varsa)
        if len(mouth_landmarks) >= 20:
            # MediaPipe'ın OUTER_LIP_INDICES ve INNER_LIP_INDICES değerlerini kullan
            # Dış dudak için önemli noktaları seçelim
            
            # Sol ve sağ köşeler (minimum ve maximum x)
            left_corner_idx = x_sorted[0]
            right_corner_idx = x_sorted[-1]
            
            # Üst ve alt dudak için en uç noktalar
            top_lip_idx = y_sorted[0]
            bottom_lip_idx = y_sorted[-1]
            
            # Ağız genişliği
            mouth_width = calculate_distance(mouth_landmarks[left_corner_idx], 
                                            mouth_landmarks[right_corner_idx])
            
            # İç dudaklar arası mesafeyi bulmak için orta bölgedeki noktaları kullan
            # Bu daha doğru bir ağız açıklığı ölçümü verir
            
            # Ağız merkezi x koordinatı
            center_x = (mouth_landmarks[left_corner_idx][0] + mouth_landmarks[right_corner_idx][0]) / 2
            
            # İç dudak için üst ve alt noktaları bul
            # Tüm noktalar içinden merkeze yakın olanları filtrele
            center_region_width = mouth_width * 0.3  # Merkez bölge genişliği
            center_region_points = [
                i for i, p in enumerate(mouth_landmarks) 
                if abs(p[0] - center_x) < center_region_width
            ]
            
            if center_region_points:
                # Merkez bölgeden en üstteki ve en alttaki noktalar
                center_y_values = [(i, mouth_landmarks[i][1]) for i in center_region_points]
                center_y_sorted = sorted(center_y_values, key=lambda x: x[1])
                
                top_center_idx = center_y_sorted[0][0]
                bottom_center_idx = center_y_sorted[-1][0]
                
                # İç dudaklar arası yükseklik
                inner_height = calculate_distance(
                    mouth_landmarks[top_center_idx], 
                    mouth_landmarks[bottom_center_idx]
                )
                
                # MAR hesaplama - iç dudak yüksekliğini ağız genişliğine oranla
                mar = inner_height / max(mouth_width, 1e-6)
                
                # Kalibrasyon faktörü - kapalı ağız için daha düşük değerler vermesi için
                # Kapalı ağız için 0.05-0.15, açık ağız için 0.3-0.7 aralığında olmalı
                calibration_factor = 0.6
                mar = mar * calibration_factor
                
                # MAR değerini sınırla
                mar = min(max(mar, 0.05), 0.7)
                
                return float(mar)
            
        # Yeterli nokta yoksa veya merkez bölge bulunamadıysa basit hesaplamaya dön
        
        # En uç noktaları kullan
        left = points[x_sorted[0]]
        right = points[x_sorted[-1]]
        top = points[y_sorted[0]]
        bottom = points[y_sorted[-1]]
        
        mouth_width = calculate_distance(left, right)
        mouth_height = calculate_distance(top, bottom)
        
        # Kalibrasyon faktörü
        calibration_factor = 0.6
        mar = (mouth_height / max(mouth_width, 1e-6)) * calibration_factor
        
        # MAR değerini sınırla
        mar = min(max(mar, 0.05), 0.7)
        
        return float(mar)
    
    except Exception as e:
        print(f"MAR hesaplanırken hata oluştu: {str(e)}")
        return 0.1  # Hata durumunda varsayılan değer


def get_perclos(eye_state_history: List[int], window_seconds: int = 60, fps: int = 30) -> float:
    """
    Calculate PERCLOS (percentage of eye closure) from eye state history.
    
    Args:
        eye_state_history: List of eye states (0 for closed, 1 for open)
        window_seconds: Time window in seconds for PERCLOS calculation
        fps: Frames per second
        
    Returns:
        PERCLOS value (0.0-1.0)
    """
    if not eye_state_history:
        return 0.0
    
    # Calculate window size in frames
    window_size = min(len(eye_state_history), window_seconds * fps)
    
    # Get recent history within the window
    recent_history = eye_state_history[-window_size:]
    
    # Calculate PERCLOS
    closed_frames = sum(1 for state in recent_history if state == 0)
    perclos = (closed_frames / window_size) * 100.0 if window_size > 0 else 0.0
    
    return perclos


def is_blinking(ear: float, threshold: float = 0.21, 
               consecutive_frames: int = 3, 
               ear_history: Optional[List[float]] = None) -> Tuple[bool, List[float]]:
    """
    Detect if the eye is blinking based on EAR values.
    
    Args:
        ear: Current eye aspect ratio
        threshold: EAR threshold for closed eyes
        consecutive_frames: Number of consecutive frames below threshold to consider a blink
        ear_history: History of EAR values
        
    Returns:
        Tuple of (is_blinking, updated_ear_history)
    """
    # Initialize history if not provided
    if ear_history is None:
        ear_history = []
    
    # Add current EAR to history
    ear_history.append(ear)
    
    # Keep only recent history
    max_history_len = consecutive_frames * 3  # 3x to include before, during, after blink
    if len(ear_history) > max_history_len:
        ear_history = ear_history[-max_history_len:]
    
    # Check if the eye is considered blinking
    is_blinking_now = False
    
    # Only check if we have enough history
    if len(ear_history) >= consecutive_frames:
        # Get the most recent EAR values
        recent_ear = ear_history[-consecutive_frames:]
        # Check if all recent EAR values are below threshold
        is_blinking_now = all(e < threshold for e in recent_ear)
    
    return is_blinking_now, ear_history


def is_yawning(mar: float, threshold: float = 0.5, 
              consecutive_frames: int = 5,
              mar_history: Optional[List[float]] = None) -> Tuple[bool, List[float]]:
    """
    Detect if the person is yawning based on MAR values.
    
    Args:
        mar: Current mouth aspect ratio
        threshold: MAR threshold for open mouth
        consecutive_frames: Number of consecutive frames above threshold to consider a yawn
        mar_history: History of MAR values
        
    Returns:
        Tuple of (is_yawning, updated_mar_history)
    """
    # Initialize history if not provided
    if mar_history is None:
        mar_history = []
    
    # Add current MAR to history
    mar_history.append(mar)
    
    # Keep only recent history
    max_history_len = consecutive_frames * 3  # 3x to include before, during, after yawn
    if len(mar_history) > max_history_len:
        mar_history = mar_history[-max_history_len:]
    
    # Check if the person is considered yawning
    is_yawning_now = False
    
    # Only check if we have enough history
    if len(mar_history) >= consecutive_frames:
        # Get the most recent MAR values
        recent_mar = mar_history[-consecutive_frames:]
        # Check if all recent MAR values are above threshold
        is_yawning_now = all(m > threshold for m in recent_mar)
    
    return is_yawning_now, mar_history


def calculate_ear_mar(landmarks: List[List[float]]) -> Tuple[float, float, float, float]:
    """
    Calculate EAR and MAR values from facial landmarks.
    
    This function combines EAR and MAR calculations for convenience.
    
    Args:
        landmarks: Full list of facial landmarks
        
    Returns:
        Tuple of (left_ear, right_ear, avg_ear, mar)
    """
    # Get eye landmarks
    left_eye = [landmarks[i] for i in LEFT_EYE_INDICES if i < len(landmarks)]
    right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES if i < len(landmarks)]
    
    # Get mouth landmarks
    mouth = [landmarks[i] for i in MOUTH_INDICES if i < len(landmarks)]
    
    # Calculate metrics
    left_ear = get_eye_aspect_ratio(left_eye)
    right_ear = get_eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    mar = get_mouth_aspect_ratio(mouth)
    
    return left_ear, right_ear, avg_ear, mar


def detect_ear_mar(frame: np.ndarray) -> Tuple[Optional[float], Optional[float], bool, bool]:
    """
    Detect EAR and MAR values from a frame.
    
    This function integrates the functionality from EARMARDetector for direct frame processing.
    
    Args:
        frame: Input frame/image
        
    Returns:
        Tuple containing:
        - Average EAR value (None if no face detected)
        - MAR value (None if no face detected)
        - Boolean indicating if eyes are closed
        - Boolean indicating if mouth is open
    """
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get frame dimensions
    h, w, _ = frame.shape
    
    # Process the frame with MediaPipe FaceMesh
    results = face_mesh.process(rgb_frame)
    
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
        
        # Get landmarks for 6-point eye model
        left_eye = [landmarks[i] for i in LEFT_EYE_INDICES_6POINT if i < len(landmarks)]
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES_6POINT if i < len(landmarks)]
        mouth = [landmarks[i] for i in MOUTH_INDICES_6POINT if i < len(landmarks)]
        
        # Calculate EAR for each eye
        left_ear = get_eye_aspect_ratio(left_eye)
        right_ear = get_eye_aspect_ratio(right_eye)
        
        # Calculate average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR
        mar = get_mouth_aspect_ratio(mouth)
        
        # Determine if eyes are closed and mouth is open based on thresholds
        eyes_closed = avg_ear < EAR_THRESHOLD
        mouth_open = mar > MAR_THRESHOLD
    
    # Release MediaPipe resources
    face_mesh.close()
    
    return avg_ear, mar, eyes_closed, mouth_open


def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Get the project root directory (3 levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except (FileNotFoundError, yaml.YAMLError):
        print(f"Warning: Could not load config file at {config_path}. Using default values.")
        return {}