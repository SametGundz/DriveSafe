#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions for calculating facial metrics like EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio).

This module provides functions to calculate various metrics from facial landmarks
detected using MediaPipe Face Mesh.
"""

import math
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np


def calculate_distance(point1: List[float], point2: List[float]) -> float:
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
    
    # Hesaplanacak kare sayısı
    window_size = window_seconds * fps
    
    # Son window_size kadar kareyi al veya tümünü
    history = eye_state_history[-min(window_size, len(eye_state_history)):]
    
    # Kapalı göz sayısı (0 değerleri)
    closed_count = history.count(0)
    
    # PERCLOS hesaplama
    perclos = closed_count / len(history)
    
    return perclos


def is_blinking(ear: float, threshold: float = 0.21, 
               consecutive_frames: int = 3, 
               ear_history: Optional[List[float]] = None) -> Tuple[bool, Optional[List[float]]]:
    """
    Detect if the eye is blinking based on EAR value.
    
    Args:
        ear: Current Eye Aspect Ratio
        threshold: EAR threshold below which the eye is considered closed
        consecutive_frames: Number of consecutive frames below threshold to confirm blink
        ear_history: Optional history of EAR values to track blink
        
    Returns:
        Tuple containing:
        - Whether the eye is blinking
        - Updated ear_history list
    """
    # EAR geçmişini başlat
    if ear_history is None:
        ear_history = []
    
    # Mevcut EAR değerini geçmişe ekle
    ear_history.append(ear)
    
    # Geçmişi belli bir uzunlukta tut
    if len(ear_history) > consecutive_frames * 2:
        ear_history = ear_history[-consecutive_frames * 2:]
    
    # Son consecutive_frames kadar kareye bak
    recent_ears = ear_history[-consecutive_frames:]
    
    # Tüm son kareler eşik değerinin altındaysa göz kırpma
    is_blink = all(e < threshold for e in recent_ears)
    
    return is_blink, ear_history


def is_yawning(mar: float, threshold: float = 0.5, 
              consecutive_frames: int = 5,
              mar_history: Optional[List[float]] = None) -> Tuple[bool, Optional[List[float]]]:
    """
    Detect if the mouth is yawning based on MAR value.
    
    Args:
        mar: Current Mouth Aspect Ratio
        threshold: MAR threshold above which the mouth is considered yawning
        consecutive_frames: Number of consecutive frames above threshold to confirm yawn
        mar_history: Optional history of MAR values to track yawn
        
    Returns:
        Tuple containing:
        - Whether the mouth is yawning
        - Updated mar_history list
    """
    # MAR geçmişini başlat
    if mar_history is None:
        mar_history = []
    
    # Mevcut MAR değerini geçmişe ekle
    mar_history.append(mar)
    
    # Geçmişi belli bir uzunlukta tut
    if len(mar_history) > consecutive_frames * 2:
        mar_history = mar_history[-consecutive_frames * 2:]
    
    # Son consecutive_frames kadar kareye bak
    recent_mars = mar_history[-consecutive_frames:]
    
    # Tüm son kareler eşik değerinin üstündeyse esneme
    is_yawn = all(m > threshold for m in recent_mars)
    
    return is_yawn, mar_history