#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drowsiness Detector (Uykululuk Tespit) modülü.

Bu modül, sürücünün uykululuk durumunu tespit etmek için çeşitli özellikleri (göz kapalılık, 
bakış yönü, baş duruşu vb.) kullanarak bir uykululuk seviyesi belirleme mekanizması sunar.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Tuple, List, Optional, Union
from collections import deque

# Logger oluştur
logger = logging.getLogger('driver_monitoring')

def dist_euclid(a, b):
    """
    İki nokta arasındaki Öklid mesafesini hesaplar.
    
    Args:
        a: İlk nokta koordinatları [x, y]
        b: İkinci nokta koordinatları [x, y]
    
    Returns:
        float: Öklid mesafesi
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


class DrowsinessDetector:
    """
    Sürücü uykululuk tespiti için EAR ve bakış yönü tabanlı değerlendirme sınıfı.
    
    Bu sınıf, göz açıklık oranı (EAR), bakış yönü ve PERCLOS değerlerini
    kullanarak sürücünün uykululuk seviyesini hesaplar.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Uykululuk dedektörünü başlatır.
        
        Args:
            config: Dedektör yapılandırma parametreleri
        """
        # Varsayılan yapılandırma
        self.config = {
            'ear_threshold': 0.2,                # EAR eşik değeri
            'ear_closed_duration': 2.0,          # Göz kapalı kabul edilme süresi (saniye)
            'ear_history_size': 150,             # EAR geçmiş değer boyutu (5 saniye @ 30fps)
            'perclos_threshold': 0.25,           # PERCLOS eşik değeri
            'perclos_window': 150,               # PERCLOS hesaplama penceresi (5 saniye @ 30fps)
            'gaze_deviation_threshold': 30.0,    # Bakış sapma eşik değeri (derece)
            'gaze_history_size': 150,            # Bakış geçmiş değer boyutu (5 saniye @ 30fps)
            'history_duration': 5.0,             # Geçmiş verileri saklama süresi (saniye)
            'head_pose_weight': 0.2,             # Baş duruşunun uykululuk tespitindeki ağırlığı
            'non_forward_pose_threshold': 4.0,   # İleri bakmama eşik süresi (saniye)
        }
        
        # Kullanıcı yapılandırmasını uygula
        if config:
            self.config.update(config)
        
        # Yapılandırma parametrelerini değişkenlere aktar
        self.ear_threshold = self.config['ear_threshold']
        self.ear_closed_duration = self.config['ear_closed_duration']
        self.perclos_threshold = self.config['perclos_threshold']
        self.gaze_deviation_threshold = self.config['gaze_deviation_threshold']
        self.history_duration = self.config['history_duration']
        self.head_pose_weight = self.config['head_pose_weight']
        self.non_forward_pose_threshold = self.config['non_forward_pose_threshold']
        
        # Geçmiş verileri tutmak için kuyrukar
        self.ear_history = deque(maxlen=int(self.history_duration * 30))
        self.ear_values = deque(maxlen=self.config['perclos_window'])
        self.gaze_history = deque(maxlen=int(self.history_duration * 30))
        
        # Baş duruşu geçmişi
        self.head_pose_history = deque(maxlen=int(self.history_duration * 30))
        self.non_forward_pose_time = 0.0
        self.current_head_pose = "Bilinmiyor"
        
        # Durum değişkenleri
        self.last_update_time = time.time()
        self.is_eyes_closed = False
        self.eyes_closed_start_time = None
        self.perclos_value = 0.0
        self.drowsiness_level = 0.0
        self.drowsiness_state = "Uyanık"
    
    def calculate_ear(self, eye_landmarks: List[List[float]]) -> float:
        """
        Göz işaret noktalarından EAR (Göz Açıklık Oranı) hesaplar.
        
        Bu metod, bir göze ait 6 landmark noktasını kullanarak EAR değerini hesaplar.
        
        Args:
            eye_landmarks: Göz işaret noktaları koordinatları
            
        Returns:
            float: Hesaplanan EAR değeri
        """
        # Göz işaret noktaları eksikse None döndür
        if len(eye_landmarks) < 6:
            logger.warning(f"Yetersiz göz noktası sayısı: {len(eye_landmarks)}")
            return 0.0
        
        # Göz genişliği (yatay mesafe)
        a = dist_euclid(eye_landmarks[1], eye_landmarks[5])
        b = dist_euclid(eye_landmarks[2], eye_landmarks[4])
        
        # Göz yüksekliği (dikey mesafe)
        c = dist_euclid(eye_landmarks[0], eye_landmarks[3])
        
        # EAR hesapla: ((p2-p6) + (p3-p5)) / (2 * (p1-p4))
        if c > 0:
            ear = (a + b) / (2.0 * c)
        else:
            ear = 0.0
        
        return ear
    
    def calculate_perclos(self) -> float:
        """
        PERCLOS (Percentage of Eye Closure) değerini hesaplar.
        
        PERCLOS, belirli bir süre içinde gözlerin kapalı olduğu zamanın yüzdesidir.
        
        Returns:
            float: Hesaplanan PERCLOS değeri [0-1]
        """
        if not self.ear_values:
            return 0.0
        
        # Eşik değerinin altında olan EAR değerlerini say (göz kapalı)
        num_closed = sum(1 for ear in self.ear_values if ear < self.ear_threshold)
        
        # PERCLOS hesapla
        perclos = num_closed / len(self.ear_values) if self.ear_values else 0.0
        
        return perclos
    
    def update(self, 
              ear_left: Optional[float] = None, 
              ear_right: Optional[float] = None) -> Dict:
        """
        Uykululuk durumunu günceller.
        
        Bu metod, göz verilerine dayanarak uykululuk seviyesini hesaplar.
        
        Args:
            ear_left: Sol göz EAR değeri
            ear_right: Sağ göz EAR değeri
            
        Returns:
            Dict: Hesaplanan uykululuk durumu bilgilerini içeren sözlük
        """
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Ortalama EAR hesapla
        if ear_left is not None and ear_right is not None:
            avg_ear = (ear_left + ear_right) / 2.0
        elif ear_left is not None:
            avg_ear = ear_left
        elif ear_right is not None:
            avg_ear = ear_right
        else:
            avg_ear = None
        
        # EAR ve PERCLOS güncelle
        if avg_ear is not None:
            self.ear_history.append(avg_ear)
            self.ear_values.append(avg_ear)
            self.perclos_value = self.calculate_perclos()
            
            # Göz kapalılık durumunu güncelle
            currently_closed = avg_ear < self.ear_threshold
            
            if currently_closed and not self.is_eyes_closed:
                # Gözler yeni kapandı
                self.is_eyes_closed = True
                self.eyes_closed_start_time = current_time
            elif not currently_closed and self.is_eyes_closed:
                # Gözler açıldı
                self.is_eyes_closed = False
                self.eyes_closed_start_time = None
        
        # Göz kapalı kalma süresini hesapla
        eyes_closed_duration = 0.0
        if self.is_eyes_closed and self.eyes_closed_start_time is not None:
            eyes_closed_duration = current_time - self.eyes_closed_start_time
        
        # Uykululuk seviyesini hesapla
        # Sadece göz tabanlı faktörlerle 
        drowsiness_score = 0.0
        factor_count = 0
        
        # PERCLOS faktörü (0-60 puan)
        if self.perclos_value > 0:
            perclos_score = min(60.0, (self.perclos_value / self.perclos_threshold) * 60.0)
            drowsiness_score += perclos_score
            factor_count += 1
        
        # Gözlerin kapalı kalma süresi faktörü (0-40 puan)
        if eyes_closed_duration > 0:
            closed_score = min(40.0, (eyes_closed_duration / self.ear_closed_duration) * 40.0)
            drowsiness_score += closed_score
            factor_count += 1
        
        # Toplam skoru normalize et (0-100 arasında)
        if factor_count > 0:
            normalized_score = drowsiness_score / factor_count * (100.0 / 60.0)  # 60: maksimum faktör puanı
            self.drowsiness_level = min(100.0, normalized_score) / 100.0
        else:
            self.drowsiness_level = 0.0
        
        # Uykululuk durumunu belirle
        if self.drowsiness_level < 0.3:
            self.drowsiness_state = "Uyanık"
        elif self.drowsiness_level < 0.6:
            self.drowsiness_state = "Yorgun"
        elif self.drowsiness_level < 0.8:
            self.drowsiness_state = "Uykulu"
        else:
            self.drowsiness_state = "Tehlikeli"
        
        # Sonuçları içeren sözlüğü oluştur
        result = {
            "drowsiness_level": self.drowsiness_level,
            "drowsiness_state": self.drowsiness_state,
            "perclos": self.perclos_value,
            "eyes_closed_duration": eyes_closed_duration,
            "is_eyes_closed": self.is_eyes_closed,
            "ear": avg_ear
        }
        
        return result
    
    def compute_ear_from_landmarks(self, 
                                  landmarks: List, 
                                  left_eye_indices: List[int], 
                                  right_eye_indices: List[int]) -> Tuple[float, float]:
        """
        Yüz işaret noktalarından sol ve sağ göz EAR değerlerini hesaplar.
        
        Args:
            landmarks: MediaPipe yüz işaret noktaları listesi
            left_eye_indices: Sol göz işaret noktalarının indeksleri
            right_eye_indices: Sağ göz işaret noktalarının indeksleri
            
        Returns:
            Tuple[float, float]: Sol ve sağ göz EAR değerleri
        """
        # Sol göz işaret noktalarını çıkar
        left_eye_landmarks = []
        for idx in left_eye_indices:
            left_eye_landmarks.append([landmarks[idx][0], landmarks[idx][1]])
        
        # Sağ göz işaret noktalarını çıkar
        right_eye_landmarks = []
        for idx in right_eye_indices:
            right_eye_landmarks.append([landmarks[idx][0], landmarks[idx][1]])
        
        # EAR değerlerini hesapla
        left_ear = self.calculate_ear(left_eye_landmarks)
        right_ear = self.calculate_ear(right_eye_landmarks)
        
        return left_ear, right_ear
    
    def visualize(self, 
                 frame: np.ndarray, 
                 ear_left: Optional[float] = None,
                 ear_right: Optional[float] = None,
                 show_metrics: bool = True) -> np.ndarray:
        """
        Uykululuk tespit sonuçlarını görselleştirir.
        
        Args:
            frame: Giriş görüntüsü
            ear_left: Sol göz EAR değeri
            ear_right: Sağ göz EAR değeri
            show_metrics: Metrik değerlerini göster
            
        Returns:
            np.ndarray: Görselleştirilmiş görüntü
        """
        vis_frame = frame.copy()
        h, w, _ = vis_frame.shape
        
        # Sadece tehlikeli durumlarda uyarı göster
        if self.drowsiness_state == "Tehlikeli":
            # Ekranda yanıp sönen çerçeve
            flash_alpha = 0.5 * (np.sin(time.time() * 10) + 1)  # 0-1 arası değer
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, flash_alpha * 0.3, vis_frame, 1 - flash_alpha * 0.3, 0, vis_frame)
        
        return vis_frame 