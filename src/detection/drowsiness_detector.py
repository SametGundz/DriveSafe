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
from typing import Dict, Tuple, List, Optional, Union
from collections import deque


class DrowsinessDetector:
    """
    Sürücü uykululuk durumunu tespit etmek için kullanılan sınıf.
    
    Bu sınıf, göz kapalılık oranı (EAR), PERCLOS, bakış yönü, baş duruşu
    ve diğer faktörleri kullanarak sürücünün uykululuk seviyesini belirler.
    
    Attributes:
        ear_threshold: Göz kapalı sayılması için EAR eşik değeri
        perclos_window_size: PERCLOS hesaplaması için pencere boyutu (kare sayısı)
        perclos_threshold: Uykululuk uyarısı için PERCLOS yüzdesi eşiği
        gaze_deviation_threshold: Dikkatsizlik için bakış sapma eşiği (derece)
        ear_closed_duration: Gözlerin kapalı kalma süresi eşiği (saniye)
        history_duration: Geçmiş verileri saklama süresi (saniye)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        DrowsinessDetector sınıfını başlatır.
        
        Args:
            config: Konfigürasyon değerleri içeren sözlük (opsiyonel)
        """
        # Varsayılan parametreler
        self.ear_threshold = 0.21
        self.perclos_window_size = 150  # ~5 saniye (30 FPS için)
        self.perclos_threshold = 15.0   # %15
        self.gaze_deviation_threshold = 20.0  # derece
        self.ear_closed_duration = 2.0  # saniye
        self.history_duration = 5.0     # saniye
        
        # Konfigürasyon sözlüğünden parametreleri yükle
        if config and isinstance(config, dict):
            if 'detection' in config:
                det_config = config['detection']
                if 'ear_threshold' in det_config:
                    self.ear_threshold = float(det_config['ear_threshold'])
                if 'perclos' in det_config:
                    perclos_config = det_config['perclos']
                    if 'window_size' in perclos_config:
                        self.perclos_window_size = int(perclos_config['window_size'])
                    if 'threshold' in perclos_config:
                        self.perclos_threshold = float(perclos_config['threshold'])
                if 'gaze' in det_config and 'deviation_threshold' in det_config['gaze']:
                    self.gaze_deviation_threshold = float(det_config['gaze']['deviation_threshold'])
                
            if 'drowsiness' in config:
                drow_config = config['drowsiness']
                if 'ear_closed_duration' in drow_config:
                    self.ear_closed_duration = float(drow_config['ear_closed_duration'])
                if 'history_duration' in drow_config:
                    self.history_duration = float(drow_config['history_duration'])
        
        # Geçmiş veri kuyrukları
        self.ear_history = deque(maxlen=int(self.history_duration * 30))  # 30fps varsayarak
        self.gaze_history = deque(maxlen=int(self.history_duration * 30))
        self.head_pose_history = deque(maxlen=int(self.history_duration * 30))
        
        # Göz kapalılık zaman takibi
        self.eyes_closed_start_time = None
        self.last_update_time = time.time()
        
        # PERCLOS için son N kare
        self.ear_values = deque(maxlen=self.perclos_window_size)
        
        # Uykululuk durumu
        self.drowsiness_level = 0.0  # 0.0 (uyanık) ile 1.0 (uykulu) arasında
        self.is_eyes_closed = False
        self.perclos_value = 0.0
        self.drowsiness_state = "Uyanık"  # Uyanık, Yorgun, Uykulu, Tehlikeli
    
    def calculate_ear(self, eye_landmarks: List[List[float]]) -> float:
        """
        Göz Açıklık Oranını (EAR) hesaplar.
        
        EAR, göz işaret noktaları kullanılarak gözün ne kadar açık olduğunu ölçen bir orandır.
        
        Args:
            eye_landmarks: Göz işaret noktalarının koordinatları
            
        Returns:
            float: Hesaplanan EAR değeri
        """
        # Göz en az 6 nokta içermelidir
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Dikey mesafeleri hesapla (üst ve alt göz kapağı arasındaki mesafeler)
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        v3 = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # Yatay mesafeyi hesapla (göz köşeleri arasındaki mesafe)
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # Sıfıra bölme hatasını önle
        if h == 0:
            return 0.0
        
        # EAR hesapla
        ear = (v1 + v2 + v3) / (3.0 * h)
        
        return ear
    
    def calculate_perclos(self) -> float:
        """
        PERCLOS (Percentage of Eye Closure) değerini hesaplar.
        
        PERCLOS, belirli bir zaman aralığında gözlerin kapalı olduğu sürenin yüzdesidir.
        
        Returns:
            float: PERCLOS değeri (yüzde olarak)
        """
        if not self.ear_values:
            return 0.0
        
        # Kapalı göz sayısı
        closed_eyes_count = sum(1 for ear in self.ear_values if ear < self.ear_threshold)
        
        # PERCLOS hesapla
        perclos = (closed_eyes_count / len(self.ear_values)) * 100.0
        
        return perclos
    
    def calculate_gaze_deviation(self, gaze_vector: np.ndarray) -> float:
        """
        Bakış sapma açısını hesaplar.
        
        İdeal sürüş pozisyonuna göre bakışın ne kadar saptığını ölçer.
        
        Args:
            gaze_vector: 3D bakış yönü vektörü [x, y, z]
            
        Returns:
            float: Bakış sapma açısı (derece)
        """
        if gaze_vector is None:
            return 0.0
        
        # İdeal sürüş bakışı (ileri)
        ideal_gaze = np.array([0.0, 0.0, -1.0])
        
        # İki vektör arasındaki açıyı hesapla
        dot_product = np.dot(gaze_vector, ideal_gaze)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Sınırla
        
        # Radyandan dereceye dönüştür
        angle = np.arccos(dot_product) * 180.0 / np.pi
        
        return angle
    
    def calculate_head_deviation(self, head_rotation: np.ndarray) -> float:
        """
        Baş duruşu sapma açısını hesaplar.
        
        Args:
            head_rotation: Baş rotasyon vektörü
            
        Returns:
            float: Baş sapma açısı (derece)
        """
        if head_rotation is None:
            return 0.0
        
        # Baş rotasyon vektöründen açıları çıkar (radyan)
        pitch, yaw, roll = head_rotation.flatten()
        
        # Mutlak değerleri al ve dereceye dönüştür
        pitch_deg = abs(pitch * 180.0 / np.pi)
        yaw_deg = abs(yaw * 180.0 / np.pi)
        roll_deg = abs(roll * 180.0 / np.pi)
        
        # Maksimum sapmayı al
        max_deviation = max(pitch_deg, yaw_deg, roll_deg)
        
        return max_deviation
    
    def update(self, 
              ear_left: Optional[float] = None, 
              ear_right: Optional[float] = None, 
              gaze_vector: Optional[np.ndarray] = None, 
              head_rotation: Optional[np.ndarray] = None) -> Dict:
        """
        Uykululuk durumunu günceller.
        
        Bu metod, yeni göz/bakış/baş duruşu verileriyle uykululuk seviyesini hesaplar.
        
        Args:
            ear_left: Sol göz EAR değeri
            ear_right: Sağ göz EAR değeri
            gaze_vector: Bakış yönü vektörü
            head_rotation: Baş rotasyon vektörü
            
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
        
        # Bakış sapmasını hesapla
        gaze_deviation = 0.0
        if gaze_vector is not None:
            gaze_deviation = self.calculate_gaze_deviation(gaze_vector)
            self.gaze_history.append(gaze_deviation)
        
        # Baş sapmasını hesapla
        head_deviation = 0.0
        if head_rotation is not None:
            head_deviation = self.calculate_head_deviation(head_rotation)
            self.head_pose_history.append(head_deviation)
        
        # Uykululuk seviyesini hesapla
        # Her bir faktöre belirli ağırlıklar atanır ve toplam puan hesaplanır
        drowsiness_score = 0.0
        factor_count = 0
        
        # PERCLOS faktörü (0-40 puan)
        if self.perclos_value > 0:
            perclos_score = min(40.0, (self.perclos_value / self.perclos_threshold) * 40.0)
            drowsiness_score += perclos_score
            factor_count += 1
        
        # Gözlerin kapalı kalma süresi faktörü (0-30 puan)
        if eyes_closed_duration > 0:
            closed_score = min(30.0, (eyes_closed_duration / self.ear_closed_duration) * 30.0)
            drowsiness_score += closed_score
            factor_count += 1
        
        # Bakış sapması faktörü (0-15 puan)
        if gaze_deviation > 0:
            gaze_score = min(15.0, (gaze_deviation / self.gaze_deviation_threshold) * 15.0)
            drowsiness_score += gaze_score
            factor_count += 1
        
        # Baş sapması faktörü (0-15 puan)
        if head_deviation > 0:
            head_score = min(15.0, (head_deviation / 45.0) * 15.0)  # 45 derece maks sapma
            drowsiness_score += head_score
            factor_count += 1
        
        # Toplam skoru normalize et (0-100 arasında)
        if factor_count > 0:
            normalized_score = drowsiness_score / factor_count * (100.0 / 40.0)  # 40: maksimum faktör puanı
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
            "ear": avg_ear,
            "gaze_deviation": gaze_deviation,
            "head_deviation": head_deviation
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
        
        # Durum çubuğu arka planı
        cv2.rectangle(vis_frame, (0, 0), (w, 30), (0, 0, 0), -1)
        
        # Uykululuk seviyesi gösterge çubuğu
        bar_width = int(w * 0.6)
        bar_height = 20
        bar_x = int(w * 0.38)
        bar_y = 5
        
        # Arka plan çubuğu
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # Doldurma çubuğu
        filled_width = int(bar_width * self.drowsiness_level)
        
        # Renk kodunu belirle (yeşil -> sarı -> kırmızı)
        if self.drowsiness_level < 0.3:
            color = (0, 255, 0)  # Yeşil
        elif self.drowsiness_level < 0.6:
            color = (0, 255, 255)  # Sarı
        elif self.drowsiness_level < 0.8:
            color = (0, 165, 255)  # Turuncu
        else:
            color = (0, 0, 255)  # Kırmızı
        
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), color, -1)
        
        # Durum metni
        cv2.putText(vis_frame, f"Durum: {self.drowsiness_state}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # PERCLOS değeri
        cv2.putText(vis_frame, f"PERCLOS: {self.perclos_value:.1f}%", (bar_x + bar_width + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Ek metrikler
        if show_metrics:
            y_offset = 60
            
            # EAR değerleri
            if ear_left is not None:
                cv2.putText(vis_frame, f"Sol Göz EAR: {ear_left:.2f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
            
            if ear_right is not None:
                cv2.putText(vis_frame, f"Sağ Göz EAR: {ear_right:.2f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
            
            # Göz kapalı kalma süresi
            if self.is_eyes_closed and self.eyes_closed_start_time is not None:
                closed_duration = time.time() - self.eyes_closed_start_time
                cv2.putText(vis_frame, f"Göz Kapalı: {closed_duration:.1f}s", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(vis_frame, "Göz Açık", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Uykululuk uyarısı
        if self.drowsiness_state == "Tehlikeli":
            # Ekranda yanıp sönen çerçeve
            flash_alpha = 0.5 * (np.sin(time.time() * 10) + 1)  # 0-1 arası değer
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, flash_alpha * 0.3, vis_frame, 1 - flash_alpha * 0.3, 0, vis_frame)
            
            # Uyarı metni
            cv2.putText(vis_frame, "UYARI! SÜRÜCÜ UYKULU", (int(w/2) - 200, int(h/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        elif self.drowsiness_state == "Uykulu":
            # Hafif uyarı
            cv2.putText(vis_frame, "Dikkat! Uykululuk Belirtisi", (int(w/2) - 180, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        return vis_frame 