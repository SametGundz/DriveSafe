#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bakış bölgesi (gaze zone) tespiti modülü.

Bu modül, sürücünün bakış yönünü kullanarak araç içindeki hangi bölgeye 
baktığını tespit etmeye yarayan GazeZoneDetector sınıfını içerir.
"""

import numpy as np
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Logger oluşturma
logger = logging.getLogger(__name__)

class GazeZoneDetector:
    """
    Bakış yönünü araç içi bölgelerle eşleştiren sınıf.
    
    Bu sınıf, gaze detector'dan alınan bakış açılarını kullanarak
    sürücünün araç içindeki hangi bölgeye baktığını tespit eder.
    
    AB regülasyonu C(2023)4523'e uygun şekilde bölge tanımları:
    - 0: Road Center - Yol merkezi (Alan 2)
    - 1: Driving Instruments - Gösterge paneli (Alan 2)
    - 2: Infotainment - Eğlence sistemi (Alan 1)
    - 3: Left Side - Sol yan cam ve ayna (Alan 2)
    - 4: Right Side - Sağ yan cam ve ayna (Alan 2)
    - 5: Rear Mirror - Dikiz aynası (Alan 2)
    """
    
    # Bölge tanımları - (yaw_min, yaw_max, pitch_min, pitch_max)
    # Açı değerleri derece cinsindendir, GazeDurationMonitor sınıfıyla uyumlu
    ZONES = {
        0: (-15, 25, -15, 5),     # Road Center - Yol merkezi
        1: (-15, 15, -70, -10),    # Driving Instruments - Gösterge paneli 
        2: (15, 45, -70, -10),    # Infotainment - Eğlence sistemi (Yaw değeri genişletildi: -15/15 -> 15/45)
        3: (-90, -15, -40, 40),    # Sol yan cam ve ayna (Artık negatif yaw değerleri - sol taraf)
        4: (40, 90, -20, 40),      # Sağ yan cam ve ayna (Artık pozitif yaw değerleri - sağ taraf)
        5: (-15, 40, 5, 40)       # Rear Mirror - Dikiz aynası (Sağa doğru genişletildi)
    }
    
    # Bölge adları - GazeDurationMonitor sınıfıyla uyumlu
    ZONE_NAMES = [
        "Road Center",         # 0
        "Driving Instruments", # 1
        "Infotainment",        # 2
        "Left Side",           # 3
        "Right Side",          # 4
        "Rear Mirror"          # 5
    ]
    
    def __init__(self, history_size: int = 10, stability_threshold: float = 0.5):
        """
        GazeZoneDetector'ı başlat.
        
        Args:
            history_size: Bakış geçmişi uzunluğu
            stability_threshold: Bakış stabilitesi için eşik değeri (0-1)
        """
        self.history_size = history_size
        self.stability_threshold = stability_threshold
        self.zone_history = []  # Son N bakılan bölge
        self.current_zone = None
        self.zone_start_time = 0
        self.zone_durations = {i: 0 for i in range(len(self.ZONES))}  # Her bölgeye bakış süresi
        self.last_update_time = time.time()
        
        logger.info(f"GazeZoneDetector initialized with history_size={history_size}, " 
                   f"stability_threshold={stability_threshold}")
    
    def update(self, pitch: float, yaw: float, timestamp: Optional[float] = None) -> int:
        """
        Bakış açılarını kullanarak bölge tespiti yapar.
        
        Args:
            pitch: Dikey bakış açısı (derece)
            yaw: Yatay bakış açısı (derece)
            timestamp: Zaman damgası (None ise mevcut zaman kullanılır)
            
        Returns:
            int: Tespit edilen bölge ID'si veya None
        """
        if timestamp is None:
            timestamp = time.time()
        
        # NaN değerleri filtrele
        if np.isnan(pitch) or np.isnan(yaw):
            return self.current_zone
        
        # Açıları kullanarak bölgeyi tespit et
        detected_zone = self._get_zone_from_angles(pitch, yaw)
        
        # Bölge geçmişini güncelle
        self.zone_history.append(detected_zone)
        if len(self.zone_history) > self.history_size:
            self.zone_history.pop(0)
        
        # Stabil bölge tespiti
        stable_zone = self._get_stable_zone()
        
        # Bölge değişimi kontrolü
        if stable_zone != self.current_zone:
            # Önceki bölgede geçen süreyi kaydet
            if self.current_zone is not None:
                duration = timestamp - self.zone_start_time
                if duration > 0:  # Negatif süreler oluşmasını önle
                    self.zone_durations[self.current_zone] += duration
                    logger.debug(f"Zone change: {self.get_zone_name(self.current_zone)} -> {self.get_zone_name(stable_zone)}, "
                               f"duration: {duration:.2f}s")
                    logger.debug(f"Updated zone durations: {self.zone_durations}")
            
            self.current_zone = stable_zone
            self.zone_start_time = timestamp
        
        self.last_update_time = timestamp
        return self.current_zone
    
    def _get_zone_from_angles(self, pitch: float, yaw: float) -> Optional[int]:
        """
        Açıları kullanarak hangi bölgeye bakıldığını hesaplar.
        
        Args:
            pitch: Dikey bakış açısı (derece)
            yaw: Yatay bakış açısı (derece)
            
        Returns:
            Optional[int]: Bölge ID'si veya None
        """
        # Açıları konsola yazdır (debug için)
        logger.info(f"Gaze angles - Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
        
        # AÇILARIN RANGE DIŞINA ÇIKTIĞI DURUMU TEST ET
        if pitch < -70 or pitch > 40 or yaw < -90 or yaw > 90:
            logger.warning(f"Angles out of expected range - Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
        
        # Road Center (zone 0): -15 <= yaw <= 25, -15 <= pitch <= 5
        if -15 <= yaw <= 25 and -15 <= pitch <= 5:
            logger.debug("Matching Zone 0: Road Center")
            return 0
            
        # Driving Instruments (zone 1): -15 <= yaw <= 15, -70 <= pitch < -10
        if -15 <= yaw <= 15 and -70 <= pitch < -10:
            logger.debug("Matching Zone 1: Driving Instruments")
            return 1
        
        # Infotainment (zone 2): 15 <= yaw <= 45, -70 <= pitch < -10
        if 15 <= yaw <= 45 and -70 <= pitch < -10:
            logger.debug("Matching Zone 2: Infotainment")
            return 2
            
        # Left Side (zone 3): -90 <= yaw < -15, -40 <= pitch <= 40
        if -90 <= yaw < -15 and -40 <= pitch <= 40:
            logger.debug("Matching Zone 3: Left Side")
            return 3
            
        # Right Side (zone 4): 40 <= yaw <= 90, -20 <= pitch <= 40
        if 40 <= yaw <= 90 and -20 <= pitch <= 40:
            logger.debug("Matching Zone 4: Right Side")
            return 4
            
        # Rear Mirror (zone 5): -15 <= yaw <= 40, 5 < pitch <= 40
        if -15 <= yaw <= 40 and 5 < pitch <= 40:
            logger.debug("Matching Zone 5: Rear Mirror")
            return 5
        
        # Tanımlı bölgelerin dışında
        logger.warning(f"No matching zone for Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
        return None
    
    def _get_stable_zone(self) -> Optional[int]:
        """
        Geçmiş veriye bakarak stabil bölgeyi belirler.
        
        Returns:
            Optional[int]: Stabil bölge ID'si veya en çok tekrar eden bölge
        """
        if not self.zone_history:
            return None
            
        # Geçmişteki None olmayan bölgeleri filtrele
        valid_zones = [zone for zone in self.zone_history if zone is not None]
        
        # Eğer hiç geçerli bölge yoksa None döndür
        if not valid_zones:
            return None
            
        # En çok tekrar eden bölgeyi bul
        zone_counts = Counter(valid_zones)
        most_common = zone_counts.most_common(1)
        
        # En az %50 stabilite sağlandıysa güvenilir kabul et
        if most_common and most_common[0][1] >= len(valid_zones) * self.stability_threshold:
            return most_common[0][0]
        # Eğer stabilite eşiği aşılmadıysa, yine de en çok tekrar eden bölgeyi döndür
        elif most_common:
            return most_common[0][0]
        # Hiç geçerli bölge yoksa None döndür
        else:
            return None
    
    def get_zone_statistics(self) -> Dict[int, float]:
        """
        Her bölgede geçirilen süre istatistiklerini döndürür.
        
        Returns:
            Dict[int, float]: Bölge ID'leri ve süreleri (saniye)
        """
        # Mevcut durum için süreleri kopyala
        current_stats = self.zone_durations.copy()
        
        # Eğer hala bir bölgeye bakılıyorsa, o bölgedeki son süreyi de ekle
        if self.current_zone is not None:
            current_time = time.time()
            current_duration = current_time - self.zone_start_time
            if current_duration > 0:  # Negatif değer olmadığından emin ol
                current_stats[self.current_zone] += current_duration
        
        # Sonuçları logla
        total_time = sum(current_stats.values())
        logger.info(f"Total gaze time: {total_time:.2f}s, Zone statistics: {current_stats}")
        
        return current_stats
    
    def get_zone_name(self, zone_id: Optional[int]) -> str:
        """
        Bölge ID'sine karşılık gelen ismi döndürür.
        
        Args:
            zone_id: Bölge ID'si
            
        Returns:
            str: Bölge adı veya "Unknown"
        """
        if zone_id is not None and 0 <= zone_id < len(self.ZONE_NAMES):
            return self.ZONE_NAMES[zone_id]
        return "Unknown"
    
    def reset(self):
        """Tüm geçmiş verileri ve istatistikleri sıfırlar."""
        self.zone_history = []
        self.current_zone = None
        self.zone_start_time = time.time()
        self.zone_durations = {i: 0 for i in range(len(self.ZONES))}
        logger.info("GazeZoneDetector reset")

    def get_gaze_target_zone(self, gaze_vector: np.ndarray) -> Optional[int]:
        """
        Bakış vektörünü kullanarak hedef bölgeyi belirler.
        
        Args:
            gaze_vector: Bakış vektörü [pitch, yaw]
            
        Returns:
            Optional[int]: Bölge ID'si veya None
        """
        if gaze_vector is None or len(gaze_vector) < 2:
            return None
        
        # Radyan açıları dereceye çevir
        pitch, yaw = np.rad2deg(gaze_vector)
        
        # Açıları kullanarak bölgeyi tespit et
        zone_id = self._get_zone_from_angles(pitch, yaw)
        
        # Basitçe bölge değişimini takip et ve süreyi kaydet
        current_time = time.time()
        
        # Eğer yeni bir bölgeye bakılmaya başlandıysa
        if zone_id != self.current_zone:
            # Önceki bölgede geçirilen süreyi kaydet
            if self.current_zone is not None:
                duration = current_time - self.zone_start_time
                if duration > 0:  # Negatif süre olmadığından emin ol
                    self.zone_durations[self.current_zone] += duration
                    logger.info(f"Zone change: {self.get_zone_name(self.current_zone)} -> {self.get_zone_name(zone_id)}, duration: {duration:.2f}s")
            
            # Yeni bölge için zamanı sıfırla
            self.current_zone = zone_id
            self.zone_start_time = current_time
        
        return zone_id


# Singleton pattern için global instance
_gaze_zone_detector = None

def get_gaze_zone_detector(history_size: int = 10, 
                         stability_threshold: float = 0.5) -> GazeZoneDetector:
    """
    GazeZoneDetector instance'ı döndürür (singleton pattern).
    
    Args:
        history_size: Bakış geçmişi uzunluğu
        stability_threshold: Bakış stabilitesi için eşik değeri
        
    Returns:
        GazeZoneDetector: Detector instance'ı
    """
    global _gaze_zone_detector
    if _gaze_zone_detector is None:
        _gaze_zone_detector = GazeZoneDetector(
            history_size=history_size,
            stability_threshold=stability_threshold
        )
    return _gaze_zone_detector 