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
    
    Bölge tanımları:
    - 0: Right Side - sağ ekstrem
    - 1: Right Top - sağ üst açı
    - 2: Right Windshield - sağda, yukarıda
    - 3: Steering Wheel - merkezde, aşağıda
    - 4: Rear Mirror - orta üstte
    - 5: Center Console - orta konsol
    - 6: Left Windshield - solda, yukarıda
    - 7: Left Bottom - sol alt
    - 8: Left Side - sol ekstrem
    """
    
    # Bölge tanımları - (yaw_min, yaw_max, pitch_min, pitch_max)
    # Açı değerleri derece cinsindendir
    ZONES = {
        0: (35, 90, -20, 20),     # Right Side - sağ ekstrem
        1: (20, 40, 5, 45),       # Right Top - sağ üst açı
        2: (0, 30, -5, 25),       # Right Windshield - sağda, yukarıda
        3: (-15, 15, -30, 0),     # Steering Wheel - merkezde, aşağıda
        4: (-10, 10, 0, 30),      # Rear Mirror - orta üstte
        5: (-20, 0, -25, 5),      # Center Console - orta konsol
        6: (-30, 0, 0, 25),       # Left Windshield - solda, yukarıda
        7: (-40, -15, -25, -5),   # Left Bottom - sol alt
        8: (-90, -35, -20, 20),   # Left Side - sol ekstrem
    }
    
    # Bölge adları
    ZONE_NAMES = [
        "Right Side",      # 0
        "Right Top",       # 1
        "Right Windshield",# 2
        "Steering Wheel",  # 3
        "Rear Mirror",     # 4
        "Center Console",  # 5
        "Left Windshield", # 6
        "Left Bottom",     # 7
        "Left Side"        # 8
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
                self.zone_durations[self.current_zone] += duration
                logger.debug(f"Zone change: {self.current_zone} -> {stable_zone}, "
                            f"duration: {duration:.2f}s")
            
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
        logger.debug(f"Gaze angles - Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
        
        for zone_id, (yaw_min, yaw_max, pitch_min, pitch_max) in self.ZONES.items():
            if yaw_min <= yaw <= yaw_max and pitch_min <= pitch <= pitch_max:
                return zone_id
        
        # Tanımlı bölgelerin dışında
        logger.debug(f"No matching zone for Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
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
        # Eğer hala bir bölgeye bakılıyorsa mevcut süreyi de ekle
        current_stats = self.zone_durations.copy()
        if self.current_zone is not None:
            current_duration = time.time() - self.zone_start_time
            current_stats[self.current_zone] += current_duration
        
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