#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bakış bölgesi istatistikleri modülü.

Bu modül, sürücünün araç içindeki farklı bölgelere ne kadar baktığına dair
istatistikleri toplayan ve raporlayan fonksiyonları içerir.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
from typing import Dict, List, Optional, Tuple

from src.detection.gaze_zone_detector import get_gaze_zone_detector, GazeZoneDetector


class GazeStatisticsRecorder:
    """
    Sürücünün bakış bölgesi istatistiklerini kaydeden ve raporlayan sınıf.
    """
    
    def __init__(self, session_name: Optional[str] = None):
        """
        GazeStatisticsRecorder sınıfını başlat.
        
        Args:
            session_name: Oturum adı (None ise otomatik oluşturulur)
        """
        # Oturum adı
        if session_name is None:
            self.session_name = datetime.now().strftime("gaze_session_%Y%m%d_%H%M%S")
        else:
            self.session_name = session_name
            
        # Zaman damgası ve bölge ID'si kayıtları
        self.timestamps = []
        self.zone_ids = []
        
        # Başlangıç zamanı
        self.start_time = time.time()
        
        # Zone detector
        self.zone_detector = get_gaze_zone_detector()
    
    def record_gaze_zone(self, zone_id: Optional[int]):
        """
        Bakış bölgesini kaydet.
        
        Args:
            zone_id: Bölge ID'si veya None (tanımlı bir bölgeye bakmıyorsa)
        """
        current_time = time.time()
        self.timestamps.append(current_time)
        self.zone_ids.append(zone_id)
    
    def get_zone_durations(self) -> Dict[int, float]:
        """
        Her bölgede geçirilen toplam süreyi hesapla.
        
        Returns:
            Dict[int, float]: Bölge ID'leri ve süreleri (saniye)
        """
        return self.zone_detector.get_zone_statistics()
    
    def get_zone_percentages(self) -> Dict[int, float]:
        """
        Her bölgede geçirilen sürenin yüzdesini hesapla.
        
        Returns:
            Dict[int, float]: Bölge ID'leri ve yüzdeleri
        """
        durations = self.get_zone_durations()
        total_duration = sum(durations.values())
        
        if total_duration == 0:
            return {zone_id: 0.0 for zone_id in durations}
        
        return {zone_id: (duration / total_duration) * 100.0 
                for zone_id, duration in durations.items()}
    
    def save_statistics(self, output_dir: str = "recordings/gaze_stats") -> str:
        """
        İstatistikleri dosyaya kaydet.
        
        Args:
            output_dir: Çıktı dizini
            
        Returns:
            str: Kaydedilen dosyanın yolu
        """
        # Dizini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV dosya yolu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(output_dir, f"{self.session_name}_{timestamp}.csv")
        
        # Bölge süreleri
        durations = self.get_zone_durations()
        percentages = self.get_zone_percentages()
        
        # CSV dosyasına yaz
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Zone ID", "Zone Name", "Duration (s)", "Percentage (%)"])
            
            for zone_id in sorted(durations.keys()):
                zone_name = self.zone_detector.get_zone_name(zone_id)
                writer.writerow([
                    zone_id,
                    zone_name,
                    round(durations[zone_id], 2),
                    round(percentages[zone_id], 2)
                ])
        
        return csv_file
    
    def generate_pie_chart(self, output_dir: str = "recordings/gaze_stats") -> str:
        """
        Bakış bölgelerinin dağılımını gösteren pasta grafik oluştur.
        
        Args:
            output_dir: Çıktı dizini
            
        Returns:
            str: Kaydedilen dosyanın yolu
        """
        # Dizini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Görüntü dosya yolu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = os.path.join(output_dir, f"{self.session_name}_{timestamp}_pie.png")
        
        # Pasta grafik verilerini hazırla
        percentages = self.get_zone_percentages()
        labels = [f"{self.zone_detector.get_zone_name(zone_id)} ({zone_id}): {pct:.1f}%" 
                 for zone_id, pct in percentages.items() if pct > 0]
        sizes = [pct for pct in percentages.values() if pct > 0]
        
        # Pasta grafik oluştur
        plt.figure(figsize=(10, 7))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Dairesel grafik için
        plt.title(f"Bakış Bölgesi Dağılımı - {self.session_name}")
        
        # Kaydet
        plt.savefig(img_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return img_file
    
    def generate_report(self, output_dir: str = "recordings/gaze_stats") -> Dict[str, str]:
        """
        İstatistik raporu oluştur ve kaydet.
        
        Args:
            output_dir: Çıktı dizini
            
        Returns:
            Dict[str, str]: Kaydedilen dosyaların yolları
        """
        # CSV ve grafik dosyalarını oluştur
        csv_file = self.save_statistics(output_dir)
        pie_file = self.generate_pie_chart(output_dir)
        
        return {
            "csv": csv_file,
            "pie_chart": pie_file
        }


# Singleton pattern için global instance
_gaze_statistics_recorder = None

def get_gaze_statistics_recorder(session_name: Optional[str] = None) -> GazeStatisticsRecorder:
    """
    GazeStatisticsRecorder instance'ı döndürür (singleton pattern).
    
    Args:
        session_name: Oturum adı
        
    Returns:
        GazeStatisticsRecorder: İstatistik kaydedici instance'ı
    """
    global _gaze_statistics_recorder
    if _gaze_statistics_recorder is None:
        _gaze_statistics_recorder = GazeStatisticsRecorder(session_name)
    return _gaze_statistics_recorder 