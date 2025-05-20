#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bakış bölgesi istatistikleri modülü.

Bu modül, sürücünün araç içindeki farklı bölgelere ne kadar baktığına dair
istatistikleri toplayan ve raporlayan fonksiyonları içerir.
"""

import time
import numpy as np
from datetime import datetime
import os
import csv
import json
from typing import Dict, List, Optional, Tuple, Any
import logging

from src.detection.gaze_zone_detector import get_gaze_zone_detector, GazeZoneDetector
# Rapor üretme modülünü ekle
from src.utils.report_generator import generate_gaze_statistics_report

# Logger tanımı
logger = logging.getLogger(__name__)

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
        
        # Visit count'u güncelle
        self.zone_visit_counts = {}
        
        # Son için maksimum kayıt sayısı
        self.max_data_points = 1000  # You can adjust this value as needed
    
    def record_gaze_zone(self, zone_id: Optional[int]):
        """
        Mevcut bakış bölgesini kaydeder.
        
        Args:
            zone_id: Bakış bölgesi ID'si
        """
        # Zaman damgasını kaydet
        current_time = time.time()
        
        # İlk kayıtsa başlangıç zamanını güncelle
        if not self.timestamps:
            self.start_time = current_time
            
        # zone_id'nin GazeZoneDetector.ZONES'da tanımlı olduğunu kontrol et
        if zone_id is not None and zone_id not in self.zone_detector.ZONES.keys():
            logger.warning(f"Invalid zone_id: {zone_id}. Converting to None.")
            zone_id = None
        
        # Timestamp ve zone_id'yi listelere ekle
        self.timestamps.append(current_time)
        self.zone_ids.append(zone_id)
        
        # Visit count'u güncelle
        if zone_id is not None:
            if zone_id not in self.zone_visit_counts:
                self.zone_visit_counts[zone_id] = 0
            
            # Eğer önceki bölgeden farklıysa visit count'u artır
            if not self.zone_ids[:-1] or self.zone_ids[-2] != zone_id:
                self.zone_visit_counts[zone_id] += 1
                logger.info(f"Entered zone {self.zone_detector.get_zone_name(zone_id)} (ID: {zone_id}), visit count: {self.zone_visit_counts[zone_id]}")
        
        # Minimum boyutları kontrol et, belirli aralıklarla loglama
        if len(self.timestamps) % 100 == 0:  # 500 yerine 100 yaparak daha sık log alalım
            logger.info(f"Recorded {len(self.timestamps)} gaze points, {sum(self.zone_visit_counts.values())} zone transitions")
            
            # Aktif olarak duration'ları ve istatistikleri logla
            if len(self.timestamps) >= 2:
                durations = self.get_zone_durations()
                total = sum(durations.values())
                percentages = {zone: (dur/total*100 if total > 0 else 0) for zone, dur in durations.items()}
                
                logger.info(f"Current durations: {durations}")
                logger.info(f"Current percentages: {percentages}")
                logger.info(f"Visit counts: {self.zone_visit_counts}")
                
                # Zone ID'lerin kaç kez kaydedildiğini logla
                zone_id_counts = {}
                for zid in self.zone_ids:
                    if zid is not None:
                        zone_id_counts[zid] = zone_id_counts.get(zid, 0) + 1
                logger.info(f"Zone ID occurrences: {zone_id_counts}")
                
        # Son için maksimum kayıt sayısı
        if len(self.timestamps) > self.max_data_points:
            # En eski kayıtları sil
            self.timestamps = self.timestamps[-self.max_data_points:]
            self.zone_ids = self.zone_ids[-self.max_data_points:]
    
    def get_zone_durations(self) -> Dict[int, float]:
        """
        Her bölgede geçirilen toplam süreyi hesapla.
        
        Returns:
            Dict[int, float]: Bölge ID'leri ve süreleri (saniye)
        """
        # Doğrudan zone detector'dan süreleri al
        detector_durations = self.zone_detector.get_zone_statistics()
        detector_total = sum(detector_durations.values())
        
        logger.debug(f"Detector zone durations: {detector_durations} (total: {detector_total:.2f}s)")
        
        # Manuel hesaplama ile süreleri hesapla
        manual_durations = {zone_id: 0.0 for zone_id in self.zone_detector.ZONES.keys()}
        
        if len(self.timestamps) > 1:
            for i in range(1, len(self.timestamps)):
                prev_zone = self.zone_ids[i-1]
                prev_time = self.timestamps[i-1]
                curr_time = self.timestamps[i]
                
                # Eğer geçerli bir bölge ise süreyi hesapla
                if prev_zone is not None and prev_zone in manual_durations:
                    duration = curr_time - prev_time
                    if duration > 0:  # Negatif değer kontrolü
                        manual_durations[prev_zone] += duration
                        
            # Son kayıt için şu anki zaman ile hesapla
            if len(self.zone_ids) > 0 and self.zone_ids[-1] is not None:
                last_zone = self.zone_ids[-1]
                if last_zone in manual_durations:
                    last_time = self.timestamps[-1]
                    current_time = time.time()
                    
                    duration = current_time - last_time
                    if duration > 0:  # Negatif değer kontrolü
                        manual_durations[last_zone] += duration
        
        manual_total = sum(manual_durations.values())
        logger.debug(f"Manual zone durations: {manual_durations} (total: {manual_total:.2f}s)")
        
        # Detector ve manual hesaplamayı karşılaştır
        if detector_total < 0.001 or manual_total > detector_total:
            logger.info("Using manually calculated durations as they seem more complete")
            return manual_durations
        else:
            logger.info("Using detector-based durations")
            return detector_durations
    
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Kaydedilen bakış verilerinden istatistik hesapla.
        
        Returns:
            Dict[str, Any]: Bölge istatistikleri içeren sözlük.
        """
        stats = {}
        
        # Zone tanımlarını GazeZoneDetector ile aynı kullan (0-5 arası ID'ler)
        zone_names = {
            0: "Road Center",         # Yol merkezi (Alan 2)
            1: "Driving Instruments", # Gösterge paneli (Alan 2)
            2: "Infotainment",        # Bilgi-eğlence ekranı (Alan 1)
            3: "Left Side",           # Sol yan cam ve ayna (Alan 2)
            4: "Right Side",          # Sağ yan cam ve ayna (Alan 2)
            5: "Rear Mirror"          # Dikiz aynası (Alan 2)
        }
        
        # Durations'ı detector'dan al (veya veri yoksa hesapla)
        durations = self.zone_detector.get_zone_statistics()
        
        # Durations boş mu kontrol et ve gerekirse manuel hesapla
        if sum(durations.values()) < 0.001 and len(self.timestamps) > 1:
            # Manuel hesaplama
            durations = {zone_id: 0.0 for zone_id in zone_names.keys()}
            try:
                for i in range(1, len(self.timestamps)):
                    prev_zone = self.zone_ids[i-1]
                    prev_time = self.timestamps[i-1]
                    curr_time = self.timestamps[i]
                    
                    if prev_zone is not None:
                        duration = curr_time - prev_time
                        if duration > 0:
                            durations[prev_zone] += duration
                
                # Son kayıt için şu anki zamanı kullan
                if len(self.zone_ids) > 0 and self.zone_ids[-1] is not None:
                    last_zone = self.zone_ids[-1]
                    last_time = self.timestamps[-1]
                    current_time = time.time()
                    
                    duration = current_time - last_time
                    if duration > 0:
                        durations[last_zone] += duration
                        
                logger.info(f"Manually calculated durations: {durations}")
            except Exception as e:
                logger.error(f"Error calculating manual durations: {str(e)}")
        
        # Bölge ziyaret sayılarını ekle
        visit_counts = self.zone_visit_counts.copy()
        # Eksik bölgeler için 0 değerini ata
        for zone_id in zone_names.keys():
            if zone_id not in visit_counts:
                visit_counts[zone_id] = 0
        
        # Yüzdeleri hesapla
        total_duration = sum(durations.values())
        if total_duration > 0:
            percentages = {zone_id: (duration / total_duration) * 100.0 for zone_id, duration in durations.items()}
            
            # Yüzdelerin toplamını kontrol et ve gerekirse normalize et
            total_percentage = sum(percentages.values())
            if abs(total_percentage - 100.0) > 0.1:  # 0.1% tolerans
                logger.warning(f"Normalizing percentages from {total_percentage:.2f}% to 100%")
                normalize_factor = 100.0 / total_percentage
                percentages = {zone_id: pct * normalize_factor for zone_id, pct in percentages.items()}
        else:
            percentages = {zone_id: 0.0 for zone_id in durations}
        
        # Durations, Percentages ve Visit Counts değerlerini logla
        logger.info(f"Zone Durations: {durations}")
        logger.info(f"Zone Percentages: {percentages}")
        logger.info(f"Zone Visit Counts: {visit_counts}")
        
        # Zone ID'lerden zone adlarına çevirme
        zone_durations = {zone_names.get(zone_id, f"Unknown ({zone_id})"): duration 
                         for zone_id, duration in durations.items() if zone_id in zone_names}
        zone_percentages = {zone_names.get(zone_id, f"Unknown ({zone_id})"): percentage 
                          for zone_id, percentage in percentages.items() if zone_id in zone_names}
        zone_visit_counts = {zone_names.get(zone_id, f"Unknown ({zone_id})"): count 
                           for zone_id, count in visit_counts.items() if zone_id in zone_names}
        
        stats["total_duration"] = total_duration
        stats["zone_durations"] = zone_durations
        stats["zone_percentages"] = zone_percentages
        stats["zone_visit_counts"] = zone_visit_counts
        stats["data_points"] = len(self.timestamps)
        
        # Oturum adını ekle
        stats["session_name"] = self.session_name
        
        return stats
        
    def generate_report(self) -> Optional[Dict[str, str]]:
        """
        Bakış bölgesi istatistiklerinden rapor oluştur.
        
        Returns:
            Dict[str, str]: CSV ve JSON rapor dosyalarının yolları, veya None (hata durumunda).
        """
        try:
            # Log detaylı bilgi
            logger.info(f"Generating report with {len(self.timestamps)} recorded data points")
            logger.info(f"Zone visit counts: {self.zone_visit_counts}")
            
            # Tüm bölgelerin adlarını logla
            zone_names = {zone_id: self.zone_detector.get_zone_name(zone_id) 
                         for zone_id in self.zone_detector.ZONES.keys()}
            logger.info(f"Available zones: {zone_names}")
            
            # Zone ID'lerin dağılımını logla
            zone_id_counts = {}
            for zid in self.zone_ids:
                if zid is not None:
                    zone_id_counts[zid] = zone_id_counts.get(zid, 0) + 1
            logger.info(f"Zone ID occurrences: {zone_id_counts}")
            
            # İstatistikleri al
            stats = self.get_statistics()
            logger.info(f"Generated statistics: {stats}")
            
            # Raporlama için utils kullan
            from src.utils.report_generator import generate_gaze_statistics_report
            
            try:
                reports = generate_gaze_statistics_report(stats)
                logger.info(f"Report generation complete: {reports}")
                return reports
            except Exception as e:
                logger.error(f"Error during report generation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
        except Exception as e:
            logger.error(f"Error in generate_report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

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