#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sürücü bakış süresi izleme modülü.

Bu modül, AB regülasyonu C(2023)4523'e uygun şekilde farklı bakış bölgelerini izleyen
ve her bölgede geçirilen süreyi hesaplayan GazeDurationMonitor sınıfını içerir.
Sürücünün bakış davranışlarını ve sürelerini analiz ederek sürücü dalgınlık tespiti
için veri toplama ve analiz altyapısını sağlar.

Modül Özellikleri:
-----------------
1. Bakış Bölgeleri İzleme:
   * Farklı bölgelerdeki bakış sürelerini takip etme
   * Her bölgede geçirilen toplam süreyi hesaplama 
   * Bölge değişimlerini algılama ve kaydetme
   
2. Dalgınlık Analizi:
   * AB regülasyonu C(2023)4523 uyumlu uyarı seviyeleri (NORMAL, WARNING, CRITICAL)
   * Alan 1 (infotainment) bakış süreleri analizi
   * Yol merkezinden uzun süre bakış kayması tespiti
   * Alan 1'den Alan 2'ye geçiş sürelerinin analizi
   
3. İleri Düzey Metrikler:
   * Dalgınlık puanı hesaplama (0-100 arası)
   * Bakış geçiş desenleri analizi 
   * Zaman penceresi bazlı istatistikler

4. Optimizasyonlar:
   * NumPy ile vektörel hesaplamalar
   * Verimli veri yapıları (deque, NumPy dizileri)
   * Yüksek performanslı metrikler
   
5. Raporlama ve Görselleştirme:
   * JSON formatında detaylı raporlar oluşturma
   * Görselleştirme için renk kodlu veri yapıları
   * Sürüş sonrası analiz için veri hazırlama

Kullanım:
-------
Örnek:
```python
from src.detection.gaze_duration_monitor import get_gaze_duration_monitor

# Singleton örneği al
monitor = get_gaze_duration_monitor()

# Her kare için güncelle
timestamp = time.time()
zone_id = 0  # Road Center
state = monitor.update(zone_id, timestamp)

# Dalgınlık seviyesini kontrol et
distraction_level = state["distraction_level"]
if distraction_level != "NORMAL":
    warning_details = state["warning"]["reasons"]
    print(f"Uyarı: {warning_details}")

# Sürüş sonunda rapor oluştur
monitor.save_report_to_file(time.time())
```

AB Regülasyonu Uyumluluğu:
------------------------
Bu modül, EU regülasyonu C(2023)4523'te belirtilen sürücü dikkat dağınıklığı uyarı 
sistemleri için gereklilikleri karşılamaktadır:

1. Alan 1 (infotainment gibi sürüşle ilgili olmayan alanlar) bakış süresi takibi
2. Yol merkezine bakmama sürelerinin takibi
3. Alan 1'de geçirilen toplam süre yüzdesinin izlenmesi
4. Alan 1'den Alan 2'ye geçiş sürelerinin analizi

Referanslar:
----------
- EU Regulation C(2023)4523: Advanced Driver Distraction Warning
- https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=PI_COM:C(2023)4523
"""

import time
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
from enum import Enum, auto
import numpy as np
from datetime import datetime

# Import the report generator module
from src.utils.report_generator import generate_distraction_report

# Logger setup
logger = logging.getLogger(__name__)

class WarningLevel(str, Enum):
    """Sürücü dalgınlık uyarı seviyeleri."""
    NORMAL = "NORMAL"     # Normal sürüş davranışı
    WARNING = "WARNING"   # Dikkat dağınıklığı riski başlıyor
    CRITICAL = "CRITICAL" # Yüksek dikkat dağınıklığı riski

class GazeData:
    """Her bir bakış verisi için yapı."""
    def __init__(self, zone_id: int, timestamp: float):
        self.zone_id = zone_id          # Bakılan bölge ID'si
        self.timestamp = timestamp      # Bakış zamanı (saniye)
        self.duration = 0.0             # Bakış süresi (saniye)
        self.is_area1 = self._is_area1(zone_id)  # Alan 1 bölgesi mi?
    
    def _is_area1(self, zone_id: int) -> bool:
        # Alan 1 bölgeleri (sürüşle doğrudan ilgili olmayan)
        # AB regülasyonu C(2023)4523'e göre, Infotainment (zone_id=2) Alan 1'dir
        return zone_id == 2  # Infotainment
        
class GazeZone:
    """Her bakış bölgesi için temel bilgiler."""
    def __init__(self, zone_id: int, name: str, is_area1: bool):
        self.zone_id = zone_id          # Bölge ID'si
        self.name = name                # Bölge adı
        self.is_area1 = is_area1        # Alan 1 bölgesi mi?
        self.total_duration = 0.0       # Toplam bakış süresi
        self.visit_count = 0            # Ziyaret sayısı
        self.current_visit_start = None # Mevcut ziyaret başlangıç zamanı

class GazeDurationMonitor:
    """
    AB regülasyonu C(2023)4523 uyumlu bakış süresi izleme sistemi.
    
    Bu sınıf, sürücünün bakışlarını farklı bölgelerde (zonlar) izler
    ve her bölgede geçirilen süreyi hesaplar. Sürücülerin bakış davranışlarının
    AB regülasyonlarına uygun şekilde takip edilmesini sağlar.
    
    AB regülasyonu C(2023)4523'e göre bakış bölgeleri:
    - Alan 1: Sürüşle doğrudan ilgili olmayan alanlar (örn. infotainment)
      Bu alanlara uzun süre bakmak dikkat dağınıklığı olarak değerlendirilir.
      
    - Alan 2: Sürüşle doğrudan ilgili alanlar (ön cam, aynalar, göstergeler)
      Bu alanlara bakış normal sürüş davranışının parçasıdır.
      
    Regülasyon kriterleri:
    - Alan 1'e bakış süresi: 2s üzerinde uyarı, 3s üzerinde kritik
    - Yol merkezine bakmama süresi: 3s üzerinde uyarı, 4s üzerinde kritik
    - Alan 1'de geçirilen toplam süre yüzdesi: %15 üzerinde uyarı, %20 üzerinde kritik
    - Alan 1'den Alan 2'ye geçiş süresi: 1s altında olmalıdır
    """
    
    # Bölge tanımlamaları - AB regülasyonu C(2023)4523'e uygun şekilde
    ZONE_DEFINITIONS = {
        0: {"name": "Road Center", "is_area1": False},           # Ön cam merkezi görüş alanı (Alan 2)
        1: {"name": "Driving Instruments", "is_area1": False},   # Temel sürüş göstergeleri (Alan 2)
        2: {"name": "Infotainment", "is_area1": True},           # Eğlence sistemi, navigasyon (Alan 1)
        3: {"name": "Left Side", "is_area1": False},             # Sol yan cam ve ayna (Alan 2)
        4: {"name": "Right Side", "is_area1": False},            # Sağ yan cam ve ayna (Alan 2)
        5: {"name": "Rear Mirror", "is_area1": False}            # Dikiz aynası (Alan 2)
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Sistemi başlat ve yapılandırma yükle.
        
        Args:
            config: Sistem konfigürasyon ayarları (None ise yapılandırma dosyasından veya
                   varsayılan ayarlar kullanılır)
        """
        # Yapılandırmayı yükle (önce YAML, sonra varsayılan)
        if config is None:
            self.config = self._load_config_from_yaml()
        else:
            self.config = config
        
        # Bölge nesneleri
        self.zones = {}
        for zone_id, info in self.ZONE_DEFINITIONS.items():
            self.zones[zone_id] = GazeZone(zone_id, info["name"], info["is_area1"])
            
        # Bakış geçmişi (son N bakış)
        self.gaze_history = []
        self.max_history_size = 100  # Son 100 bakışı sakla
        
        # Belirli süre aralığındaki bakışları takip etmek için daha verimli yapı
        self.time_window_gazes = deque()
        
        # Mevcut durum
        self.current_zone_id = None
        self.current_gaze_start = None
        self.last_timestamp = None
        
        # Metrikler
        self.total_driving_time = 0.0
        self.total_area1_time = 0.0  # Alan 1'de geçirilen toplam süre
        self.area1_percentage = 0.0  # Alan 1'de geçirilen süre yüzdesi
        self.road_center_absence_time = 0.0  # Yol merkezine bakılmayan toplam süre
        self.max_road_center_absence_time = 0.0  # Tek seferde maksimum yol merkezine bakmama süresi
        
        # Debugging sayacı ekleyelim
        self._update_counter = 0
        
        logger.info("GazeDurationMonitor initialized with configuration")
        logger.debug(f"Thresholds: {self.config['thresholds']}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Varsayılan yapılandırma ayarlarını yükle.
        
        Returns:
            Dict[str, Any]: Varsayılan konfigürasyon
        """
        return {
            "thresholds": {
                "area1": {
                    "warning": 2.0,  # saniye
                    "critical": 3.0  # saniye
                },
                "road_center_absence": {
                    "warning": 3.0,  # saniye
                    "critical": 4.0  # saniye
                },
                "area2": {
                    "warning": 2.0,  # saniye
                    "critical": 3.0  # saniye
                },
                "area1_percentage": {
                    "warning": 15.0,  # yüzde
                    "critical": 20.0  # yüzde 
                }
            },
            "time_window": 60.0  # son 60 saniyeyi analiz et
        }
    
    def _load_config_from_yaml(self) -> Dict[str, Any]:
        """
        config/config.yaml dosyasından yapılandırma yükle.
        
        Dosya bulunamazsa veya okuma hatası olursa varsayılan değerler kullanılır.
        
        Returns:
            Dict[str, Any]: Yapılandırma ayarları
        """
        try:
            import yaml
            
            # Proje kök dizinini bul
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'config', 'config.yaml')
            
            logger.debug(f"Looking for configuration at: {config_path}")
            
            # Dosya var mı kontrol et
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found at {config_path}. Using default values.")
                return self._load_default_config()
            
            # Dosyayı oku
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                
            # Dalgınlık tespiti ayarlarını al
            gaze_config = full_config.get('gaze_monitoring', {})
            
            # Varsayılan değerlerle birleştir
            default_config = self._load_default_config()
            
            # Ayarları güncelle
            if 'thresholds' in gaze_config:
                if 'area1' in gaze_config['thresholds']:
                    default_config['thresholds']['area1'].update(gaze_config['thresholds']['area1'])
                if 'road_center_absence' in gaze_config['thresholds']:
                    default_config['thresholds']['road_center_absence'].update(
                        gaze_config['thresholds']['road_center_absence'])
                if 'area2' in gaze_config['thresholds']:
                    default_config['thresholds']['area2'].update(gaze_config['thresholds']['area2'])
                if 'area1_percentage' in gaze_config['thresholds']:
                    default_config['thresholds']['area1_percentage'].update(
                        gaze_config['thresholds']['area1_percentage'])
            
            if 'time_window' in gaze_config:
                default_config['time_window'] = gaze_config['time_window']
            
            logger.info("Configuration loaded from YAML file")
            return default_config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}. Using default values.")
            return self._load_default_config()
    
    def update(self, zone_id: int, timestamp: float) -> Dict[str, Any]:
        """
        Yeni bakış verisiyle sistem durumunu güncelle.
        
        Args:
            zone_id: Bakılan bölge ID'si
            timestamp: Bakış zamanı (saniye cinsinden)
            
        Returns:
            Dict[str, Any]: Güncel durum bilgileri ve dalgınlık analizi
        """
        # Zone ID doğrulama
        if zone_id not in self.zones:
            logger.warning(f"Invalid zone_id: {zone_id}. Ignoring update.")
            return self._get_current_state(timestamp)
            
        # İlk çağrı kontrolü
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            self.current_zone_id = zone_id
            self.current_gaze_start = timestamp
            logger.info(f"First gaze update received. Zone: {self.zones[zone_id].name}")
            return self._get_current_state(timestamp)
            
        # Süre hesapla
        time_diff = timestamp - self.last_timestamp
        if time_diff < 0:
            logger.warning(f"Timestamp error: Current ({timestamp}) earlier than previous ({self.last_timestamp})")
            return self._get_current_state(timestamp)
        
        self.total_driving_time += time_diff
        
        # Bölge değişikliği kontrolü
        if zone_id != self.current_zone_id:
            # Önceki bölgede geçirilen süreyi hesapla
            if self.current_zone_id is not None:
                gaze_duration = timestamp - self.current_gaze_start
                zone = self.zones[self.current_zone_id]
                zone.total_duration += gaze_duration
                
                # Alan 1 istatistiklerini güncelle
                if zone.is_area1:
                    self.total_area1_time += gaze_duration
                
                # Yol merkezine bakılmayan süreyi güncelle
                if self.current_zone_id != 0:  # Road Center değilse
                    self.road_center_absence_time += gaze_duration
                
                # Bakış geçmişine ekle
                gaze_data = GazeData(self.current_zone_id, self.current_gaze_start)
                gaze_data.duration = gaze_duration
                self._add_to_history(gaze_data)
                self._add_to_time_window(gaze_data)
                
                # Ziyaret sayacını artır
                zone.visit_count += 1
                
                logger.debug(f"Zone change: {zone.name} -> {self.zones[zone_id].name}, "
                           f"duration: {gaze_duration:.2f}s")
            
            # Yeni bölgeye geçiş
            self.current_zone_id = zone_id
            self.current_gaze_start = timestamp
            
        # Toplam metrikleri güncelle
        if self.total_driving_time > 0:
            self.area1_percentage = (self.total_area1_time / self.total_driving_time) * 100
            # Debug için area1_percentage hesaplamasını logla
            if self._update_counter % 10 == 0:  # Her 10 güncelleme sonrası logla
                logger.info(f"Area1 percentage calculation: {self.total_area1_time}/{self.total_driving_time}*100 = {self.area1_percentage:.2f}%")
                logger.info(f"Current Zone: {self.current_zone_id}, Is Area1: {self.zones[self.current_zone_id].is_area1 if self.current_zone_id is not None else 'N/A'}")
            
        # Son zaman damgasını güncelle
        self.last_timestamp = timestamp
        
        # Güncelleme sayacını artır
        self._update_counter += 1
        
        # Güncel durum ve dalgınlık analizi
        current_state = self._get_current_state(timestamp)
        
        # Dalgınlık analizini ekle
        current_state["warning"] = self.get_warning_details(timestamp)
        
        return current_state
    
    def _add_to_history(self, gaze_data: GazeData) -> None:
        """
        Bakış verilerini geçmişe ekle ve boyutu kontrol et.
        
        Args:
            gaze_data: Eklenecek bakış verisi
        """
        self.gaze_history.append(gaze_data)
        if len(self.gaze_history) > self.max_history_size:
            self.gaze_history.pop(0)  # En eski veriyi kaldır
    
    def _add_to_time_window(self, gaze_data: GazeData) -> None:
        """
        Zaman penceresi için yeni bakış verisi ekle.
        
        Args:
            gaze_data: Eklenecek bakış verisi
        """
        self.time_window_gazes.append(gaze_data)
        
        # Zaman penceresi dışındaki eski bakışları kaldır
        window_size = self.config["time_window"]
        cutoff_time = gaze_data.timestamp - window_size
        
        while self.time_window_gazes and self.time_window_gazes[0].timestamp < cutoff_time:
            self.time_window_gazes.popleft()
    
    def get_current_gaze_duration(self, timestamp: float) -> float:
        """
        Mevcut bakışın süresini hesapla.
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            float: Mevcut bakış süresi (saniye)
        """
        if self.current_gaze_start is None:
            return 0.0
        return timestamp - self.current_gaze_start
    
    def get_zone_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Tüm bölgeler için istatistikler.
        
        Returns:
            Dict[int, Dict[str, Any]]: Her bölge için istatistikler
        """
        stats = {}
        
        # Bölge sürelerini al
        durations = {}
        for zone_id, zone in self.zones.items():
            durations[zone_id] = zone.total_duration
        
        # Toplam süre
        total_duration = sum(durations.values())
        
        # Süre durumu hakkında bilgi logla
        logger.info(f"Zone durations: {durations}, total: {total_duration:.2f}s")
        
        # Eğer toplam süre 0 veya çok küçükse, geçerli veri yok anlamına gelir
        if total_duration < 0.001:  # Çok küçük değerler için epsilon
            logger.warning("Total zone duration is zero or very small - possible data collection issue")
            # Bu durumda her bölge için 0 değeri döndür
            for zone_id, zone in self.zones.items():
                stats[zone_id] = {
                    "name": zone.name,
                    "is_area1": zone.is_area1,
                    "total_duration": 0.0,
                    "visit_count": 0,
                    "average_duration": 0.0,
                    "percentage": 0.0
                }
            return stats
        
        # Her bölge için istatistikleri hesapla
        for zone_id, zone in self.zones.items():
            # Toplam yüzde hesapla
            percentage = (durations[zone_id] / total_duration) * 100.0 if total_duration > 0 else 0.0
            
            stats[zone_id] = {
                "name": zone.name,
                "is_area1": zone.is_area1,
                "total_duration": durations[zone_id],
                "visit_count": zone.visit_count,
                "average_duration": durations[zone_id] / max(1, zone.visit_count),
                "percentage": percentage
            }
        
        # Calculated percentages for verification
        total_percentage = sum(stats[zone_id]["percentage"] for zone_id in stats)
        logger.info(f"Total percentage: {total_percentage:.2f}%, should be close to 100%")
        
        return stats
    
    def get_time_window_statistics(self, current_timestamp: float, window_size: Optional[float] = None) -> Dict[str, float]:
        """
        Belirli zaman penceresi içindeki istatistikler.
        
        Args:
            current_timestamp: Mevcut zaman
            window_size: Zaman penceresi boyutu (saniye) (None ise varsayılan pencere kullanılır)
            
        Returns:
            Dict[str, float]: Zaman penceresi içindeki istatistikler
        """
        if window_size is None:
            window_size = self.config["time_window"]
            
        start_time = current_timestamp - window_size
        window_stats = {
            "total_time": 0.0,
            "area1_time": 0.0,
            "area1_percentage": 0.0,
            "road_center_time": 0.0,
            "road_center_percentage": 0.0,
            "road_center_absence_time": 0.0,
            "zone_durations": {zone_id: 0.0 for zone_id in self.zones}
        }
        
        # Zaman penceresi içindeki bakışları filtrele
        recent_gazes = [g for g in self.time_window_gazes]
        
        # Mevcut bakışı da dahil et
        if self.current_zone_id is not None and self.current_gaze_start is not None:
            current_duration = current_timestamp - self.current_gaze_start
            if self.current_gaze_start >= start_time:
                # Eğer bakış pencere içinde başladıysa
                window_stats["total_time"] += current_duration
                window_stats["zone_durations"][self.current_zone_id] += current_duration
                
                if self.zones[self.current_zone_id].is_area1:
                    window_stats["area1_time"] += current_duration
                
                if self.current_zone_id == 0:  # Road Center
                    window_stats["road_center_time"] += current_duration
                else:
                    window_stats["road_center_absence_time"] += current_duration
            else:
                # Bakış pencere dışında başladıysa, sadece pencere içindeki kısmını hesapla
                overlap_duration = current_timestamp - start_time
                window_stats["total_time"] += overlap_duration
                window_stats["zone_durations"][self.current_zone_id] += overlap_duration
                
                if self.zones[self.current_zone_id].is_area1:
                    window_stats["area1_time"] += overlap_duration
                
                if self.current_zone_id == 0:  # Road Center
                    window_stats["road_center_time"] += overlap_duration
                else:
                    window_stats["road_center_absence_time"] += overlap_duration
        
        # Geçmiş bakışları hesapla
        for gaze in recent_gazes:
            window_stats["total_time"] += gaze.duration
            window_stats["zone_durations"][gaze.zone_id] += gaze.duration
            
            if gaze.is_area1:
                window_stats["area1_time"] += gaze.duration
                
            if gaze.zone_id == 0:  # Road Center
                window_stats["road_center_time"] += gaze.duration
            else:
                window_stats["road_center_absence_time"] += gaze.duration
        
        # Yüzdeleri hesapla
        if window_stats["total_time"] > 0:
            window_stats["area1_percentage"] = (window_stats["area1_time"] / window_stats["total_time"]) * 100
            window_stats["road_center_percentage"] = (window_stats["road_center_time"] / window_stats["total_time"]) * 100
            
        return window_stats
    
    def _get_current_state(self, timestamp: float) -> Dict[str, Any]:
        """
        Sistemin güncel durumunu döndür.
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            Dict[str, Any]: Güncel durum
        """
        current_zone_name = "Unknown"
        current_zone_is_area1 = False
        current_gaze_duration = 0.0
        
        if self.current_zone_id is not None:
            current_zone_name = self.zones[self.current_zone_id].name
            current_zone_is_area1 = self.zones[self.current_zone_id].is_area1
            current_gaze_duration = self.get_current_gaze_duration(timestamp)
        
        # Temel dalgınlık seviyesi
        distraction_level = self.get_distraction_level(timestamp)
        
        return {
            "current_zone": {
                "id": self.current_zone_id,
                "name": current_zone_name,
                "is_area1": current_zone_is_area1,
                "duration": current_gaze_duration
            },
            "metrics": {
                "total_driving_time": self.total_driving_time,
                "total_area1_time": self.total_area1_time,
                "area1_percentage": self.area1_percentage,
                "road_center_absence_time": self.road_center_absence_time
            },
            "distraction_level": distraction_level
        }
    
    def reset(self) -> None:
        """Tüm geçmiş verileri ve istatistikleri sıfırlar."""
        # Bölge verilerini sıfırla
        for zone in self.zones.values():
            zone.total_duration = 0.0
            zone.visit_count = 0
            zone.current_visit_start = None
        
        # Bakış geçmişini temizle
        self.gaze_history = []
        self.time_window_gazes = deque()
        
        # Mevcut durumu sıfırla
        self.current_zone_id = None
        self.current_gaze_start = None
        self.last_timestamp = None
        
        # Metrikleri sıfırla
        self.total_driving_time = 0.0
        self.total_area1_time = 0.0
        self.area1_percentage = 0.0
        self.road_center_absence_time = 0.0
        self.max_road_center_absence_time = 0.0
        
        logger.info("GazeDurationMonitor reset")

    def _get_last_zone_timestamp(self, target_zone_id: int) -> Optional[float]:
        """
        Belirli bir bölgeye son bakış zamanını bul.
        
        Args:
            target_zone_id: Aranan bölge ID'si
            
        Returns:
            Optional[float]: Bölgeye son bakışın bitiş zamanı veya None
        """
        # Geçmiş bakışlarda sondan başa doğru ara
        for gaze in reversed(self.gaze_history):
            if gaze.zone_id == target_zone_id:
                return gaze.timestamp + gaze.duration
        return None

    def check_current_gaze_duration(self, timestamp: float) -> WarningLevel:
        """
        Mevcut bakışın süresinin uyarı seviyesini kontrol et.
        
        AB regülasyonu C(2023)4523'e göre, farklı bölge türleri için 
        farklı maksimum bakış süreleri tanımlanmıştır.
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            WarningLevel: Uyarı seviyesi (NORMAL, WARNING, CRITICAL)
        """
        if self.current_zone_id is None:
            return WarningLevel.NORMAL
            
        duration = self.get_current_gaze_duration(timestamp)
        zone = self.zones[self.current_zone_id]
        
        # Alan 1 kontrolü (Infotainment, vb.)
        if zone.is_area1:
            if duration >= self.config["thresholds"]["area1"]["critical"]:
                logger.warning(f"CRITICAL: Looking at {zone.name} for {duration:.2f}s")
                return WarningLevel.CRITICAL
            elif duration >= self.config["thresholds"]["area1"]["warning"]:
                logger.info(f"WARNING: Looking at {zone.name} for {duration:.2f}s")
                return WarningLevel.WARNING
        
        # Diğer Alan 2 bölgeleri (Road Center hariç)
        elif self.current_zone_id != 0:  # Road Center değilse
            if duration >= self.config["thresholds"]["area2"]["critical"]:
                logger.warning(f"CRITICAL: Looking at {zone.name} for {duration:.2f}s")
                return WarningLevel.CRITICAL
            elif duration >= self.config["thresholds"]["area2"]["warning"]:
                logger.info(f"WARNING: Looking at {zone.name} for {duration:.2f}s")
                return WarningLevel.WARNING
                
        return WarningLevel.NORMAL
    
    def check_road_center_absence(self, timestamp: float) -> WarningLevel:
        """
        Yol merkezine bakmama süresini kontrol et.
        
        AB regülasyonu C(2023)4523'e göre, sürücünün yol merkezine belirli bir süre
        bakmaması dikkat dağınıklığı olarak değerlendirilir.
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            WarningLevel: Uyarı seviyesi (NORMAL, WARNING, CRITICAL)
        """
        # Zaten Road Center'a bakıyorsa, kontrol gerekmiyor
        if self.current_zone_id == 0:
            return WarningLevel.NORMAL
            
        # Son ne zaman Road Center'a bakıldı?
        if self.current_zone_id is not None:
            last_road_center_time = self._get_last_zone_timestamp(0)
            if last_road_center_time is not None:
                road_center_absence = timestamp - last_road_center_time
                
                # Maksimum road center absence değerini güncelle
                self.max_road_center_absence_time = max(self.max_road_center_absence_time, road_center_absence)
                
                if road_center_absence >= self.config["thresholds"]["road_center_absence"]["critical"]:
                    logger.warning(f"CRITICAL: Eyes off road for {road_center_absence:.2f}s")
                    return WarningLevel.CRITICAL
                elif road_center_absence >= self.config["thresholds"]["road_center_absence"]["warning"]:
                    logger.info(f"WARNING: Eyes off road for {road_center_absence:.2f}s")
                    return WarningLevel.WARNING
                    
        return WarningLevel.NORMAL
    
    def check_total_area1_percentage(self) -> WarningLevel:
        """
        Alan 1'de geçirilen toplam süre yüzdesini kontrol et.
        
        AB regülasyonu C(2023)4523'e göre, sürüş dışı alanlara fazla bakılması
        dikkat dağınıklığı olarak değerlendirilir.
        
        Returns:
            WarningLevel: Uyarı seviyesi (NORMAL, WARNING, CRITICAL)
        """
        if self.total_driving_time < 10.0:  # En az 10 saniye veri topla
            return WarningLevel.NORMAL
            
        if self.area1_percentage >= self.config["thresholds"]["area1_percentage"]["critical"]:
            logger.warning(f"CRITICAL: Too much time on non-driving areas ({self.area1_percentage:.1f}%)")
            return WarningLevel.CRITICAL
        elif self.area1_percentage >= self.config["thresholds"]["area1_percentage"]["warning"]:
            logger.info(f"WARNING: Increased time on non-driving areas ({self.area1_percentage:.1f}%)")
            return WarningLevel.WARNING
            
        return WarningLevel.NORMAL
    
    def get_distraction_level(self, timestamp: float) -> WarningLevel:
        """
        Genel dalgınlık seviyesini değerlendir.
        
        Bu metot, tüm dalgınlık göstergelerini değerlendirerek
        genel bir dalgınlık seviyesi belirler.
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            WarningLevel: Genel dalgınlık seviyesi (NORMAL, WARNING, CRITICAL)
        """
        # Tüm metrikleri değerlendir
        current_gaze_level = self.check_current_gaze_duration(timestamp)
        road_center_absence_level = self.check_road_center_absence(timestamp)
        area1_percentage_level = self.check_total_area1_percentage()
        
        # En kritik seviyeyi döndür
        if (current_gaze_level == WarningLevel.CRITICAL or 
            road_center_absence_level == WarningLevel.CRITICAL or
            area1_percentage_level == WarningLevel.CRITICAL):
            return WarningLevel.CRITICAL
        elif (current_gaze_level == WarningLevel.WARNING or 
              road_center_absence_level == WarningLevel.WARNING or
              area1_percentage_level == WarningLevel.WARNING):
            return WarningLevel.WARNING
            
        return WarningLevel.NORMAL
    
    def get_warning_details(self, timestamp: float) -> Dict[str, Any]:
        """
        Uyarı detaylarını ve nedenlerini döndür.
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            Dict[str, Any]: Uyarı seviyesi ve nedenleri
        """
        distraction_level = self.get_distraction_level(timestamp)
        
        if distraction_level == WarningLevel.NORMAL:
            return {"level": WarningLevel.NORMAL, "reasons": ["Normal driving behavior"]}
            
        # Uyarı nedenleri
        reasons = []
        
        # Mevcut bakış kontrolü
        if self.current_zone_id is not None:
            duration = self.get_current_gaze_duration(timestamp)
            zone = self.zones[self.current_zone_id]
            
            # Alan 1 kontrolü
            if zone.is_area1:
                if duration >= self.config["thresholds"]["area1"]["critical"]:
                    reasons.append(f"Looking at {zone.name} for too long ({duration:.1f}s)")
                elif duration >= self.config["thresholds"]["area1"]["warning"]:
                    reasons.append(f"Long gaze at {zone.name} ({duration:.1f}s)")
            
            # Diğer Alan 2 bölgeleri için
            elif self.current_zone_id != 0:  # Road Center değilse
                if duration >= self.config["thresholds"]["area2"]["critical"]:
                    reasons.append(f"Looking at {zone.name} for too long ({duration:.1f}s)")
                elif duration >= self.config["thresholds"]["area2"]["warning"]:
                    reasons.append(f"Long gaze at {zone.name} ({duration:.1f}s)")
        
        # Road Center'dan ayrılma kontrolü
        if self.current_zone_id != 0:  # Road Center değilse
            last_road_center_time = self._get_last_zone_timestamp(0)
            if last_road_center_time is not None:
                road_center_absence = timestamp - last_road_center_time
                
                if road_center_absence >= self.config["thresholds"]["road_center_absence"]["critical"]:
                    reasons.append(f"Eyes off road for too long ({road_center_absence:.1f}s)")
                elif road_center_absence >= self.config["thresholds"]["road_center_absence"]["warning"]:
                    reasons.append(f"Eyes off road ({road_center_absence:.1f}s)")
        
        # Alan 1 yüzde kontrolü
        if self.area1_percentage >= self.config["thresholds"]["area1_percentage"]["critical"]:
            reasons.append(f"Too much time on non-driving areas ({self.area1_percentage:.1f}%)")
        elif self.area1_percentage >= self.config["thresholds"]["area1_percentage"]["warning"]:
            reasons.append(f"Increased time on non-driving areas ({self.area1_percentage:.1f}%)")
            
        return {
            "level": distraction_level,
            "reasons": reasons
        }

    def _optimize_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Geçmiş verileri NumPy dizilerine dönüştür ve hesaplamaları hızlandır.
        
        Bakış geçmişi verilerini NumPy dizilerine dönüştürerek vektörel hesaplamalara
        olanak sağlar ve böylece performansı artırır.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (timestamps, durations, zone_ids, is_area1) NumPy dizileri
        """
        if not self.gaze_history:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        # Geçmiş verilerden NumPy dizileri oluştur
        timestamps = np.array([g.timestamp for g in self.gaze_history])
        durations = np.array([g.duration for g in self.gaze_history])
        zone_ids = np.array([g.zone_id for g in self.gaze_history])
        is_area1 = np.array([1 if g.is_area1 else 0 for g in self.gaze_history], dtype=bool)
        
        return timestamps, durations, zone_ids, is_area1
    
    def get_advanced_time_window_statistics(self, current_timestamp: float, window_size: Optional[float] = None) -> Dict[str, Any]:
        """
        NumPy kullanarak zaman penceresi istatistiklerini hızla hesapla.
        
        Belirli bir zaman penceresi içindeki bakış verilerini NumPy ile analiz eder.
        Bu metot, özellikle büyük veri setlerinde standart metoda göre daha hızlıdır.
        
        Args:
            current_timestamp: Mevcut zaman
            window_size: Zaman penceresi boyutu (saniye) (None ise varsayılan pencere kullanılır)
            
        Returns:
            Dict[str, Any]: Zaman penceresi içindeki gelişmiş istatistikler
        """
        if window_size is None:
            window_size = self.config["time_window"]
            
        # NumPy dizileri oluştur
        timestamps, durations, zone_ids, is_area1 = self._optimize_history()
        
        if len(timestamps) == 0:
            return self._empty_window_stats()
        
        # Zaman penceresi içindeki verileri filtrele
        start_time = current_timestamp - window_size
        time_mask = timestamps >= start_time
        
        if not np.any(time_mask):
            return self._empty_window_stats()
        
        # Filtrelenmiş veriler
        window_durations = durations[time_mask]
        window_zone_ids = zone_ids[time_mask]
        window_is_area1 = is_area1[time_mask]
        
        # Toplam süre
        total_time = np.sum(window_durations)
        
        # Alan 1 süresi
        area1_time = np.sum(window_durations[window_is_area1])
        area1_percentage = (area1_time / total_time * 100) if total_time > 0 else 0
        
        # Road Center süresi
        road_center_mask = window_zone_ids == 0
        road_center_time = np.sum(window_durations[road_center_mask])
        road_center_percentage = (road_center_time / total_time * 100) if total_time > 0 else 0
        
        # Road Center'a bakılmayan süre
        road_center_absence_time = total_time - road_center_time
        
        # Her bölge için toplam süreler
        zone_times = {}
        zone_percentages = {}
        for zone_id in range(len(self.ZONE_DEFINITIONS)):
            zone_mask = window_zone_ids == zone_id
            zone_time = np.sum(window_durations[zone_mask])
            zone_times[zone_id] = float(zone_time)
            zone_percentages[zone_id] = float(zone_time / total_time * 100) if total_time > 0 else 0.0
        
        # Geçiş sayısı: bakış bölgesi değişim sayısı
        transitions_count = 0
        if len(window_zone_ids) > 1:
            transitions_count = np.sum(np.diff(window_zone_ids) != 0)
        
        return {
            "total_time": float(total_time),
            "area1_time": float(area1_time),
            "area1_percentage": float(area1_percentage),
            "road_center_time": float(road_center_time),
            "road_center_percentage": float(road_center_percentage),
            "road_center_absence_time": float(road_center_absence_time),
            "zone_times": zone_times,
            "zone_percentages": zone_percentages,
            "transitions_count": int(transitions_count)
        }
    
    def _empty_window_stats(self) -> Dict[str, Any]:
        """
        Boş istatistik şablonu.
        
        Veri olmadığında veya filtre sonrası veri kalmadığında 
        kullanılacak boş istatistik şablonu.
        
        Returns:
            Dict[str, Any]: Boş istatistik yapısı
        """
        return {
            "total_time": 0.0,
            "area1_time": 0.0,
            "area1_percentage": 0.0,
            "road_center_time": 0.0,
            "road_center_percentage": 0.0,
            "road_center_absence_time": 0.0,
            "zone_times": {zone_id: 0.0 for zone_id in range(len(self.ZONE_DEFINITIONS))},
            "zone_percentages": {zone_id: 0.0 for zone_id in range(len(self.ZONE_DEFINITIONS))},
            "transitions_count": 0
        }

    def analyze_gaze_transitions(self, current_timestamp: float, window_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Bölgeler arası bakış geçişlerini analiz et.
        
        AB regülasyonu C(2023)4523'e göre, özellikle Alan 1'den (infotainment) 
        sürüşle ilgili alanlara geçiş süreleri önemlidir. Bu metot, geçiş
        sürelerini ve sıklığını analiz eder.
        
        Args:
            current_timestamp: Mevcut zaman
            window_size: Zaman penceresi boyutu (saniye)
            
        Returns:
            Dict[str, Any]: Geçiş analizleri
        """
        if window_size is None:
            window_size = self.config["time_window"]
            
        start_time = current_timestamp - window_size
        
        # Zaman penceresi içindeki bakışları filtrele
        recent_gazes = [g for g in self.gaze_history if g.timestamp >= start_time]
        
        if len(recent_gazes) < 2:
            return {
                "transitions": {},
                "transition_matrix": {i: {j: 0 for j in range(len(self.ZONE_DEFINITIONS))} for i in range(len(self.ZONE_DEFINITIONS))},
                "road_center_returns": [],
                "area1_to_area2_times": [],
                "area1_to_area2_time_avg": 0.0,
                "transition_frequency": 0.0,
                "transition_count": 0
            }
        
        # Geçiş matrisi oluştur
        transitions = {}
        transition_matrix = {i: {j: 0 for j in range(len(self.ZONE_DEFINITIONS))} for i in range(len(self.ZONE_DEFINITIONS))}
        prev_zone = None
        road_center_returns = []  # Road Center'a dönüş süreleri
        last_rc_exit_time = None
        
        # Area1'den Area2'ye geçiş süreleri
        area1_exit_time = None
        area1_to_area2_times = []
        transition_count = 0
        
        for gaze in recent_gazes:
            # Geçişleri izle
            if prev_zone is not None and prev_zone != gaze.zone_id:
                # Geçiş sayısını artır
                transition_count += 1
                transition_key = f"{prev_zone}->{gaze.zone_id}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
                
                # Geçiş matrisini güncelle
                transition_matrix[prev_zone][gaze.zone_id] = transition_matrix[prev_zone].get(gaze.zone_id, 0) + 1
                
                # Road Center'dan çıkış zamanını kaydet
                if prev_zone == 0:  # Road Center'dan çıkış
                    last_rc_exit_time = gaze.timestamp
                
                # Road Center'a dönüş süresini kaydet
                if gaze.zone_id == 0 and last_rc_exit_time is not None:  # Road Center'a dönüş
                    return_time = gaze.timestamp - last_rc_exit_time
                    road_center_returns.append(float(return_time))
                    last_rc_exit_time = None
                    
                # Alan 1'den çıkış zamanını kaydet
                if prev_zone == 2:  # Infotainment'dan çıkış
                    area1_exit_time = gaze.timestamp
                    
                # Alan 2'ye geçiş süresini kaydet
                if area1_exit_time is not None and not self.zones[gaze.zone_id].is_area1:
                    transition_time = gaze.timestamp - area1_exit_time
                    area1_to_area2_times.append(float(transition_time))
                    area1_exit_time = None
                    
            prev_zone = gaze.zone_id
        
        # Alan 1 -> Alan 2 geçiş süresi ortalaması
        avg_area1_to_area2_time = float(np.mean(area1_to_area2_times)) if area1_to_area2_times else 0.0
        
        # Geçiş sıklığı (dakikada geçiş sayısı)
        total_time_minutes = (recent_gazes[-1].timestamp + recent_gazes[-1].duration - recent_gazes[0].timestamp) / 60.0
        transition_frequency = transition_count / max(total_time_minutes, 0.01)
        
        return {
            "transitions": transitions,
            "transition_matrix": transition_matrix,
            "road_center_returns": road_center_returns,
            "area1_to_area2_times": area1_to_area2_times,
            "area1_to_area2_time_avg": avg_area1_to_area2_time,
            "transition_frequency": float(transition_frequency),
            "transition_count": transition_count
        }
    
    def calculate_distraction_score(self, timestamp: float) -> float:
        """
        Dalgınlık puanı hesapla (0-100).
        
        Farklı metrikleri birleştirerek 0-100 arasında bir dalgınlık puanı oluşturur.
        Düşük puan daha iyi sürüş davranışını gösterir.
        
        - 0-25: Mükemmel sürüş davranışı
        - 26-50: İyi sürüş davranışı
        - 51-75: Geliştirilmesi gereken sürüş davranışı
        - 76-100: Kritik dalgınlık seviyesi
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            float: Dalgınlık puanı (0-100)
        """
        # Mevcut durumu kontrol et
        if self.current_zone_id is None or self.total_driving_time < 10.0:
            return 0.0
            
        # Anlık metrikler
        current_gaze_duration = self.get_current_gaze_duration(timestamp)
        zone = self.zones[self.current_zone_id]
        
        # Son 60 saniyedeki istatistikler
        window_stats = self.get_advanced_time_window_statistics(timestamp, 60.0)
        transitions = self.analyze_gaze_transitions(timestamp, 60.0)
        
        # Skor bileşenleri (her biri 0-20 arasında)
        score_components = []
        
        # 1. Anlık bakış süresi (0-20 puan)
        if zone.is_area1:
            # Alan 1 için daha sıkı kurallar
            area1_critical = self.config["thresholds"]["area1"]["critical"]
            area1_warning = self.config["thresholds"]["area1"]["warning"]
            
            if current_gaze_duration >= area1_critical:
                score_components.append(20.0)
            elif current_gaze_duration >= area1_warning:
                normalized = (current_gaze_duration - area1_warning) / (area1_critical - area1_warning)
                score_components.append(10.0 + normalized * 10.0)
            else:
                normalized = min(1.0, current_gaze_duration / area1_warning)
                score_components.append(normalized * 10.0)
        elif self.current_zone_id != 0:  # Road Center değil
            # Alan 2 bölgeleri için (Road Center hariç)
            area2_critical = self.config["thresholds"]["area2"]["critical"]
            area2_warning = self.config["thresholds"]["area2"]["warning"]
            
            if current_gaze_duration >= area2_critical:
                score_components.append(15.0)
            elif current_gaze_duration >= area2_warning:
                normalized = (current_gaze_duration - area2_warning) / (area2_critical - area2_warning)
                score_components.append(7.5 + normalized * 7.5)
            else:
                normalized = min(1.0, current_gaze_duration / area2_warning)
                score_components.append(normalized * 7.5)
        else:
            # Road Center için puan yok (ideal durum)
            score_components.append(0.0)
            
        # 2. Yola bakmama süresi (0-20 puan)
        if self.current_zone_id != 0:
            # Son road center bakışından geçen süre
            last_road_center_time = self._get_last_zone_timestamp(0)
            if last_road_center_time is not None:
                road_center_absence = timestamp - last_road_center_time
                rc_critical = self.config["thresholds"]["road_center_absence"]["critical"]
                rc_warning = self.config["thresholds"]["road_center_absence"]["warning"]
                
                if road_center_absence >= rc_critical:
                    score_components.append(20.0)
                elif road_center_absence >= rc_warning:
                    normalized = (road_center_absence - rc_warning) / (rc_critical - rc_warning)
                    score_components.append(10.0 + normalized * 10.0)
                else:
                    normalized = min(1.0, road_center_absence / rc_warning)
                    score_components.append(normalized * 10.0)
            else:
                # Road Center'a hiç bakılmadıysa yüksek puan
                score_components.append(15.0)
        else:
            # Road Center'a bakılıyorsa puan yok
            score_components.append(0.0)
            
        # 3. Alan 1 yüzdesi (0-20 puan)
        area1_percentage = window_stats["area1_percentage"]
        area1_pct_critical = self.config["thresholds"]["area1_percentage"]["critical"]
        area1_pct_warning = self.config["thresholds"]["area1_percentage"]["warning"]
        
        if area1_percentage >= area1_pct_critical:
            score_components.append(20.0)
        elif area1_percentage >= area1_pct_warning:
            normalized = (area1_percentage - area1_pct_warning) / (area1_pct_critical - area1_pct_warning)
            score_components.append(10.0 + normalized * 10.0)
        else:
            normalized = min(1.0, area1_percentage / area1_pct_warning)
            score_components.append(normalized * 10.0)
            
        # 4. Alan 1'den Alan 2'ye geçiş süresi (0-20 puan)
        area1_to_area2_time = transitions["area1_to_area2_time_avg"]
        if area1_to_area2_time > 0:
            # 1 saniye üzerindeki geçişler giderek daha fazla cezalandırılır
            if area1_to_area2_time >= 2.0:
                score_components.append(20.0)
            elif area1_to_area2_time >= 1.0:
                normalized = (area1_to_area2_time - 1.0)
                score_components.append(10.0 + normalized * 10.0)
            else:
                normalized = min(1.0, area1_to_area2_time)
                score_components.append(normalized * 10.0)
        else:
            # Geçiş yoksa düşük puan (geçiş yapılmamış olabilir)
            score_components.append(5.0)
            
        # 5. Geçiş sıklığı (0-20 puan)
        # Çok fazla veya çok az geçiş sorunlu olabilir
        transition_frequency = transitions["transition_frequency"]
        if transition_frequency > 30.0:  # Dakikada 30'dan fazla
            # Çok fazla geçiş (gözler çok hızlı hareket ediyor)
            score_components.append(20.0)
        elif transition_frequency < 3.0 and window_stats["total_time"] > 10.0:
            # Çok az geçiş (tek bir yere odaklanmış olabilir)
            score_components.append(15.0)
        elif transition_frequency >= 15.0:
            # Yüksek ama kritik olmayan geçiş
            normalized = (transition_frequency - 15.0) / 15.0
            score_components.append(10.0 + normalized * 10.0)
        else:
            # İdeal aralık
            normalized = min(1.0, transition_frequency / 15.0)
            score_components.append(normalized * 10.0)
            
        # Toplam skor hesapla
        final_score = sum(score_components) / len(score_components) * 5.0  # 0-100 aralığına normalize et
        
        return min(100.0, max(0.0, final_score))  # 0-100 aralığında sınırla

    def generate_report(self, timestamp: float) -> Dict[str, Any]:
        """
        Sürücü dalgınlık istatistik raporu oluştur.
        
        Bu metot, AB regülasyonu C(2023)4523'e uygunluk analizi dahil
        kapsamlı bir dalgınlık istatistik raporu oluşturur.
        
        Args:
            timestamp: Rapor oluşturma zamanı
            
        Returns:
            Dict[str, Any]: Kapsamlı dalgınlık raporu
        """
        # Temel metrikleri topla
        distraction_score = self.calculate_distraction_score(timestamp)
        distraction_level = self.get_distraction_level(timestamp)
        advanced_stats = self.get_advanced_time_window_statistics(timestamp, 60.0)
        transitions = self.analyze_gaze_transitions(timestamp, 60.0)
        
        # Ana rapor yapısı
        report = {
            "timestamp": timestamp,
            "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "total_driving_time": self.total_driving_time,
            "zone_statistics": self.get_zone_statistics(),
            "area1_statistics": {
                "total_time": self.total_area1_time,
                "percentage": self.area1_percentage
            },
            "distraction": {
                "level": distraction_level,
                "score": distraction_score,
                "details": self.get_warning_details(timestamp)
            },
            "last_minute": advanced_stats,
            "transitions": transitions
        }
        
        # AB regülasyonu metrikleri (C(2023)4523 uyumluluk analizi)
        report["eu_regulation"] = {
            "area1_percentage_limit": self.config["thresholds"]["area1_percentage"]["warning"],
            "area1_percentage_actual": self.area1_percentage,
            "area1_percentage_compliant": self.area1_percentage < self.config["thresholds"]["area1_percentage"]["warning"],
            "road_center_absence_limit": self.config["thresholds"]["road_center_absence"]["warning"],
            "road_center_absence_actual": self.max_road_center_absence_time,
            "road_center_absence_compliant": True,  # Varsayılan, güncellenecek
            "area1_to_area2_transition_time": transitions["area1_to_area2_time_avg"],
            "area1_to_area2_compliant": transitions["area1_to_area2_time_avg"] < 1.0,
            "overall_compliance": "COMPLIANT"  # Varsayılan, güncellenecek
        }
        
        # Debug bilgisi ekleyelim
        logger.info(f"Report Generation Debug:")
        logger.info(f"  total_driving_time: {self.total_driving_time:.2f}s")
        logger.info(f"  total_area1_time: {self.total_area1_time:.2f}s")
        logger.info(f"  area1_percentage: {self.area1_percentage:.2f}%")
        logger.info(f"  zone_statistics: {self.get_zone_statistics()}")
        
        # Road Center'dan uzak kalma süresi kontrolü
        if self.current_zone_id != 0:  # Road Center değilse
            last_road_center_time = self._get_last_zone_timestamp(0)
            if last_road_center_time is not None:
                road_center_absence = timestamp - last_road_center_time
                # Maksimum değeri son kez güncelle
                self.max_road_center_absence_time = max(self.max_road_center_absence_time, road_center_absence)
                road_center_absence_compliant = self.max_road_center_absence_time < self.config["thresholds"]["road_center_absence"]["warning"]
                report["eu_regulation"]["road_center_absence_compliant"] = road_center_absence_compliant
                report["eu_regulation"]["road_center_absence_actual"] = self.max_road_center_absence_time
            else:
                # Road Center'a hiç bakılmadıysa uyumlu değil
                report["eu_regulation"]["road_center_absence_compliant"] = False
                
        # Genel uyumluluk kontrolü
        if (not report["eu_regulation"]["area1_percentage_compliant"] or
            not report["eu_regulation"]["road_center_absence_compliant"] or
            not report["eu_regulation"]["area1_to_area2_compliant"]):
            report["eu_regulation"]["overall_compliance"] = "NON-COMPLIANT"
            
        return report
    
    def save_report_to_file(self, timestamp: float, file_path: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Dalgınlık raporunu JSON dosyasına kaydet.
        
        Args:
            timestamp: Rapor oluşturma zamanı
            file_path: Kaydedilecek dosya yolu (None ise otomatik oluşturulur)
            
        Returns:
            Dict[str, str]: Kaydedilen dosyaların yolları veya hata durumunda None
        """
        try:
            # Rapor oluştur
            report = self.generate_report(timestamp)
            
            # Özel bir dosya yolu belirtilmişse onu kullan
            if file_path is not None:
                try:
                    # Dizin yoksa oluştur
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Custom dosya yolunu kullan
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=4, ensure_ascii=False)
                    logger.info(f"Distraction report saved to {file_path}")
                    return {'json': file_path}
                except Exception as e:
                    logger.error(f"Error saving distraction report to {file_path}: {str(e)}")
                    # Print stack trace for debugging
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
            
            # Merkezi rapor oluşturma fonksiyonunu kullan
            try:
                result_paths = generate_distraction_report(report)
                if result_paths:
                    logger.info(f"Distraction report saved using report generator")
                    return result_paths
                else:
                    logger.warning("Failed to generate distraction report - report generator returned None")
                    return None
            except Exception as e:
                logger.error(f"Error generating distraction report: {str(e)}")
                # Print stack trace for debugging
                import traceback
                logger.error(traceback.format_exc())
                return None
        except Exception as e:
            logger.error(f"Error in save_report_to_file: {str(e)}")
            # Print stack trace for debugging
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_visualization_data(self, timestamp: float) -> Dict[str, Any]:
        """
        Görselleştirme için renk kodlu veri yapısı oluştur.
        
        Bu metot, kullanıcı arayüzünde görsel geri bildirim için
        renk kodlaması dahil veri hazırlar.
        
        Args:
            timestamp: Mevcut zaman
            
        Returns:
            Dict[str, Any]: Görselleştirme verileri
        """
        # Dalgınlık seviyesi
        distraction_level = self.get_distraction_level(timestamp)
        distraction_score = self.calculate_distraction_score(timestamp)
        
        # Renk kodları (RGB formatında)
        colors = {
            WarningLevel.NORMAL: (0, 255, 0),     # Yeşil
            WarningLevel.WARNING: (255, 165, 0),  # Turuncu
            WarningLevel.CRITICAL: (255, 0, 0)    # Kırmızı
        }
        
        # Mevcut bakış bölgesi
        current_zone = {
            "id": self.current_zone_id,
            "name": "Unknown",
            "duration": 0.0,
            "color": (200, 200, 200)  # Varsayılan gri
        }
        
        if self.current_zone_id is not None:
            zone = self.zones[self.current_zone_id]
            current_zone["name"] = zone.name
            current_zone["duration"] = self.get_current_gaze_duration(timestamp)
            
            # Bölge rengini belirle
            if zone.is_area1:
                # Alan 1 bölgeleri için daha dikkatli renk kodlaması
                area1_critical = self.config["thresholds"]["area1"]["critical"]
                area1_warning = self.config["thresholds"]["area1"]["warning"]
                
                if current_zone["duration"] >= area1_critical:
                    current_zone["color"] = colors[WarningLevel.CRITICAL]
                elif current_zone["duration"] >= area1_warning:
                    current_zone["color"] = colors[WarningLevel.WARNING]
                else:
                    current_zone["color"] = colors[WarningLevel.NORMAL]
            elif self.current_zone_id != 0:  # Road Center değil
                # Diğer Alan 2 bölgeleri için
                area2_critical = self.config["thresholds"]["area2"]["critical"]
                area2_warning = self.config["thresholds"]["area2"]["warning"]
                
                if current_zone["duration"] >= area2_critical:
                    current_zone["color"] = colors[WarningLevel.CRITICAL]
                elif current_zone["duration"] >= area2_warning:
                    current_zone["color"] = colors[WarningLevel.WARNING]
                else:
                    current_zone["color"] = colors[WarningLevel.NORMAL]
            else:
                # Road Center her zaman yeşil
                current_zone["color"] = colors[WarningLevel.NORMAL]
        
        # Yol merkezine bakılmayan süre renk kodlaması
        road_center_absence_color = colors[WarningLevel.NORMAL]
        road_center_absence_time = 0.0
        
        if self.current_zone_id != 0:  # Road Center değilse
            last_road_center_time = self._get_last_zone_timestamp(0)
            if last_road_center_time is not None:
                road_center_absence_time = timestamp - last_road_center_time
                rc_critical = self.config["thresholds"]["road_center_absence"]["critical"]
                rc_warning = self.config["thresholds"]["road_center_absence"]["warning"]
                
                if road_center_absence_time >= rc_critical:
                    road_center_absence_color = colors[WarningLevel.CRITICAL]
                elif road_center_absence_time >= rc_warning:
                    road_center_absence_color = colors[WarningLevel.WARNING]
        
        return {
            "distraction_level": distraction_level,
            "distraction_score": distraction_score,
            "level_color": colors[distraction_level],
            "current_zone": current_zone,
            "road_center_absence": {
                "time": road_center_absence_time,
                "color": road_center_absence_color
            },
            "area1_percentage": {
                "value": self.area1_percentage,
                "color": self._get_color_for_area1_percentage()
            },
            "warning_details": self.get_warning_details(timestamp)
        }
    
    def _get_color_for_area1_percentage(self) -> Tuple[int, int, int]:
        """
        Alan 1 yüzdesi için renk kodu belirle.
        
        Returns:
            Tuple[int, int, int]: RGB renk kodu
        """
        area1_pct_critical = self.config["thresholds"]["area1_percentage"]["critical"]
        area1_pct_warning = self.config["thresholds"]["area1_percentage"]["warning"]
        
        if self.area1_percentage >= area1_pct_critical:
            return (255, 0, 0)  # Kırmızı
        elif self.area1_percentage >= area1_pct_warning:
            return (255, 165, 0)  # Turuncu
        return (0, 255, 0)  # Yeşil

    def integrate_with_gaze_detector(self, gaze_detector, timestamp: float, pitch: float, yaw: float) -> Dict[str, Any]:
        """
        Gaze Zone Detector ile entegrasyon için yardımcı metot.
        
        GazeZoneDetector'dan gelen verileri kullanarak bölge tespiti
        ve dalgınlık analizini gerçekleştirir.
        
        Args:
            gaze_detector: GazeZoneDetector nesnesi
            timestamp: Mevcut zaman
            pitch: Pitch açısı (derece)
            yaw: Yaw açısı (derece)
            
        Returns:
            Dict[str, Any]: Güncel durum ve dalgınlık analizi
        """
        # GazeZoneDetector'ı kullanarak bölge tespiti
        detected_zone = gaze_detector.update(pitch, yaw, timestamp)
        
        # GazeDurationMonitor'u güncelle
        return self.update(detected_zone, timestamp)

# Singleton pattern için global instance
_gaze_duration_monitor = None

def get_gaze_duration_monitor(config: Optional[Dict[str, Any]] = None) -> GazeDurationMonitor:
    """
    GazeDurationMonitor instance'ı döndürür (singleton pattern).
    
    Args:
        config: Sistem konfigürasyon ayarları
        
    Returns:
        GazeDurationMonitor: Monitor instance'ı
    """
    global _gaze_duration_monitor
    if _gaze_duration_monitor is None:
        _gaze_duration_monitor = GazeDurationMonitor(config=config)
    return _gaze_duration_monitor 