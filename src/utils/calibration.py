#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration module for the driver drowsiness detection system.

This module provides functionality to calibrate the system based on the driver's
facial metrics during an initial calibration period.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import threading

# Logger oluşturma
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Calibration")


class EARCalibrator:
    """
    Calibrates the Eye Aspect Ratio (EAR) values for a specific driver.
    
    This class collects EAR values during an initial calibration period,
    analyzes them to determine minimum and maximum values, and provides
    normalized EAR values based on the driver's specific eye characteristics.
    """
    
    def __init__(self, 
                 calibration_time: int = 60,  # Kalibrasyon süresi (saniye)
                 min_samples: int = 30,       # Minimum örnek sayısı
                 percentile_min: int = 5,     # Minimum değer için yüzdelik dilim
                 percentile_max: int = 95,    # Maksimum değer için yüzdelik dilim
                 default_min_ear: float = 0.15,  # Varsayılan minimum EAR
                 default_max_ear: float = 0.35,  # Varsayılan maksimum EAR
                 store_raw_data: bool = True,    # Ham verileri sakla
                 auto_calibrate: bool = False,   # Otomatik kalibrasyon
                 calibration_callback: Optional[callable] = None  # Kalibrasyon tamamlandığında çağrılacak fonksiyon
                ):
        """
        Initialize the EAR calibrator.
        
        Args:
            calibration_time: Duration of calibration period in seconds
            min_samples: Minimum number of samples required for valid calibration
            percentile_min: Percentile to use for minimum EAR (filters outliers)
            percentile_max: Percentile to use for maximum EAR (filters outliers)
            default_min_ear: Default minimum EAR value if calibration fails
            default_max_ear: Default maximum EAR value if calibration fails
            store_raw_data: Whether to store raw EAR values or discard after calibration
            auto_calibrate: Whether to start calibration automatically on first update
            calibration_callback: Function to call when calibration completes
        """
        # Kalibrasyon parametreleri
        self.calibration_time = calibration_time
        self.min_samples = min_samples
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.store_raw_data = store_raw_data
        self.auto_calibrate = auto_calibrate
        self.calibration_callback = calibration_callback
        
        # Varsayılan değerler
        self.default_min_ear = default_min_ear
        self.default_max_ear = default_max_ear
        
        # Kalibrasyon durumu
        self.is_calibrating = False
        self.is_calibrated = False
        self.start_time = None
        self.end_time = None
        
        # Kalibrasyon verileri
        self.ear_values = []
        self.min_ear = default_min_ear
        self.max_ear = default_max_ear
        self.ear_range = default_max_ear - default_min_ear
        
        # İstatistikler için ek değişkenler
        self.mean_ear = (default_min_ear + default_max_ear) / 2
        self.std_ear = (default_max_ear - default_min_ear) / 4
        self.blink_threshold = None
        
        # Kalibrasyon sonuçları
        self.calibration_results = {
            "min_ear": default_min_ear,
            "max_ear": default_max_ear,
            "ear_range": default_max_ear - default_min_ear,
            "mean_ear": (default_min_ear + default_max_ear) / 2,
            "std_ear": (default_max_ear - default_min_ear) / 4,
            "sample_count": 0,
            "calibration_duration": 0,
            "is_valid": False,
            "histogram": None,
            "blink_threshold": None,
            "timestamp": None
        }
        
        # İlerleme takibi
        self.progress = 0.0  # 0.0 - 1.0 arası
        
        # Asenkron işlem için thread havuzu
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._lock = threading.Lock()
        
        # İlk güncelleme için otomatik kalibrasyon flag'i
        self._first_update = True
        
        # Performans optimizasyonu için ara bellek
        self._temp_buffer = []
        self._buffer_size = 10  # 10 değer toplandıktan sonra ana listeye ekle
    
    def start_calibration(self, 
                         reset_previous: bool = False, 
                         calibration_time: Optional[int] = None) -> bool:
        """
        Start the calibration process.
        
        Args:
            reset_previous: Whether to reset previous calibration results
            calibration_time: Optional custom calibration time (seconds)
            
        Returns:
            bool: True if calibration started, False otherwise
        """
        with self._lock:
            if self.is_calibrating:
                logger.warning("Calibration is already in progress.")
                return False
                
            logger.info("Starting EAR calibration...")
            
            # Önceki kalibrasyonu sıfırla
            if reset_previous:
                self.reset(keep_defaults=True)
            
            # Kalibrasyon süresini güncelle (isteğe bağlı)
            if calibration_time is not None:
                self.calibration_time = calibration_time
            
            self.is_calibrating = True
            self.is_calibrated = False
            self.start_time = time.time()
            self.end_time = None
            
            # Kalibrasyon verilerini temizle
            self.ear_values = []
            self._temp_buffer = []
            self.progress = 0.0
            
            return True
    
    def update(self, ear_value: float) -> float:
        """
        Update the calibration with a new EAR value.
        
        Args:
            ear_value: Current EAR value
            
        Returns:
            float: Normalized EAR value (0.0-1.0)
        """
        # İlk güncelleme ise ve otomatik kalibrasyon etkinse başlat
        if self._first_update and self.auto_calibrate and not self.is_calibrating and not self.is_calibrated:
            self._first_update = False
            self.start_calibration()
        
        # Geçersiz değerleri filtrele
        if ear_value is None or ear_value <= 0:
            return self.normalize_ear(ear_value)
        
        # Kalibrasyon devam ediyorsa değeri kaydet
        if self.is_calibrating:
            with self._lock:
                # Ara belleğe ekle (performans için)
                self._temp_buffer.append(ear_value)
                
                # Belirli sayıda değer toplandığında ana listeye ekle
                if len(self._temp_buffer) >= self._buffer_size:
                    self.ear_values.extend(self._temp_buffer)
                    self._temp_buffer = []
                
                # Kalibrasyon süresini kontrol et
                elapsed_time = time.time() - self.start_time
                self.progress = min(1.0, elapsed_time / self.calibration_time)
                
                # Kalibrasyon tamamlandı mı?
                if elapsed_time >= self.calibration_time and self._future is None:
                    # Kalan değerleri ana listeye ekle
                    self.ear_values.extend(self._temp_buffer)
                    self._temp_buffer = []
                    
                    # Asenkron olarak kalibrasyonu bitir
                    self._future = self._executor.submit(
                        self._finalize_calibration, elapsed_time
                    )
        
        # Değeri normalize et ve döndür
        return self.normalize_ear(ear_value)
    
    def _finalize_calibration(self, elapsed_time: float):
        """
        Finalize the calibration process. This is called asynchronously.
        
        Args:
            elapsed_time: Elapsed time in seconds
        """
        with self._lock:
            self.is_calibrating = False
            self.end_time = time.time()
            
            # Yeterli örnek var mı kontrol et
            if len(self.ear_values) < self.min_samples:
                logger.warning(f"Not enough samples for calibration: {len(self.ear_values)} < {self.min_samples}")
                self.is_calibrated = False
                self._future = None
                
                # Callback'i çağır
                if self.calibration_callback:
                    self.calibration_callback(False, self.calibration_results)
                
                return
            
            try:
                # Kalan değerleri tekrar kontrol et ve ekle
                if self._temp_buffer:
                    self.ear_values.extend(self._temp_buffer)
                    self._temp_buffer = []
                
                # Aykırı değerleri filtrele
                ear_array = np.array(self.ear_values)
                
                # Debug logging
                logger.info(f"Calibration raw values: min={np.min(ear_array):.4f}, max={np.max(ear_array):.4f}, mean={np.mean(ear_array):.4f}")
                
                # Temel istatistikleri hesapla
                self.mean_ear = np.mean(ear_array)
                self.std_ear = np.std(ear_array)
                
                # Yüzdelik dilimlere göre min ve max değerleri belirle
                self.min_ear = np.percentile(ear_array, self.percentile_min)
                self.max_ear = np.percentile(ear_array, self.percentile_max)
                
                # Debug logging
                logger.info(f"Calibration percentile values: {self.percentile_min}%={self.min_ear:.4f}, {self.percentile_max}%={self.max_ear:.4f}")
                
                # Histogram oluştur (veri dağılımını anlamak için)
                hist, bin_edges = np.histogram(ear_array, bins=20)
                
                # Göz kırpma eşiğini tahmin et
                # Genellikle mean - 1.5*std iyi bir eşik değeridir
                self.blink_threshold = max(self.min_ear, self.mean_ear - 1.5 * self.std_ear)
                
                # Makul bir aralık olduğundan emin ol
                if self.max_ear - self.min_ear < 0.05:
                    logger.warning("Calibration range too small, adjusting values")
                    # Mean etrafında makul bir aralık oluştur
                    self.min_ear = max(0.1, self.mean_ear - 0.1)
                    self.max_ear = min(0.5, self.mean_ear + 0.1)
                
                self.ear_range = max(0.05, self.max_ear - self.min_ear)
                self.is_calibrated = True
                
                # Kalibrasyon sonuçlarını güncelle
                self.calibration_results = {
                    "min_ear": self.min_ear,
                    "max_ear": self.max_ear,
                    "ear_range": self.ear_range,
                    "mean_ear": self.mean_ear,
                    "std_ear": self.std_ear,
                    "sample_count": len(self.ear_values),
                    "calibration_duration": elapsed_time,
                    "is_valid": self.is_calibrated,
                    "histogram": {
                        "counts": hist.tolist(),
                        "bin_edges": bin_edges.tolist()
                    },
                    "blink_threshold": self.blink_threshold,
                    "timestamp": self.end_time
                }
                
                # Ham verileri sakla seçeneği
                if not self.store_raw_data:
                    self.ear_values = []
                
                logger.info(f"Calibration completed: min_ear={self.min_ear:.3f}, max_ear={self.max_ear:.3f}, threshold={self.blink_threshold:.3f}")
                
                # Callback'i çağır
                if self.calibration_callback:
                    self.calibration_callback(True, self.calibration_results)
            
            except Exception as e:
                logger.error(f"Error during calibration finalization: {str(e)}")
                self.is_calibrated = False
                
                # Varsayılan değerlere dön
                self.min_ear = self.default_min_ear
                self.max_ear = self.default_max_ear
                self.ear_range = self.default_max_ear - self.default_min_ear
                
                # Callback'i çağır
                if self.calibration_callback:
                    self.calibration_callback(False, {"error": str(e)})
            
            finally:
                self._future = None
    
    def normalize_ear(self, ear_value: float) -> float:
        """
        Normalize an EAR value based on calibration.
        
        Args:
            ear_value: Raw EAR value
            
        Returns:
            float: Normalized EAR value (0.0-1.0)
        """
        if ear_value is None or ear_value <= 0:
            return 0.0
            
        # Değeri 0-1 aralığına normalize et
        normalized = (ear_value - self.min_ear) / max(self.ear_range, 0.001)
        
        # 0-1 aralığında sınırla
        return max(0.0, min(1.0, normalized))
    
    def is_eye_closed(self, ear_value: float) -> bool:
        """
        Determine if the eye is closed based on the calibrated threshold.
        
        Args:
            ear_value: Current EAR value
            
        Returns:
            bool: True if the eye is likely closed, False otherwise
        """
        if ear_value is None or ear_value <= 0:
            return True
            
        # Eğer göz kırpma eşiği kalibre edilmişse kullan, değilse heuristic
        threshold = self.blink_threshold if self.blink_threshold else (self.min_ear + self.ear_range * 0.3)
        
        return ear_value < threshold
    
    def get_calibration_status(self) -> Dict:
        """
        Get the current calibration status.
        
        Returns:
            Dict: Calibration status information
        """
        with self._lock:
            status = {
                "is_calibrating": self.is_calibrating,
                "is_calibrated": self.is_calibrated,
                "progress": self.progress,
                "sample_count": len(self.ear_values) + len(self._temp_buffer),
                "min_ear": self.min_ear,
                "max_ear": self.max_ear,
                "ear_range": self.ear_range,
                "mean_ear": self.mean_ear,
                "std_ear": self.std_ear,
                "blink_threshold": self.blink_threshold
            }
            
            if self.is_calibrating:
                elapsed_time = time.time() - self.start_time
                status["elapsed_time"] = elapsed_time
                status["remaining_time"] = max(0, self.calibration_time - elapsed_time)
                status["sample_rate"] = status["sample_count"] / max(0.1, elapsed_time)
            elif self.is_calibrated and self.end_time:
                status["calibration_duration"] = self.end_time - self.start_time
                status["timestamp"] = self.end_time
            
            return status
    
    def get_threshold_ear(self, normalized_threshold: float = 0.3) -> float:
        """
        Convert a normalized threshold to an absolute EAR value.
        
        Args:
            normalized_threshold: Normalized threshold (0.0-1.0)
            
        Returns:
            float: Absolute EAR threshold
        """
        return self.min_ear + (normalized_threshold * self.ear_range)
    
    def save_calibration(self, file_path: str) -> bool:
        """
        Save calibration results to a file.
        
        Args:
            file_path: Path to save the calibration data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Kaydetme işlemi
            import json
            
            # Raw değerleri hariç tut veya sınırla
            save_data = self.calibration_results.copy()
            if not self.store_raw_data:
                save_data["raw_values"] = []
            else:
                # Sadece ilk ve son 100 değeri sakla (dosya boyutunu sınırlamak için)
                values = self.ear_values.copy()
                if len(values) > 200:
                    save_data["raw_values"] = values[:100] + values[-100:]
                else:
                    save_data["raw_values"] = values
            
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Calibration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {str(e)}")
            return False
    
    def load_calibration(self, file_path: str) -> bool:
        """
        Load calibration results from a file.
        
        Args:
            file_path: Path to load the calibration data from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Yükleme işlemi
            import json
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Temel değerleri güncelle
            self.min_ear = data.get("min_ear", self.default_min_ear)
            self.max_ear = data.get("max_ear", self.default_max_ear)
            self.ear_range = data.get("ear_range", self.max_ear - self.min_ear)
            self.mean_ear = data.get("mean_ear", (self.min_ear + self.max_ear) / 2)
            self.std_ear = data.get("std_ear", self.ear_range / 4)
            self.blink_threshold = data.get("blink_threshold", self.mean_ear - 1.5 * self.std_ear)
            
            # Kalibrasyon sonuçlarını güncelle
            self.calibration_results = data
            
            # Durumu güncelle
            self.is_calibrated = True
            self.is_calibrating = False
            
            logger.info(f"Calibration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {str(e)}")
            return False
    
    def reset(self, keep_defaults: bool = False):
        """
        Reset the calibration.
        
        Args:
            keep_defaults: Whether to keep default values or reset them too
        """
        with self._lock:
            # Kalibrasyon devam ediyorsa iptal et
            if self.is_calibrating and self._future:
                self._future.cancel()
                self._future = None
            
            self.is_calibrating = False
            self.is_calibrated = False
            self.start_time = None
            self.end_time = None
            self.ear_values = []
            self._temp_buffer = []
            self.progress = 0.0
            
            # Varsayılan değerlere dön
            if not keep_defaults:
                self.min_ear = self.default_min_ear
                self.max_ear = self.default_max_ear
                self.ear_range = self.default_max_ear - self.default_min_ear
                self.mean_ear = (self.default_min_ear + self.default_max_ear) / 2
                self.std_ear = (self.default_max_ear - self.default_min_ear) / 4
                self.blink_threshold = None
                
                self.calibration_results = {
                    "min_ear": self.default_min_ear,
                    "max_ear": self.default_max_ear,
                    "ear_range": self.default_max_ear - self.default_min_ear,
                    "mean_ear": (self.default_min_ear + self.default_max_ear) / 2,
                    "std_ear": (self.default_max_ear - self.default_min_ear) / 4,
                    "sample_count": 0,
                    "calibration_duration": 0,
                    "is_valid": False,
                    "histogram": None,
                    "blink_threshold": None,
                    "timestamp": None
                }
            
            # İlk güncelleme flag'ini sıfırla
            self._first_update = True
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Singleton pattern için thread-safe implementasyon
_ear_calibrator_lock = threading.Lock()
_ear_calibrator = None

def get_ear_calibrator() -> EARCalibrator:
    """
    Get or create the EAR calibrator instance (thread-safe singleton).
    
    Returns:
        EARCalibrator: The EAR calibrator instance
    """
    global _ear_calibrator
    
    with _ear_calibrator_lock:
        if _ear_calibrator is None:
            _ear_calibrator = EARCalibrator()
    
    return _ear_calibrator