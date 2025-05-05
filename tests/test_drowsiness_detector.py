#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drowsiness Detector testleri.

Bu script, DrowsinessDetector sınıfının doğru çalışıp çalışmadığını test eder.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Proje kök dizinini PATH'e ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detection.drowsiness_detector import DrowsinessDetector


class TestDrowsinessDetector(unittest.TestCase):
    """DrowsinessDetector için test sınıfı."""
    
    def setUp(self):
        """Her test öncesinde çalıştırılır."""
        # Test için konfigürasyon
        self.test_config = {
            'detection': {
                'ear_threshold': 0.25,
                'perclos': {
                    'window_size': 10,
                    'threshold': 20.0
                },
                'gaze': {
                    'deviation_threshold': 30.0
                }
            },
            'drowsiness': {
                'ear_closed_duration': 1.5,
                'history_duration': 3.0
            }
        }
        
        # DrowsinessDetector örneği oluştur
        self.detector = DrowsinessDetector(self.test_config)
    
    def test_initialization(self):
        """DrowsinessDetector'ın doğru şekilde başlatılıp başlatılmadığını test eder."""
        # Konfigürasyon değerlerinin doğru yüklendiğini kontrol et
        self.assertEqual(self.detector.ear_threshold, 0.25)
        self.assertEqual(self.detector.perclos_window_size, 10)
        self.assertEqual(self.detector.perclos_threshold, 20.0)
        self.assertEqual(self.detector.gaze_deviation_threshold, 30.0)
        self.assertEqual(self.detector.ear_closed_duration, 1.5)
        self.assertEqual(self.detector.history_duration, 3.0)
        
        # Başlangıç değerlerini kontrol et
        self.assertEqual(self.detector.drowsiness_level, 0.0)
        self.assertEqual(self.detector.is_eyes_closed, False)
        self.assertEqual(self.detector.perclos_value, 0.0)
        self.assertEqual(self.detector.drowsiness_state, "Uyanık")
    
    def test_calculate_ear(self):
        """EAR hesaplamasının doğru çalışıp çalışmadığını test eder."""
        # Test için yapay göz işaret noktaları
        # Açık göz (yüksek EAR)
        open_eye_landmarks = [
            [0, 0], # sol köşe
            [2, 5], # üst sol
            [5, 6], # üst orta
            [8, 0], # sağ köşe
            [5, -6], # alt orta
            [2, -5]  # alt sol
        ]
        
        # Kapalı göz (düşük EAR)
        closed_eye_landmarks = [
            [0, 0], # sol köşe
            [2, 1], # üst sol
            [5, 1], # üst orta
            [8, 0], # sağ köşe
            [5, -1], # alt orta
            [2, -1]  # alt sol
        ]
        
        # EAR değerlerini hesapla
        open_ear = self.detector.calculate_ear(open_eye_landmarks)
        closed_ear = self.detector.calculate_ear(closed_eye_landmarks)
        
        # Açık göz için EAR değeri kapalı göz için EAR değerinden büyük olmalı
        self.assertGreater(open_ear, closed_ear)
        
        # EAR değerleri makul aralıkta olmalı (0-1)
        self.assertLess(open_ear, 1.0)
        self.assertGreater(open_ear, 0.0)
        self.assertLess(closed_ear, 1.0)
        self.assertGreater(closed_ear, 0.0)
    
    def test_calculate_perclos(self):
        """PERCLOS hesaplamasının doğru çalışıp çalışmadığını test eder."""
        # EAR değerleri ekle (5 açık, 5 kapalı)
        for _ in range(5):
            self.detector.ear_values.append(0.3)  # Açık (> ear_threshold)
        
        for _ in range(5):
            self.detector.ear_values.append(0.2)  # Kapalı (< ear_threshold)
        
        # PERCLOS'u hesapla
        perclos = self.detector.calculate_perclos()
        
        # PERCLOS değeri %50 olmalı (10 kareden 5'i kapalı)
        self.assertEqual(perclos, 50.0)
    
    def test_calculate_gaze_deviation(self):
        """Bakış sapma açısı hesaplamasının doğru çalışıp çalışmadığını test eder."""
        # İdeal bakış (tam ileri)
        forward_gaze = np.array([0.0, 0.0, -1.0])
        forward_deviation = self.detector.calculate_gaze_deviation(forward_gaze)
        
        # İdeal bakış için sapma 0 olmalı
        self.assertAlmostEqual(forward_deviation, 0.0)
        
        # 45 derece sağa bakış
        right_gaze = np.array([np.sin(np.pi/4), 0.0, -np.cos(np.pi/4)])
        right_deviation = self.detector.calculate_gaze_deviation(right_gaze)
        
        # 45 derece sapma beklenir
        self.assertAlmostEqual(right_deviation, 45.0, delta=0.01)
        
        # 90 derece yukarı bakış
        up_gaze = np.array([0.0, -1.0, 0.0])
        up_deviation = self.detector.calculate_gaze_deviation(up_gaze)
        
        # 90 derece sapma beklenir
        self.assertAlmostEqual(up_deviation, 90.0, delta=0.01)
    
    def test_calculate_head_deviation(self):
        """Baş sapma açısı hesaplamasının doğru çalışıp çalışmadığını test eder."""
        # Nötr baş pozisyonu (tüm açılar 0)
        neutral_rotation = np.array([[0.0, 0.0, 0.0]])
        neutral_deviation = self.detector.calculate_head_deviation(neutral_rotation)
        
        # Nötr pozisyon için sapma 0 olmalı
        self.assertAlmostEqual(neutral_deviation, 0.0)
        
        # 30 derece sağa dönük baş
        right_rotation = np.array([[0.0, np.pi/6, 0.0]])  # 30 derece yaw
        right_deviation = self.detector.calculate_head_deviation(right_rotation)
        
        # 30 derece sapma beklenir
        self.assertAlmostEqual(right_deviation, 30.0, delta=0.01)
        
        # Karışık rotasyon
        mixed_rotation = np.array([[np.pi/12, np.pi/6, np.pi/8]])  # 15, 30, 22.5 derece
        mixed_deviation = self.detector.calculate_head_deviation(mixed_rotation)
        
        # En büyük açı (30 derece) dönmeli
        self.assertAlmostEqual(mixed_deviation, 30.0, delta=0.01)
    
    def test_compute_ear_from_landmarks(self):
        """Yüz işaretlerinden EAR hesaplamasının doğru çalışıp çalışmadığını test eder."""
        # Yapay yüz işaretleri
        landmarks = [[0, 0, 0] for _ in range(468)]  # 468 adet dummy landmark
        
        # Sol göz işaretleri
        left_eye_indices = [0, 1, 2, 3, 4, 5]
        for i, idx in enumerate(left_eye_indices):
            if i < 3:  # Üst kapak
                landmarks[idx] = [i * 2, 5, 0]
            else:  # Alt kapak
                landmarks[idx] = [(i - 3) * 2, -5, 0]
        
        # Sağ göz işaretleri
        right_eye_indices = [10, 11, 12, 13, 14, 15]
        for i, idx in enumerate(right_eye_indices):
            if i < 3:  # Üst kapak (kapalıya yakın)
                landmarks[idx] = [100 + i * 2, 1, 0]
            else:  # Alt kapak
                landmarks[idx] = [100 + (i - 3) * 2, -1, 0]
        
        # EAR değerlerini hesapla
        left_ear, right_ear = self.detector.compute_ear_from_landmarks(
            landmarks, left_eye_indices, right_eye_indices
        )
        
        # Sol göz daha açık olduğundan EAR daha yüksek olmalı
        self.assertGreater(left_ear, right_ear)
    
    def test_update_drowsiness(self):
        """Uykululuk durumu güncellemesinin doğru çalışıp çalışmadığını test eder."""
        # Zamanı mock'la
        with patch('time.time') as mock_time:
            # İlk zaman
            mock_time.return_value = 100.0
            
            # 1. Durum: Uyanık (yüksek EAR, normal bakış)
            result1 = self.detector.update(
                ear_left=0.35,
                ear_right=0.35,
                gaze_vector=np.array([0.0, 0.0, -1.0]),
                head_rotation=np.array([[0.0, 0.0, 0.0]])
            )
            
            # Uyanık durumunda olmalı
            self.assertEqual(self.detector.drowsiness_state, "Uyanık")
            self.assertLess(self.detector.drowsiness_level, 0.3)
            
            # Gözler kapalı değil
            self.assertFalse(self.detector.is_eyes_closed)
            
            # 2. Durum: Gözler kapanıyor (düşük EAR)
            mock_time.return_value = 101.0  # 1 saniye sonra
            
            result2 = self.detector.update(
                ear_left=0.2,
                ear_right=0.2,
                gaze_vector=np.array([0.0, 0.0, -1.0]),
                head_rotation=np.array([[0.0, 0.0, 0.0]])
            )
            
            # Gözler kapalı olmalı
            self.assertTrue(self.detector.is_eyes_closed)
            self.assertEqual(self.detector.eyes_closed_start_time, 101.0)
            
            # 3. Durum: Gözler bir süre kapalı kalıyor
            mock_time.return_value = 102.5  # 1.5 saniye daha (toplam 2.5s)
            
            result3 = self.detector.update(
                ear_left=0.15,
                ear_right=0.15,
                gaze_vector=np.array([0.0, 0.0, -1.0]),
                head_rotation=np.array([[0.0, 0.0, 0.0]])
            )
            
            # Uykululuk seviyesi artmalı
            self.assertGreater(self.detector.drowsiness_level, result2["drowsiness_level"])
            
            # Duruma göre uykululuk durumu değişmiş olmalı
            self.assertIn(self.detector.drowsiness_state, ["Yorgun", "Uykulu", "Tehlikeli"])
    
    def test_drowsiness_level_calculation(self):
        """Uykululuk seviyesi hesaplamasının doğru çalışıp çalışmadığını test eder."""
        # PERCLOS değerlerini ayarla
        for _ in range(8):
            self.detector.ear_values.append(0.3)  # Açık (> ear_threshold)
        
        for _ in range(2):
            self.detector.ear_values.append(0.2)  # Kapalı (< ear_threshold)
        
        # Hesaplanan PERCLOS %20 olmalı
        self.assertEqual(self.detector.calculate_perclos(), 20.0)
        
        # Zamanı mock'la
        with patch('time.time') as mock_time:
            # Başlangıç zamanı
            mock_time.return_value = 100.0
            
            # Başlangıç durumunu ayarla
            self.detector.last_update_time = 100.0
            
            # Gözler yeni kapandı
            mock_time.return_value = 101.0
            
            result = self.detector.update(
                ear_left=0.2,
                ear_right=0.2,
                gaze_vector=np.array([0.1, 0.0, -0.9]),  # Hafif sapma
                head_rotation=np.array([[0.0, np.pi/12, 0.0]])  # 15 derece dönüş
            )
            
            # Doğru göz kapalılık durumu
            self.assertTrue(self.detector.is_eyes_closed)
            
            # Hesaplanan uykululuk seviyesini kontrol et
            self.assertGreaterEqual(self.detector.drowsiness_level, 0.0)
            self.assertLessEqual(self.detector.drowsiness_level, 1.0)
            
            # Farklı durumlar oluştur ve seviyeyi gözlemle
            # 1. Uzun süre göz kapalı
            mock_time.return_value = 103.0  # 2 saniye kapalı
            
            result_long_closure = self.detector.update(
                ear_left=0.2,
                ear_right=0.2,
                gaze_vector=np.array([0.1, 0.0, -0.9]),
                head_rotation=np.array([[0.0, np.pi/12, 0.0]])
            )
            
            # 2. Yüksek PERCLOS
            for _ in range(10):
                self.detector.ear_values.append(0.2)  # Tüm değerler kapalı
            
            mock_time.return_value = 104.0
            
            result_high_perclos = self.detector.update(
                ear_left=0.2,
                ear_right=0.2,
                gaze_vector=np.array([0.1, 0.0, -0.9]),
                head_rotation=np.array([[0.0, np.pi/12, 0.0]])
            )
            
            # Uykululuk seviyesi artmalı
            self.assertGreater(
                result_high_perclos["drowsiness_level"],
                result_long_closure["drowsiness_level"]
            )
            
            # 3. Gözler açıldı
            mock_time.return_value = 105.0
            
            result_eyes_open = self.detector.update(
                ear_left=0.35,
                ear_right=0.35,
                gaze_vector=np.array([0.0, 0.0, -1.0]),
                head_rotation=np.array([[0.0, 0.0, 0.0]])
            )
            
            # Gözler açık olmalı
            self.assertFalse(self.detector.is_eyes_closed)
            self.assertIsNone(self.detector.eyes_closed_start_time)


if __name__ == '__main__':
    unittest.main() 