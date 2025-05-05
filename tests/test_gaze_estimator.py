#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaze Estimator testleri.

Bu script, GazeEstimator sınıfının ve ilgili model türlerinin (PyTorch & ONNX) 
doğru çalışıp çalışmadığını test eder.
"""

import os
import sys
import unittest
import numpy as np
import cv2
import torch
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Proje kök dizinini PATH'e ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Logları susturmak için (opsiyonel)
logging.basicConfig(level=logging.ERROR)

from src.detection.gaze_estimator import GazeEstimator, GazeResNet
from src.utils.model_loader import ONNXGazeModel, ModelLoader


class TestGazeEstimator(unittest.TestCase):
    """Gaze Estimator için test sınıfı."""
    
    @classmethod
    def setUpClass(cls):
        """Test sınıfı kurulumunda bir kez çalıştırılır."""
        # Test modelleri için klasörü oluştur
        cls.test_model_dir = os.path.join(project_root, 'models', 'test')
        os.makedirs(cls.test_model_dir, exist_ok=True)
        
        # Test görüntüsünü yükle
        test_image_path = os.path.join(project_root, 'tests', 'data', 'test_face.jpg')
        if os.path.exists(test_image_path):
            cls.test_image = cv2.imread(test_image_path)
        else:
            # Test görüntüsü yoksa, boş bir görüntü oluştur
            cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            print(f"Uyarı: Test görüntüsü bulunamadı: {test_image_path}")
    
    @classmethod
    def tearDownClass(cls):
        """Test sınıfı sonlandığında bir kez çalıştırılır."""
        # Test model klasörünü temizle
        if os.path.exists(cls.test_model_dir):
            shutil.rmtree(cls.test_model_dir)
    
    def setUp(self):
        """
        Her test öncesinde çalışacak hazırlık fonksiyonu
        """
        try:
            # Gaze Estimator'ı oluşturmayı deneyelim, ancak model bulunamazsa bu testi atlayacağız
            self.gaze_estimator = GazeEstimator()
            self.has_model = True
        except FileNotFoundError:
            self.has_model = False
    
    def test_gaze_estimator_initialization(self):
        """GazeEstimator'ın doğru şekilde başlatılıp başlatılmadığını test eder."""
        # Mock model yüklemesi
        with patch('src.detection.gaze_estimator.GazeEstimator._load_model') as mock_load_model:
            # Sahte bir model döndür
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            
            # GazeEstimator'ı başlat
            estimator = GazeEstimator()
            
            # _load_model'in çağrıldığını doğrula
            self.assertTrue(mock_load_model.called)
            
            # Modelin atandığını doğrula
            self.assertEqual(estimator.model, mock_model)
    
    def test_model_loader_pytorch(self):
        """PyTorch modelinin doğru şekilde yüklenip yüklenmediğini test eder."""
        # Bu test sadece model_loader.py için geçerli, burada GazeEstimator test ediyoruz
        # Bu yüzden test içeriğini GazeEstimator._load_model ile değiştiriyoruz
        
        # Geçici model yolu
        test_model_path = os.path.join(self.test_model_dir, 'eth_xgaze_model.pth')
        
        # Gerçek model dosyası olmadan mock ile test edeceğiz
        with patch('torch.load') as mock_torch_load:
            # Sahte model durumu oluştur
            mock_state_dict = {
                'gaze_fc.0.weight': torch.randn(2, 2048),
                'gaze_fc.0.bias': torch.randn(2)
            }
            mock_torch_load.return_value = {'model_state': mock_state_dict}
            
            # Modeli yüklemeyi dene
            with patch('src.detection.gaze_estimator.GazeResNet') as mock_gaze_resnet:
                mock_model = MagicMock()
                mock_gaze_resnet.return_value = mock_model
                
                # Modeli yükle
                estimator = GazeEstimator(model_path=test_model_path)
                
                # torch.load'un çağrıldığını doğrula
                mock_torch_load.assert_called_once()
    
    def test_model_loader_onnx(self):
        """ONNX modelinin doğru şekilde yüklenip yüklenmediğini test eder."""
        # GazeEstimator şu anda doğrudan ONNX kullanmıyor, bu yüzden testi atla
        self.skipTest("GazeEstimator doğrudan ONNX modellerini desteklemiyor.")
    
    def test_preprocess_eye_image(self):
        """Göz görüntüsü ön işleminin doğru çalışıp çalışmadığını test eder."""
        # GazeEstimator transform özelliğini kullanarak ön işleme yapıyor
        estimator = GazeEstimator()
        
        # Test için bir göz görüntüsü oluştur
        eye_image = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
        
        # Ön işleme uygula (transform özelliğini kullanarak)
        preprocessed = estimator.transform(eye_image)
        
        # Çıktı tipini kontrol et
        self.assertIsInstance(preprocessed, torch.Tensor)
        
        # Çıktı boyutunu kontrol et (batch, channels, height, width)
        self.assertEqual(preprocessed.shape[0], 3)  # RGB channels 
    
    def test_postprocess_gaze(self):
        """GazeEstimator.estimate_gaze çıktı formatını test eder."""
        # GazeEstimator oluştur
        with patch('src.detection.gaze_estimator.GazeEstimator._load_model') as mock_load_model:
            mock_model = MagicMock()
            mock_model.return_value = torch.tensor([[0.1, 0.2]])
            mock_load_model.return_value = mock_model
            
            estimator = GazeEstimator()
            
            # Test için bir normalize edilmiş yüz görüntüsü oluştur
            face_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Model çıktısını ayarla
            estimator.model = MagicMock()
            estimator.model.return_value = torch.tensor([[0.1, 0.2]])
            
            # estimate_gaze metodunu çağır
            with patch.object(estimator, 'transform') as mock_transform:
                mock_transform.return_value = torch.zeros((3, 224, 224))
                
                success, gaze_angles = estimator.estimate_gaze(face_img)
                
                # Çıktıları kontrol et
                self.assertTrue(success)
                self.assertIsInstance(gaze_angles, tuple)
                self.assertEqual(len(gaze_angles), 2)  # (yaw, pitch)
    
    @patch('cv2.warpPerspective')
    @patch('cv2.solvePnP')
    def test_normalize_face(self, mock_solve_pnp, mock_warp_perspective):
        """Yüz normalizasyonunun doğru çalışıp çalışmadığını test eder."""
        # Mock değerleri ayarla
        rotation_vec = np.array([[0.1], [0.2], [0.3]])
        translation_vec = np.array([[10], [20], [30]])
        mock_solve_pnp.return_value = (True, rotation_vec, translation_vec)
        
        normalized_face = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_warp_perspective.return_value = normalized_face
        
        # GazeEstimator'ı mock model ile başlat
        with patch('src.detection.gaze_estimator.GazeEstimator._load_model'):
            estimator = GazeEstimator()
            
            # Test için yüz işaretleri oluştur (6 nokta, 2 koordinat)
            landmarks = np.array([
                [100, 100],  # Sol göz sol köşesi
                [140, 100],  # Sol göz sağ köşesi
                [180, 100],  # Sağ göz sol köşesi
                [220, 100],  # Sağ göz sağ köşesi
                [160, 140],  # Burun sol noktası
                [180, 140]   # Burun sağ noktası
            ])
            
            # _normalize_face fonksiyonunu çağır
            normalized_img = estimator._normalize_face(self.test_image, landmarks, rotation_vec, translation_vec)
            
            # Çıktıları kontrol et
            self.assertIsInstance(normalized_img, np.ndarray)
            self.assertEqual(normalized_img.shape[:2], (224, 224))
            
            # solvePnP'nin çağrıldığını doğrula
            self.assertFalse(mock_solve_pnp.called)  # Bu fonksiyon içinde _normalize_face zaten rvec/tvec alıyor
            
            # warpPerspective'in çağrıldığını doğrula
            self.assertTrue(mock_warp_perspective.called)
    
    @patch('src.detection.gaze_estimator.GazeEstimator.get_face_from_landmarks')
    @patch('src.detection.gaze_estimator.GazeEstimator.estimate_gaze')
    def test_detect(self, mock_estimate_gaze, mock_get_face):
        """detect() metodunun doğru çalışıp çalışmadığını test eder."""
        # Mock değerleri ayarla
        normalized_face = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_get_face.return_value = normalized_face
        
        gaze_angles = (10.0, 20.0)  # (yaw, pitch)
        mock_estimate_gaze.return_value = (True, gaze_angles)
        
        # GazeEstimator'ı mock model ile başlat
        with patch('src.detection.gaze_estimator.GazeEstimator._load_model'):
            estimator = GazeEstimator()
            
            # Sahte landmarks sözlüğü oluştur
            landmarks = {'all_landmarks': [[100, 100, 0]] * 468}
            
            # detect() metodunu çağır
            success, angles, face_img = estimator.detect(self.test_image, landmarks)
            
            # Çıktıları kontrol et
            self.assertTrue(success)
            self.assertEqual(angles, gaze_angles)
            # NumPy dizileri için np.array_equal kullan
            self.assertTrue(np.array_equal(face_img, normalized_face))
            
            # get_face_from_landmarks'in çağrıldığını doğrula
            mock_get_face.assert_called_once()
            
            # estimate_gaze'in çağrıldığını doğrula
            mock_estimate_gaze.assert_called_once()
    
    def test_end_to_end_pytorch(self):
        """PyTorch modeli ile uçtan uca test (basitleştirilmiş)."""
        # Bu test gerçek bir model dosyası gerektirir, bu yüzden modeli mock'luyoruz
        with patch('src.detection.gaze_estimator.GazeEstimator._load_model') as mock_load_model:
            # Sahte model
            model = MagicMock()
            model.return_value = torch.tensor([[0.1, 0.2]])
            mock_load_model.return_value = model
            
            # GazeEstimator oluştur
            estimator = GazeEstimator()
            
            # Yüz çıkarma ve bakış tahmini kısmını mock'la
            with patch.object(estimator, 'get_face_from_landmarks') as mock_get_face:
                # Normalize edilmiş yüz mock'u
                normalized_face = np.zeros((224, 224, 3), dtype=np.uint8)
                mock_get_face.return_value = normalized_face
                
                # Bakış tahmini kısmını mock'la
                with patch.object(estimator, 'estimate_gaze') as mock_estimate_gaze:
                    mock_estimate_gaze.return_value = (True, (10.0, 20.0))
                    
                    # Sahte landmark sözlüğü
                    landmarks = {
                        'all_landmarks': [[100, 100, 0]] * 468,
                        'left_eye': np.array([[100, 100]]),
                        'right_eye': np.array([[200, 100]])
                    }
                    
                    # detect() metodunu çağır
                    success, angles, face_img = estimator.detect(self.test_image, landmarks)
                    
                    # Çıktıları kontrol et
                    self.assertTrue(success)
                    self.assertEqual(angles, (10.0, 20.0))
                    # NumPy dizileri için np.array_equal kullan
                    self.assertTrue(np.array_equal(face_img, normalized_face))
    
    def test_end_to_end_onnx(self):
        """ONNX modeli ile uçtan uca test (basitleştirilmiş)."""
        # GazeEstimator doğrudan ONNX desteği sağlamadığı için bu testi atlıyoruz
        self.skipTest("GazeEstimator doğrudan ONNX modellerini desteklemiyor.")
    
    def test_simplified_face_model(self):
        """
        Basitleştirilmiş yüz modelinin doğru şekilde oluşturulup oluşturulmadığını test eder
        """
        # Basitleştirilmiş yüz modelini elde etmek için doğrudan fonksiyonu çağıralım
        # GazeEstimator'ı oluşturmasak bile bu fonksiyon test edilebilir
        model = GazeResNet()
        self.assertIsNotNone(model, "GazeResNet modeli oluşturulamadı")
        
        # Basitleştirilmiş yüz modeli fonksiyonunu test edelim
        # (GazeEstimator oluşturmadan da test edilebilir)
        face_model = GazeEstimator._get_simplified_face_model(None)
        
        # Modelin doğru şekle sahip olduğunu kontrol edelim
        self.assertEqual(face_model.shape, (6, 3), "Basitleştirilmiş yüz modeli yanlış şekle sahip")
        
        # Modelin içeriğini hızlıca kontrol edelim
        # Sol göz sol köşesi
        self.assertEqual(face_model[0, 0], -35.0)
        self.assertEqual(face_model[0, 1], -50.0)
        self.assertEqual(face_model[0, 2], -30.0)
    
    def test_face_model_loading(self):
        """
        Yüz modelinin doğru yüklenip yüklenmediğini test eder
        """
        if not self.has_model:
            self.skipTest("Model dosyası bulunamadı, bu test atlanıyor")
        
        # Yüz modelinin boyutunu kontrol et
        self.assertEqual(self.gaze_estimator.face_model.shape, (6, 3), 
                        "Yüz modeli yanlış boyuta sahip")
    
    def test_head_pose_estimation(self):
        """
        Baş pozisyonu tahmini işlevini test eder
        """
        if not self.has_model:
            self.skipTest("Model dosyası bulunamadı, bu test atlanıyor")
        
        # Basit bir örnek landmark dizisi oluştur
        landmarks = np.array([
            [100, 100],  # Sol göz sol köşesi
            [140, 100],  # Sol göz sağ köşesi
            [180, 100],  # Sağ göz sol köşesi
            [220, 100],  # Sağ göz sağ köşesi
            [160, 140],  # Burun sol noktası
            [180, 140]   # Burun sağ noktası
        ])
        
        # Baş pozisyonu tahmini yapabilmemiz gerekir
        try:
            rvec, tvec = self.gaze_estimator._estimate_head_pose(landmarks)
            
            # Döndürme vektörünün 3x1 boyutunda olduğunu kontrol et
            self.assertEqual(rvec.shape, (3, 1), "Döndürme vektörü yanlış boyuta sahip")
            
            # Dönüşüm vektörünün 3x1 boyutunda olduğunu kontrol et
            self.assertEqual(tvec.shape, (3, 1), "Dönüşüm vektörü yanlış boyuta sahip")
        except Exception as e:
            self.fail(f"Baş pozisyonu tahmini beklenmeyen hata verdi: {e}")
    
    def test_normalize_face(self):
        """
        Yüz normalizasyonu işlevini test eder
        """
        if not self.has_model:
            self.skipTest("Model dosyası bulunamadı, bu test atlanıyor")
        
        # Basit bir test görüntüsü oluştur (300x300 siyah kare)
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Görüntüye basit bir yüz çiz (test amaçlı)
        cv2.circle(test_image, (150, 100), 80, (255, 255, 255), -1)  # Yüz
        cv2.circle(test_image, (120, 80), 10, (0, 0, 0), -1)  # Sol göz
        cv2.circle(test_image, (180, 80), 10, (0, 0, 0), -1)  # Sağ göz
        cv2.ellipse(test_image, (150, 120), (20, 10), 0, 0, 180, (0, 0, 0), -1)  # Ağız
        
        # Basit bir örnek landmark dizisi oluştur
        landmarks = np.array([
            [100, 80],   # Sol göz sol köşesi
            [140, 80],   # Sol göz sağ köşesi
            [160, 80],   # Sağ göz sol köşesi
            [200, 80],   # Sağ göz sağ köşesi
            [130, 120],  # Burun sol noktası
            [170, 120]   # Burun sağ noktası
        ])
        
        # Önce baş pozisyonu tahmini yap
        rvec, tvec = self.gaze_estimator._estimate_head_pose(landmarks)
        
        # Şimdi normalizasyon yap
        try:
            normalized_face = self.gaze_estimator._normalize_face(test_image, landmarks, rvec, tvec)
            
            # Normalize edilmiş yüz 224x224 boyutunda olmalı
            self.assertEqual(normalized_face.shape, (224, 224, 3), 
                           "Normalize edilmiş yüz yanlış boyuta sahip")
        except Exception as e:
            self.fail(f"Yüz normalizasyonu beklenmeyen hata verdi: {e}")
    
    def test_gaze_estimation_format(self):
        """
        Bakış tahmini çıktı formatını test eder
        (gerçek bir kamera veya görüntü olmadan)
        """
        if not self.has_model:
            self.skipTest("Model dosyası bulunamadı, bu test atlanıyor")
        
        # Sahte bir normalize edilmiş yüz görüntüsü oluştur
        # Bu gerçek bir görüntü olmadığı için tahmin doğru olmayacak
        # Ama en azından fonksiyonun çalışıp çalışmadığını test edebiliriz
        test_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Bakış tahminini test et
        success, gaze_angles = self.gaze_estimator.estimate_gaze(test_face)
        
        # Başarısız olsa bile doğru türde değer döndürüyor mu kontrol et
        self.assertIsInstance(success, bool, "Başarı değeri Boolean türünde değil")
        self.assertIsInstance(gaze_angles, tuple, "Bakış açıları bir tuple değil")
        self.assertEqual(len(gaze_angles), 2, "Bakış açıları tuple'ı 2 eleman içermiyor")
        
        # Açıların sayısal değerler olduğunu kontrol et
        self.assertIsInstance(gaze_angles[0], float, "Yaw açısı float türünde değil")
        self.assertIsInstance(gaze_angles[1], float, "Pitch açısı float türünde değil")


if __name__ == '__main__':
    unittest.main()
