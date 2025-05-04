#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Loader testleri.

Bu script, ModelLoader sınıfının ve ETH-XGaze modelinin farklı formatlarda 
(PyTorch ve ONNX) yüklenmesini test eder.
"""

import os
import sys
import unittest
import torch
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Proje kök dizinini PATH'e ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.model_loader import ModelLoader, GazeResNet, ONNXGazeModel


class TestModelLoader(unittest.TestCase):
    """Model Loader için test sınıfı."""
    
    @classmethod
    def setUpClass(cls):
        """Test sınıfı kurulumunda bir kez çalıştırılır."""
        # Test modelleri için klasörü oluştur
        cls.test_model_dir = os.path.join(project_root, 'models', 'test')
        os.makedirs(cls.test_model_dir, exist_ok=True)
        
        # Test modelini (PyTorch) oluştur
        model = GazeResNet()
        # Model'i eval moduna al
        model.eval()
        cls.pth_path = os.path.join(cls.test_model_dir, 'eth_xgaze_model.pth')
        torch.save(model.state_dict(), cls.pth_path)
        
        # Test modelini (ONNX) için sahte dosya oluştur
        cls.onnx_path = os.path.join(cls.test_model_dir, 'eth_xgaze_model.onnx')
        with open(cls.onnx_path, 'w') as f:
            f.write('mock onnx model')
    
    @classmethod
    def tearDownClass(cls):
        """Test sınıfı sonlandığında bir kez çalıştırılır."""
        # Test model klasörünü temizle
        if os.path.exists(cls.test_model_dir):
            shutil.rmtree(cls.test_model_dir)
    
    def test_model_loader_initialization(self):
        """ModelLoader'ın doğru şekilde başlatılıp başlatılmadığını test eder."""
        loader = ModelLoader()
        
        # base_model_dir'in doğru olduğunu kontrol et
        expected_path = os.path.join(project_root, 'models')
        self.assertEqual(loader.base_model_dir, expected_path)
        
        # config'in yüklendiğini kontrol et
        self.assertIsInstance(loader.config, dict)
    
    @patch('yaml.safe_load')
    def test_load_config(self, mock_yaml_load):
        """_load_config metodunun doğru çalışıp çalışmadığını test eder."""
        # Mock config dosyası içeriği
        mock_config = {
            'camera': {
                'matrix': [1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0],
                'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
        mock_yaml_load.return_value = mock_config
        
        # ModelLoader oluştur ve config'i yükle
        loader = ModelLoader()
        config = loader._load_config()
        
        # Config'in mock değeri ile eşleştiğini kontrol et
        self.assertEqual(config, mock_config)
        
        # YAML yükleme metodunun çağrıldığını doğrula
        self.assertTrue(mock_yaml_load.called)
    
    def test_gaze_resnet_initialization(self):
        """GazeResNet modelinin doğru şekilde başlatılıp başlatılmadığını test eder."""
        # GazeResNet modeli oluştur
        model = GazeResNet()
        
        # Modelin doğru tipte olduğunu kontrol et
        self.assertIsInstance(model, torch.nn.Module)
        
        # Backbone'un var olduğunu kontrol et
        self.assertTrue(hasattr(model, 'backbone'))
        
        # FC katmanının 2 çıktılı olduğunu kontrol et (pitch, yaw)
        self.assertEqual(model.backbone.fc.out_features, 2)
    
    def test_gaze_resnet_forward(self):
        """GazeResNet'in forward metodunun doğru çalışıp çalışmadığını test eder."""
        # GazeResNet modeli oluştur
        model = GazeResNet()
        # Modeli eval moduna al
        model.eval()
        
        # Test girişi oluştur (batch_size=1, channels=3, height=224, width=224)
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Forward metodunu çağır
        with torch.no_grad():
            output = model(input_tensor)
        
        # Çıktı şeklini kontrol et (batch_size=1, output_dim=2)
        self.assertEqual(output.shape, (1, 2))
    
    def test_load_pytorch_model(self):
        """PyTorch modelinin doğru şekilde yüklenip yüklenmediğini test eder."""
        # ModelLoader oluştur
        loader = ModelLoader()
        
        # Orjinal base_model_dir değerini kaydet
        original_base_dir = loader.base_model_dir
        
        try:
            # Test dizinini kullanmak için base_model_dir'i değiştir
            loader.base_model_dir = self.test_model_dir
            
            # Mock PyTorch modelini oluştur - Gerçek torch.nn.Module'den türet
            with patch('src.utils.model_loader.GazeResNet') as mock_gaze_resnet:
                # Gerçek torch.nn.Module kullanarak mock modeli oluştur
                real_model = torch.nn.Sequential(torch.nn.Linear(10, 2))
                real_model.eval()  # Eval moduna al
                
                # to metodu için mock ekle
                to_mock = MagicMock(return_value=real_model)
                real_model.to = to_mock
                
                mock_gaze_resnet.return_value = real_model
                
                # PyTorch modelini yükle
                model = loader._load_pytorch_model(self.pth_path, 'cpu')
                
                # Modelin doğru tipte olduğunu kontrol et
                self.assertIsInstance(model, torch.nn.Module)
                
                # Modelin eval modunda olduğunu kontrol et
                self.assertFalse(model.training)
        finally:
            # Orijinal dizini geri yükle
            loader.base_model_dir = original_base_dir
    
    def test_load_nonexistent_pytorch_model(self):
        """Var olmayan PyTorch modeli için fallback mekanizmasını test eder."""
        # Var olmayan model yolu
        nonexistent_path = os.path.join(self.test_model_dir, 'nonexistent.pth')
        
        # ModelLoader oluştur
        loader = ModelLoader()
        
        # Orjinal base_model_dir değerini kaydet
        original_base_dir = loader.base_model_dir
        
        try:
            # Test dizinini kullanmak için base_model_dir'i değiştir
            loader.base_model_dir = self.test_model_dir
            
            # Mock PyTorch modelini oluştur - Gerçek torch.nn.Module'den türet
            with patch('src.utils.model_loader.GazeResNet') as mock_gaze_resnet:
                # Gerçek torch.nn.Module kullanarak mock modeli oluştur
                real_model = torch.nn.Sequential(torch.nn.Linear(10, 2))
                real_model.eval()  # Eval moduna al
                
                # to metodu için mock ekle
                to_mock = MagicMock(return_value=real_model)
                real_model.to = to_mock
                
                mock_gaze_resnet.return_value = real_model
                
                # Var olmayan PyTorch modelini yüklemeyi dene
                model = loader._load_pytorch_model(nonexistent_path, 'cpu')
                
                # Fallback modelin oluşturulduğunu kontrol et
                self.assertIsInstance(model, torch.nn.Module)
                
                # Modelin eval modunda olduğunu kontrol et
                self.assertFalse(model.training)
        finally:
            # Orijinal dizini geri yükle
            loader.base_model_dir = original_base_dir
    
    def test_onnx_gaze_model(self):
        """ONNXGazeModel sınıfını test eder."""
        try:
            import onnxruntime
            has_onnxruntime = True
        except ImportError:
            has_onnxruntime = False
        
        if not has_onnxruntime:
            self.skipTest("onnxruntime paketi yüklü değil, ONNX testi atlanıyor")
            return
        
        # Mock bir ONNX InferenceSession oluştur
        with patch('onnxruntime.InferenceSession') as mock_session, \
             patch('onnxruntime.SessionOptions'), \
             patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
            
            # Mock session için gerekli özellikleri ayarla
            input_mock = MagicMock()
            input_mock.name = 'input'
            input_mock.shape = [1, 3, 224, 224]
            
            output_mock = MagicMock()
            output_mock.name = 'output'
            
            mock_session.return_value.get_inputs.return_value = [input_mock]
            mock_session.return_value.get_outputs.return_value = [output_mock]
            mock_session.return_value.run.return_value = [np.array([[0.1, 0.2]])]
            
            # ONNXGazeModel oluştur
            onnx_model = ONNXGazeModel(self.onnx_path, 'cpu')
            
            # Modelin özelliklerini kontrol et
            self.assertEqual(onnx_model.input_name, 'input')
            self.assertEqual(onnx_model.output_name, 'output')
            self.assertEqual(list(onnx_model.input_shape), [1, 3, 224, 224])
            
            # Modelin eval metodunu test et
            self.assertEqual(onnx_model.eval(), onnx_model)
            
            # Forward metodu (__call__) test et
            input_tensor = torch.randn(1, 3, 224, 224)
            output = onnx_model(input_tensor)
            
            # Çıktı tipini kontrol et
            self.assertIsInstance(output, torch.Tensor)
            
            # Session run'ın çağrıldığını doğrula
            mock_session.return_value.run.assert_called_once()
    
    def test_load_eth_xgaze_model_onnx_first(self):
        """load_eth_xgaze_model metodunun önce ONNX modelini yüklemeyi denediğini test eder."""
        try:
            import onnxruntime
            has_onnxruntime = True
        except ImportError:
            has_onnxruntime = False
        
        if not has_onnxruntime:
            self.skipTest("onnxruntime paketi yüklü değil, ONNX testi atlanıyor")
            return
        
        # ModelLoader oluştur
        loader = ModelLoader()
        
        # Orjinal base_model_dir değerini kaydet
        original_base_dir = loader.base_model_dir
        
        try:
            # Test dizinini kullanmak için base_model_dir'i değiştir
            loader.base_model_dir = self.test_model_dir
            
            # ONNXGazeModel sınıfını mock'la
            with patch('src.utils.model_loader.ONNXGazeModel') as mock_onnx_model:
                mock_onnx_instance = MagicMock()
                mock_onnx_model.return_value = mock_onnx_instance
                
                # ETH-XGaze modelini yükle
                model = loader.load_eth_xgaze_model('cpu')
                
                # ONNXGazeModel'in oluşturulduğunu doğrula
                mock_onnx_model.assert_called_once_with(self.onnx_path, 'cpu')
                
                # Dönülen modelin ONNX model olduğunu kontrol et
                self.assertEqual(model, mock_onnx_instance)
        finally:
            # Orijinal dizini geri yükle
            loader.base_model_dir = original_base_dir
    
    def test_load_eth_xgaze_model_fallback_to_pytorch(self):
        """ONNX model yoksa PyTorch modeline geçiş yapıldığını test eder."""
        # ModelLoader oluştur
        loader = ModelLoader()
        
        # Orjinal base_model_dir değerini kaydet
        original_base_dir = loader.base_model_dir
        
        try:
            # Test dizinini kullanmak için base_model_dir'i değiştir
            loader.base_model_dir = self.test_model_dir
            
            # Geçici dosya yolları
            temp_onnx_path = self.onnx_path + '.bak'
            
            # Eğer .bak dosya hala varsa silinmesini sağla
            if os.path.exists(temp_onnx_path):
                os.remove(temp_onnx_path)
                
            # ONNX dosyasını geçici olarak yeniden adlandır
            os.rename(self.onnx_path, temp_onnx_path)
            
            # GazeResNet sınıfının yüklenmesini mock'la
            with patch('src.utils.model_loader.GazeResNet') as mock_gaze_resnet:
                mock_model = MagicMock(spec=torch.nn.Module)
                mock_model.training = False
                mock_gaze_resnet.return_value = mock_model
                
                # PyTorch modelinin yüklenmesini mock'la
                with patch.object(loader, '_load_pytorch_model') as mock_load_pytorch:
                    mock_load_pytorch.return_value = mock_model
                    
                    # ETH-XGaze modelini yükle
                    model = loader.load_eth_xgaze_model('cpu')
                    
                    # PyTorch modelinin yüklenmesinin denendiğini doğrula
                    expected_path = os.path.join(self.test_model_dir, 'eth_xgaze_model.pth')
                    mock_load_pytorch.assert_called_once_with(expected_path, 'cpu')
            
            # ONNX dosyasını geri al
            os.rename(temp_onnx_path, self.onnx_path)
        finally:
            # Orijinal dizini geri yükle
            loader.base_model_dir = original_base_dir
            
            # Eğer test başarısız olursa dosyayı geri almayı dene
            if os.path.exists(temp_onnx_path):
                os.rename(temp_onnx_path, self.onnx_path)
    
    def test_onnx_import_error_fallback(self):
        """onnxruntime import edilemezse PyTorch'a geçiş yapıldığını test eder."""
        # ModelLoader oluştur
        loader = ModelLoader()
        
        # Orjinal base_model_dir değerini kaydet
        original_base_dir = loader.base_model_dir
        
        try:
            # Test dizinini kullanmak için base_model_dir'i değiştir
            loader.base_model_dir = self.test_model_dir
            
            # onnxruntime import edilirken ImportError fırlatmasını sağla
            with patch('src.utils.model_loader.ONNXGazeModel') as mock_onnx_model, \
                 patch('builtins.__import__', side_effect=lambda name, *args: 
                    __import__(name, *args) if name != 'onnxruntime' else exec('raise ImportError("Mock ImportError")')):
                
                # PyTorch modelinin yüklenmesini mock'la
                with patch.object(loader, '_load_pytorch_model') as mock_load_pytorch:
                    mock_model = MagicMock(spec=torch.nn.Module)
                    mock_model.training = False
                    mock_load_pytorch.return_value = mock_model
                    
                    # ETH-XGaze modelini yükle
                    model = loader.load_eth_xgaze_model('cpu')
                    
                    # PyTorch modelinin yüklenmesinin denendiğini doğrula
                    expected_path = os.path.join(self.test_model_dir, 'eth_xgaze_model.pth')
                    mock_load_pytorch.assert_called_once_with(expected_path, 'cpu')
                
                # ONNXGazeModel'in hiç çağrılmadığını doğrula
                mock_onnx_model.assert_not_called()
        finally:
            # Orijinal dizini geri yükle
            loader.base_model_dir = original_base_dir


if __name__ == '__main__':
    unittest.main() 