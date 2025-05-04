#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETH-XGaze Model Dönüştürücü

Bu script, ETH-XGaze reposundaki model dosyalarını (epoch_X_ckpt.pth.tar formatı) 
bizim sistemimizin kullandığı formata (eth_xgaze_model.pth veya eth_xgaze_model.onnx) 
dönüştürür.

Kullanım:
    python convert_ethxgaze_model.py <kaynak_model> <hedef_model> [--export_onnx]
    
Örnek:
    python convert_ethxgaze_model.py ../ETH-XGaze/ckpt/epoch_24_ckpt.pth.tar ../models/eth_xgaze_model.pth
    python convert_ethxgaze_model.py ../ETH-XGaze/ckpt/epoch_24_ckpt.pth.tar ../models/eth_xgaze_model.onnx --export_onnx
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Proje kök dizinini ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.model_loader import GazeResNet


def parse_args():
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(description="ETH-XGaze Model Dönüştürücü")
    parser.add_argument("source_model", type=str, help="Kaynak model dosyasının yolu (ETH-XGaze formatı)")
    parser.add_argument("target_model", type=str, help="Hedef model dosyasının yolu")
    parser.add_argument("--export_onnx", action="store_true", help="ONNX formatına dışa aktar")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Modelin dönüştürüleceği cihaz ('cpu' veya 'cuda')")
    return parser.parse_args()


def create_dummy_input(device='cpu'):
    """ONNX dönüşümü için gerekli örnek bir girdi oluşturur."""
    return torch.randn(1, 3, 224, 224, device=device)


def convert_ethxgaze_model(source_path, target_path, export_onnx=False, device='cpu'):
    """
    ETH-XGaze model formatını dönüştürür
    
    Args:
        source_path: Kaynak model dosyasının yolu
        target_path: Hedef model dosyasının yolu
        export_onnx: ONNX formatına dışa aktarma seçeneği
        device: Modelin dönüştürüleceği cihaz ('cpu' veya 'cuda')
    """
    print(f"Kaynak model: {source_path}")
    print(f"Hedef model: {target_path}")
    
    # Hedef klasörün varlığını kontrol et
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    
    # Cihazı seç
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    try:
        # Kaynak modeli yükle
        print("Kaynak ETH-XGaze modelini yükleme...")
        checkpoint = torch.load(source_path, map_location=device)
        
        # Model yapısını oluştur
        print("Model yapısını oluşturma...")
        model = GazeResNet()
        
        # ETH-XGaze formatını kontrol et
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            print("ETH-XGaze checkpoint formatı (model_state anahtarı ile) algılandı")
            model.load_state_dict(checkpoint['model_state'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            print("Standart checkpoint formatı (state_dict anahtarı ile) algılandı")
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Bilinmeyen format, doğrudan state_dict olarak yüklemeye çalışılıyor...")
            model.load_state_dict(checkpoint)
        
        # Modeli değerlendirme moduna al
        model.eval()
        model = model.to(device)
        
        # ONNX dönüşümünü yap
        if export_onnx or target_path.endswith('.onnx'):
            if not target_path.endswith('.onnx'):
                target_path = target_path + '.onnx'
            
            print(f"Modeli ONNX formatına dönüştürme: {target_path}")
            
            try:
                # ONNX dönüşümü için örnek girdi oluştur
                dummy_input = create_dummy_input(device)
                
                # Dönüşüm için gerekli argümanları ayarla
                input_names = ["input"]
                output_names = ["output"]
                dynamic_axes = {
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
                
                # ONNX dışa aktarma
                torch.onnx.export(
                    model,                     # Dışa aktarılacak model
                    dummy_input,               # Örnek girdi
                    target_path,               # Çıktı dosyası
                    export_params=True,        # Model parametrelerini de dışa aktar
                    opset_version=11,          # ONNX opset versiyonu
                    do_constant_folding=True,  # Sabit katlamasını etkinleştir
                    input_names=input_names,   # Girdi isimleri
                    output_names=output_names, # Çıktı isimleri
                    dynamic_axes=dynamic_axes  # Dinamik eksenler
                )
                
                print("ONNX dönüşümü başarılı!")
                
                # onnxruntime ile modeli test et (opsiyonel)
                try:
                    import onnxruntime as ort
                    print("ONNX modelini doğrulama...")
                    
                    # ONNX Runtime oturumu oluştur
                    ort_session = ort.InferenceSession(target_path)
                    
                    # ONNX Runtime girişi için numpy dizisi oluştur
                    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
                    
                    # ONNX Runtime çıktısını al
                    ort_outputs = ort_session.run(None, ort_inputs)
                    
                    print("ONNX modeli doğrulandı ve çalışıyor!")
                except ImportError:
                    print("onnxruntime yüklü değil, model doğrulanamadı.")
                except Exception as e:
                    print(f"ONNX model doğrulaması başarısız: {e}")
            
            except Exception as e:
                print(f"ONNX dönüşümü başarısız: {e}")
                print("PyTorch modeli olarak kaydetmeye çalışılıyor...")
                
                # PyTorch modeli olarak kaydet
                torch.save(model.state_dict(), target_path.replace('.onnx', '.pth'))
                print(f"Model PyTorch formatında kaydedildi: {target_path.replace('.onnx', '.pth')}")
        else:
            # PyTorch modeli olarak kaydet
            if not target_path.endswith('.pth'):
                target_path = target_path + '.pth'
                
            print(f"Modeli PyTorch formatında kaydetme: {target_path}")
            torch.save(model.state_dict(), target_path)
            print("Model başarıyla kaydedildi!")
        
        return True
    
    except Exception as e:
        print(f"Model dönüşümü sırasında hata oluştu: {e}")
        return False


def main():
    args = parse_args()
    
    # Modeli dönüştür
    success = convert_ethxgaze_model(
        args.source_model, 
        args.target_model, 
        args.export_onnx,
        args.device
    )
    
    if success:
        print("Model dönüşümü başarıyla tamamlandı!")
    else:
        print("Model dönüşümü başarısız oldu.")
        sys.exit(1)


if __name__ == "__main__":
    main() 