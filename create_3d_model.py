#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Yüz Modeli Oluşturucu

Bu betik, baş pozisyonu tahmini için kullanılan 3D yüz modelini oluşturup
'models/headpose_3d_model.npy' dosyasına kaydeder.
"""

import os
import numpy as np

def create_3d_face_model():
    """
    Baş pozisyonu tahmini için 3D yüz modeli oluşturur.
    
    Returns:
        np.ndarray: 3D yüz modeli noktaları
    """
    # Standart 3D yüz referans noktaları
    model_points = np.array([
        # Burun
        (0.0, 0.0, 0.0),             # Burun ucu
        (0.0, -20.0, -15.0),         # Burun alt merkezi
        (0.0, 20.0, -30.0),          # Burun üst merkezi
        
        # Çene ve Yüz Konturu
        (0.0, -120.0, -40.0),        # Çene ucu
        (-70.0, -100.0, -40.0),      # Sol çene köşesi
        (70.0, -100.0, -40.0),       # Sağ çene köşesi
        (-100.0, -40.0, -70.0),      # Sol yüz konturu orta nokta
        (100.0, -40.0, -70.0),       # Sağ yüz konturu orta nokta
        (-80.0, 40.0, -80.0),        # Sol yüz konturu üst nokta
        (80.0, 40.0, -80.0),         # Sağ yüz konturu üst nokta
        
        # Gözler
        (-35.0, 40.0, -60.0),        # Sol göz dış köşe
        (-10.0, 40.0, -55.0),        # Sol göz iç köşe
        (10.0, 40.0, -55.0),         # Sağ göz iç köşe
        (35.0, 40.0, -60.0),         # Sağ göz dış köşe
        (-20.0, 42.0, -65.0),        # Sol göz merkezi
        (20.0, 42.0, -65.0),         # Sağ göz merkezi
        
        # Kaşlar
        (-35.0, 60.0, -60.0),        # Sol kaş dış köşe
        (-10.0, 60.0, -55.0),        # Sol kaş iç köşe
        (10.0, 60.0, -55.0),         # Sağ kaş iç köşe
        (35.0, 60.0, -60.0),         # Sağ kaş dış köşe
        
        # Ağız
        (-30.0, -60.0, -55.0),       # Sol ağız köşesi
        (30.0, -60.0, -55.0),        # Sağ ağız köşesi
        (0.0, -60.0, -50.0),         # Ağız üst merkezi
        (0.0, -70.0, -50.0),         # Ağız alt merkezi
        
        # Kulaklar
        (-85.0, 0.0, -90.0),         # Sol kulak
        (85.0, 0.0, -90.0),          # Sağ kulak
    ])
    
    return model_points

def main():
    # 3D yüz modelini oluştur
    model_3d = create_3d_face_model()
    
    # Modeller dizinini oluştur
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 3D modeli .npy dosyasına kaydet
    model_path = os.path.join(models_dir, 'headpose_3d_model.npy')
    np.save(model_path, model_3d)
    
    print(f"3D yüz modeli başarıyla kaydedildi: {model_path}")
    print(f"Model boyutu: {model_3d.shape}")

if __name__ == "__main__":
    main() 