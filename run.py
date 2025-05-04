#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sürücü Uykululuk Tespiti için Başlatıcı Betik

Bu betik, proje kök dizinini Python'un modül arama yoluna ekler ve
gerekli argümanları ileterek ana uygulamayı (src/main.py) çalıştırır.
"""

import os
import sys

# Proje kök dizinini belirle ve Python yoluna ekle
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Ana modülü import et ve main fonksiyonunu çağır
    from src.main import main
    main() 