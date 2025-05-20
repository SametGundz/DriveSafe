#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GazeDurationMonitor Test Script
Bu script, GazeDurationMonitor sınıfını test etmek ve sonuçları görmek için kullanılır.
Farklı test senaryoları simüle eder ve uyarı sisteminin davranışını gözlemler.
"""

import time
import os
import sys
import numpy as np
from datetime import datetime

# Proje kök dizinini ekleyin
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Gerekli modülleri içe aktarın
from src.detection.gaze_duration_monitor import get_gaze_duration_monitor, WarningLevel

def print_colored(text, color):
    """Renkli çıktı yazdırma"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def print_warning_level(level, details=None):
    """Uyarı seviyesini renkli yazdırma"""
    if level == WarningLevel.CRITICAL:
        print_colored(f"[KRITIK UYARI] {details if details else ''}", "red")
    elif level == WarningLevel.WARNING:
        print_colored(f"[UYARI] {details if details else ''}", "yellow")
    else:
        print_colored("[NORMAL]", "green")

def test_area1_gaze_duration():
    """Alan 1'e (infotainment) uzun süre bakma testi"""
    print_colored("\n=== TEST 1: Alan 1 (Infotainment) Uzun Süre Bakış Testi ===", "cyan")
    
    # Monitor oluştur
    monitor = get_gaze_duration_monitor()
    monitor.reset()  # Önceki test verilerini temizle
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # İlk olarak yol merkezine bak (zone_id = 0)
    print("Yol merkezine bakış (1 saniye)")
    for i in range(10):
        new_time = start_time + (i * 0.1)
        state = monitor.update(0, new_time)
        if i == 9:
            print_warning_level(state["distraction_level"])
        time.sleep(0.05)  # Simülasyon için hızlı çalıştır
    
    # Infotainment'a bak (zone_id = 2)
    print("\nInfotainment'a bakış başladı (0s)")
    for i in range(35):  # 3.5 saniye boyunca
        new_time = start_time + 1.0 + (i * 0.1)
        state = monitor.update(2, new_time)
        
        # Belirli noktalarda durumu yazdır
        if i == 9:  # 1 saniye sonra
            print("1.0 saniye geçti:")
            print_warning_level(state["distraction_level"])
        
        elif i == 19:  # 2 saniye sonra
            print("2.0 saniye geçti:")
            print_warning_level(state["distraction_level"], state.get("warning", {}).get("reasons", []))
        
        elif i == 29:  # 3 saniye sonra
            print("3.0 saniye geçti:")
            print_warning_level(state["distraction_level"], state.get("warning", {}).get("reasons", []))
        
        time.sleep(0.05)
    
    # Yol merkezine dön
    print("\nYol merkezine dönüş")
    new_time = start_time + 4.5
    state = monitor.update(0, new_time)
    print_warning_level(state["distraction_level"])
    
    # Sonuçlar
    print_colored("\nYUKARIDAKİ SONUÇ: 2 saniyeden sonra WARNING, 3 saniyeden sonra CRITICAL seviye uyarı görmeliyiz.", "magenta")

def test_road_center_absence():
    """Yol merkezinden uzun süre bakış kaydırma testi"""
    print_colored("\n=== TEST 2: Yol Merkezinden Uzun Süre Bakış Kaydırma Testi ===", "cyan")
    
    # Monitor oluştur
    monitor = get_gaze_duration_monitor()
    monitor.reset()  # Önceki test verilerini temizle
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # İlk olarak yol merkezine bak (zone_id = 0)
    print("Yol merkezine bakış (1 saniye)")
    for i in range(10):
        new_time = start_time + (i * 0.1)
        state = monitor.update(0, new_time)
        if i == 9:
            print_warning_level(state["distraction_level"])
        time.sleep(0.05)
    
    # Sol yana bak (zone_id = 3, Alan 2)
    print("\nSol yana bakış başladı (0s)")
    for i in range(45):  # 4.5 saniye boyunca
        new_time = start_time + 1.0 + (i * 0.1)
        state = monitor.update(3, new_time)
        
        # Belirli noktalarda durumu yazdır
        if i == 19:  # 2 saniye sonra
            print("2.0 saniye geçti:")
            print_warning_level(state["distraction_level"], state.get("warning", {}).get("reasons", []))
        
        elif i == 29:  # 3 saniye sonra
            print("3.0 saniye geçti:")
            print_warning_level(state["distraction_level"], state.get("warning", {}).get("reasons", []))
        
        elif i == 39:  # 4 saniye sonra
            print("4.0 saniye geçti:")
            print_warning_level(state["distraction_level"], state.get("warning", {}).get("reasons", []))
        
        time.sleep(0.05)
    
    # Yol merkezine dön
    print("\nYol merkezine dönüş")
    new_time = start_time + 5.5
    state = monitor.update(0, new_time)
    print_warning_level(state["distraction_level"])
    
    # Sonuçlar
    print_colored("\nYUKARIDAKİ SONUÇ: 3 saniyeden sonra WARNING, 4 saniyeden sonra CRITICAL uyarı görmeliyiz.", "magenta")

def test_area1_percentage():
    """Alan 1 yüzde testi"""
    print_colored("\n=== TEST 3: Alan 1 Yüzde Testi ===", "cyan")
    
    # Monitor oluştur
    monitor = get_gaze_duration_monitor()
    monitor.reset()  # Önceki test verilerini temizle
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # Toplam 60 saniye simüle et, %20 infotainment
    print("Simülasyon başladı: Her 5 saniyede 1 saniye infotainment (~%20)")
    
    # Hızlı simülasyon için 1 saniyeyi 0.05 saniyede simüle edelim
    for i in range(600):  # 60 saniye simülasyonu, 100ms aralıklarla
        new_time = start_time + (i * 0.1)
        
        # Her 5 saniyede 1 saniye infotainment'a bak (toplam %20)
        zone_id = 2 if (i % 50) < 10 else 0
        
        state = monitor.update(zone_id, new_time)
        
        # Her 10 saniyede bir durumu rapor et
        if i % 100 == 0 and i > 0:
            secs = i / 10
            area1_pct = state["metrics"]["area1_percentage"]
            print(f"\n{secs:.1f} saniye geçti - Alan 1: %{area1_pct:.1f}")
            print_warning_level(state["distraction_level"], state.get("warning", {}).get("reasons", []))
        
        time.sleep(0.005)  # Simülasyonu hızlandır
    
    # Sonuçlar
    print_colored("\nYUKARIDAKİ SONUÇ: Zaman ilerledikçe Alan 1 yüzdesi ~%20 civarına ulaşmalı ve kritik seviye uyarısı görmeliyiz.", "magenta")

def test_area1_to_area2_transition():
    """Alan 1'den Alan 2'ye geçiş süresi testi"""
    print_colored("\n=== TEST 4: Alan 1'den Alan 2'ye Geçiş Süresi Testi ===", "cyan")
    
    # Monitor oluştur
    monitor = get_gaze_duration_monitor()
    monitor.reset()  # Önceki test verilerini temizle
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # İlk olarak infotainment'a bak (zone_id = 2)
    print("Infotainment'a bakış (2 saniye)")
    for i in range(20):
        new_time = start_time + (i * 0.1)
        state = monitor.update(2, new_time)
        if i == 19:
            print_warning_level(state["distraction_level"])
        time.sleep(0.05)
    
    # 2 saniye sonra yol merkezine dön (yavaş geçiş simülasyonu)
    new_time = start_time + 4.0  # 2s infotainment + 2s geçiş
    print("\n2 saniye sonra yol merkezine geçiş:")
    state = monitor.update(0, new_time)
    print_warning_level(state["distraction_level"])
    
    # Rapor oluştur ve geçiş süresini yazdır
    report = monitor.generate_report(new_time + 0.1)
    transition_time = report["transitions"]["area1_to_area2_time_avg"]
    compliant = report["eu_regulation"]["area1_to_area2_compliant"]
    
    print(f"\nGeçiş Raporu:")
    print(f"- Alan 1->Alan 2 geçiş süresi: {transition_time:.2f} saniye")
    print(f"- AB düzenlemesi uyumluluğu: {'UYUMLU' if compliant else 'UYUMSUZ'}")
    
    # Sonuçlar
    print_colored("\nYUKARIDAKİ SONUÇ: Alan 1'den Alan 2'ye geçiş süresi 2.0 saniye civarında olmalı ve AB düzenlemesine uyumsuz olarak işaretlenmeli.", "magenta")

def test_distraction_score():
    """Dalgınlık puanı testi"""
    print_colored("\n=== TEST 5: Dalgınlık Puanı Testi ===", "cyan")
    
    # Monitor oluştur
    monitor = get_gaze_duration_monitor()
    monitor.reset()  # Önceki test verilerini temizle
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # Çeşitli bakış davranışları simüle et
    scenarios = [
        {"zone": 0, "duration": 10, "desc": "Normal sürüş (yol merkezi)"},
        {"zone": 2, "duration": 3, "desc": "Kısa infotainment bakışı"},
        {"zone": 0, "duration": 5, "desc": "Yol merkezine dönüş"},
        {"zone": 2, "duration": 5, "desc": "Uzun infotainment bakışı"},
        {"zone": 3, "duration": 4, "desc": "Sol yana uzun bakış"},
        {"zone": 0, "duration": 10, "desc": "Yol merkezine dönüş"}
    ]
    
    current_time = start_time
    
    for scenario in scenarios:
        zone_id = scenario["zone"]
        duration = scenario["duration"]
        
        print(f"\n{scenario['desc']} ({duration} saniye)")
        
        # Senaryoyu 100ms adımlarla simüle et
        steps = duration * 10
        for i in range(steps):
            new_time = current_time + (i * 0.1)
            state = monitor.update(zone_id, new_time)
            
            # Belirli noktalarda durumu rapor et
            if i % 10 == 0:
                score = monitor.calculate_distraction_score(new_time)
                print(f"- {i/10:.1f}s geçti - Dalgınlık puanı: {score:.1f}/100")
                if i == 0 or i == steps - 10:
                    print_warning_level(state["distraction_level"])
            
            time.sleep(0.005)  # Simülasyonu hızlandır
        
        current_time += duration
    
    # Final raporu
    final_time = current_time
    report = monitor.generate_report(final_time)
    
    print("\n=== Final Raporu ===")
    print(f"Toplam sürüş süresi: {report['total_driving_time']:.1f} saniye")
    print(f"Alan 1 süresi: {report['area1_statistics']['total_time']:.1f} saniye")
    print(f"Alan 1 yüzdesi: %{report['area1_statistics']['percentage']:.1f}")
    print(f"Dalgınlık seviyesi: {report['distraction']['level']}")
    print(f"Dalgınlık puanı: {report['distraction']['score']:.1f}/100")
    
    # AB regülasyonu uyumluluğu
    print("\nAB Regülasyonu Uyumluluğu:")
    print(f"- Alan 1 yüzdesi: {'UYUMLU' if report['eu_regulation']['area1_percentage_compliant'] else 'UYUMSUZ'}")
    print(f"- Yol merkezi yokluğu: {'UYUMLU' if report['eu_regulation']['road_center_absence_compliant'] else 'UYUMSUZ'}")
    print(f"- Alan 1->Alan 2 geçiş: {'UYUMLU' if report['eu_regulation']['area1_to_area2_compliant'] else 'UYUMSUZ'}")
    print(f"- Genel uyumluluk: {report['eu_regulation']['overall_compliance']}")
    
    # Raporu kaydet
    file_path = monitor.save_report_to_file(final_time)
    print(f"\nRapor kaydedildi: {file_path}")
    
    # Sonuçlar
    print_colored("\nYUKARIDAKİ SONUÇ: Dalgınlık puanının simüle edilen davranışlara göre değiştiğini görmeliyiz.", "magenta")

def test_advanced_metrics():
    """İleri düzey metrikler testi"""
    print_colored("\n=== TEST 6: İleri Düzey Metrikler Testi ===", "cyan")
    
    # Monitor oluştur
    monitor = get_gaze_duration_monitor()
    monitor.reset()  # Önceki test verilerini temizle
    
    # Başlangıç zamanı
    start_time = time.time()
    
    # Karmaşık bakış deseni simüle et
    print("Karmaşık bakış deseni simülasyonu (30 saniye)...")
    
    # Çeşitli bakış kalıpları yaratmak için
    # belirli olasılıklarla farklı bölgelere bak
    zones = [0, 1, 2, 3, 4, 5]  # Tüm zonlar
    
    current_time = start_time
    current_zone = 0  # Road Center'dan başla
    
    np.random.seed(42)  # Tekrarlanabilirlik için
    
    # 30 saniye boyunca simüle et
    for i in range(300):  # 30 saniye, 100ms adımlarla
        new_time = start_time + (i * 0.1)
        
        # Her 1-3 saniye arasında bölge değiştir
        if i % np.random.randint(10, 30) == 0:
            # Alan 1'e %20 olasılıkla, diğer bölgelere eşit dağılımla bak
            if np.random.random() < 0.2:
                current_zone = 2  # Infotainment (Alan 1)
            else:
                # Road Center'a daha yüksek olasılıkla dön
                if np.random.random() < 0.5:
                    current_zone = 0  # Road Center
                else:
                    # Diğer Alan 2 bölgelerinden rastgele seç
                    current_zone = np.random.choice([1, 3, 4, 5])
        
        state = monitor.update(current_zone, new_time)
        
        # Her 5 saniyede bir metrikleri rapor et
        if i % 50 == 0 and i > 0:
            print(f"\n{i/10:.1f} saniye geçti:")
            print(f"- Mevcut bölge: {state['current_zone']['name']} (Alan {'1' if state['current_zone']['is_area1'] else '2'})")
            print(f"- Dalgınlık seviyesi: {state['distraction_level']}")
            if "warning" in state and state["warning"]["level"] != "NORMAL":
                print(f"- Uyarı nedenleri: {', '.join(state['warning']['reasons'])}")
            
            # NumPy ile optimize edilmiş istatistikler
            advanced_stats = monitor.get_advanced_time_window_statistics(new_time, 10.0)  # Son 10 saniye
            transitions = monitor.analyze_gaze_transitions(new_time, 10.0)
            score = monitor.calculate_distraction_score(new_time)
            
            print(f"- Son 10s Alan 1 yüzdesi: %{advanced_stats['area1_percentage']:.1f}")
            print(f"- Son 10s geçiş sayısı: {transitions['transition_count']}")
            print(f"- Dalgınlık puanı: {score:.1f}/100")
        
        time.sleep(0.005)  # Simülasyonu hızlandır
    
    # Final raporu
    final_time = start_time + 30.0
    print("\n=== İleri Düzey Analiz Sonuçları ===")
    
    # Geçiş matrisi analizi
    transitions = monitor.analyze_gaze_transitions(final_time, 30.0)
    
    print("Geçiş Matrisi:")
    for from_zone in range(6):
        for to_zone in range(6):
            count = transitions["transition_matrix"][from_zone][to_zone]
            if count > 0:
                print(f"- {from_zone} -> {to_zone}: {count} kez")
    
    print(f"\nAlan 1->Alan 2 ort. geçiş süresi: {transitions['area1_to_area2_time_avg']:.2f}s")
    print(f"Geçiş sıklığı: {transitions['transition_frequency']:.1f} geçiş/dakika")
    
    # Raporu kaydet
    file_path = monitor.save_report_to_file(final_time)
    print(f"\nDetaylı rapor kaydedildi: {file_path}")
    
    # Sonuçlar
    print_colored("\nYUKARIDAKİ SONUÇ: Karmaşık bir sürüş davranışının detaylı analizini görmeliyiz.", "magenta")

def main():
    """Ana test fonksiyonu"""
    print_colored("=== GAZE DURATION MONITOR TESTİ ===", "cyan")
    print("AB Regülasyonu C(2023)4523'e uygun bakış dalgınlık tespit sistemi")
    print(f"Tarih/Saat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Kullanıcıya menü göster
    while True:
        print_colored("\nTest Menüsü:", "cyan")
        print("1. Alan 1 (Infotainment) Uzun Süre Bakış Testi")
        print("2. Yol Merkezinden Uzun Süre Bakış Kaydırma Testi")
        print("3. Alan 1 Yüzde Testi")
        print("4. Alan 1'den Alan 2'ye Geçiş Süresi Testi")
        print("5. Dalgınlık Puanı Testi")
        print("6. İleri Düzey Metrikler Testi")
        print("7. Tüm Testleri Çalıştır")
        print("0. Çıkış")
        
        choice = input("\nTest numarasını seçin (0-7): ")
        
        if choice == "1":
            test_area1_gaze_duration()
        elif choice == "2":
            test_road_center_absence()
        elif choice == "3":
            test_area1_percentage()
        elif choice == "4":
            test_area1_to_area2_transition()
        elif choice == "5":
            test_distraction_score()
        elif choice == "6":
            test_advanced_metrics()
        elif choice == "7":
            test_area1_gaze_duration()
            test_road_center_absence()
            test_area1_percentage()
            test_area1_to_area2_transition()
            test_distraction_score()
            test_advanced_metrics()
        elif choice == "0":
            print_colored("\nTest programı sonlandırılıyor...", "cyan")
            break
        else:
            print_colored("Geçersiz seçim! Lütfen 0-7 arasında bir değer girin.", "yellow")
        
        input("\nDevam etmek için Enter tuşuna basın...")

if __name__ == "__main__":
    main() 