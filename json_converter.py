#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON Gaze Zone Dönüştürücü Arayüzü

Bu uygulama, gaze zone annotation JSON dosyalarını sistem formatına 
dönüştürmek için basit bir kullanıcı arayüzü sağlar.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox,
    QProgressBar, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon


class JSONConverterWorker(QThread):
    """JSON dönüştürme işlemini arka planda yapan worker thread."""
    
    progress_updated = pyqtSignal(str)  # İlerleme mesajları
    conversion_finished = pyqtSignal(dict)  # Sonuç verileri
    error_occurred = pyqtSignal(str)  # Hata mesajları
    
    def __init__(self, input_file_path, output_dir):
        super().__init__()
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        
        # Zone eşleme tablosu - Ground truth ile uyumlu zone_id formatı
        # NOT: Sol ve sağ bölgelerin ID'leri güncellendi (GazeZoneDetector ile uyumlu olması için)
        self.zone_map = {
            "gaze_zone/front": 0,              # Road Center
            "gaze_zone/steering_wheel": 1,     # Driving Instruments  
            "gaze_zone/infotainment": 2,       # Infotainment
            "gaze_zone/right": 3,              # Right Side (ID değiştirildi: 4 -> 3)
            "gaze_zone/front_right": 3,        # Right Side (ID değiştirildi: 4 -> 3)
            "gaze_zone/right_mirror": 3,       # Right Side (ID değiştirildi: 4 -> 3)
            "gaze_zone/left": 4,               # Left Side (ID değiştirildi: 3 -> 4)
            "gaze_zone/left_mirror": 4,        # Left Side (ID değiştirildi: 3 -> 4)
            "gaze_zone/center_mirror": 5,      # Rear Mirror
            "gaze_zone/not_valid": None        # Not a valid gaze zone
        }
    
    def run(self):
        """Dönüştürme işlemini gerçekleştir."""
        try:
            self.progress_updated.emit("JSON dosyası okunuyor...")
            
            # JSON dosyasını yükle
            with open(self.input_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.progress_updated.emit("Veriler işleniyor...")
            
            actions = data.get("openlabel", {}).get("actions", {})
            
            # Frame → Zone eşlemesi
            frames_to_zones = {}
            converted_output = defaultdict(list)
            unknown_types = set()
            zone_stats = defaultdict(int)
            
            for action_id, action in actions.items():
                action_type = action.get("type", "").lower()
                intervals = action.get("frame_intervals", [])
                
                # Bilinmeyen zone türlerini topla
                if action_type.startswith("gaze_zone/") and action_type not in self.zone_map:
                    unknown_types.add(action_type)
                
                zone_id = self.zone_map.get(action_type, None)
                
                for interval in intervals:
                    start = interval["frame_start"]
                    end = interval["frame_end"]
                    frame_count = end - start + 1
                    
                    # Sadece geçerli zone_id'ler için istatistik tut
                    if zone_id is not None:
                        zone_stats[zone_id] += frame_count
                    
                    for frame_id in range(start, end + 1):
                        frames_to_zones[str(frame_id)] = zone_id
                        if zone_id is not None:
                            converted_output[zone_id].append(frame_id)
            
            self.progress_updated.emit("Çıktı dosyaları kaydediliyor...")
            
            # Çıktı dosyalarını kaydet
            converted_output_path = os.path.join(self.output_dir, "converted_output.json")
            frames_to_zones_path = os.path.join(self.output_dir, "frames_to_zones.json")
            
            # converted_output.json - zone_id'leri sırala
            with open(converted_output_path, "w", encoding="utf-8") as f_out:
                # Zone ID'leri sıralayarak kaydet
                sorted_output = {}
                for zone_id in sorted(converted_output.keys()):
                    sorted_output[zone_id] = sorted(converted_output[zone_id])
                json.dump(sorted_output, f_out, indent=2, ensure_ascii=False)
            
            # frames_to_zones.json
            with open(frames_to_zones_path, "w", encoding="utf-8") as f_out:
                json.dump(frames_to_zones, f_out, indent=2, ensure_ascii=False)
            
            # Sonuç verilerini hazırla
            result_data = {
                "converted_output_path": converted_output_path,
                "frames_to_zones_path": frames_to_zones_path,
                "total_frames": len(frames_to_zones),
                "zone_stats": dict(zone_stats),
                "unknown_types": list(unknown_types),
                "zone_map": self.zone_map
            }
            
            self.progress_updated.emit("Dönüştürme tamamlandı!")
            self.conversion_finished.emit(result_data)
            
        except Exception as e:
            self.error_occurred.emit(f"Hata oluştu: {str(e)}")


class JSONConverterUI(QMainWindow):
    """JSON Gaze Zone Dönüştürücü ana penceresi."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
        
    def init_ui(self):
        """Kullanıcı arayüzünü başlat."""
        self.setWindowTitle("JSON Gaze Zone Dönüştürücü")
        self.setMinimumSize(800, 600)
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Başlık
        title_label = QLabel("JSON Gaze Zone Dönüştürücü")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Dosya seçimi grubu
        file_group = QGroupBox("Dosya Seçimi")
        file_layout = QVBoxLayout(file_group)
        
        # Input dosyası seçimi
        input_layout = QHBoxLayout()
        self.input_path_label = QLabel("Henüz dosya seçilmedi")
        self.input_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background-color: #f9f9f9;")
        self.input_browse_button = QPushButton("JSON Dosyası Seç")
        self.input_browse_button.clicked.connect(self.browse_input_file)
        
        input_layout.addWidget(QLabel("Input Dosyası:"))
        input_layout.addWidget(self.input_path_label, 1)
        input_layout.addWidget(self.input_browse_button)
        file_layout.addLayout(input_layout)
        
        # Output klasörü seçimi
        output_layout = QHBoxLayout()
        self.output_path_label = QLabel("Mevcut dizin kullanılacak")
        self.output_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background-color: #f9f9f9;")
        self.output_browse_button = QPushButton("Çıktı Klasörü Seç")
        self.output_browse_button.clicked.connect(self.browse_output_dir)
        
        output_layout.addWidget(QLabel("Çıktı Klasörü:"))
        output_layout.addWidget(self.output_path_label, 1)
        output_layout.addWidget(self.output_browse_button)
        file_layout.addLayout(output_layout)
        
        layout.addWidget(file_group)
        
        # Kontrol butonları
        button_layout = QHBoxLayout()
        
        self.convert_button = QPushButton("Dönüştürmeyi Başlat")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.start_conversion)
        self.convert_button.setStyleSheet("""
            QPushButton {
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.clear_button = QPushButton("Temizle")
        self.clear_button.clicked.connect(self.clear_all)
        
        button_layout.addStretch()
        button_layout.addWidget(self.convert_button)
        button_layout.addWidget(self.clear_button)
        layout.addLayout(button_layout)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Sonuç grubu
        result_group = QGroupBox("Sonuçlar")
        result_layout = QVBoxLayout(result_group)
        
        # Log alanı
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        result_layout.addWidget(self.log_text)
        
        # Çıktı dosya yolları
        self.output_info_layout = QVBoxLayout()
        result_layout.addLayout(self.output_info_layout)
        
        layout.addWidget(result_group)
        
        # Varsayılan değerler
        self.input_file_path = ""
        self.output_dir = os.getcwd()  # Mevcut dizin
        self.update_output_label()
        
    def browse_input_file(self):
        """Input JSON dosyasını seç."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "JSON Annotation Dosyası Seç",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.input_file_path = file_path
            self.input_path_label.setText(os.path.basename(file_path))
            self.input_path_label.setToolTip(file_path)
            self.convert_button.setEnabled(True)
            self.log_text.append(f"✅ Input dosyası seçildi: {file_path}")
    
    def browse_output_dir(self):
        """Çıktı klasörünü seç."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Çıktı Klasörü Seç",
            self.output_dir
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.update_output_label()
            self.log_text.append(f"📁 Çıktı klasörü değiştirildi: {dir_path}")
    
    def update_output_label(self):
        """Çıktı klasörü etiketini güncelle."""
        self.output_path_label.setText(os.path.basename(self.output_dir) or self.output_dir)
        self.output_path_label.setToolTip(self.output_dir)
    
    def start_conversion(self):
        """Dönüştürme işlemini başlat."""
        if not self.input_file_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir JSON dosyası seçin!")
            return
        
        # UI durumunu güncelle
        self.convert_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Belirsiz ilerleme
        self.log_text.clear()
        self.clear_output_info()
        
        # Worker thread başlat
        self.worker = JSONConverterWorker(self.input_file_path, self.output_dir)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.conversion_finished.connect(self.on_conversion_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def update_progress(self, message):
        """İlerleme mesajını güncelle."""
        self.log_text.append(f"⏳ {message}")
        self.log_text.ensureCursorVisible()
    
    def on_conversion_finished(self, result_data):
        """Dönüştürme tamamlandığında çağrılır."""
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        
        # Başarı mesajı
        self.log_text.append("\n" + "="*50)
        self.log_text.append("🎉 DÖNÜŞTÜRME BAŞARIYLA TAMAMLANDI!")
        self.log_text.append("="*50)
        
        # İstatistikler
        self.log_text.append(f"\n📊 İstatistikler:")
        self.log_text.append(f"   • Toplam frame sayısı: {result_data['total_frames']}")
        
        # Zone istatistikleri
        self.log_text.append(f"\n🎯 Zone Dağılımı:")
        for zone_id in sorted(result_data['zone_stats'].keys()):
            count = result_data['zone_stats'][zone_id]
            zone_name = self.get_zone_description(zone_id)
            self.log_text.append(f"   • Zone {zone_id} ({zone_name}): {count} frame")
        
        # Bilinmeyen türler
        if result_data['unknown_types']:
            self.log_text.append(f"\n⚠️ Tanımsız gaze_zone türleri:")
            for unknown in result_data['unknown_types']:
                self.log_text.append(f"   • {unknown}")
        
        # Çıktı dosyalarını göster
        self.show_output_files(result_data)
        
        # Başarı popup'ı
        QMessageBox.information(
            self, 
            "Başarılı", 
            f"Dönüştürme tamamlandı!\n\n"
            f"Toplam {result_data['total_frames']} frame işlendi.\n"
            f"Çıktı dosyaları: {self.output_dir}"
        )
    
    def on_error(self, error_message):
        """Hata oluştuğunda çağrılır."""
        self.progress_bar.setVisible(False)
        self.convert_button.setEnabled(True)
        
        self.log_text.append(f"\n❌ HATA: {error_message}")
        QMessageBox.critical(self, "Hata", f"Dönüştürme sırasında hata oluştu:\n\n{error_message}")
    
    def show_output_files(self, result_data):
        """Çıktı dosya bilgilerini göster."""
        self.clear_output_info()
        
        # Çıktı dosya yolları
        files_info = [
            ("converted_output.json", result_data['converted_output_path']),
            ("frames_to_zones.json", result_data['frames_to_zones_path'])
        ]
        
        for filename, filepath in files_info:
            file_layout = QHBoxLayout()
            
            label = QLabel(f"📄 {filename}:")
            label.setMinimumWidth(200)
            
            path_label = QLabel(filepath)
            path_label.setStyleSheet("border: 1px solid #28a745; padding: 5px; background-color: #d4edda; color: #155724;")
            path_label.setWordWrap(True)
            
            open_button = QPushButton("Klasör Aç")
            open_button.clicked.connect(lambda checked, path=filepath: self.open_file_location(path))
            
            file_layout.addWidget(label)
            file_layout.addWidget(path_label, 1)
            file_layout.addWidget(open_button)
            
            self.output_info_layout.addLayout(file_layout)
    
    def clear_output_info(self):
        """Çıktı dosya bilgilerini temizle."""
        while self.output_info_layout.count():
            layout = self.output_info_layout.takeAt(0)
            self.clear_layout(layout)
    
    def clear_layout(self, layout):
        """Layout'taki tüm widget'ları temizle."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())
    
    def open_file_location(self, file_path):
        """Dosya konumunu dosya yöneticisinde aç."""
        import subprocess
        import platform
        
        try:
            if platform.system() == "Windows":
                subprocess.run(f'explorer /select,"{file_path}"', shell=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-R", file_path])
            else:  # Linux
                subprocess.run(["xdg-open", os.path.dirname(file_path)])
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Dosya konumu açılamadı: {str(e)}")
    
    def get_zone_description(self, zone_id):
        """Zone ID'sine göre açıklama döndür."""
        descriptions = {
            0: "Road Center",
            1: "Driving Instruments", 
            2: "Infotainment",
            3: "Right Side",  # Değiştirildi: Left -> Right
            4: "Left Side",   # Değiştirildi: Right -> Left
            5: "Rear Mirror",
            None: "Invalid Zone"
        }
        return descriptions.get(zone_id, "Unknown")
    
    def clear_all(self):
        """Tüm verileri temizle."""
        self.input_file_path = ""
        self.input_path_label.setText("Henüz dosya seçilmedi")
        self.input_path_label.setToolTip("")
        self.convert_button.setEnabled(False)
        self.log_text.clear()
        self.clear_output_info()
        self.progress_bar.setVisible(False)


def main():
    """Ana uygulama fonksiyonu."""
    app = QApplication(sys.argv)
    
    # Uygulama ayarları
    app.setApplicationName("JSON Gaze Zone Dönüştürücü")
    app.setApplicationVersion("1.0")
    
    # Ana pencereyi oluştur ve göster
    window = JSONConverterUI()
    window.show()
    
    # Uygulamayı çalıştır
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 