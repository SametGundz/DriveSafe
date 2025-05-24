#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON Gaze Zone Doğruluk Hesaplayıcı Arayüzü

Bu uygulama, gaze zone tahmin sonuçları ile ground truth verilerini 
karşılaştırarak doğruluk skorları hesaplar.
"""

import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox,
    QProgressBar, QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QTabWidget, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap, QPainter, QColor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np


class AccuracyCalculatorWorker(QThread):
    """Doğruluk hesaplama işlemini arka planda yapan worker thread."""
    
    progress_updated = pyqtSignal(str)  # İlerleme mesajları
    calculation_finished = pyqtSignal(dict)  # Sonuç verileri
    error_occurred = pyqtSignal(str)  # Hata mesajları
    
    def __init__(self, ground_truth_path, prediction_path):
        super().__init__()
        self.ground_truth_path = ground_truth_path
        self.prediction_path = prediction_path
        
        # Zone açıklamaları
        self.zone_descriptions = {
            0: "Road Center",
            1: "Driving Instruments", 
            2: "Infotainment",
            3: "Left Side",
            4: "Right Side",
            5: "Rear Mirror",
            None: "No Detection"
        }
    
    def run(self):
        """Doğruluk hesaplama işlemini gerçekleştir."""
        try:
            self.progress_updated.emit("Ground truth dosyası okunuyor...")
            
            # Ground truth dosyasını yükle
            with open(self.ground_truth_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
            
            self.progress_updated.emit("Tahmin dosyası okunuyor...")
            
            # Tahmin dosyasını yükle
            with open(self.prediction_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            
            self.progress_updated.emit("Veriler karşılaştırılıyor...")
            
            # Doğruluk hesaplama
            results = self.calculate_accuracy(ground_truth, predictions)
            
            self.progress_updated.emit("Detaylı analiz yapılıyor...")
            
            # Detaylı analiz
            detailed_results = self.detailed_analysis(ground_truth, predictions)
            results.update(detailed_results)
            
            self.progress_updated.emit("Hesaplama tamamlandı!")
            self.calculation_finished.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"Hata oluştu: {str(e)}")
    
    def calculate_accuracy(self, ground_truth, predictions):
        """Temel doğruluk skorunu hesapla."""
        
        # Ortak frame'leri bul
        gt_frames = set(ground_truth.keys())
        pred_frames = set(predictions.keys())
        common_frames = gt_frames.intersection(pred_frames)
        
        # Null olmayan değerleri filtrele
        valid_comparisons = []
        total_frames = 0
        correct_predictions = 0
        
        for frame_id in common_frames:
            gt_value = ground_truth[frame_id]
            pred_value = predictions[frame_id]
            
            # Null değerleri atla
            if gt_value is not None and pred_value is not None:
                valid_comparisons.append((frame_id, gt_value, pred_value))
                total_frames += 1
                
                if gt_value == pred_value:
                    correct_predictions += 1
        
        # Doğruluk oranı hesapla
        overall_accuracy = (correct_predictions / total_frames * 100) if total_frames > 0 else 0.0
        
        return {
            "overall_accuracy": overall_accuracy,
            "total_frames": total_frames,
            "correct_predictions": correct_predictions,
            "common_frames": len(common_frames),
            "gt_total_frames": len(gt_frames),
            "pred_total_frames": len(pred_frames),
            "valid_comparisons": valid_comparisons
        }
    
    def detailed_analysis(self, ground_truth, predictions):
        """Detaylı analiz ve confusion matrix hesapla."""
        
        # Zone bazında analiz
        zone_stats = defaultdict(lambda: {"correct": 0, "total": 0, "gt_count": 0, "pred_count": 0})
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        # Ground truth zone sayıları
        gt_zone_counts = Counter()
        pred_zone_counts = Counter()
        
        for frame_id, gt_value in ground_truth.items():
            if gt_value is not None:
                gt_zone_counts[gt_value] += 1
                zone_stats[gt_value]["gt_count"] += 1
        
        for frame_id, pred_value in predictions.items():
            if pred_value is not None:
                pred_zone_counts[pred_value] += 1
                zone_stats[pred_value]["pred_count"] += 1
        
        # Ortak frame'lerde karşılaştırma
        common_frames = set(ground_truth.keys()).intersection(set(predictions.keys()))
        
        for frame_id in common_frames:
            gt_value = ground_truth[frame_id]
            pred_value = predictions[frame_id]
            
            if gt_value is not None and pred_value is not None:
                # Confusion matrix
                confusion_matrix[gt_value][pred_value] += 1
                
                # Zone istatistikleri
                zone_stats[gt_value]["total"] += 1
                if gt_value == pred_value:
                    zone_stats[gt_value]["correct"] += 1
        
        # Zone bazında accuracy hesapla
        zone_accuracies = {}
        for zone_id, stats in zone_stats.items():
            if stats["total"] > 0:
                zone_accuracies[zone_id] = (stats["correct"] / stats["total"]) * 100
            else:
                zone_accuracies[zone_id] = 0.0
        
        # Precision, Recall, F1-Score hesapla
        metrics_per_zone = {}
        all_zones = set(list(gt_zone_counts.keys()) + list(pred_zone_counts.keys()))
        
        for zone in all_zones:
            if zone is not None:
                # True Positive
                tp = confusion_matrix[zone][zone]
                
                # False Positive (diğer zone'lardan bu zone'a yanlış tahmin)
                fp = sum(confusion_matrix[other_zone][zone] for other_zone in all_zones if other_zone != zone)
                
                # False Negative (bu zone'dan diğer zone'lara yanlış tahmin)
                fn = sum(confusion_matrix[zone][other_zone] for other_zone in all_zones if other_zone != zone)
                
                # Precision, Recall, F1
                precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
                recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
                f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                
                metrics_per_zone[zone] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn
                }
        
        return {
            "zone_accuracies": zone_accuracies,
            "zone_stats": dict(zone_stats),
            "confusion_matrix": dict(confusion_matrix),
            "gt_zone_counts": dict(gt_zone_counts),
            "pred_zone_counts": dict(pred_zone_counts),
            "metrics_per_zone": metrics_per_zone
        }


class ConfusionMatrixWidget(QWidget):
    """Confusion Matrix görselleştirme widget'ı."""
    
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
    
    def update_matrix(self, confusion_matrix, zone_descriptions):
        """Confusion matrix'i güncelle."""
        self.figure.clear()
        
        # Zone'ları sırala
        zones = sorted([z for z in confusion_matrix.keys() if z is not None])
        
        if not zones:
            return
        
        # Matrix oluştur
        matrix_size = len(zones)
        matrix = np.zeros((matrix_size, matrix_size))
        
        zone_to_idx = {zone: idx for idx, zone in enumerate(zones)}
        
        for true_zone in zones:
            for pred_zone in zones:
                true_idx = zone_to_idx[true_zone]
                pred_idx = zone_to_idx[pred_zone]
                matrix[true_idx][pred_idx] = confusion_matrix[true_zone].get(pred_zone, 0)
        
        # Heatmap çiz
        ax = self.figure.add_subplot(111)
        
        # Zone isimlerini kısalt
        zone_labels = [f"Zone {z}" for z in zones]
        
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                   xticklabels=zone_labels, yticklabels=zone_labels, ax=ax)
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Zone', fontsize=12)
        ax.set_ylabel('True Zone', fontsize=12)
        
        # Etiketleri döndür
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        self.figure.tight_layout()
        self.canvas.draw()


class AccuracyCalculatorUI(QMainWindow):
    """JSON Gaze Zone Doğruluk Hesaplayıcı ana penceresi."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
        self.results_data = None
        
    def init_ui(self):
        """Kullanıcı arayüzünü başlat."""
        self.setWindowTitle("Gaze Zone Doğruluk Hesaplayıcı")
        self.setMinimumSize(1200, 800)
        
        # Ana widget ve splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Başlık
        title_label = QLabel("Gaze Zone Doğruluk Hesaplayıcı")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Dosya seçimi grubu
        file_group = QGroupBox("Dosya Seçimi")
        file_layout = QVBoxLayout(file_group)
        
        # Ground Truth dosyası seçimi
        gt_layout = QHBoxLayout()
        self.gt_path_label = QLabel("Henüz dosya seçilmedi")
        self.gt_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background-color: #f9f9f9;")
        self.gt_browse_button = QPushButton("Ground Truth Seç")
        self.gt_browse_button.clicked.connect(self.browse_ground_truth)
        
        gt_layout.addWidget(QLabel("Ground Truth:"), 0)
        gt_layout.addWidget(self.gt_path_label, 1)
        gt_layout.addWidget(self.gt_browse_button, 0)
        file_layout.addLayout(gt_layout)
        
        # Prediction dosyası seçimi
        pred_layout = QHBoxLayout()
        self.pred_path_label = QLabel("Henüz dosya seçilmedi")
        self.pred_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px; background-color: #f9f9f9;")
        self.pred_browse_button = QPushButton("Tahmin Dosyası Seç")
        self.pred_browse_button.clicked.connect(self.browse_prediction)
        
        pred_layout.addWidget(QLabel("Prediction:"), 0)
        pred_layout.addWidget(self.pred_path_label, 1)
        pred_layout.addWidget(self.pred_browse_button, 0)
        file_layout.addLayout(pred_layout)
        
        main_layout.addWidget(file_group)
        
        # Kontrol butonları
        button_layout = QHBoxLayout()
        
        self.calculate_button = QPushButton("Doğruluk Hesapla")
        self.calculate_button.setEnabled(False)
        self.calculate_button.clicked.connect(self.start_calculation)
        self.calculate_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        self.clear_button = QPushButton("Temizle")
        self.clear_button.clicked.connect(self.clear_all)
        
        self.export_button = QPushButton("Sonuçları Dışa Aktar")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        
        button_layout.addStretch()
        button_layout.addWidget(self.calculate_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.export_button)
        main_layout.addLayout(button_layout)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Sonuçlar için tab widget
        self.results_tabs = QTabWidget()
        main_layout.addWidget(self.results_tabs)
        
        # Sonuç tabları oluştur
        self.create_results_tabs()
        
        # Varsayılan değerler
        self.ground_truth_path = ""
        self.prediction_path = ""
        
    def create_results_tabs(self):
        """Sonuç tablarını oluştur."""
        
        # 1. Genel Sonuçlar Tabı
        self.overview_tab = QWidget()
        overview_layout = QVBoxLayout(self.overview_tab)
        
        # Genel sonuçlar için büyük kartlar
        cards_layout = QGridLayout()
        
        # Ana doğruluk kartı
        self.accuracy_card = self.create_metric_card("Genel Doğruluk", "-%", "#28a745")
        cards_layout.addWidget(self.accuracy_card, 0, 0)
        
        # Toplam frame kartı
        self.frames_card = self.create_metric_card("Toplam Frame", "-", "#17a2b8")
        cards_layout.addWidget(self.frames_card, 0, 1)
        
        # Doğru tahmin kartı
        self.correct_card = self.create_metric_card("Doğru Tahmin", "-", "#ffc107")
        cards_layout.addWidget(self.correct_card, 0, 2)
        
        overview_layout.addLayout(cards_layout)
        
        # Log alanı
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        overview_layout.addWidget(self.log_text)
        
        self.results_tabs.addTab(self.overview_tab, "📊 Genel Sonuçlar")
        
        # 2. Zone Detay Tabı
        self.zone_detail_tab = QWidget()
        zone_layout = QVBoxLayout(self.zone_detail_tab)
        
        self.zone_table = QTableWidget()
        zone_layout.addWidget(self.zone_table)
        
        self.results_tabs.addTab(self.zone_detail_tab, "🎯 Zone Detayları")
        
        # 3. Confusion Matrix Tabı
        self.confusion_tab = QWidget()
        confusion_layout = QVBoxLayout(self.confusion_tab)
        
        self.confusion_widget = ConfusionMatrixWidget()
        confusion_layout.addWidget(self.confusion_widget)
        
        self.results_tabs.addTab(self.confusion_tab, "📈 Confusion Matrix")
        
    def create_metric_card(self, title, value, color):
        """Metrik kartı oluştur."""
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.Box)
        card.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {color};
                border-radius: 10px;
                background-color: white;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"color: {color};")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        # Değeri güncellemek için referansı sakla
        card.value_label = value_label
        
        return card
        
    def browse_ground_truth(self):
        """Ground truth JSON dosyasını seç."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Ground Truth JSON Dosyası Seç",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.ground_truth_path = file_path
            self.gt_path_label.setText(os.path.basename(file_path))
            self.gt_path_label.setToolTip(file_path)
            self.check_ready_to_calculate()
            self.log_text.append(f"✅ Ground truth dosyası seçildi: {file_path}")
    
    def browse_prediction(self):
        """Prediction JSON dosyasını seç."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Prediction JSON Dosyası Seç",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            self.prediction_path = file_path
            self.pred_path_label.setText(os.path.basename(file_path))
            self.pred_path_label.setToolTip(file_path)
            self.check_ready_to_calculate()
            self.log_text.append(f"✅ Prediction dosyası seçildi: {file_path}")
    
    def check_ready_to_calculate(self):
        """Hesaplama için hazır olup olmadığını kontrol et."""
        ready = bool(self.ground_truth_path and self.prediction_path)
        self.calculate_button.setEnabled(ready)
    
    def start_calculation(self):
        """Doğruluk hesaplama işlemini başlat."""
        if not self.ground_truth_path or not self.prediction_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce her iki JSON dosyasını da seçin!")
            return
        
        # UI durumunu güncelle
        self.calculate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Belirsiz ilerleme
        self.log_text.clear()
        self.export_button.setEnabled(False)
        
        # Worker thread başlat
        self.worker = AccuracyCalculatorWorker(self.ground_truth_path, self.prediction_path)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.calculation_finished.connect(self.on_calculation_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def update_progress(self, message):
        """İlerleme mesajını güncelle."""
        self.log_text.append(f"⏳ {message}")
        self.log_text.ensureCursorVisible()
    
    def on_calculation_finished(self, results_data):
        """Hesaplama tamamlandığında çağrılır."""
        self.progress_bar.setVisible(False)
        self.calculate_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.results_data = results_data
        
        # Başarı mesajı
        self.log_text.append("\n" + "="*50)
        self.log_text.append("🎉 DOĞRULUK HESAPLAMA TAMAMLANDI!")
        self.log_text.append("="*50)
        
        # Sonuçları güncelle
        self.update_results_display(results_data)
        
        # Başarı popup'ı
        QMessageBox.information(
            self, 
            "Başarılı", 
            f"Doğruluk hesaplama tamamlandı!\n\n"
            f"Genel Doğruluk: {results_data['overall_accuracy']:.2f}%\n"
            f"Toplam Frame: {results_data['total_frames']}"
        )
    
    def update_results_display(self, results_data):
        """Sonuç ekranlarını güncelle."""
        
        # 1. Genel sonuçlar kartlarını güncelle
        self.accuracy_card.value_label.setText(f"{results_data['overall_accuracy']:.2f}%")
        self.frames_card.value_label.setText(str(results_data['total_frames']))
        self.correct_card.value_label.setText(str(results_data['correct_predictions']))
        
        # 2. Zone detay tablosunu güncelle
        self.update_zone_table(results_data)
        
        # 3. Confusion matrix'i güncelle
        zone_descriptions = {
            0: "Road Center",
            1: "Driving Instruments", 
            2: "Infotainment",
            3: "Left Side",
            4: "Right Side",
            5: "Rear Mirror"
        }
        self.confusion_widget.update_matrix(results_data['confusion_matrix'], zone_descriptions)
        
        # 4. Log'a detaylı sonuçları ekle
        self.log_detailed_results(results_data)
    
    def update_zone_table(self, results_data):
        """Zone detay tablosunu güncelle."""
        zone_descriptions = {
            0: "Road Center",
            1: "Driving Instruments", 
            2: "Infotainment",
            3: "Left Side",
            4: "Right Side",
            5: "Rear Mirror"
        }
        
        zones = sorted([z for z in results_data['zone_accuracies'].keys() if z is not None])
        
        # Tablo başlıkları
        headers = ["Zone ID", "Zone Adı", "Doğruluk (%)", "Doğru/Toplam", "Precision (%)", "Recall (%)", "F1-Score (%)"]
        self.zone_table.setColumnCount(len(headers))
        self.zone_table.setHorizontalHeaderLabels(headers)
        self.zone_table.setRowCount(len(zones))
        
        for i, zone_id in enumerate(zones):
            accuracy = results_data['zone_accuracies'][zone_id]
            stats = results_data['zone_stats'][zone_id]
            metrics = results_data['metrics_per_zone'].get(zone_id, {})
            
            # Tablo öğelerini oluştur
            items = [
                str(zone_id),
                zone_descriptions.get(zone_id, f"Zone {zone_id}"),
                f"{accuracy:.2f}",
                f"{stats['correct']}/{stats['total']}",
                f"{metrics.get('precision', 0):.2f}",
                f"{metrics.get('recall', 0):.2f}",
                f"{metrics.get('f1_score', 0):.2f}"
            ]
            
            for j, item_text in enumerate(items):
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Doğruluk sütununu renklendir
                if j == 2:  # Doğruluk sütunu
                    if accuracy >= 90:
                        item.setBackground(QColor(144, 238, 144))  # Açık yeşil
                    elif accuracy >= 70:
                        item.setBackground(QColor(255, 255, 224))  # Açık sarı
                    else:
                        item.setBackground(QColor(255, 182, 193))  # Açık kırmızı
                
                self.zone_table.setItem(i, j, item)
        
        # Sütun genişliklerini ayarla
        self.zone_table.resizeColumnsToContents()
    
    def log_detailed_results(self, results_data):
        """Detaylı sonuçları log'a yaz."""
        
        self.log_text.append(f"\n📊 Detaylı Sonuçlar:")
        self.log_text.append(f"   • Genel Doğruluk: {results_data['overall_accuracy']:.2f}%")
        self.log_text.append(f"   • Toplam Karşılaştırılan Frame: {results_data['total_frames']}")
        self.log_text.append(f"   • Doğru Tahmin: {results_data['correct_predictions']}")
        self.log_text.append(f"   • Yanlış Tahmin: {results_data['total_frames'] - results_data['correct_predictions']}")
        
        self.log_text.append(f"\n🎯 Zone Bazında Doğruluk:")
        zone_descriptions = {
            0: "Road Center",
            1: "Driving Instruments", 
            2: "Infotainment",
            3: "Left Side",
            4: "Right Side",
            5: "Rear Mirror"
        }
        
        for zone_id in sorted(results_data['zone_accuracies'].keys()):
            if zone_id is not None:
                accuracy = results_data['zone_accuracies'][zone_id]
                zone_name = zone_descriptions.get(zone_id, f"Zone {zone_id}")
                stats = results_data['zone_stats'][zone_id]
                
                self.log_text.append(f"   • {zone_name} (ID: {zone_id}): {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
    
    def on_error(self, error_message):
        """Hata oluştuğunda çağrılır."""
        self.progress_bar.setVisible(False)
        self.calculate_button.setEnabled(True)
        
        self.log_text.append(f"\n❌ HATA: {error_message}")
        QMessageBox.critical(self, "Hata", f"Hesaplama sırasında hata oluştu:\n\n{error_message}")
    
    def export_results(self):
        """Sonuçları dışa aktar."""
        if not self.results_data:
            QMessageBox.warning(self, "Uyarı", "Önce doğruluk hesaplama işlemini yapın!")
            return
        
        # Dosya yolu seç
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Sonuçları Kaydet",
            f"accuracy_results_{self.results_data['overall_accuracy']:.2f}percent.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Sonuçları JSON formatında kaydet
                export_data = {
                    "analysis_info": {
                        "ground_truth_file": os.path.basename(self.ground_truth_path),
                        "prediction_file": os.path.basename(self.prediction_path),
                        "analysis_date": str(QDateTime.currentDateTime().toString()),
                    },
                    "overall_results": {
                        "accuracy_percentage": self.results_data['overall_accuracy'],
                        "total_frames": self.results_data['total_frames'],
                        "correct_predictions": self.results_data['correct_predictions'],
                        "wrong_predictions": self.results_data['total_frames'] - self.results_data['correct_predictions']
                    },
                    "zone_results": {},
                    "confusion_matrix": self.results_data['confusion_matrix'],
                    "detailed_metrics": self.results_data['metrics_per_zone']
                }
                
                # Zone sonuçlarını ekle
                zone_descriptions = {
                    0: "Road Center",
                    1: "Driving Instruments", 
                    2: "Infotainment",
                    3: "Left Side",
                    4: "Right Side",
                    5: "Rear Mirror"
                }
                
                for zone_id, accuracy in self.results_data['zone_accuracies'].items():
                    if zone_id is not None:
                        stats = self.results_data['zone_stats'][zone_id]
                        metrics = self.results_data['metrics_per_zone'].get(zone_id, {})
                        
                        export_data["zone_results"][str(zone_id)] = {
                            "zone_name": zone_descriptions.get(zone_id, f"Zone {zone_id}"),
                            "accuracy_percentage": accuracy,
                            "correct_predictions": stats['correct'],
                            "total_predictions": stats['total'],
                            "precision": metrics.get('precision', 0),
                            "recall": metrics.get('recall', 0),
                            "f1_score": metrics.get('f1_score', 0)
                        }
                
                # JSON dosyasını kaydet
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "Başarılı", f"Sonuçlar başarıyla kaydedildi:\n{file_path}")
                self.log_text.append(f"✅ Sonuçlar dışa aktarıldı: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dışa aktarma sırasında hata oluştu:\n{str(e)}")
    
    def clear_all(self):
        """Tüm verileri temizle."""
        self.ground_truth_path = ""
        self.prediction_path = ""
        self.gt_path_label.setText("Henüz dosya seçilmedi")
        self.pred_path_label.setText("Henüz dosya seçilmedi")
        self.gt_path_label.setToolTip("")
        self.pred_path_label.setToolTip("")
        self.calculate_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setVisible(False)
        self.results_data = None
        
        # Sonuç kartlarını sıfırla
        self.accuracy_card.value_label.setText("-%")
        self.frames_card.value_label.setText("-")
        self.correct_card.value_label.setText("-")
        
        # Tabloyu temizle
        self.zone_table.setRowCount(0)
        
        # Confusion matrix'i temizle
        self.confusion_widget.figure.clear()
        self.confusion_widget.canvas.draw()


def main():
    """Ana uygulama fonksiyonu."""
    app = QApplication(sys.argv)
    
    # Uygulama ayarları
    app.setApplicationName("Gaze Zone Doğruluk Hesaplayıcı")
    app.setApplicationVersion("1.0")
    
    # Ana pencereyi oluştur ve göster
    window = AccuracyCalculatorUI()
    window.show()
    
    # Uygulamayı çalıştır
    sys.exit(app.exec())


if __name__ == "__main__":
    main()