#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scientific Gaze Zone Evaluation GUI

PyQt6 tabanlı bilimsel değerlendirme arayüzü
"""

import sys
import os
import json
import traceback
from pathlib import Path
import numpy as np

# İsteğe bağlı kütüphaneler için kontrol
HAS_PANDAS = False
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas is not installed. Excel export will be disabled.")

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    classification_report, cohen_kappa_score, 
    matthews_corrcoef, balanced_accuracy_score
)
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox,
    QGroupBox, QProgressBar, QGridLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

# Import the existing classes from the original file
from evaulation_gui import (
    ScientificGazeEvaluator, create_publication_figures, 
    export_detailed_results
)

# Add this class after the imports
class NumpyEncoder(json.JSONEncoder):
    """NumPy veri tiplerini JSON formatına dönüştürebilen özel encoder."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class EvaluationWorker(QThread):
    """Evaluation işlemini arka planda yapan worker thread."""
    
    progress_updated = pyqtSignal(str)
    evaluation_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    create_figures_signal = pyqtSignal(dict, str)  # Signal for creating figures in the main thread
    create_excel_signal = pyqtSignal(dict, str)    # Signal for creating Excel in the main thread
    
    def __init__(self, gt_file, pred_file, output_dir):
        super().__init__()
        self.gt_file = gt_file
        self.pred_file = pred_file
        self.output_dir = output_dir
        self.figure_paths = []
    
    def run(self):
        """Evaluation işlemini çalıştır."""
        try:
            self.progress_updated.emit("Loading JSON files...")
            
            # Dosyaları yükle
            with open(self.gt_file, 'r') as f:
                ground_truth = json.load(f)
            
            with open(self.pred_file, 'r') as f:
                predictions = json.load(f)
            
            self.progress_updated.emit("Performing scientific analysis...")
            
            # Evaluator oluştur ve analiz yap
            evaluator = ScientificGazeEvaluator()
            results = evaluator.comprehensive_analysis(ground_truth, predictions)
            
            # NumPy değerlerini Python veri tiplerine dönüştür
            results = self._convert_numpy_types(results)
            
            self.progress_updated.emit("Generating reports and figures...")
            
            # Çıktı klasörünü oluştur
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Akademik rapor oluştur
            report_text = evaluator.generate_academic_report(
                results, self.gt_file, self.pred_file
            )
            
            report_file = os.path.join(self.output_dir, "scientific_evaluation_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # Görselleştirmeler için ana thread'e sinyal gönder
            figures_dir = os.path.join(self.output_dir, "figures")
            self.create_figures_signal.emit(results, figures_dir)
            
            # Excel raporu için sinyal gönder
            excel_file = os.path.join(self.output_dir, "detailed_metrics.xlsx")
            self.create_excel_signal.emit(results, excel_file)
            
            # JSON sonuçları kaydet
            results_file = os.path.join(self.output_dir, "evaluation_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Biraz bekleyerek figürlerin oluşturulmasını bekle
            QThread.msleep(500)
            
            # Sonuç paketini hazırla
            summary = {
                "results": results,
                "overall_accuracy": results.get("basic_metrics", {}).get("overall_accuracy", 0),
                "cohens_kappa": results.get("advanced_metrics", {}).get("cohens_kappa", 0),
                "agreement_strength": results.get("advanced_metrics", {}).get("agreement_strength", "Unknown"),
                "report_file": report_file,
                "figures": self.figure_paths,
                "excel_file": excel_file,
                "results_file": results_file,
                "output_dir": self.output_dir
            }
            
            self.progress_updated.emit("Evaluation completed successfully!")
            self.evaluation_completed.emit(summary)
            
        except Exception as e:
            error_msg = f"Error during evaluation: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
    def _convert_numpy_types(self, obj):
        """NumPy değerlerini Python değerlerine dönüştür."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            # Sözlüğün her anahtarını ve değerini dönüştür
            return {self._convert_numpy_types(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Listenin her elemanını dönüştür
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            # Tuple'ın her elemanını dönüştür
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def set_figure_paths(self, paths):
        """Oluşturulan figür yollarını kaydet."""
        self.figure_paths = paths


class ScientificEvaluationGUI(QMainWindow):
    """Bilimsel değerlendirme GUI ana penceresi."""
    
    def __init__(self):
        super().__init__()
        self.gt_file = ""
        self.pred_file = ""
        self.output_dir = "evaluation_results"
        self.worker = None
        self.evaluation_results = None
        
        self.init_ui()
        self.setup_styles()
    
    def init_ui(self):
        """Kullanıcı arayüzünü başlat."""
        self.setWindowTitle("Scientific Gaze Zone Evaluation")
        self.setMinimumSize(1000, 700)
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Ana layout
        main_layout = QVBoxLayout(central_widget)
        
        # Başlık
        title_label = QLabel("Scientific Gaze Zone Classification Evaluation")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # Splitter ile bölme
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        
        # Üst panel - Dosya seçimi ve kontroller
        top_panel = self.create_control_panel()
        splitter.addWidget(top_panel)
        
        # Alt panel - Sonuçlar
        bottom_panel = self.create_results_panel()
        splitter.addWidget(bottom_panel)
        
        # Splitter oranları
        splitter.setSizes([300, 400])
    
    def create_control_panel(self) -> QWidget:
        """Kontrol panelini oluştur."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Dosya seçim grubu
        file_group = QGroupBox("Input Files Selection")
        file_layout = QGridLayout(file_group)
        
        # Ground Truth dosyası
        file_layout.addWidget(QLabel("Ground Truth JSON:"), 0, 0)
        self.gt_label = QLabel("No file selected")
        self.gt_label.setStyleSheet("border: 1px solid #bdc3c7; padding: 8px; background-color: #ecf0f1; border-radius: 4px;")
        self.gt_button = QPushButton("Browse...")
        self.gt_button.clicked.connect(self.select_ground_truth)
        
        file_layout.addWidget(self.gt_label, 0, 1)
        file_layout.addWidget(self.gt_button, 0, 2)
        
        # Predictions dosyası
        file_layout.addWidget(QLabel("Predictions JSON:"), 1, 0)
        self.pred_label = QLabel("No file selected")
        self.pred_label.setStyleSheet("border: 1px solid #bdc3c7; padding: 8px; background-color: #ecf0f1; border-radius: 4px;")
        self.pred_button = QPushButton("Browse...")
        self.pred_button.clicked.connect(self.select_predictions)
        
        file_layout.addWidget(self.pred_label, 1, 1)
        file_layout.addWidget(self.pred_button, 1, 2)
        
        # Output directory
        file_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        self.output_label = QLabel(self.output_dir)
        self.output_label.setStyleSheet("border: 1px solid #bdc3c7; padding: 8px; background-color: #ecf0f1; border-radius: 4px;")
        self.output_button = QPushButton("Change...")
        self.output_button.clicked.connect(self.select_output_dir)
        
        file_layout.addWidget(self.output_label, 2, 1)
        file_layout.addWidget(self.output_button, 2, 2)
        
        layout.addWidget(file_group)
        
        # Kontrol butonları
        button_layout = QHBoxLayout()
        
        self.evaluate_button = QPushButton("🔬 Run Scientific Evaluation")
        self.evaluate_button.setEnabled(False)
        self.evaluate_button.clicked.connect(self.run_evaluation)
        self.evaluate_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        
        self.clear_button = QPushButton("🗑️ Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        
        self.open_results_button = QPushButton("📁 Open Results Folder")
        self.open_results_button.setEnabled(False)
        self.open_results_button.clicked.connect(self.open_results_folder)
        
        button_layout.addStretch()
        button_layout.addWidget(self.evaluate_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.open_results_button)
        layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def create_results_panel(self) -> QWidget:
        """Sonuçlar panelini oluştur."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget
        self.results_tabs = QTabWidget()
        
        # Log tab
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setReadOnly(True)
        self.results_tabs.addTab(self.log_text, "📝 Process Log")
        
        # Metrics tab
        self.metrics_table = QTableWidget()
        self.results_tabs.addTab(self.metrics_table, "📊 Metrics")
        
        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setFont(QFont("Consolas", 10))
        self.summary_text.setReadOnly(True)
        self.results_tabs.addTab(self.summary_text, "📋 Summary Report")
        
        layout.addWidget(self.results_tabs)
        
        return panel
    
    def setup_styles(self):
        """Stilleri ayarla."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin: 10px 0px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
            }
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
        """)
    
    def select_ground_truth(self):
        """Ground truth dosyası seç."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Ground Truth JSON File", "", 
            "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            self.gt_file = file_path
            self.gt_label.setText(os.path.basename(file_path))
            self.gt_label.setToolTip(file_path)
            self.log_text.append(f"✅ Ground Truth selected: {os.path.basename(file_path)}")
            self.check_ready()
    
    def select_predictions(self):
        """Predictions dosyası seç."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Predictions JSON File", "", 
            "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            self.pred_file = file_path
            self.pred_label.setText(os.path.basename(file_path))
            self.pred_label.setToolTip(file_path)
            self.log_text.append(f"✅ Predictions selected: {os.path.basename(file_path)}")
            self.check_ready()
    
    def select_output_dir(self):
        """Output klasörü seç."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir)
        
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(dir_path)
            self.output_label.setToolTip(dir_path)
            self.log_text.append(f"📁 Output directory: {dir_path}")
    
    def check_ready(self):
        """Evaluation için hazır mı kontrol et."""
        ready = bool(self.gt_file and self.pred_file)
        self.evaluate_button.setEnabled(ready)
        
        if ready:
            self.log_text.append("🚀 Ready for scientific evaluation!")
    
    def run_evaluation(self):
        """Evaluation'ı başlat."""
        if not self.gt_file or not self.pred_file:
            QMessageBox.warning(self, "Warning", "Please select both JSON files!")
            return
        
        # UI durumunu güncelle
        self.evaluate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Belirsiz progress
        self.log_text.clear()
        self.clear_results()
        
        # İlk log mesajı
        self.log_text.append("🚀 Starting scientific evaluation...")
        self.log_text.append(f"📄 Ground Truth: {os.path.basename(self.gt_file)}")
        self.log_text.append(f"📄 Predictions: {os.path.basename(self.pred_file)}")
        self.log_text.append(f"📁 Output directory: {self.output_dir}")
        self.log_text.append("="*50)
        
        # Worker thread başlat
        self.worker = EvaluationWorker(self.gt_file, self.pred_file, self.output_dir)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.evaluation_completed.connect(self.on_evaluation_completed)
        self.worker.error_occurred.connect(self.on_error_occurred)
        self.worker.create_figures_signal.connect(self.on_create_figures)
        self.worker.create_excel_signal.connect(self.on_create_excel)
        self.worker.start()
    
    def update_progress(self, message):
        """Progress mesajını güncelle."""
        self.log_text.append(f"⏳ {message}")
        self.log_text.ensureCursorVisible()
    
    def on_evaluation_completed(self, summary):
        """Evaluation tamamlandığında çağrılır."""
        self.evaluation_results = summary
        
        # UI durumunu güncelle
        self.progress_bar.setVisible(False)
        self.evaluate_button.setEnabled(True)
        self.open_results_button.setEnabled(True)
        
        # Başarı mesajı
        self.log_text.append("\n" + "="*60)
        self.log_text.append("🎉 SCIENTIFIC EVALUATION COMPLETED SUCCESSFULLY!")
        self.log_text.append("="*60)
        self.log_text.append(f"📊 Overall Accuracy: {summary['overall_accuracy']:.2f}%")
        self.log_text.append(f"📈 Cohen's Kappa: {summary['cohens_kappa']:.4f}")
        self.log_text.append(f"🤝 Agreement Level: {summary['agreement_strength']}")
        self.log_text.append(f"📁 Results saved to: {summary['output_dir']}")
        
        # Log figures
        figure_count = len(summary.get('figures', []))
        self.log_text.append(f"📈 Generated {figure_count} visualization figures")
        
        # Metrics tabını güncelle
        self.update_metrics_table(summary['results'])
        
        # Summary tabını güncelle
        self.update_summary_tab(summary)
        
        # Results tabına geç
        self.results_tabs.setCurrentIndex(1)
        
        # Başarı popup'ı
        QMessageBox.information(
            self, "Evaluation Completed", 
            f"Scientific evaluation completed successfully!\n\n"
            f"📊 Overall Accuracy: {summary['overall_accuracy']:.2f}%\n"
            f"📈 Cohen's Kappa: {summary['cohens_kappa']:.4f}\n"
            f"🤝 Agreement: {summary['agreement_strength']}\n\n"
            f"Results saved to: {summary['output_dir']}")
    
    def on_error_occurred(self, error_message):
        """Hata oluştuğunda çağrılır."""
        self.progress_bar.setVisible(False)
        self.evaluate_button.setEnabled(True)
        
        self.log_text.append(f"\n❌ ERROR OCCURRED:")
        self.log_text.append(error_message)
        
        QMessageBox.critical(self, "Evaluation Error", 
                           f"An error occurred during evaluation:\n\n{error_message}")
    
    def update_metrics_table(self, results):
        """Metrics tablosunu güncelle."""
        metrics_data = []
        
        # Basic metrics
        if "basic_metrics" in results:
            basic = results["basic_metrics"]
            metrics_data.extend([
                ("Overall Accuracy", f"{basic['overall_accuracy']:.2f}%"),
                ("Balanced Accuracy", f"{basic['balanced_accuracy']:.2f}%"),
                ("Macro Precision", f"{basic['macro_precision']:.2f}%"),
                ("Macro Recall", f"{basic['macro_recall']:.2f}%"),
                ("Macro F1-Score", f"{basic['macro_f1']:.2f}%"),
            ])
        
        # Advanced metrics
        if "advanced_metrics" in results:
            advanced = results["advanced_metrics"]
            metrics_data.extend([
                ("", ""),  # Boş satır
                ("Cohen's Kappa", f"{advanced['cohens_kappa']:.4f}"),
                ("Agreement Strength", advanced['agreement_strength']),
                ("Matthews Correlation", f"{advanced['matthews_correlation_coeff']:.4f}"),
                ("Krippendorff's Alpha", f"{advanced['krippendorff_alpha']:.4f}"),
            ])
        
        # Category Analysis
        if "category_analysis" in results:
            metrics_data.append(("", ""))  # Boş satır
            for category, data in results["category_analysis"].items():
                metrics_data.append((f"{category} Accuracy", f"{data['accuracy']:.2f}%"))
        
        # Tabloyu oluştur
        self.metrics_table.setRowCount(len(metrics_data))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        
        for i, (metric, value) in enumerate(metrics_data):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
        
        # Tablo ayarları
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    
    def update_summary_tab(self, summary):
        """Summary tabını güncelle."""
        report_text = f"""
SCIENTIFIC GAZE ZONE EVALUATION SUMMARY
{'='*50}

FILES:
  Ground Truth: {os.path.basename(self.gt_file)}
  Predictions:  {os.path.basename(self.pred_file)}
  
OVERALL PERFORMANCE:
  Accuracy:           {summary['overall_accuracy']:.2f}%
  Cohen's Kappa:      {summary['cohens_kappa']:.4f}
  Agreement Level:    {summary['agreement_strength']}

GENERATED FILES:
  📄 Academic Report:  {os.path.basename(summary['report_file'])}
  📊 Excel Analysis:   {os.path.basename(summary['excel_file'])}
  📈 Figures:          {len(summary['figures'])} publication-quality plots
  📋 JSON Results:     {os.path.basename(summary['results_file'])}

INTERPRETATION:
"""
        
        # Otomatik yorum ekleme
        kappa = summary['cohens_kappa']
        accuracy = summary['overall_accuracy']
        
        if accuracy >= 90:
            report_text += "  ✅ Excellent classification performance achieved\n"
        elif accuracy >= 80:
            report_text += "  ✅ Good classification performance achieved\n"
        elif accuracy >= 70:
            report_text += "  ⚠️  Moderate performance - room for improvement\n"
        else:
            report_text += "  ❌ Performance requires significant improvement\n"
        
        if kappa >= 0.8:
            report_text += "  ✅ Almost perfect inter-rater agreement\n"
        elif kappa >= 0.6:
            report_text += "  ✅ Substantial agreement achieved\n"
        elif kappa >= 0.4:
            report_text += "  ⚠️  Moderate agreement - consider improvements\n"
        else:
            report_text += "  ❌ Low agreement - significant issues detected\n"
        
        report_text += f"\nAll results saved to: {summary['output_dir']}"
        
        self.summary_text.setPlainText(report_text)
    
    def clear_results(self):
        """Sonuçları temizle."""
        self.metrics_table.setRowCount(0)
        self.summary_text.clear()
        self.evaluation_results = None
        self.open_results_button.setEnabled(False)
    
    def clear_all(self):
        """Tüm verileri temizle."""
        self.gt_file = ""
        self.pred_file = ""
        self.gt_label.setText("No file selected")
        self.pred_label.setText("No file selected")
        self.gt_label.setToolTip("")
        self.pred_label.setToolTip("")
        self.evaluate_button.setEnabled(False)
        self.log_text.clear()
        self.clear_results()
        self.progress_bar.setVisible(False)
    
    def open_results_folder(self):
        """Sonuçlar klasörünü aç."""
        if self.evaluation_results:
            import subprocess
            import platform
            
            path = self.evaluation_results['output_dir']
            
            try:
                if platform.system() == "Windows":
                    subprocess.run(f'explorer "{path}"', shell=True)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", path])
                else:  # Linux
                    subprocess.run(["xdg-open", path])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open folder: {str(e)}")
    
    def on_create_figures(self, results, figures_dir):
        """Ana thread'de görselleştirmeler oluştur."""
        try:
            figure_paths = create_publication_figures(results, figures_dir)
            self.worker.set_figure_paths(figure_paths)
            self.log_text.append(f"📊 Created {len(figure_paths)} visualization figures")
        except Exception as e:
            self.log_text.append(f"⚠️ Warning: Could not create figures: {str(e)}")
    
    def on_create_excel(self, results, excel_file):
        """Ana thread'de Excel raporu oluştur."""
        try:
            # İlk önce pandas'ın yüklü olup olmadığını kontrol edelim
            try:
                import pandas as pd
                has_pandas = True
            except ImportError:
                has_pandas = False
                self.log_text.append("⚠️ Warning: pandas kütüphanesi yüklü değil.")
                self.log_text.append("   Excel raporu oluşturulamıyor.")
                self.log_text.append("   pip install pandas komutu ile pandas'ı yükleyebilirsiniz.")
                return

            # pandas varsa devam et
            if has_pandas:
                export_detailed_results(results, excel_file)
                self.log_text.append(f"📈 Exported detailed metrics to Excel")
        except Exception as e:
            self.log_text.append(f"⚠️ Warning: Could not create Excel report: {str(e)}")
            self.log_text.append(f"   This may be due to missing dependencies")


def main():
    """Ana GUI uygulamasını başlat."""
    
    print("Starting Scientific Gaze Zone Evaluation GUI...")
    
    # Gerekli bağımlılıkları kontrol et
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import PyQt6
    except ImportError:
        missing_deps.append("PyQt6")
    
    # Opsiyonel bağımlılıklar
    optional_missing = []
    if not HAS_PANDAS:
        optional_missing.append("pandas")
    
    try:
        import openpyxl
    except ImportError:
        optional_missing.append("openpyxl")
    
    # Eksik bağımlılıklar varsa uyarı ver
    if missing_deps:
        print("Error: Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install them with:")
        print(f"  pip install {' '.join(missing_deps)}")
        sys.exit(1)
    
    if optional_missing:
        print("Warning: Missing optional dependencies:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print("\nSome features may be disabled. To enable all features, install:")
        print(f"  pip install {' '.join(optional_missing)}")
    
    # QApplication oluştur
    app = QApplication(sys.argv)
    
    # Uygulama ayarları
    app.setApplicationName("Scientific Gaze Zone Evaluation")
    app.setApplicationVersion("1.0")
    
    try:
        # Ana pencereyi oluştur
        window = ScientificEvaluationGUI()
        window.show()
        
        # Uygulamayı çalıştır
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 