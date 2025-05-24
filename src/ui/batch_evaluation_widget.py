#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Evaluation Widget for the UI.

This module provides a UI widget for batch evaluation of multiple file pairs.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QFileDialog, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QTabWidget, QRadioButton, QButtonGroup, QGroupBox, QGridLayout,
    QMessageBox, QCheckBox, QSpinBox, QHeaderView, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QFont, QIcon

from src.evaluation.batch_evaluator import BatchEvaluator

# Configure logger
logger = logging.getLogger(__name__)

class BatchEvaluationWidget(QWidget):
    """
    Widget for batch evaluation of multiple gaze zone prediction files.
    
    This class provides functionality to:
    1. Select directories containing system output and ground truth files
    2. Configure file matching options
    3. Run batch evaluation
    4. Display aggregated results and comparisons
    5. Export reports
    """
    
    # Define signals
    batch_evaluation_started = pyqtSignal()
    batch_evaluation_completed = pyqtSignal(dict)
    
    def __init__(self, config, parent=None):
        """
        Initialize the BatchEvaluationWidget.
        
        Args:
            config: Application configuration
            parent: Parent widget
        """
        super().__init__(parent)
        self.config = config
        self.batch_evaluator = BatchEvaluator()
        
        # Initialize state variables
        self.system_dir = ""
        self.ground_truth_dir = ""
        self.matching_mode = "suffix"
        self.matching_pattern = "_system"
        self.results = None
        
        # Initialize UI
        self._init_ui()
        
        # Connect signals from batch evaluator
        self.batch_evaluator.evaluation_started.connect(self._on_evaluation_started)
        self.batch_evaluator.file_evaluation_started.connect(self._on_file_evaluation_started)
        self.batch_evaluator.file_evaluation_progress.connect(self._on_file_evaluation_progress)
        self.batch_evaluator.file_evaluation_completed.connect(self._on_file_evaluation_completed)
        self.batch_evaluator.file_evaluation_error.connect(self._on_file_evaluation_error)
        self.batch_evaluator.batch_evaluation_completed.connect(self._on_batch_evaluation_completed)
    
    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create sections
        self._create_directory_selection_section(main_layout)
        self._create_matching_options_section(main_layout)
        self._create_control_panel_section(main_layout)
        self._create_results_section(main_layout)
        
        # Set widget style
        self._set_widget_style()
    
    def _create_directory_selection_section(self, parent_layout):
        """Create directory selection section."""
        group_box = QGroupBox("Dizin Seçimi")
        layout = QVBoxLayout(group_box)
        
        # System output directory
        sys_layout = QHBoxLayout()
        sys_label = QLabel("Sistem Çıktı Dizini:")
        self.sys_dir_edit = QLineEdit()
        self.sys_dir_edit.setReadOnly(True)
        self.sys_dir_edit.setPlaceholderText("Sistem çıktı dosyalarını içeren dizini seçin")
        sys_browse_btn = QPushButton("Gözat...")
        sys_browse_btn.clicked.connect(self._on_system_dir_browse_clicked)
        
        sys_layout.addWidget(sys_label)
        sys_layout.addWidget(self.sys_dir_edit, 1)
        sys_layout.addWidget(sys_browse_btn)
        
        # Ground truth directory
        gt_layout = QHBoxLayout()
        gt_label = QLabel("Ground Truth Dizini:")
        self.gt_dir_edit = QLineEdit()
        self.gt_dir_edit.setReadOnly(True)
        self.gt_dir_edit.setPlaceholderText("Ground truth dosyalarını içeren dizini seçin")
        gt_browse_btn = QPushButton("Gözat...")
        gt_browse_btn.clicked.connect(self._on_ground_truth_dir_browse_clicked)
        
        gt_layout.addWidget(gt_label)
        gt_layout.addWidget(self.gt_dir_edit, 1)
        gt_layout.addWidget(gt_browse_btn)
        
        # Add scan button
        scan_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Dosyaları Tara")
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        self.scan_btn.setEnabled(False)
        self.files_found_label = QLabel("Dosya Çifti Bulunamadı")
        
        scan_layout.addWidget(self.scan_btn)
        scan_layout.addWidget(self.files_found_label, 1)
        
        # Add layouts to section
        layout.addLayout(sys_layout)
        layout.addLayout(gt_layout)
        layout.addLayout(scan_layout)
        
        parent_layout.addWidget(group_box)
    
    def _create_matching_options_section(self, parent_layout):
        """Create matching options section."""
        group_box = QGroupBox("Dosya Eşleştirme Seçenekleri")
        layout = QGridLayout(group_box)
        
        # Matching mode
        mode_label = QLabel("Eşleştirme Modu:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Tam Eşleşme (video1.json ↔ video1.json)",
            "Önek (sys_video1.json ↔ video1.json)",
            "Sonek (video1_system.json ↔ video1.json)",
            "Regex Kalıbı"
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        
        # Pattern input
        pattern_label = QLabel("Kalıp:")
        self.pattern_edit = QLineEdit("_system")
        self.pattern_edit.setPlaceholderText("Dosya eşleştirme kalıbı")
        self.pattern_edit.setToolTip("Sistem çıktı dosyaları için kalıp")
        
        # Pattern explanation
        self.pattern_explanation = QLabel("Sistem çıktı dosyaları '*_system.json' şeklinde, ground truth dosyaları '*.json' şeklinde olmalıdır.")
        self.pattern_explanation.setWordWrap(True)
        
        # Add widgets to layout
        layout.addWidget(mode_label, 0, 0)
        layout.addWidget(self.mode_combo, 0, 1)
        layout.addWidget(pattern_label, 1, 0)
        layout.addWidget(self.pattern_edit, 1, 1)
        layout.addWidget(self.pattern_explanation, 2, 0, 1, 2)
        
        parent_layout.addWidget(group_box)
    
    def _create_control_panel_section(self, parent_layout):
        """Create control panel section."""
        control_layout = QHBoxLayout()
        
        # Status section
        status_layout = QVBoxLayout()
        status_label = QLabel("Durum:")
        self.status_text = QLabel("Hazır")
        self.status_text.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_text)
        status_layout.addWidget(self.progress_bar)
        
        # Buttons section
        buttons_layout = QVBoxLayout()
        self.evaluate_btn = QPushButton("Toplu Değerlendirme Başlat")
        self.evaluate_btn.clicked.connect(self._on_evaluate_clicked)
        self.evaluate_btn.setEnabled(False)
        
        self.export_reports_btn = QPushButton("Raporları Dışa Aktar")
        self.export_reports_btn.clicked.connect(self._on_export_reports_clicked)
        self.export_reports_btn.setEnabled(False)
        
        buttons_layout.addWidget(self.evaluate_btn)
        buttons_layout.addWidget(self.export_reports_btn)
        buttons_layout.addStretch()
        
        # Add layouts to control layout
        control_layout.addLayout(status_layout, 7)
        control_layout.addLayout(buttons_layout, 3)
        
        parent_layout.addLayout(control_layout)
    
    def _create_results_section(self, parent_layout):
        """Create results section."""
        group_box = QGroupBox("Değerlendirme Sonuçları")
        layout = QVBoxLayout(group_box)
        
        # Create tab widget for results
        self.results_tabs = QTabWidget()
        
        # Summary tab
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        self.summary_label = QLabel("Henüz değerlendirme yapılmadı.")
        summary_layout.addWidget(self.summary_label)
        self.results_tabs.addTab(summary_widget, "Özet")
        
        # Files comparison tab
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(5)
        self.comparison_table.setHorizontalHeaderLabels([
            "Dosya Adı", "Çerçeve Sayısı", "Doğruluk", "Ağırlıklı F1", "Cohen's Kappa"
        ])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        comparison_layout.addWidget(self.comparison_table)
        self.results_tabs.addTab(comparison_widget, "Dosya Karşılaştırma")
        
        # Add tab widget to layout
        layout.addWidget(self.results_tabs)
        
        parent_layout.addWidget(group_box)
    
    def _set_widget_style(self):
        """Set widget style."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QPushButton {
                padding: 5px 10px;
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:disabled {
                background-color: #f8f8f8;
                color: #888888;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
    
    def _on_system_dir_browse_clicked(self):
        """Handle system directory browse button click."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Sistem Çıktı Dizinini Seç",
            os.path.expanduser("~"),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directory:
            self.system_dir = directory
            self.sys_dir_edit.setText(directory)
            self._update_scan_button_state()
    
    def _on_ground_truth_dir_browse_clicked(self):
        """Handle ground truth directory browse button click."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Ground Truth Dizinini Seç",
            os.path.expanduser("~"),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directory:
            self.ground_truth_dir = directory
            self.gt_dir_edit.setText(directory)
            self._update_scan_button_state()
    
    def _update_scan_button_state(self):
        """Update scan button state based on directory selections."""
        self.scan_btn.setEnabled(bool(self.system_dir and self.ground_truth_dir))
        self.evaluate_btn.setEnabled(False)
        self.files_found_label.setText("Dosya Çifti Bulunamadı")
    
    def _on_mode_changed(self, index):
        """Handle matching mode change."""
        modes = ["exact", "prefix", "suffix", "regex"]
        if index < len(modes):
            self.matching_mode = modes[index]
        
        # Update pattern explanation
        explanations = [
            "Sistem çıktı dosyaları ve ground truth dosyaları aynı isme sahip olmalıdır (video1.json ↔ video1.json).",
            "Sistem çıktı dosyaları önek içerir, örneğin: 'sys_video1.json' ↔ 'video1.json'.",
            "Sistem çıktı dosyaları sonek içerir, örneğin: 'video1_system.json' ↔ 'video1.json'.",
            "Özel regex kalıbı. Örnek: '(.*?)_sys\\.json' kalıbı 'video1_sys.json' dosyasını 'video1.json' ile eşleştirir."
        ]
        
        if index < len(explanations):
            self.pattern_explanation.setText(explanations[index])
        
        # Enable/disable pattern edit
        if index == 0:  # Exact match
            self.pattern_edit.setEnabled(False)
            self.pattern_edit.setText("")
        else:
            self.pattern_edit.setEnabled(True)
            # Set default patterns
            if index == 1:  # Prefix
                self.pattern_edit.setText("sys_")
            elif index == 2:  # Suffix
                self.pattern_edit.setText("_system")
            elif index == 3:  # Regex
                self.pattern_edit.setText(r"(.*?)_system\.json")
    
    def _on_scan_clicked(self):
        """Handle scan button click."""
        if not self.system_dir or not self.ground_truth_dir:
            QMessageBox.warning(
                self,
                "Dizin Seçimi Gerekli",
                "Lütfen önce sistem çıktı ve ground truth dizinlerini seçin."
            )
            return
        
        try:
            # Set directories and matching pattern
            self.batch_evaluator.set_directories(self.system_dir, self.ground_truth_dir)
            self.matching_pattern = self.pattern_edit.text()
            self.batch_evaluator.set_matching_pattern(self.matching_pattern, self.matching_mode)
            
            # Find file pairs
            file_pairs = self.batch_evaluator.find_file_pairs()
            
            if file_pairs:
                self.files_found_label.setText(f"{len(file_pairs)} dosya çifti bulundu")
                self.evaluate_btn.setEnabled(True)
                self.status_text.setText(f"{len(file_pairs)} dosya çifti tarandı. Değerlendirmeye hazır.")
            else:
                self.files_found_label.setText("Eşleşen dosya çifti bulunamadı")
                self.evaluate_btn.setEnabled(False)
                self.status_text.setText("Eşleşen dosya çifti bulunamadı. Lütfen dizinleri ve eşleştirme ayarlarını kontrol edin.")
                
                QMessageBox.information(
                    self,
                    "Dosya Bulunamadı",
                    "Belirtilen dizinlerde eşleşen dosya çifti bulunamadı.\n"
                    "Lütfen dizinleri ve eşleştirme ayarlarını kontrol edin."
                )
        
        except Exception as e:
            logger.error(f"Error during file scan: {str(e)}")
            self.status_text.setText(f"Hata: {str(e)}")
            self.files_found_label.setText("Tarama başarısız")
            
            QMessageBox.critical(
                self,
                "Tarama Hatası",
                f"Dosyalar taranırken bir hata oluştu:\n{str(e)}"
            )
    
    def _on_evaluate_clicked(self):
        """Handle evaluate button click."""
        # Clear previous results
        self.summary_label.setText("Değerlendirme işleniyor...")
        self.comparison_table.setRowCount(0)
        self.export_reports_btn.setEnabled(False)
        
        # Disable UI elements during evaluation
        self.evaluate_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        
        # Start evaluation
        try:
            self.batch_evaluation_started.emit()
            self.batch_evaluator.run_batch_evaluation()
            
        except Exception as e:
            logger.error(f"Error starting batch evaluation: {str(e)}")
            self.status_text.setText(f"Hata: {str(e)}")
            
            QMessageBox.critical(
                self,
                "Değerlendirme Hatası",
                f"Toplu değerlendirme başlatılırken bir hata oluştu:\n{str(e)}"
            )
            
            # Re-enable UI elements
            self.evaluate_btn.setEnabled(True)
            self.scan_btn.setEnabled(True)
    
    def _on_export_reports_clicked(self):
        """Handle export reports button click."""
        if not self.results:
            QMessageBox.warning(
                self,
                "Sonuç Bulunamadı",
                "Dışa aktarılacak sonuç bulunamadı. Lütfen önce değerlendirme yapın."
            )
            return
        
        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Rapor Çıktı Dizinini Seç",
            os.path.expanduser("~"),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_dir:
            return
        
        try:
            # Export reports
            self.status_text.setText("Raporlar dışa aktarılıyor...")
            
            # Export aggregated report
            aggregated_report_path = self.batch_evaluator.export_aggregated_report(
                os.path.join(output_dir, "toplu_degerlendirme_raporu.json")
            )
            
            # Export comparison CSV
            csv_path = self.batch_evaluator.export_comparison_csv(
                os.path.join(output_dir, "dosya_karsilastirma.csv")
            )
            
            # Generate individual reports
            report_paths = self.batch_evaluator.generate_individual_reports(output_dir)
            
            self.status_text.setText(f"Raporlar başarıyla dışa aktarıldı: {output_dir}")
            
            QMessageBox.information(
                self,
                "Raporlar Dışa Aktarıldı",
                f"Raporlar başarıyla şu dizine kaydedildi:\n{output_dir}\n\n"
                f"Oluşturulan raporlar:\n"
                f"- Toplu değerlendirme raporu (JSON)\n"
                f"- Dosya karşılaştırma tablosu (CSV)\n"
                f"- {len(report_paths)} adet bireysel dosya raporu"
            )
            
        except Exception as e:
            logger.error(f"Error exporting reports: {str(e)}")
            self.status_text.setText(f"Rapor dışa aktarma hatası: {str(e)}")
            
            QMessageBox.critical(
                self,
                "Dışa Aktarma Hatası",
                f"Raporlar dışa aktarılırken bir hata oluştu:\n{str(e)}"
            )
    
    @pyqtSlot(int)
    def _on_evaluation_started(self, total_files):
        """Handle evaluation started signal."""
        self.status_text.setText(f"{total_files} dosya değerlendirilmeye başlanıyor...")
        self.progress_bar.setRange(0, total_files)
        self.progress_bar.setValue(0)
    
    @pyqtSlot(str, int)
    def _on_file_evaluation_started(self, filename, index):
        """Handle file evaluation started signal."""
        self.status_text.setText(f"Dosya değerlendiriliyor ({index+1}/{self.batch_evaluator.total_files}): {filename}")
    
    @pyqtSlot(int, int)
    def _on_file_evaluation_progress(self, file_index, progress):
        """Handle file evaluation progress signal."""
        # Update file-specific progress if needed
        pass
    
    @pyqtSlot(dict, int)
    def _on_file_evaluation_completed(self, results, index):
        """Handle file evaluation completed signal."""
        self.progress_bar.setValue(index + 1)
        file_name = results.get("file_name", f"Dosya {index+1}")
        self.status_text.setText(f"Tamamlandı ({index+1}/{self.batch_evaluator.total_files}): {file_name}")
    
    @pyqtSlot(str, int)
    def _on_file_evaluation_error(self, error_message, index):
        """Handle file evaluation error signal."""
        self.progress_bar.setValue(index + 1)
        self.status_text.setText(f"Hata ({index+1}/{self.batch_evaluator.total_files}): {error_message}")
    
    @pyqtSlot(dict)
    def _on_batch_evaluation_completed(self, results):
        """Handle batch evaluation completed signal."""
        self.results = results
        
        # Update UI
        self.status_text.setText(f"Değerlendirme tamamlandı. {results.get('total_files', 0)} dosya işlendi.")
        self.evaluate_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.export_reports_btn.setEnabled(True)
        
        # Display results
        self._update_results_ui(results)
        
        # Emit completion signal
        self.batch_evaluation_completed.emit(results)
    
    def _update_results_ui(self, results):
        """Update UI with evaluation results."""
        # Update summary tab
        summary_html = f"""
        <h2>Toplu Değerlendirme Özeti</h2>
        <p><b>Toplam Dosya:</b> {results.get('total_files', 0)}</p>
        <p><b>Toplam Çerçeve:</b> {results.get('total_frames', 0)}</p>
        
        <h3>Ortalama Metrikler</h3>
        """
        
        avg_metrics = results.get('average_metrics', {})
        if avg_metrics:
            summary_html += f"""
            <p><b>Doğruluk:</b> {avg_metrics.get('accuracy', 0):.4f}</p>
            <p><b>Ağırlıklı F1:</b> {avg_metrics.get('f1_weighted', 0):.4f}</p>
            <p><b>Makro F1:</b> {avg_metrics.get('f1_macro', 0):.4f}</p>
            <p><b>Cohen's Kappa:</b> {avg_metrics.get('cohen_kappa', 0):.4f}</p>
            """
        
        # Add best and worst performing files
        best = results.get('best_performing', {})
        worst = results.get('worst_performing', {})
        
        if best:
            summary_html += f"""
            <h3>En İyi Performans Gösteren Dosyalar</h3>
            <p><b>Doğruluk:</b> {best.get('accuracy', 'N/A')}</p>
            <p><b>Ağırlıklı F1:</b> {best.get('f1_weighted', 'N/A')}</p>
            <p><b>Cohen's Kappa:</b> {best.get('cohen_kappa', 'N/A')}</p>
            """
        
        if worst:
            summary_html += f"""
            <h3>En Kötü Performans Gösteren Dosyalar</h3>
            <p><b>Doğruluk:</b> {worst.get('accuracy', 'N/A')}</p>
            <p><b>Ağırlıklı F1:</b> {worst.get('f1_weighted', 'N/A')}</p>
            <p><b>Cohen's Kappa:</b> {worst.get('cohen_kappa', 'N/A')}</p>
            """
        
        self.summary_label.setText(summary_html)
        
        # Update files comparison table
        files_summary = results.get('files_summary', [])
        self.comparison_table.setRowCount(len(files_summary))
        
        for i, file_data in enumerate(files_summary):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(file_data.get('file_name', 'Unknown')))
            self.comparison_table.setItem(i, 1, QTableWidgetItem(str(file_data.get('frame_count', 0))))
            self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{file_data.get('accuracy', 0):.4f}"))
            self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{file_data.get('f1_weighted', 0):.4f}"))
            self.comparison_table.setItem(i, 4, QTableWidgetItem(f"{file_data.get('cohen_kappa', 0):.4f}"))
        
        # Resize table columns to content
        self.comparison_table.resizeColumnsToContents()
    
    def generate_sample_files(self, num_videos=3, num_frames=100):
        """
        Generate sample files for testing batch evaluation.
        
        Args:
            num_videos (int): Number of sample video files to create
            num_frames (int): Number of frames per video
            
        Returns:
            Tuple[str, str]: Paths to system output and ground truth directories
        """
        from src.evaluation.examples.batch_evaluator_example import create_sample_files
        
        # Create sample files
        output_dir = os.path.join(os.getcwd(), "data", "batch_evaluation")
        sample_files = create_sample_files(output_dir, num_videos, num_frames)
        
        # Update UI
        self.system_dir = sample_files["system_dir"]
        self.ground_truth_dir = sample_files["ground_truth_dir"]
        
        self.sys_dir_edit.setText(self.system_dir)
        self.gt_dir_edit.setText(self.ground_truth_dir)
        
        # Update scan button state
        self._update_scan_button_state()
        
        # Show message
        QMessageBox.information(
            self,
            "Örnek Dosyalar Oluşturuldu",
            f"{num_videos} video için örnek dosyalar oluşturuldu.\n"
            f"Sistem çıktı dizini: {self.system_dir}\n"
            f"Ground truth dizini: {self.ground_truth_dir}\n\n"
            "Dosyaları taramak için 'Dosyaları Tara' butonuna tıklayın."
        )
        
        return self.system_dir, self.ground_truth_dir 