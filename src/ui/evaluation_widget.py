#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation widget for the driver drowsiness detection application.

This module implements a widget for evaluating gaze zone predictions against ground truth data.
"""

import os
import logging
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QProgressBar, QSizePolicy, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QGridLayout, QGroupBox, QMessageBox,
    QTabWidget, QSplitter, QDialog, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QPixmap
import json

from src.evaluation.gaze_zone_evaluator import GazeZoneEvaluator
from src.evaluation.visualization import (
    plot_confusion_matrix, plot_zone_performance, create_classification_table, 
    plot_roc_curves, generate_report_figures
)
from src.evaluation.data_validator import (
    validate_files, get_validation_example,
    validate_system_json, validate_ground_truth_json
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

# Get module-specific logger
logger = logging.getLogger(__name__)

class EvaluationWidget(QWidget):
    """
    Widget for evaluating gaze zone predictions.
    
    This class provides functionality to select system output and ground truth files,
    run evaluation, and display results.
    """
    
    # Define signals
    evaluation_started = pyqtSignal()  # Emitted when evaluation starts
    evaluation_progress = pyqtSignal(int)  # Emitted during evaluation (progress %)
    evaluation_completed = pyqtSignal(dict)  # Emitted when evaluation is complete with results
    
    def __init__(self, config, parent=None):
        """
        Initialize the evaluation widget.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        self.system_output_path = None
        self.ground_truth_path = None
        self.evaluator = GazeZoneEvaluator()
        self.evaluation_results = None
        
        # Set up the UI
        self._init_ui()
        
        logger.debug("EvaluationWidget initialized")
    
    def _init_ui(self):
        """Initialize the UI components of the widget."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("Gaze Zone Değerlendirme")
        title_font = QFont(self.config.get('fonts', {}).get('family', 'Arial'), 14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # File selection section
        self._create_file_selection_section(main_layout)
        
        # Control panel section
        self._create_control_panel_section(main_layout)
        
        # Results section with tabs
        self.results_tabs = QTabWidget()
        
        # Summary tab
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()
        
        # Overall metrics group
        metrics_group = QGroupBox("Overall Metrics")
        metrics_layout = QGridLayout()
        
        # Create labels for metrics
        self.accuracy_label = QLabel("Accuracy: -")
        self.precision_label = QLabel("Precision: -")
        self.recall_label = QLabel("Recall: -")
        self.f1_label = QLabel("F1 Score: -")
        self.macro_f1_label = QLabel("Macro F1: -")
        self.weighted_f1_label = QLabel("Weighted F1: -")
        self.kappa_label = QLabel("Cohen's Kappa: -")
        self.accuracy_ci_label = QLabel("Accuracy CI: [-]")
        
        # Add metrics to grid layout
        metrics_layout.addWidget(self.accuracy_label, 0, 0)
        metrics_layout.addWidget(self.precision_label, 0, 1)
        metrics_layout.addWidget(self.recall_label, 1, 0)
        metrics_layout.addWidget(self.f1_label, 1, 1)
        metrics_layout.addWidget(self.macro_f1_label, 2, 0)
        metrics_layout.addWidget(self.weighted_f1_label, 2, 1)
        metrics_layout.addWidget(self.kappa_label, 3, 0)
        metrics_layout.addWidget(self.accuracy_ci_label, 3, 1)
        
        metrics_group.setLayout(metrics_layout)
        summary_layout.addWidget(metrics_group)
        
        # Zone-wise performance table
        zone_group = QGroupBox("Zone-wise Performance")
        zone_layout = QVBoxLayout()
        
        self.zone_table = QTableWidget(0, 6)  # Rows will be added dynamically
        self.zone_table.setHorizontalHeaderLabels(["Zone ID", "Precision", "Recall", "F1 Score", "Support", "Accuracy"])
        self.zone_table.horizontalHeader().setStretchLastSection(True)
        
        zone_layout.addWidget(self.zone_table)
        zone_group.setLayout(zone_layout)
        summary_layout.addWidget(zone_group)
        
        # Detailed report button
        self.detailed_report_btn = QPushButton("Generate Detailed Report")
        self.detailed_report_btn.setMinimumHeight(40)
        self.detailed_report_btn.clicked.connect(self.generate_detailed_report)
        self.detailed_report_btn.setEnabled(False)
        summary_layout.addWidget(self.detailed_report_btn)
        
        summary_widget.setLayout(summary_layout)
        self.results_tabs.addTab(summary_widget, "Summary")
        
        # Confusion Matrix tab
        self.confusion_matrix_widget = QWidget()
        cm_layout = QVBoxLayout()
        
        # Placeholder for confusion matrix visualization
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        cm_layout.addWidget(self.canvas)
        
        self.confusion_matrix_widget.setLayout(cm_layout)
        self.results_tabs.addTab(self.confusion_matrix_widget, "Confusion Matrix")
        
        # Zone Performance Visualization tab
        self.zone_performance_widget = QWidget()
        zone_perf_layout = QVBoxLayout()
        
        # Placeholders for zone performance visualizations
        self.zone_perf_figure = Figure(figsize=(8, 6))
        self.zone_perf_canvas = FigureCanvas(self.zone_perf_figure)
        zone_perf_layout.addWidget(self.zone_perf_canvas)
        
        self.zone_performance_widget.setLayout(zone_perf_layout)
        self.results_tabs.addTab(self.zone_performance_widget, "Zone Performance")
        
        # Error Analysis tab
        error_widget = QWidget()
        error_layout = QVBoxLayout()
        
        # Error statistics group
        error_stats_group = QGroupBox("Error Statistics")
        error_stats_layout = QGridLayout()
        
        self.total_samples_label = QLabel("Total Samples: -")
        self.correct_predictions_label = QLabel("Correct Predictions: -")
        self.total_errors_label = QLabel("Total Errors: -")
        self.error_rate_label = QLabel("Error Rate: -")
        
        error_stats_layout.addWidget(self.total_samples_label, 0, 0)
        error_stats_layout.addWidget(self.correct_predictions_label, 0, 1)
        error_stats_layout.addWidget(self.total_errors_label, 1, 0)
        error_stats_layout.addWidget(self.error_rate_label, 1, 1)
        
        error_stats_group.setLayout(error_stats_layout)
        error_layout.addWidget(error_stats_group)
        
        # Most confused pairs group
        confused_pairs_group = QGroupBox("Most Confused Zone Pairs")
        confused_pairs_layout = QVBoxLayout()
        
        self.confused_pairs_table = QTableWidget(0, 4)  # Rows will be added dynamically
        self.confused_pairs_table.setHorizontalHeaderLabels(["True Zone", "Predicted Zone", "Count", "Percentage"])
        self.confused_pairs_table.horizontalHeader().setStretchLastSection(True)
        
        confused_pairs_layout.addWidget(self.confused_pairs_table)
        confused_pairs_group.setLayout(confused_pairs_layout)
        error_layout.addWidget(confused_pairs_group)
        
        error_widget.setLayout(error_layout)
        self.results_tabs.addTab(error_widget, "Error Analysis")
        
        # Statistical Tests tab
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        
        # McNemar's Test group
        mcnemar_group = QGroupBox("McNemar's Test")
        mcnemar_layout = QGridLayout()
        
        self.mcnemar_statistic_label = QLabel("Test Statistic: -")
        self.mcnemar_pvalue_label = QLabel("p-value: -")
        self.mcnemar_result_label = QLabel("Result: -")
        
        mcnemar_layout.addWidget(self.mcnemar_statistic_label, 0, 0)
        mcnemar_layout.addWidget(self.mcnemar_pvalue_label, 0, 1)
        mcnemar_layout.addWidget(self.mcnemar_result_label, 1, 0, 1, 2)
        
        mcnemar_group.setLayout(mcnemar_layout)
        stats_layout.addWidget(mcnemar_group)
        
        stats_widget.setLayout(stats_layout)
        self.results_tabs.addTab(stats_widget, "Statistical Tests")
        
        # Initially hide the results tabs (will be shown after evaluation)
        self.results_tabs.setVisible(False)
        
        # Add to main layout
        main_layout.addWidget(self.results_tabs)
        
        # Set overall widget style
        self._set_widget_style()
        
        logger.debug("EvaluationWidget UI initialized")
    
    def _create_file_selection_section(self, parent_layout):
        """Create the file selection section."""
        file_group = QGroupBox("Dosya Seçimi")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(10)
        
        # System output file selection
        system_layout = QHBoxLayout()
        system_label = QLabel("Sistem Çıktısı:")
        system_label.setFixedWidth(100)
        system_layout.addWidget(system_label)
        
        self.system_path_label = QLabel("Henüz dosya seçilmedi")
        self.system_path_label.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
            padding: 8px;
        """)
        self.system_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        system_layout.addWidget(self.system_path_label)
        
        self.system_browse_button = QPushButton("Dosya Seç")
        self.system_browse_button.setFixedSize(
            self.config.get('controls', {}).get('button_width', 120),
            self.config.get('controls', {}).get('button_height', 40)
        )
        self.system_browse_button.clicked.connect(self._on_system_browse_clicked)
        system_layout.addWidget(self.system_browse_button)
        
        file_layout.addLayout(system_layout)
        
        # Ground truth file selection
        ground_truth_layout = QHBoxLayout()
        ground_truth_label = QLabel("Ground Truth:")
        ground_truth_label.setFixedWidth(100)
        ground_truth_layout.addWidget(ground_truth_label)
        
        self.ground_truth_path_label = QLabel("Henüz dosya seçilmedi")
        self.ground_truth_path_label.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
            padding: 8px;
        """)
        self.ground_truth_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        ground_truth_layout.addWidget(self.ground_truth_path_label)
        
        self.ground_truth_browse_button = QPushButton("Dosya Seç")
        self.ground_truth_browse_button.setFixedSize(
            self.config.get('controls', {}).get('button_width', 120),
            self.config.get('controls', {}).get('button_height', 40)
        )
        self.ground_truth_browse_button.clicked.connect(self._on_ground_truth_browse_clicked)
        ground_truth_layout.addWidget(self.ground_truth_browse_button)
        
        file_layout.addLayout(ground_truth_layout)
        
        # Add help button to show expected file formats
        help_layout = QHBoxLayout()
        help_layout.addStretch()
        
        help_button = QPushButton("Dosya Formatı Yardımı")
        help_button.setStyleSheet("""
            background-color: #f0f0f0;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            color: #007aff;
            font-weight: bold;
        """)
        help_button.clicked.connect(self._show_format_help)
        help_layout.addWidget(help_button)
        
        file_layout.addLayout(help_layout)
        
        parent_layout.addWidget(file_group)
    
    def _create_control_panel_section(self, parent_layout):
        """Create the control panel section."""
        control_group = QGroupBox("Kontrol Paneli")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(10)
        
        # Button and progress layout
        button_progress_layout = QHBoxLayout()
        
        # Evaluate button
        self.evaluate_button = QPushButton("Değerlendirmeyi Başlat")
        self.evaluate_button.setFixedSize(
            int(self.config.get('controls', {}).get('button_width', 120) * 1.5),
            self.config.get('controls', {}).get('button_height', 40)
        )
        self.evaluate_button.setObjectName("start_button")  # For styling
        self.evaluate_button.clicked.connect(self._on_evaluate_clicked)
        self.evaluate_button.setEnabled(False)  # Disabled until files are selected
        button_progress_layout.addWidget(self.evaluate_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(
            self.config.get('indicators', {}).get('progress_bar_height', 20)
        )
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% Tamamlandı")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e1e1e1;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007aff;
                border-radius: 3px;
            }
        """)
        button_progress_layout.addWidget(self.progress_bar)
        
        control_layout.addLayout(button_progress_layout)
        
        # Status message
        self.status_label = QLabel("Durum: Hazır")
        self.status_label.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
            padding: 8px;
        """)
        control_layout.addWidget(self.status_label)
        
        parent_layout.addWidget(control_group)
    
    def _set_widget_style(self):
        """Set the widget style."""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #333333;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e1e1e1;
            }
            QPushButton:pressed {
                background-color: #d1d1d1;
            }
            QPushButton:disabled {
                background-color: #f8f8f8;
                color: #aaaaaa;
            }
            QPushButton#start_button {
                background-color: #007aff;
                color: white;
            }
            QPushButton#start_button:hover {
                background-color: #0069d9;
            }
            QPushButton#start_button:pressed {
                background-color: #0062cc;
            }
            QPushButton#start_button:disabled {
                background-color: #66a9ff;
            }
            QTableWidget {
                border: 1px solid #e1e1e1;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #e1f0ff;
                color: #000000;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #e1e1e1;
                font-weight: bold;
            }
        """)
    
    def _on_system_browse_clicked(self):
        """Handle system output file selection."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("JSON files (*.json)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                selected_file = selected_files[0]
                
                # Validate the file format
                try:
                    with open(selected_file, 'r') as f:
                        data = json.load(f)
                    
                    valid, error_message = validate_system_json(data)
                    if valid:
                        self.system_output_path = selected_file
                        self.system_path_label.setText(os.path.basename(self.system_output_path))
                        self._check_files_selected()
                        self.status_label.setText("Durum: Sistem çıktısı geçerli")
                        logger.info(f"System output file selected: {self.system_output_path}")
                    else:
                        QMessageBox.warning(
                            self,
                            "Geçersiz Sistem Çıktısı",
                            f"Seçilen dosya geçerli bir sistem çıktısı değil:\n\n{error_message}\n\n"
                            f"Lütfen geçerli bir dosya seçin veya örnek formatı inceleyin."
                        )
                        logger.warning(f"Invalid system output file selected: {error_message}")
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Dosya Okuma Hatası",
                        f"Dosya okunurken hata oluştu:\n\n{str(e)}"
                    )
                    logger.error(f"Error reading system output file: {str(e)}")
    
    def _on_ground_truth_browse_clicked(self):
        """Handle ground truth file selection."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("JSON files (*.json)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                selected_file = selected_files[0]
                
                # Validate the file format
                try:
                    with open(selected_file, 'r') as f:
                        data = json.load(f)
                    
                    valid, error_message = validate_ground_truth_json(data)
                    if valid:
                        self.ground_truth_path = selected_file
                        self.ground_truth_path_label.setText(os.path.basename(self.ground_truth_path))
                        self._check_files_selected()
                        self.status_label.setText("Durum: Ground truth geçerli")
                        logger.info(f"Ground truth file selected: {self.ground_truth_path}")
                    else:
                        QMessageBox.warning(
                            self,
                            "Geçersiz Ground Truth",
                            f"Seçilen dosya geçerli bir ground truth değil:\n\n{error_message}\n\n"
                            f"Lütfen geçerli bir dosya seçin veya örnek formatı inceleyin."
                        )
                        logger.warning(f"Invalid ground truth file selected: {error_message}")
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Dosya Okuma Hatası",
                        f"Dosya okunurken hata oluştu:\n\n{str(e)}"
                    )
                    logger.error(f"Error reading ground truth file: {str(e)}")
    
    def _check_files_selected(self):
        """Check if both files are selected and enable the evaluate button if they are."""
        if self.system_output_path and self.ground_truth_path:
            self.evaluate_button.setEnabled(True)
        else:
            self.evaluate_button.setEnabled(False)
    
    def _on_evaluate_clicked(self):
        """Handle evaluate button click."""
        if not self.system_output_path or not self.ground_truth_path:
            self.status_label.setText("Durum: Lütfen önce dosyaları seçin.")
            return
        
        # Validate files and check alignment
        valid, message, sys_data, gt_data = validate_files(self.system_output_path, self.ground_truth_path)
        
        if not valid:
            QMessageBox.warning(
                self,
                "Doğrulama Hatası",
                f"Dosya doğrulama hatası:\n\n{message}\n\n"
                f"Lütfen geçerli dosyaları seçtiğinizden emin olun."
            )
            self.status_label.setText(f"Durum: Doğrulama hatası")
            logger.error(f"Validation error: {message}")
            return
        
        # If there's a warning in the message, show it but continue
        if message.startswith("Warning:"):
            response = QMessageBox.question(
                self,
                "Doğrulama Uyarısı",
                f"{message}\n\nYine de devam etmek istiyor musunuz?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if response == QMessageBox.StandardButton.No:
                self.status_label.setText("Durum: Değerlendirme iptal edildi")
                return
        
        # Disable UI elements during evaluation
        self.evaluate_button.setEnabled(False)
        self.system_browse_button.setEnabled(False)
        self.ground_truth_browse_button.setEnabled(False)
        
        # Reset progress and status
        self.progress_bar.setValue(0)
        self.status_label.setText("Durum: Değerlendirme başlatılıyor...")
        
        # Hide results section if it was visible
        self.results_tabs.setVisible(False)
        
        # Emit evaluation started signal
        self.evaluation_started.emit()
        
        # Start evaluation in a separate thread to keep UI responsive
        # For simplicity in this example, we'll just call the methods directly
        self._run_evaluation()
    
    def _run_evaluation(self):
        """Run the evaluation process."""
        try:
            # Load system output
            system_loaded = self.evaluator.load_system_output(self.system_output_path)
            if not system_loaded:
                self._handle_evaluation_error("Failed to load system output")
                return
            
            self.evaluation_progress.emit(30)
            
            # Load ground truth
            ground_truth_loaded = self.evaluator.load_ground_truth(self.ground_truth_path)
            if not ground_truth_loaded:
                self._handle_evaluation_error("Failed to load ground truth")
                return
            
            self.evaluation_progress.emit(60)
            
            # Align data
            aligned_data = self.evaluator.align_data()
            if not aligned_data:
                self._handle_evaluation_error("Failed to align data")
                return
            
            self.evaluation_progress.emit(80)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics()
            if not metrics:
                self._handle_evaluation_error("Failed to calculate metrics")
                return
            
            # Generate report
            detailed_report = self.evaluator.generate_detailed_report()
            if not detailed_report:
                logger.warning("Failed to generate detailed report")
            
            self.evaluation_progress.emit(100)
            
            # Save results
            self.evaluation_results = detailed_report
            
            # Update UI
            self._update_results_ui()
            
            # Enable detailed report button
            self.detailed_report_btn.setEnabled(True)
            
            # Show results tabs
            self.results_tabs.setVisible(True)
            
            # Update status
            self.status_label.setText("Durum: Değerlendirme tamamlandı")
            
            # Emit completion signal
            self.evaluation_completed.emit(detailed_report)
            
            logger.info(f"Evaluation completed successfully. Processed {aligned_data['num_frames']} frames.")
            
        except Exception as e:
            self._handle_evaluation_error(str(e))
    
    def _handle_evaluation_error(self, error_message):
        """Handle evaluation errors."""
        self.status_label.setText(f"Durum: Hata - {error_message}")
        
        # Re-enable UI elements
        self.system_browse_button.setEnabled(True)
        self.ground_truth_browse_button.setEnabled(True)
        self.evaluate_button.setEnabled(True)
        
        logger.error(f"Evaluation error: {error_message}")
    
    def _update_results_ui(self):
        """Update the UI with evaluation results."""
        if not self.evaluation_results:
            return
        
        # Get the summary metrics
        summary = self.evaluation_results["summary"]
        
        # Update basic metrics
        self.accuracy_label.setText(f"Accuracy: {summary['accuracy']:.4f}")
        self.precision_label.setText(f"Precision: {summary['precision']:.4f}")
        self.recall_label.setText(f"Recall: {summary['recall']:.4f}")
        self.f1_label.setText(f"F1 Score: {summary['f1_score']:.4f}")
        
        # Update advanced metrics if available
        if "macro_f1" in summary:
            self.macro_f1_label.setText(f"Macro F1: {summary['macro_f1']:.4f}")
        if "weighted_f1" in summary:
            self.weighted_f1_label.setText(f"Weighted F1: {summary['weighted_f1']:.4f}")
        if "cohen_kappa" in summary:
            self.kappa_label.setText(f"Cohen's Kappa: {summary['cohen_kappa']:.4f}")
        if "accuracy_ci_lower" in summary and "accuracy_ci_upper" in summary:
            ci_text = f"Accuracy CI: [{summary['accuracy_ci_lower']:.4f}, {summary['accuracy_ci_upper']:.4f}]"
            self.accuracy_ci_label.setText(ci_text)
        
        # Update zone performance table
        self.zone_table.setRowCount(0)  # Clear existing rows
        zone_performance = self.evaluation_results["zone_performance"]
        
        for zone_id, metrics in zone_performance.items():
            row_position = self.zone_table.rowCount()
            self.zone_table.insertRow(row_position)
            
            # Set zone ID
            self.zone_table.setItem(row_position, 0, QTableWidgetItem(zone_id))
            
            # Set metrics
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1_score = metrics.get("f1_score", 0)
            support = metrics.get("support", 0)
            accuracy = metrics.get("accuracy", 0)
            
            self.zone_table.setItem(row_position, 1, QTableWidgetItem(f"{precision:.4f}"))
            self.zone_table.setItem(row_position, 2, QTableWidgetItem(f"{recall:.4f}"))
            self.zone_table.setItem(row_position, 3, QTableWidgetItem(f"{f1_score:.4f}"))
            self.zone_table.setItem(row_position, 4, QTableWidgetItem(str(support)))
            self.zone_table.setItem(row_position, 5, QTableWidgetItem(f"{accuracy:.4f}"))
        
        # Resize columns to content
        self.zone_table.resizeColumnsToContents()
        
        # Update confusion matrix visualization
        self.plot_confusion_matrix()
        
        # Update zone performance visualization
        self.plot_zone_performance()
        
        # Update error analysis if available
        if "error_analysis" in self.evaluation_results:
            error_analysis = self.evaluation_results["error_analysis"]
            
            # Update error statistics
            if "total_samples" in error_analysis:
                self.total_samples_label.setText(f"Total Samples: {error_analysis.get('total_samples', 0)}")
            if "correct_predictions" in error_analysis:
                self.correct_predictions_label.setText(f"Correct Predictions: {error_analysis.get('correct_predictions', 0)}")
            if "total_errors" in error_analysis:
                self.total_errors_label.setText(f"Total Errors: {error_analysis.get('total_errors', 0)}")
            if "error_rate" in error_analysis:
                self.error_rate_label.setText(f"Error Rate: {error_analysis.get('error_rate', 0):.4f}")
            
            # Update confused pairs table
            self.update_confused_pairs_table(error_analysis)
            
            # Update McNemar's test if available
            if "mcnemar_test" in error_analysis:
                mcnemar = error_analysis["mcnemar_test"]
                self.mcnemar_statistic_label.setText(f"Test Statistic: {mcnemar.get('statistic', 0):.4f}")
                self.mcnemar_pvalue_label.setText(f"p-value: {mcnemar.get('pvalue', 0):.4f}")
                
                # Interpret p-value
                p_value = mcnemar.get('pvalue', 1)
                if p_value < 0.05:
                    result = "Significant difference detected (p < 0.05)"
                else:
                    result = "No significant difference detected (p ≥ 0.05)"
                self.mcnemar_result_label.setText(f"Result: {result}")

    def update_confused_pairs_table(self, error_analysis):
        """Update the confused pairs table"""
        # Get confused pairs
        confused_pairs = error_analysis.get("confused_pairs", [])
        
        # Clear table
        self.confused_pairs_table.setRowCount(0)
        
        # Add rows for each confused pair
        for pair in confused_pairs[:10]:  # Show top 10
            row_position = self.confused_pairs_table.rowCount()
            self.confused_pairs_table.insertRow(row_position)
            
            # Set data
            self.confused_pairs_table.setItem(row_position, 0, QTableWidgetItem(str(pair.get("true_zone", ""))))
            self.confused_pairs_table.setItem(row_position, 1, QTableWidgetItem(str(pair.get("predicted_zone", ""))))
            self.confused_pairs_table.setItem(row_position, 2, QTableWidgetItem(str(pair.get("count", 0))))
            self.confused_pairs_table.setItem(row_position, 3, QTableWidgetItem(f"{pair.get('percentage', 0):.2f}%"))
        
        # Resize columns to content
        self.confused_pairs_table.resizeColumnsToContents()

    def plot_confusion_matrix(self):
        """Plot confusion matrix as heatmap using the visualization module"""
        # Get confusion matrix and zones
        if not self.evaluation_results or "confusion_matrix" not in self.evaluator.evaluation_results:
            return
        
        cm = np.array(self.evaluator.evaluation_results["confusion_matrix"])
        zones = [str(z) for z in self.evaluator.evaluation_results["zones"]]
        
        if len(cm) == 0 or len(zones) == 0:
            return
        
        # Clear previous plot
        self.figure.clear()
        
        # Create new plot using the visualization module
        fig = plot_confusion_matrix(cm, zones, normalize=True, title="Normalized Confusion Matrix")
        
        # Copy the figure to our canvas figure
        for ax in fig.get_axes():
            new_ax = self.figure.add_subplot(111)
            new_ax.imshow(ax.get_images()[0].get_array(), cmap=ax.get_images()[0].get_cmap())
            
            # Copy text annotations
            for text in ax.texts:
                new_ax.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                           ha=text.get_ha(), va=text.get_va(), color=text.get_color(),
                           fontsize=text.get_fontsize())
            
            # Copy labels and title
            new_ax.set_xticks(ax.get_xticks())
            new_ax.set_yticks(ax.get_yticks())
            new_ax.set_xticklabels(ax.get_xticklabels())
            new_ax.set_yticklabels(ax.get_yticklabels())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_title(ax.get_title())
            
            # Set axes
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())
        
        # Update canvas
        self.canvas.draw()

    def plot_zone_performance(self):
        """Plot zone performance metrics using the visualization module"""
        if not self.evaluation_results or "zone_performance" not in self.evaluation_results:
            return
        
        # Prepare zone metrics for visualization
        zone_metrics = {}
        for zone_id, metrics in self.evaluation_results["zone_performance"].items():
            zone_metrics[zone_id] = {
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "support": metrics.get("support", 0),
                "accuracy": metrics.get("accuracy", 0)
            }
        
        if not zone_metrics:
            return
        
        # Clear previous plots
        self.zone_perf_figure.clear()
        
        # Create subplots for different metrics
        grid_spec = self.zone_perf_figure.add_gridspec(2, 1, height_ratios=[1.5, 1])
        
        # Plot F1 scores in top subplot
        ax_f1 = self.zone_perf_figure.add_subplot(grid_spec[0, 0])
        
        # Get F1 score plot from visualization module
        f1_fig = plot_zone_performance(zone_metrics, metric="f1_score", title="Per-Zone F1 Score")
        
        # Copy content to our axis
        for ax in f1_fig.get_axes():
            # Copy the barplot
            for patch in ax.patches:
                rect = patch.get_rect()
                new_patch = ax_f1.barh(
                    rect.y + rect.height/2, rect.width, 
                    height=rect.height, color=patch.get_facecolor()
                )
            
            # Copy labels and title
            ax_f1.set_yticks(ax.get_yticks())
            ax_f1.set_yticklabels(ax.get_yticklabels())
            ax_f1.set_xlabel(ax.get_xlabel())
            ax_f1.set_ylabel(ax.get_ylabel())
            ax_f1.set_title(ax.get_title())
            
            # Set limits
            ax_f1.set_xlim(0, 1.0)
            
            # Add gridlines
            ax_f1.grid(True, linestyle='--', alpha=0.6)
            
            # Add value annotations
            for i, zone in enumerate(zone_metrics.keys()):
                ax_f1.text(
                    zone_metrics[zone]["f1_score"] + 0.02, i, 
                    f'{zone_metrics[zone]["f1_score"]:.3f}', 
                    va='center'
                )
        
        # Plot precision and recall in bottom subplot
        ax_pr = self.zone_perf_figure.add_subplot(grid_spec[1, 0])
        
        # Prepare data
        zones = list(zone_metrics.keys())
        precision = [zone_metrics[z]["precision"] for z in zones]
        recall = [zone_metrics[z]["recall"] for z in zones]
        
        # Set width of bars
        bar_width = 0.35
        indices = np.arange(len(zones))
        
        # Create bars
        ax_pr.barh(indices - bar_width/2, precision, bar_width, 
                 label='Precision', color='tab:blue', alpha=0.7)
        ax_pr.barh(indices + bar_width/2, recall, bar_width,
                 label='Recall', color='tab:orange', alpha=0.7)
        
        # Add labels
        ax_pr.set_yticks(indices)
        ax_pr.set_yticklabels(zones)
        ax_pr.set_xlabel('Score')
        ax_pr.set_ylabel('Zone')
        ax_pr.set_title('Precision vs Recall by Zone')
        ax_pr.legend(loc='lower right')
        
        # Set limits and grid
        ax_pr.set_xlim(0, 1.0)
        ax_pr.grid(True, linestyle='--', alpha=0.6)
        
        # Adjust layout
        self.zone_perf_figure.tight_layout()
        
        # Update canvas
        self.zone_perf_canvas.draw()

    def get_evaluation_results(self):
        """Get the evaluation results."""
        return self.evaluation_results 

    def generate_detailed_report(self):
        """Generate and display/export detailed report"""
        # Check if evaluation has been done
        if not self.evaluator.evaluation_results:
            QMessageBox.warning(self, "No Data", "Please run evaluation first.")
            return
        
        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Report"
        )
        
        if not output_dir:
            return
            
        try:
            # Generate visualization report
            self.evaluator.generate_visualization_report(output_dir=output_dir)
            
            # Show success message
            QMessageBox.information(
                self, "Report Generated", 
                f"Detailed report and visualizations have been saved to:\n{output_dir}"
            )
            
            logger.info(f"Detailed report generated and saved to {output_dir}")
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to generate report: {str(e)}"
            )
            logger.error(f"Error generating detailed report: {str(e)}")

    def _show_format_help(self):
        """Show a dialog with information about expected file formats."""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Dosya Formatı Yardımı")
        help_dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(help_dialog)
        
        # Title
        title_label = QLabel("Beklenen Dosya Formatları")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Tabs for different formats
        tabs = QTabWidget()
        
        # System Output Tab
        system_widget = QWidget()
        system_layout = QVBoxLayout(system_widget)
        
        system_info = QLabel(
            "Sistem çıktısı, her bir video karesi için göz bakış bölgesi tahminlerini içeren bir JSON dosyasıdır. "
            "Aşağıdaki formatı takip etmelidir:"
        )
        system_info.setWordWrap(True)
        system_layout.addWidget(system_info)
        
        system_example = QTextEdit()
        system_example.setReadOnly(True)
        system_example.setPlainText(get_validation_example("system"))
        system_layout.addWidget(system_example)
        
        system_notes = QLabel(
            "Önemli alanlar:\n"
            "- frame_analysis: Kare analizlerini içeren dizi\n"
            "- frame_number: Her kare için benzersiz numara\n"
            "- gaze: Bakış verisi\n"
            "- zone_id: Bakış bölgesi ID'si (0-5 arasında)"
        )
        system_layout.addWidget(system_notes)
        
        tabs.addTab(system_widget, "Sistem Çıktısı")
        
        # Ground Truth Tab
        gt_widget = QWidget()
        gt_layout = QVBoxLayout(gt_widget)
        
        gt_info = QLabel(
            "Ground truth, her bir video karesi için gerçek göz bakış bölgelerini içeren bir JSON dosyasıdır. "
            "Aşağıdaki formatı takip etmelidir:"
        )
        gt_info.setWordWrap(True)
        gt_layout.addWidget(gt_info)
        
        gt_example = QTextEdit()
        gt_example.setReadOnly(True)
        gt_example.setPlainText(get_validation_example("ground_truth"))
        gt_layout.addWidget(gt_example)
        
        gt_notes = QLabel(
            "Önemli noktalar:\n"
            "- Anahtarlar: Kare numaraları (sayısal değer olmalı)\n"
            "- Değerler: Bakış bölgesi ID'leri (0-5 arasında)\n"
            "- Null değerler kabul edilmez"
        )
        gt_layout.addWidget(gt_notes)
        
        tabs.addTab(gt_widget, "Ground Truth")
        
        # Validation Tab
        validation_widget = QWidget()
        validation_layout = QVBoxLayout(validation_widget)
        
        validation_info = QLabel(
            "Dosya doğrulama kuralları:\n\n"
            "1. Sistem çıktısı doğrulama:\n"
            "   - JSON formatında olmalı\n"
            "   - 'frame_analysis' alanı bir dizi olmalı\n"
            "   - Her karede 'frame_number' ve 'gaze.zone_id' bulunmalı\n"
            "   - 'zone_id' değerleri 0-5 arasında olmalı\n\n"
            "2. Ground truth doğrulama:\n"
            "   - JSON formatında olmalı\n"
            "   - Anahtarlar geçerli kare numaraları olmalı\n"
            "   - Değerler 0-5 arasında tamsayılar olmalı\n"
            "   - Null değerler kabul edilmez\n\n"
            "3. Veri uyumu doğrulama:\n"
            "   - En az 50 ortak kare bulunmalı\n"
            "   - Sistem ve ground truth arasında yeterli örtüşme olmalı"
        )
        validation_info.setWordWrap(True)
        validation_layout.addWidget(validation_info)
        
        tabs.addTab(validation_widget, "Doğrulama Kuralları")
        
        layout.addWidget(tabs)
        
        # Close button
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(help_dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        help_dialog.exec() 