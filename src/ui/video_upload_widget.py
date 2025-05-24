#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video upload widget for the driver drowsiness detection application.

This module implements a widget for uploading and analyzing video files.
"""

import os
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFileDialog, QProgressBar, QSizePolicy, QFrame,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPixmap, QImage

from src.analysis.video_analyzer import VideoAnalyzer

# Get module-specific logger
logger = logging.getLogger(__name__)

class VideoUploadWidget(QWidget):
    """
    Widget for uploading and analyzing video files.
    
    This class provides functionality to select video files,
    display a preview, and start analysis.
    """
    
    # Define signals
    video_selected = pyqtSignal(str)  # Emitted when a video is selected (path)
    analysis_started = pyqtSignal(str)  # Emitted when analysis starts (path)
    analysis_progress = pyqtSignal(int)  # Emitted during analysis (progress %)
    analysis_completed = pyqtSignal()  # Emitted when analysis is complete
    
    def __init__(self, config, parent=None):
        """
        Initialize the video upload widget.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        self.video_path = None
        self.preview_frame = None
        self.video_analyzer = None
        
        # Set up the UI
        self._init_ui()
        
        logger.debug("VideoUploadWidget initialized")
    
    def _init_ui(self):
        """Initialize the UI components of the widget."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Video Yükleme ve Analiz")
        title_font = QFont(self.config.get('fonts', {}).get('family', 'Arial'), 14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # File selection section
        file_section_layout = QHBoxLayout()
        
        # File path label
        self.file_path_label = QLabel("Henüz video seçilmedi")
        self.file_path_label.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
            padding: 8px;
        """)
        self.file_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        file_section_layout.addWidget(self.file_path_label)
        
        # Browse button
        self.browse_button = QPushButton("Video Seç")
        self.browse_button.setFixedSize(
            self.config.get('controls', {}).get('button_width', 120),
            self.config.get('controls', {}).get('button_height', 40)
        )
        self.browse_button.clicked.connect(self._on_browse_clicked)
        file_section_layout.addWidget(self.browse_button)
        
        main_layout.addLayout(file_section_layout)
        
        # Video preview section
        preview_section_layout = QVBoxLayout()
        
        # Preview label
        preview_label = QLabel("Video Önizleme:")
        preview_label.setFont(QFont(self.config.get('fonts', {}).get('family', 'Arial'), 12))
        preview_section_layout.addWidget(preview_label)
        
        # Preview frame
        self.preview_frame_label = QLabel("Video seçildiğinde önizleme burada gösterilecek")
        self.preview_frame_label.setFixedSize(
            self.config.get('video_frame', {}).get('width', 640) // 2,  # Half the normal size
            self.config.get('video_frame', {}).get('height', 480) // 2
        )
        self.preview_frame_label.setFrameShape(QFrame.Shape.Box)
        self.preview_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_frame_label.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
        """)
        preview_section_layout.addWidget(self.preview_frame_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addLayout(preview_section_layout)
        
        # Result info section
        self.result_label = QLabel("")
        self.result_label.setFont(QFont(self.config.get('fonts', {}).get('family', 'Arial'), 10))
        self.result_label.setStyleSheet("""
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 8px;
            margin-top: 10px;
        """)
        self.result_label.setVisible(False)
        main_layout.addWidget(self.result_label)
        
        # Progress section
        progress_section_layout = QVBoxLayout()
        
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
        progress_section_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(progress_section_layout)
        
        # Control buttons section
        buttons_layout = QHBoxLayout()
        
        # Start analysis button
        self.start_button = QPushButton("Analizi Başlat")
        self.start_button.setFixedSize(
            self.config.get('controls', {}).get('button_width', 120),
            self.config.get('controls', {}).get('button_height', 40)
        )
        self.start_button.setObjectName("start_button")  # For styling
        self.start_button.clicked.connect(self._on_start_clicked)
        self.start_button.setEnabled(False)  # Disabled until video is selected
        buttons_layout.addWidget(self.start_button)
        
        # Cancel button
        self.cancel_button = QPushButton("İptal")
        self.cancel_button.setFixedSize(
            self.config.get('controls', {}).get('button_width', 120),
            self.config.get('controls', {}).get('button_height', 40)
        )
        self.cancel_button.setObjectName("stop_button")  # For styling
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        self.cancel_button.setEnabled(False)  # Disabled until analysis starts
        buttons_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Set widget style
        self.setStyleSheet("""
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
            QPushButton#stop_button {
                background-color: #ff3b30;
                color: white;
            }
            QPushButton#stop_button:hover {
                background-color: #dc3545;
            }
            QPushButton#stop_button:pressed {
                background-color: #c82333;
            }
            QPushButton#stop_button:disabled {
                background-color: #ff8680;
            }
        """)
    
    def _on_browse_clicked(self):
        """Handle browse button click to select a video file."""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mkv)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.video_path = selected_files[0]
                self.file_path_label.setText(os.path.basename(self.video_path))
                self.start_button.setEnabled(True)
                
                # Reset result label
                self.result_label.setVisible(False)
                
                # Emit signal
                self.video_selected.emit(self.video_path)
                
                # Update preview
                self._update_preview()
                
                logger.info(f"Video selected: {self.video_path}")
    
    def _update_preview(self):
        """Update the preview frame with the first frame of the selected video."""
        import cv2
        
        if not self.video_path or not os.path.exists(self.video_path):
            logger.warning("Cannot update preview: Invalid video path")
            return
        
        try:
            # Open video and get first frame
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to QImage
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Convert to QPixmap and set to label
                pixmap = QPixmap.fromImage(q_img)
                
                # Scale pixmap to fit label while maintaining aspect ratio
                self.preview_frame_label.setPixmap(pixmap.scaled(
                    self.preview_frame_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
                
                # Save preview frame for later use
                self.preview_frame = frame
                
                logger.debug(f"Preview updated with first frame: {w}x{h}")
            else:
                self.preview_frame_label.setText("Önizleme oluşturulamadı")
                logger.warning("Could not read first frame from video")
        except Exception as e:
            self.preview_frame_label.setText(f"Önizleme hatası: {str(e)}")
            logger.error(f"Error updating preview: {str(e)}")
    
    def _on_start_clicked(self):
        """Handle start analysis button click."""
        if not self.video_path:
            logger.warning("Cannot start analysis: No video selected")
            return
        
        # Update UI state
        self.browse_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.result_label.setVisible(False)
        
        # Initialize video analyzer if not already done
        if not self.video_analyzer:
            self.video_analyzer = VideoAnalyzer(self.config)
            
            # Connect signals
            self.video_analyzer.progress_updated.connect(self.update_progress)
            self.video_analyzer.analysis_completed.connect(self._on_analysis_completed)
            self.video_analyzer.error_occurred.connect(self._on_analysis_error)
        
        # Emit signal to start analysis
        self.analysis_started.emit(self.video_path)
        
        # Start analysis
        success = self.video_analyzer.analyze_video(self.video_path)
        
        if not success:
            # Reset UI state
            self.browse_button.setEnabled(True)
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.progress_bar.setValue(0)
            
            # Show error message
            QMessageBox.critical(
                self,
                "Analiz Hatası",
                "Video analizi başlatılamadı. Lütfen videoyu kontrol edin ve tekrar deneyin."
            )
        
        logger.info(f"Analysis started for video: {self.video_path}")
    
    def _on_cancel_clicked(self):
        """Handle cancel button click."""
        # Stop analysis
        if self.video_analyzer and self.video_analyzer.is_analyzing:
            self.video_analyzer.stop_analysis()
        
        # Reset UI state
        self.browse_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        logger.info("Analysis cancelled")
    
    def update_progress(self, value):
        """
        Update the progress bar value.
        
        Args:
            value: Progress percentage (0-100)
        """
        self.progress_bar.setValue(value)
        self.analysis_progress.emit(value)
        
        # If progress is complete, reset UI
        if value >= 100:
            self.browse_button.setEnabled(True)
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.analysis_completed.emit()
            logger.info("Analysis completed")
    
    def _on_analysis_completed(self, statistics):
        """
        Handle analysis completion.
        
        Args:
            statistics: Analysis statistics
        """
        # Check if the video analyzer has saved frame zone data
        if hasattr(self.video_analyzer, '_save_frame_zone_data'):
            # _save_frame_zone_data metodunu çağır ve dosya yolunu al
            output_path = self.video_analyzer._save_frame_zone_data()
            
            if output_path and os.path.exists(output_path):
                # Sonuç bilgisi göster
                self.result_label.setText(
                    f"<b>Analiz tamamlandı!</b> Gaze zone verisi şu konuma kaydedildi: "
                    f"<a href=\"file://{output_path}\">{os.path.basename(output_path)}</a>"
                )
                self.result_label.setOpenExternalLinks(True)
                self.result_label.setVisible(True)
                
                # Bilgi mesajı göster
                QMessageBox.information(
                    self,
                    "Analiz Tamamlandı",
                    f"Video analizi başarıyla tamamlandı!\n\n"
                    f"Her frame için gaze zone bilgisi JSON formatında kaydedildi:\n{output_path}"
                )
            else:
                # Hata mesajı göster
                self.result_label.setText(
                    "<b>Analiz tamamlandı!</b> Ancak gaze zone verileri kaydedilemedi."
                )
                self.result_label.setStyleSheet("""
                    background-color: #fff3cd;
                    color: #856404;
                    border: 1px solid #ffeeba;
                    border-radius: 4px;
                    padding: 8px;
                    margin-top: 10px;
                """)
                self.result_label.setVisible(True)
        
        logger.info("Analysis results processed")
    
    def _on_analysis_error(self, error_message):
        """
        Handle analysis error.
        
        Args:
            error_message: Error message
        """
        # Reset UI state
        self.browse_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Show error message
        QMessageBox.critical(
            self,
            "Analiz Hatası",
            f"Video analizi sırasında bir hata oluştu:\n\n{error_message}"
        )
        
        logger.error(f"Analysis error: {error_message}")
    
    def get_video_path(self):
        """
        Get the selected video file path.
        
        Returns:
            str: Path to the selected video file
        """
        return self.video_path 