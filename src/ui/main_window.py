#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main window for the driver drowsiness detection application.

This module implements the main window GUI for the driver drowsiness
detection system, using modular UI components.
"""

import sys
import os
import cv2
import logging
import time
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStatusBar, QLabel, QSizePolicy, QSlider, QCheckBox, QPushButton, QDialog, QMessageBox, QTextEdit
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QFont, QPixmap, QImage

# Import utilities - update to use the new modular architecture
from src.utils import MediaPipeHelper, get_mediapipe_helper, load_ui_config

# Import UI components
from src.ui.video_panel import VideoPanel
from src.ui.metrics_panel import MetricsPanel
from src.ui.control_panel import ControlPanel
from src.ui.chart_panel import ChartPanel
from src.ui.menu_manager import MenuManager
from src.ui.dialogs import SettingsDialog, AboutDialog
from src.ui.graph_utils import ExpandedChartsWindow, create_chart_data_from_series
from src.ui.video_upload_widget import VideoUploadWidget  # Import the VideoUploadWidget

# Import logger for application
from src.utils.logger import DrowsinessLogger

# Import gaze statistics recorder
from src.detection.gaze_statistics import get_gaze_statistics_recorder
from src.detection.gaze_zone_detector import get_gaze_zone_detector
from src.detection.gaze_duration_monitor import get_gaze_duration_monitor, WarningLevel

# Import video analyzer
from src.analysis.video_analyzer import VideoAnalyzer, FrameData

class DriverDrowsinessMainWindow(QMainWindow):
    """
    Main window for the driver drowsiness detection application.
    
    This class implements the main window GUI as specified, with a menu bar,
    central widget with video display and metrics,
    and a status bar.
    """
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Set window title and icon
        self.setWindowTitle("Sürücü Uykululuk Tespit Sistemi")
        
        # Initialize variables
        self.cap = None
        self.mediapipe_helper = None  # Update variable name for clarity
        self.is_capturing = False
        self.show_landmarks = False
        self.show_head_pose = False
        self.show_gaze = False
        self.show_gaze_zone = False
        self.current_gaze_zone = None
        self.gaze_duration_monitor = None
        self.distraction_level = "NORMAL"
        self.distraction_reasons = []
        
        # Load configuration
        self.config = load_ui_config()
        
        # Camera settings - ana konfigürasyon dosyasından al
        self.camera_id = self.config.get('camera', {}).get('device_id', 0)
        self.camera_width = self.config.get('camera', {}).get('width', 640)
        self.camera_height = self.config.get('camera', {}).get('height', 480)
        self.camera_fps = self.config.get('camera', {}).get('fps', 30)
        
        # UI konfigürasyonunu ayrıca sakla
        self.ui_config = self.config.get('ui', {})
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Create drowsiness logger instance
        self.drowsiness_logger_instance = DrowsinessLogger(
            name='ui',
            console_level='info',
            file_level='debug',
            config=self.config
        )
        # Get the actual logger from the instance
        self.drowsiness_logger = self.drowsiness_logger_instance
        
        self.logger.info("Main window initialization started")
        
        # Initialize UI
        self._init_ui()
        
        # Initialize timers
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.update_interval_ms = 33  # ~30 FPS
        
        # Initialize chart timer
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self._update_chart_data)
        self.update_interval_sec = 0.05  # 50ms update for chart
        self.time_counter = 0.0  # Time counter for chart (seconds)
        
        # Initialize drowsiness metrics
        self.eye_closure_history = []
        self.max_history_frames = int(self.config.get('detection', {}).get('perclos_window_sec', 60) * self.camera_fps)
        
        self.logger.info("Main window initialized successfully")
    
    def _init_ui(self):
        """
        Initialize the UI components of the main window.
        
        This method sets up all the UI components, including the menu bar,
        central widget, and status bar.
        """
        # Set window properties
        self.setWindowTitle(self.ui_config['window']['title'])
        self.resize(
            self.ui_config['window']['width'],
            self.ui_config['window']['height']
        )
        self.setMinimumSize(
            self.ui_config['window']['min_width'],
            self.ui_config['window']['min_height']
        )
        
        # Set application style to be minimalist and clean
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: white;
                color: #333333;
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
            QPushButton:checked {
                background-color: #d1d1d1;
                border: 2px solid #007aff;
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
            QStatusBar {
                background-color: #f5f5f5;
                color: #666666;
            }
        """)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central widget
        self._create_central_widget()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır")
        
        # Set initial status
        self.update_status("Sistem hazır. Başlamak için 'Başlat' düğmesine tıklayın.")
    
    def _create_menu_bar(self):
        """Create the menu bar with File, View, and Help menus."""
        self.menu_manager = MenuManager(self.ui_config, self)
        self.setMenuBar(self.menu_manager)
        
        # Connect menu signals
        self.menu_manager.upload_video_triggered.connect(self.on_upload_video)  # Connect the new signal
        self.menu_manager.start_triggered.connect(self.on_start)
        self.menu_manager.stop_triggered.connect(self.on_stop)
        self.menu_manager.settings_triggered.connect(self.on_settings)
        self.menu_manager.exit_triggered.connect(self.close)
        self.menu_manager.stats_toggle_triggered.connect(self._toggle_stats_panel)
        self.menu_manager.chart_toggle_triggered.connect(self._toggle_chart)
        self.menu_manager.about_triggered.connect(self.on_about)
    
    def _create_central_widget(self):
        """Create the central widget with video display, metrics, and controls."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Set main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(
            self.ui_config['layout']['margin'],
            self.ui_config['layout']['margin'],
            self.ui_config['layout']['margin'],
            self.ui_config['layout']['margin']
        )
        main_layout.setSpacing(self.ui_config['layout']['spacing'])
        
        # Top area (video + 3D model + stats)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(5)  # Paneller arasında daha az boşluk
        
        # Üst layout'un ağırlıklarını eşit ayarla
        top_layout.setStretch(0, 1)  # Video paneli ağırlığı
        top_layout.setStretch(1, 1)  # 3D model ağırlığı
        top_layout.setStretch(2, 1)  # Metrik ve kontrol paneli ağırlığı
        
        # ----- SOL BÖLGE: VİDEO PANEL -----
        # Video panel layoutu
        video_container = QVBoxLayout()
        
        # Video panel
        self.video_panel = VideoPanel(self.ui_config)
        video_container.addWidget(self.video_panel)
        
        # Sol tarafta video paneli için bir container widget oluştur ve hizalamayı ayarla
        video_widget = QWidget()
        video_widget.setLayout(video_container)
        video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # Stretch faktörünü 1 olarak ayarla (sabit genişlik oranı)
        top_layout.addWidget(video_widget, 1)
        
        # ----- MERKEZ BÖLGE: 3D HEAD POSE MODEL -----
        from src.ui.head_pose_model import Head3DPanel, HeadPoseModelWidget
        
        # 3D modelin kontrolleri için widget ve layout oluştur
        self.head_pose_panel = Head3DPanel(self.ui_config)
        # 3D modelin ana bölgede olması için boyut ayarlarını düzenle
        self.head_pose_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Merkezi bölgede 3D model için bir container oluştur
        model_container = QVBoxLayout()
        model_container.setContentsMargins(1, 1, 1, 1)  # Minimum boşluk (1 piksel)
        model_container.setSpacing(0)  # İç boşlukları tamamen kaldır
        model_container.addWidget(self.head_pose_panel)
        
        model_widget = QWidget()
        model_widget.setLayout(model_container)
        model_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # 3D model container'ı için arka plan rengini ayarla, sınırı en aza indir
        model_widget.setStyleSheet("background-color: black; border: none; padding: 0px; margin: 0px;")
        # Stretch faktörünü 1 olarak ayarla (sabit genişlik oranı)
        top_layout.addWidget(model_widget, 1)
        
        # ----- SAĞ BÖLGE: METRİK PANEL ve 3D KONTROLLER -----
        # Düzen değişikliği - 3D model kontrollerinin metrik barlarının altında olmasını sağlayacak
        right_container = QVBoxLayout()
        right_container.setContentsMargins(0, 0, 0, 0)
        right_container.setSpacing(10)
        
        # Metrics panel
        self.metrics_panel = MetricsPanel(self.ui_config)
        # Daha kompakt bir görünüm için boyut politikasını ayarla
        self.metrics_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        right_container.addWidget(self.metrics_panel)
        
        # ----- 3D MODEL KONTROL PANEL -----
        # 3D model kontrolleri için yeni widget oluştur
        from src.ui.head_pose_model import Head3DPanel
        
        # SADECE kontrol paneli için bir wrapper widget oluştur
        control_wrapper = QWidget()
        control_wrapper.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        # Sabit genişlik sınırlamasını kaldır
        
        # 3D model kontrol panel bilgilerini al ve kontrol panele yerleştir
        control_layout = QVBoxLayout(control_wrapper)
        control_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model boyutu kontrolü
        scale_widget = QWidget()
        scale_layout = QVBoxLayout(scale_widget)
        scale_layout.setContentsMargins(0, 5, 0, 5)
        
        scale_label = QLabel("Model Boyutu:")
        scale_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        scale_layout.addWidget(scale_label)
        
        scale_slider_layout = QHBoxLayout()
        self.model_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.model_scale_slider.setMinimum(50)  # 0.5x
        self.model_scale_slider.setMaximum(200)  # 2.0x
        self.model_scale_slider.setValue(100)  # 1.0x (varsayılan)
        self.model_scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.model_scale_slider.setTickInterval(25)
        scale_slider_layout.addWidget(self.model_scale_slider)
        
        self.scale_value_label = QLabel("1.0x")
        self.scale_value_label.setFixedWidth(40)
        self.scale_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        scale_slider_layout.addWidget(self.scale_value_label)
        
        scale_layout.addLayout(scale_slider_layout)
        control_layout.addWidget(scale_widget)
        
        # "Görünümü Sıfırla" butonu
        reset_button = QPushButton("Görünümü Sıfırla")
        reset_button.clicked.connect(lambda: self.head_pose_panel.reset_view())
        reset_button.setStyleSheet("font-weight: bold; padding: 8px; margin-top: 8px;")
        control_layout.addWidget(reset_button)
        
        # Görselleştirme ayarları
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 5, 0, 5)
        
        viz_label = QLabel("Görselleştirme Ayarları:")
        viz_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        viz_layout.addWidget(viz_label)
        
        # Debug gösterimi için checkbox
        self.debug_toggle = QCheckBox("Debug Bilgilerini Göster")
        self.debug_toggle.setChecked(False)
        viz_layout.addWidget(self.debug_toggle)
        
        # Eksenler için checkbox
        self.axes_toggle = QCheckBox("Eksenleri Göster")
        self.axes_toggle.setChecked(True)  # Varsayılan olarak true
        viz_layout.addWidget(self.axes_toggle)
        
        # Grid için checkbox
        self.grid_toggle = QCheckBox("Grid Göster")
        self.grid_toggle.setChecked(False)  # Varsayılan olarak false
        viz_layout.addWidget(self.grid_toggle)
        
        control_layout.addWidget(viz_widget)
        
        # Kontrol panelini sağ tarafa ekle
        right_container.addWidget(control_wrapper)
        
        # Sağ taraf widget'ı
        right_widget = QWidget()
        right_widget.setLayout(right_container)
        right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # Stretch faktörünü 1 olarak ayarla (sabit genişlik oranı)
        top_layout.addWidget(right_widget, 1)
        
        # Yatay boşluğu azalt
        top_layout.setSpacing(5)
        
        main_layout.addLayout(top_layout, 1)  # Top area should take more space
        
        # Kontrol paneli - daha kompakt bir görünüm için
        self.control_panel = ControlPanel(self.ui_config)
        main_layout.addWidget(self.control_panel, 0)  # Bottom control panel should take less space
        
        # Connect control panel signals
        self.control_panel.start_clicked.connect(self.on_start)
        self.control_panel.stop_clicked.connect(self.on_stop)
        self.control_panel.settings_clicked.connect(self.on_settings)
        self.control_panel.landmarks_toggled.connect(self._toggle_landmarks)
        self.control_panel.head_pose_toggled.connect(self._toggle_head_pose)
        self.control_panel.gaze_toggled.connect(self._toggle_gaze)
        self.control_panel.gaze_zone_toggled.connect(self._toggle_gaze_zone)
        self.control_panel.expand_charts_clicked.connect(self._show_expanded_charts)
        
        # Grafik paneli - başlık olmadan direkt paneli ekle
        self.chart_panel = ChartPanel(self.ui_config)
        main_layout.addWidget(self.chart_panel, 0)  # Chart panel should take less space
        
        # Kontrol wrapper sinyal bağlantıları
        # Model boyutu slideri
        self.model_scale_slider.valueChanged.connect(self._on_model_scale_changed)
        # Görselleştirme kontrolleri
        self.debug_toggle.toggled.connect(self._on_debug_toggle)
        self.axes_toggle.toggled.connect(self._on_axes_toggle)
        self.grid_toggle.toggled.connect(self._on_grid_toggle)
    
    def showEvent(self, event):
        """Handle window show event - maximize after UI is fully initialized."""
        super().showEvent(event)
        
        # UI tam olarak yüklendikten sonra tam ekran yap
        if self.ui_config['window'].get('start_maximized', True):
            # QTimer kullanarak bir sonraki event loop'ta maximize yap
            QTimer.singleShot(0, self.showMaximized)
    
    def update_status(self, message):
        """
        Update the status bar message.
        
        Args:
            message: The message to display in the status bar
        """
        self.status_bar.showMessage(message)
    
    def _toggle_stats_panel(self, checked):
        """Toggle the visibility of the stats panel."""
        self.metrics_panel.setVisible(checked)
    
    def _toggle_chart(self, checked):
        """Toggle the visibility of the chart."""
        self.chart_panel.setVisible(checked)
    
    def _toggle_landmarks(self, checked):
        """Toggle visibility of facial landmarks."""
        self.show_landmarks = checked
        status = "gösteriliyor" if self.show_landmarks else "gizleniyor"
        self.update_status(f"Yüz işaretleri {status}")
    
    def _toggle_head_pose(self, checked):
        """Toggle the display of head pose."""
        self.show_head_pose = checked
        status = "gösteriliyor" if self.show_head_pose else "gizleniyor"
        self.update_status(f"Baş duruşu {status}")
    
    def _toggle_gaze(self, checked):
        """Toggle the display of gaze direction."""
        self.show_gaze = checked
        status = "gösteriliyor" if self.show_gaze else "gizleniyor"
        self.update_status(f"Bakış yönü {status}")
    
    def _toggle_gaze_zone(self, checked):
        """Toggle the display of gaze zone."""
        self.show_gaze_zone = checked
        status = "gösteriliyor" if self.show_gaze_zone else "gizleniyor"
        self.update_status(f"Bakış bölgesi {status}")
    
    def on_start(self):
        """Start drowsiness detection."""
        if self.is_capturing:
            self.logger.warning("Detection already running, start button ignored")
            return
            
        self.logger.info("Starting drowsiness detection")
        
        try:
            # Initialize MediaPipe helper
            if not self.mediapipe_helper:
                self.mediapipe_helper = get_mediapipe_helper()
                self.logger.debug("MediaPipe helper initialized")
            
            # Initialize gaze statistics recorder
            self.gaze_stats_recorder = get_gaze_statistics_recorder()
            
            # Initialize GazeDurationMonitor
            self.gaze_duration_monitor = get_gaze_duration_monitor()
            self.logger.info("GazeDurationMonitor initialized")
            
            # Open camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                error_msg = "Camera açılamadı! Kamera bağlantısını kontrol edin."
                self.logger.error(f"Camera open failed: {error_msg}")
                self.update_status(error_msg)
                return
                
            # Read camera properties (may be different from requested)
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera opened: ID={self.camera_id}, resolution={actual_width}x{actual_height}, FPS={actual_fps}")
            
            # Reset counter and start camera timer
            self._frame_counter = 0
            self.camera_timer.start(self.update_interval_ms)
            
            # Start chart timer
            self.time_counter = 0.0
            self.chart_timer.start(int(self.update_interval_sec * 1000))
            
            # Initialize drowsiness detection
            from src.detection.drowsiness_detector import DrowsinessDetector
            self.drowsiness_detector = DrowsinessDetector()
            
            # Update UI state
            self.is_capturing = True
            self.control_panel.update_start_stop_state(True)
            
            # Reset metrics chart
            self.chart_panel.reset()
            
            # Update status
            self.update_status("Uykululuk tespiti başlatıldı.")
            self.logger.info("Drowsiness detection started successfully")
            
        except Exception as e:
            error_msg = f"Kamera başlatma hatası: {str(e)}"
            self.logger.exception(f"Error starting camera: {str(e)}")
            self.update_status(error_msg)
    
    def on_stop(self):
        """Stop drowsiness detection."""
        if not self.is_capturing:
            self.logger.warning("Detection not running, stop button ignored")
            return
            
        self.logger.info("Stopping drowsiness detection")
        
        # Generate gaze statistics report if we have data
        if hasattr(self, 'gaze_stats_recorder'):
            try:
                # Rapor oluşturmadan önce son durumları görelim
                durations = self.gaze_stats_recorder.get_zone_durations()
                percentages = self.gaze_stats_recorder.get_zone_percentages()
                total_time = sum(durations.values())
                
                self.logger.info(f"Final gaze statistics - Total time: {total_time:.2f}s")
                self.logger.info(f"Zone durations: {durations}")
                self.logger.info(f"Zone percentages: {percentages}")
                
                # Rapor oluştur
                self.logger.info("Generating gaze statistics report...")
                report_files = self.gaze_stats_recorder.generate_report()
                
                if report_files and 'csv' in report_files:
                    session_folder = os.path.dirname(report_files['csv'])
                    self.logger.info(f"Gaze statistics report files: {report_files}")
                    self.update_status(f"Gaze statistics saved to: {session_folder}")
                else:
                    self.logger.warning("Gaze statistics report generation failed or returned no paths")
                    self.update_status("Failed to generate gaze statistics report")
            except Exception as e:
                self.logger.error(f"Error generating gaze statistics report: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.update_status("Error generating gaze statistics report")
        
        # Generate gaze duration monitor report if we have data
        if hasattr(self, 'gaze_duration_monitor') and self.gaze_duration_monitor is not None:
            try:
                now = time.time()
                self.logger.info("Generating distraction report...")
                report_paths = self.gaze_duration_monitor.save_report_to_file(now)
                if report_paths:
                    # Eğer PDF rapor varsa, onun klasörünü kullan
                    if isinstance(report_paths, dict) and 'pdf' in report_paths:
                        session_folder = os.path.dirname(report_paths['pdf'])
                        self.logger.info(f"Distraction report saved to: {report_paths['pdf']}")
                    # Eğer JSON rapor varsa, onun klasörünü kullan
                    elif isinstance(report_paths, dict) and 'json' in report_paths:
                        session_folder = os.path.dirname(report_paths['json'])
                        self.logger.info(f"Distraction report saved to: {report_paths['json']}")
                    # Geriye uyumluluk için - string döndürürse
                    elif isinstance(report_paths, str):
                        session_folder = os.path.dirname(report_paths)
                        self.logger.info(f"Distraction report saved to: {report_paths}")
                    else:
                        session_folder = None
                    
                    # Eğer geçerli bir klasör yolu varsa, bilgiyi göster
                    if session_folder:
                        self.update_status(f"Distraction report saved to: {session_folder}")
                    else:
                        self.logger.warning("Could not determine session folder from report paths")
                        self.update_status("Distraction report generated successfully")
                else:
                    self.logger.warning("Distraction report generation failed or returned no path")
                    self.update_status("Failed to generate distraction report")
            except Exception as e:
                self.logger.error(f"Error generating gaze duration report: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.update_status("Error generating distraction report")
        
        # Stop timers
        if self.camera_timer.isActive():
            self.camera_timer.stop()
            
        if self.chart_timer.isActive():
            self.chart_timer.stop()
        
        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.logger.debug("Camera released")
        
        # Reset UI state
        self.is_capturing = False
        self.control_panel.update_start_stop_state(False)
        
        # Free MediaPipe resources
        if self.mediapipe_helper:
            self.mediapipe_helper.release()
            self.mediapipe_helper = None
            self.logger.debug("MediaPipe helper resources released")
        
        # Update status
        self.update_status("Uykululuk tespiti durduruldu.")
        self.logger.info("Drowsiness detection stopped")
    
    def _update_camera_frame(self):
        """Update the video frame with current camera image."""
        if not self.cap or not self.is_capturing:
            return
        
        # FPS ölçümü için zaman ölçümü başlat
        frame_start_time = time.time()
            
        # Read a frame from the camera
        ret, frame = self.cap.read()
        
        if not ret:
            error_msg = "Kameradan görüntü alınamadı!"
            self.logger.error(f"Failed to read frame from camera: {error_msg}")
            self.update_status(error_msg)
            self.on_stop()
            return
            
        # Mirror the frame horizontally (selfie view)
        frame = cv2.flip(frame, 1)
        
        # Process the frame with MediaPipe
        landmarks, face_detected = self.mediapipe_helper.detect_face_landmarks(frame)
        
        # Initialize metrics with default values
        ear = 0.0
        mar = 0.0
        perclos = 0.0
        
        # Variables for drowsiness detection
        left_ear = 0.0
        right_ear = 0.0
        
        # Head pose değişkenlerini ekle
        self._frame_counter = getattr(self, '_frame_counter', 0)
        self._head_pose_update_frequency = 2  # Kaç karede bir head pose güncelleneceği
        
        if face_detected:
            # Head pose tespiti için daha güvenli bir kontrol yap
            try:
                # Her karede değil, belirli aralıklarla head pose hesaplaması yap (performans iyileştirmesi)
                if self._frame_counter % self._head_pose_update_frequency == 0:
                    # Önce MediaPipeHelper'in head pose işlevlerine sahip olup olmadığını kontrol et
                    if hasattr(self.mediapipe_helper, 'get_head_pose') and self.mediapipe_helper.can_detect_head_pose():
                        # Head pose hesapla 
                        pitch, yaw, roll = self.mediapipe_helper.get_head_pose(landmarks, frame)
                        
                        # Head pose değerlerini sakla
                        self.head_pitch = pitch
                        self.head_yaw = yaw
                        self.head_roll = roll
                        
                        # Değerlerin geçerli olduğundan emin ol
                        if not (np.isnan(pitch) or np.isnan(yaw) or np.isnan(roll)):
                            # 3D modeli güncelle - 3D panel tanımlandı mı emin olalım
                            if hasattr(self, 'head_pose_panel'):
                                self.head_pose_panel.update_pose(pitch, yaw, roll)
                                self.logger.debug(f"Head pose updated: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}")
            except Exception as e:
                self.logger.error(f"Error processing head pose: {str(e)}")       
            
            # Get eye landmarks
            left_eye_landmarks = self.mediapipe_helper.get_eye_landmarks(landmarks, left_eye=True)
            right_eye_landmarks = self.mediapipe_helper.get_eye_landmarks(landmarks, left_eye=False)
            
            # Get mouth landmarks - only inner lip
            inner_lip_landmarks = self.mediapipe_helper.get_inner_lip_landmarks(landmarks)
            
            # Calculate metrics
            left_ear = self.mediapipe_helper.get_eye_aspect_ratio(left_eye_landmarks)
            right_ear = self.mediapipe_helper.get_eye_aspect_ratio(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2.0  # Average EAR
            
            # Calculate MAR using landmarks
            mar = self.mediapipe_helper.get_mouth_aspect_ratio(landmarks)
            
            # Update PERCLOS
            is_eye_closed = ear < self.config.get('detection', {}).get('ear_threshold', 0.21)
            self.eye_closure_history.append(1 if is_eye_closed else 0)
            
            # Keep history within window size
            if len(self.eye_closure_history) > self.max_history_frames:
                self.eye_closure_history = self.eye_closure_history[-self.max_history_frames:]
            
            # Calculate PERCLOS as percentage of closed eyes in the window
            if self.eye_closure_history:
                # Minimum veri miktarı kontrolü
                min_history_frames = min(10, self.max_history_frames // 10)
                if len(self.eye_closure_history) < min_history_frames:
                    perclos = 0.0
                else:
                    perclos = (sum(self.eye_closure_history) / len(self.eye_closure_history)) * 100.0
                    
            # Log drowsiness data periodically (every 30 frames)
            if getattr(self, '_frame_counter', 0) % 30 == 0:
                # Get the drowsiness level from the detector
                drowsiness_level = 0
                if hasattr(self, 'drowsiness_detector'):
                    # Get the drowsiness level from the update method's return value
                    drowsiness_result = self.drowsiness_detector.update(
                        ear_value=ear,
                        head_pose=None,
                        gaze_direction=None
                    )
                    drowsiness_level = drowsiness_result['drowsiness_level']
                
                alert_status = drowsiness_level > 0.5
                # Use the instance directly
                try:
                    self.drowsiness_logger_instance.log_drowsiness_data(
                        ear=ear,
                        mar=mar,
                        perclos=perclos,
                        kss_score=None,  # We don't have KSS score here
                        alert_status=alert_status
                    )
                    
                    # Also log system status
                    current_fps = 1.0 / (time.time() - frame_start_time) if (time.time() - frame_start_time) > 0 else 0
                    self.drowsiness_logger_instance.log_system_status(
                        fps=current_fps,
                        frame_count=getattr(self, '_frame_counter', 0),
                        status="Running" if self.is_capturing else "Stopped"
                    )
                except Exception as e:
                    self.logger.error(f"Error logging drowsiness data: {str(e)}")
                
            # Görüntünün bir kopyasını oluştur
            processed_frame = frame.copy()
            
            # Visualize head pose if enabled
            if self.show_head_pose:
                processed_frame = self.mediapipe_helper.visualize_head_pose(
                    processed_frame, 
                    landmarks,
                    visualization_type='cube'
                )
            
            # Visualize gaze direction if enabled
            if self.show_gaze:
                ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
                frame_skip = self.config.get('detection', {}).get('gaze', {}).get('frame_skip', 3)
                processed_frame, normalized_face = self.mediapipe_helper.visualize_gaze(
                    processed_frame, 
                    landmarks,
                    ear_value=ear,
                    ear_threshold=ear_threshold,
                    frame_skip=frame_skip
                )
                
                # Eğer normalize edilmiş yüz görüntüsü varsa, küçük bir pencerede göster
                if normalized_face is not None:
                    # Normalize edilmiş yüz görüntüsünü yeniden boyutlandır
                    norm_face_display = cv2.resize(normalized_face, (112, 112))
                    
                    # Görüntüyü ana kareye yerleştir (sağ üst köşe)
                    h, w = norm_face_display.shape[:2]
                    processed_frame[10:10+h, processed_frame.shape[1]-w-10:processed_frame.shape[1]-10] = norm_face_display
                
                # GAZE ZONE TESPİTİ VE KAYIT - Her durumda yap (show_gaze veya show_gaze_zone durumundan bağımsız)
                # Sadece gaze vector geçerli ise ve gözler açıksa işlem yap
                ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
                if (hasattr(self.mediapipe_helper.gaze_detector, '_last_gaze_vector') and 
                        self.mediapipe_helper.gaze_detector._last_gaze_vector is not None and 
                        (ear is None or ear >= ear_threshold)):
                    
                    # Bakış vektörü alınıyor
                    gaze_vector = self.mediapipe_helper.gaze_detector._last_gaze_vector
                    
                    # Basitleştirilmiş yaklaşım: GazeZoneDetector, zone_durations'ı otomatik olarak güncelleyecek
                    zone_id = self.mediapipe_helper.gaze_detector.get_gaze_target_zone(gaze_vector)
                    self.current_gaze_zone = zone_id
                    
                    # Tespit edilen bölgeyi istatistiklerde kaydet - sadece geçerli bir bölge varsa
                    if hasattr(self, 'gaze_stats_recorder') and zone_id is not None:
                        # İstatistiklere ekle - süre hesabı GazeZoneDetector tarafında yapılacak
                        self.gaze_stats_recorder.record_gaze_zone(zone_id)
                        
                        # Her 300 kayıtta bir (yaklaşık 10 saniyede bir) zone durations'ı log'a yaz
                        if getattr(self, '_gaze_record_counter', 0) % 300 == 0:
                            durations = self.gaze_stats_recorder.get_zone_durations()
                            percentages = self.gaze_stats_recorder.get_zone_percentages()
                            total_time = sum(durations.values())
                            self.logger.info(f"Total recorded gaze time: {total_time:.2f}s")
                            self.logger.info(f"Current zone durations: {durations}")
                            self.logger.info(f"Current zone percentages: {percentages}")
                        
                        # Sayacı artır
                        self._gaze_record_counter = getattr(self, '_gaze_record_counter', 0) + 1
                    
                    # GazeDurationMonitor ile bakış sürelerini takip et - her zaman takip et
                    if hasattr(self, 'gaze_duration_monitor') and self.gaze_duration_monitor is not None and zone_id is not None:
                        try:
                            # Dalgınlık analizi
                            monitor_result = self.gaze_duration_monitor.update(zone_id, time.time())
                            
                            # Dalgınlık seviyesi ve nedenlerini sakla
                            self.distraction_level = monitor_result["distraction_level"]
                            if "warning" in monitor_result and "reasons" in monitor_result["warning"]:
                                self.distraction_reasons = monitor_result["warning"]["reasons"]
                            else:
                                self.distraction_reasons = []
                            
                            # Dalgınlık seviyesine göre uyarı göster
                            if self.distraction_level != "NORMAL":
                                self.logger.warning(f"Distraction detected: {self.distraction_level} - {self.distraction_reasons}")
                        except Exception as e:
                            self.logger.error(f"Error updating gaze duration monitor: {str(e)}")
                    
                    # Bakış açılarını ekranda göster (FPS stilinde)
                    # Sadece gaze vector geçerli ise ve gözler açıksa göster
                    pitch, yaw = np.rad2deg(gaze_vector)
                    
                    # Bakış yönü metnini oluştur - İngilizce olarak
                    pitch_yaw_text = f"Pitch: {pitch:.1f}° Yaw: {yaw:.1f}°"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4  # Daha küçük
                    thickness = 1
                    font_color = (0, 255, 255)
                    
                    # Metin boyutunu al
                    (text_width, text_height), baseline = cv2.getTextSize(pitch_yaw_text, font, font_scale, thickness)
                    
                    # Sağ üst köşe pozisyonu
                    padding = 5
                    text_x = processed_frame.shape[1] - text_width - padding
                    text_y = 30  # Üstten mesafe
                    
                    # Yarı saydam arka plan kutusu çiz
                    overlay = processed_frame.copy()
                    cv2.rectangle(
                        overlay,
                        (text_x - padding, text_y - text_height - padding),
                        (text_x + text_width + padding * 2, text_y + padding),
                        (0, 0, 0),  # Siyah arkaplan
                        -1
                    )
                    # Şeffaflık uygula
                    cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
                    
                    # Pitch ve Yaw yazısını ekle
                    cv2.putText(
                        processed_frame,
                        pitch_yaw_text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        font_color,
                        thickness
                    )
                    
                    # Bakış bölgesi görselleştirme - sadece show_gaze_zone etkinse göster
                    if self.show_gaze_zone and zone_id is not None:
                        # Bölge adını al
                        zone_detector = get_gaze_zone_detector()
                        zone_name = zone_detector.get_zone_name(zone_id)
                        
                        # Bakış bölgesi metnini oluştur
                        gaze_zone_text = f"Gaze Zone: {zone_name} ({zone_id})"
                        
                        # Metin boyutunu al
                        (text_width, text_height), baseline = cv2.getTextSize(gaze_zone_text, font, font_scale, thickness)
                        
                        # Sağ üst köşe pozisyonu - pitch/yaw'dan sonra
                        text_x = processed_frame.shape[1] - text_width - padding
                        text_y = 60  # Pitch/yaw metninin altında
                        
                        # Yarı saydam arka plan kutusu çiz
                        overlay = processed_frame.copy()
                        cv2.rectangle(
                            overlay,
                            (text_x - padding, text_y - text_height - padding),
                            (text_x + text_width + padding * 2, text_y + padding),
                            (0, 0, 0),  # Siyah arkaplan
                            -1
                        )
                        # Şeffaflık uygula
                        cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
                        
                        # Bakış bölgesi yazısını ekle
                        cv2.putText(
                            processed_frame,
                            gaze_zone_text,
                            (text_x, text_y),
                            font,
                            font_scale,
                            font_color,
                            thickness
                        )
            
            # Draw landmarks if enabled - SON OLARAK YÜZ İŞARETLERİNİ ÇİZ
            if self.show_landmarks:
                # Create connections for eyes (to form a polygon)
                left_eye_connections = [(i, i+1) for i in range(len(left_eye_landmarks)-1)]
                left_eye_connections.append((len(left_eye_landmarks)-1, 0))  # Close the loop
                
                right_eye_connections = [(i, i+1) for i in range(len(right_eye_landmarks)-1)]
                right_eye_connections.append((len(right_eye_landmarks)-1, 0))  # Close the loop
                
                # Create connections for inner lip (to form a polygon)
                inner_lip_connections = [(i, i+1) for i in range(len(inner_lip_landmarks)-1)]
                inner_lip_connections.append((len(inner_lip_landmarks)-1, 0))  # Close the loop
                
                # Draw eye landmarks and connections
                processed_frame = self.mediapipe_helper.draw_facial_landmarks(
                    processed_frame, left_eye_landmarks, 
                    connections=left_eye_connections,
                    landmark_color=(0, 255, 0), 
                    connection_color=(0, 255, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
                processed_frame = self.mediapipe_helper.draw_facial_landmarks(
                    processed_frame, right_eye_landmarks, 
                    connections=right_eye_connections,
                    landmark_color=(0, 255, 0), 
                    connection_color=(0, 255, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
                
                # Draw only inner lip landmarks and connections
                processed_frame = self.mediapipe_helper.draw_facial_landmarks(
                    processed_frame, inner_lip_landmarks, 
                    connections=inner_lip_connections,
                    landmark_color=(255, 0, 0), 
                    connection_color=(255, 0, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
            
            # Update drowsiness detection
            drowsiness_result = self.drowsiness_detector.update(
                ear_value=ear,
                head_pose=None,
                gaze_direction=None
            )
            
            # Visualize drowsiness detection results
            processed_frame = self.drowsiness_detector.visualize(
                processed_frame, 
                ear_left=left_ear, 
                ear_right=right_ear,
                show_metrics=True
            )
            
            # Dalgınlık uyarısını göster (GazeDurationMonitor'dan)
            if hasattr(self, 'distraction_level') and self.distraction_level != "NORMAL":
                # Uyarı rengi belirle
                if self.distraction_level == "CRITICAL":
                    warning_color = (0, 0, 255)  # Kırmızı (BGR)
                else:  # WARNING
                    warning_color = (0, 165, 255)  # Turuncu (BGR)
                
                # Uyarı metni - İngilizce olarak değiştir
                warning_text = f"DISTRACTION: {self.distraction_level}"
                
                # Metin özellikleri - Daha küçük metin
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6  # Daha küçük yazı boyutu
                thickness = 1  # Daha ince çizgi
                
                # Metin boyutunu al
                text_size = cv2.getTextSize(warning_text, font, font_scale, thickness)[0]
                
                # Sağ üst köşeye yerleştir (yatayda sağ tarafta, dikeyde üstte)
                padding = 5  # Kenarlardan uzaklık
                text_x = processed_frame.shape[1] - text_size[0] - padding
                text_y = 30  # Üstten mesafe
                
                # Arkaplan dikdörtgeni - Yarı şeffaf arka plan
                overlay = processed_frame.copy()
                cv2.rectangle(
                    overlay,
                    (text_x - padding, text_y - text_size[1] - padding),
                    (text_x + text_size[0] + padding * 2, text_y + padding),
                    (0, 0, 0),  # Siyah arkaplan
                    -1
                )
                # Şeffaflık uygula (alpha = 0.6)
                cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
                
                # Uyarı metni
                cv2.putText(
                    processed_frame,
                    warning_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    warning_color,
                    thickness
                )
                
                # Uyarı nedenlerini göster
                if hasattr(self, 'distraction_reasons') and self.distraction_reasons:
                    # Daha küçük font
                    font_scale_reason = 0.4
                    thickness_reason = 1
                    
                    # Her neden için - maksimum 2 neden göster
                    y_pos = text_y + 20
                    for i, reason in enumerate(self.distraction_reasons[:2]):  # En fazla 2 neden göster
                        # Doğrudan İngilizce gelen metni kullan
                        reason_text = f"- {reason}"
                        text_size_reason = cv2.getTextSize(reason_text, font, font_scale_reason, thickness_reason)[0]
                        
                        # Nedenleri de sağ tarafta hizala
                        reason_x = processed_frame.shape[1] - text_size_reason[0] - padding
                        
                        # Arkaplan dikdörtgeni - Yarı şeffaf
                        overlay = processed_frame.copy()
                        cv2.rectangle(
                            overlay,
                            (reason_x - padding, y_pos - text_size_reason[1] - padding),
                            (reason_x + text_size_reason[0] + padding * 2, y_pos + padding),
                            (0, 0, 0),  # Siyah arkaplan
                            -1
                        )
                        # Şeffaflık uygula (alpha = 0.6)
                        cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)
                        
                        # Neden metni
                        cv2.putText(
                            processed_frame,
                            reason_text,
                            (reason_x, y_pos),
                            font,
                            font_scale_reason,
                            warning_color,
                            thickness_reason
                        )
                        
                        y_pos += 15  # Satırlar arası daha az boşluk
            
            # İşlenmiş kareyi kullan
            frame = processed_frame
        
        # FPS hesapla
        frame_processing_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
        
        # FPS'i göster
        if self.config.get('visualization', {}).get('show_fps', True):
            # FPS metni
            fps_text = f"FPS: {current_fps:.1f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # Daha küçük
            thickness = 1
            font_color = (0, 255, 255)
            
            # Metin boyutunu al
            (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
            
            # Sol üst köşe pozisyonu
            text_x = 10
            text_y = 20
            
            # Yarı saydam arka plan kutusu çiz
            padding = 5
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (text_x - padding, text_y - text_height - padding),
                (text_x + text_width + padding, text_y + padding),
                (0, 0, 0),
                -1
            )
            # Şeffaflık uygula
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # FPS yazısını ekle
            cv2.putText(
                frame,
                fps_text,
                (text_x, text_y),
                font,
                font_scale,
                font_color,
                thickness
            )
        
        # Display the frame
        self.video_panel.update_frame(frame)
        
        # Update metrics UI
        self.metrics_panel.update_metrics(ear, mar, perclos)
        
        # Increment frame counter
        self._frame_counter = getattr(self, '_frame_counter', 0) + 1
    
    def _update_chart_data(self):
        """Update the chart with new data points."""
        if not self.is_capturing:
            return
            
        # Get current metrics
        ear_value = self.metrics_panel.ear_indicator.last_value
        mar_value = self.metrics_panel.mar_indicator.last_value
        perclos_value = self.metrics_panel.perclos_indicator.last_value
        
        # Update chart data
        self.chart_panel.update_chart_data(
            ear_value, 
            mar_value, 
            perclos_value, 
            self.update_interval_sec
        )
        
        # Update time counter
        self.time_counter = self.chart_panel.time_counter
    
    def on_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.ui_config, self)
        dialog.exec()
    
    def on_about(self):
        """Show about dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

    def closeEvent(self, event):
        """Handle window close event."""
        self.logger.info("Application closing")
        
        # Stop detection if running
        if self.is_capturing:
            self.on_stop()
        
        # Log application shutdown
        try:
            self.drowsiness_logger_instance.log_shutdown()
        except Exception as e:
            self.logger.error(f"Error logging shutdown: {str(e)}")
        
        # Accept the close event
        event.accept()

    def _show_expanded_charts(self):
        """Genişletilmiş grafikleri ayrı bir pencerede göster."""
        # Eğer pencere zaten açıksa odaklan, yoksa yeni pencere oluştur
        if hasattr(self, 'expanded_charts_window') and self.expanded_charts_window.isVisible():
            self.expanded_charts_window.activateWindow()
        else:
            self.expanded_charts_window = ExpandedChartsWindow(self.ui_config, parent=self)
            
            # Pencere kapatıldığında ana grafikleri tekrar etkinleştirmek için sinyal bağlantısı
            self.expanded_charts_window.closeEvent = self._on_expanded_charts_close
            
            # Ana grafik panelini geçici olarak devre dışı bırak ve bilgilendirici mesaj göster
            self.chart_panel.setVisible(False)
            
            # Bilgilendirme etiketi oluştur
            if not hasattr(self, 'charts_info_label'):
                # Bilgilendirme paneli layoutu
                self.charts_info_container = QVBoxLayout()
                
                # Bilgilendirme etiketi
                self.charts_info_label = QLabel("Grafikler şu an genişletilmiş istatistikler penceresinde çiziliyor")
                self.charts_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.charts_info_label.setFixedHeight(250)  # Chart panel ile aynı yükseklik
                self.charts_info_label.setSizePolicy(
                    QSizePolicy.Policy.Expanding,  # Genişlik için expanding
                    QSizePolicy.Policy.Fixed       # Yükseklik için fixed
                )
                self.charts_info_label.setStyleSheet("""
                    background-color: #f8f8f8;
                    border: 1px solid #e1e1e1;
                    border-radius: 8px;
                    padding: 15px;
                    color: #007aff;
                    font-weight: bold;
                    font-size: 14px;
                """)
                
                # Container'a bileşenleri ekle
                self.charts_info_container.addWidget(self.charts_info_label)
                
                # Merkezi widget'ın layout'una container'ı ekle
                central_layout = self.centralWidget().layout()
                central_layout.addLayout(self.charts_info_container)
            else:
                # Eğer zaten oluşturulmuşsa sadece görünür yap
                self.charts_info_label.setVisible(True)
            
            self.expanded_charts_window.show()
            
            # Eğer veri toplanıyorsa, genişletilmiş grafiklere de veri gönder
            if self.is_capturing:
                # Mevcut verileri genişletilmiş grafiklere aktar
                ear_data = create_chart_data_from_series(self.chart_panel.ear_series)
                mar_data = create_chart_data_from_series(self.chart_panel.mar_series)
                perclos_data = create_chart_data_from_series(self.chart_panel.perclos_series)
                
                self.expanded_charts_window.initialize_with_data(ear_data, mar_data, perclos_data)
                
                # Timer'ı genişletilmiş grafiklere bağla
                self.chart_timer.timeout.connect(self.expanded_charts_window.update_charts)
    
    def _on_expanded_charts_close(self, event):
        """Genişletilmiş grafikler penceresi kapatıldığında ana grafikleri tekrar göster."""
        # Timer bağlantısını kaldır
        if hasattr(self, "chart_timer"):
            try:
                self.chart_timer.timeout.disconnect(self.expanded_charts_window.update_charts)
            except TypeError:
                # Bağlantı zaten yoksa hata oluşabilir, yoksay
                pass
        
        # Ana grafik panelini tekrar göster
        self.chart_panel.setVisible(True)
        
        # Bilgilendirme etiketini gizle
        if hasattr(self, 'charts_info_label'):
            self.charts_info_label.setVisible(False)
        
        # Orijinal closeEvent çağrı
        event.accept()

    # 3D model sinyal bağlantıları için metodlar
    def _on_model_scale_changed(self, value):
        """Model ölçek faktörünü değiştir."""
        if hasattr(self, 'head_pose_panel'):
            # Ölçeği 0.5-2.0 aralığına dönüştür
            scale_value = value / 100.0
            self.scale_value_label.setText(f"{scale_value:.1f}x")
            
            # Model ölçeğini güncelle
            self.head_pose_panel.set_user_scale(scale_value)
    
    def _on_debug_toggle(self, checked):
        """Debug bilgilerini göster/gizle."""
        if hasattr(self, 'head_pose_panel'):
            self.head_pose_panel.set_debug_info(checked)
    
    def _on_axes_toggle(self, checked):
        """Eksenleri göster/gizle."""
        if hasattr(self, 'head_pose_panel'):
            self.head_pose_panel.set_axes_visible(checked)
    
    def _on_grid_toggle(self, checked):
        """Grid göster/gizle."""
        if hasattr(self, 'head_pose_panel'):
            self.head_pose_panel.set_grid_visible(checked)

    def on_upload_video(self):
        """Handle the video upload menu action."""
        # Create a dialog to host the video upload widget
        upload_dialog = QDialog(self)
        upload_dialog.setWindowTitle("Video Yükleme")
        upload_dialog.setMinimumSize(800, 600)
        
        # Create layout for the dialog
        layout = QVBoxLayout(upload_dialog)
        
        # Create the video upload widget
        self.video_upload_widget = VideoUploadWidget(self.ui_config)
        layout.addWidget(self.video_upload_widget)
        
        # Connect signals
        self.video_upload_widget.video_selected.connect(self._on_video_selected)
        self.video_upload_widget.analysis_started.connect(self._on_video_analysis_started)
        self.video_upload_widget.analysis_completed.connect(self._on_video_analysis_completed)
        
        # Show the dialog
        upload_dialog.exec()
    
    def _on_video_selected(self, video_path):
        """Handle video selection."""
        self.logger.info(f"Video selected: {video_path}")
        self.update_status(f"Video seçildi: {os.path.basename(video_path)}")
    
    def _on_video_analysis_started(self, video_path):
        """Handle video analysis start."""
        self.logger.info(f"Starting analysis of video: {video_path}")
        self.update_status(f"Video analizi başlatıldı: {os.path.basename(video_path)}")
        
        # Initialize the video analyzer if not already done
        if not hasattr(self, 'video_analyzer'):
            self.video_analyzer = VideoAnalyzer(self.config)
            
            # Connect signals
            self.video_analyzer.progress_updated.connect(self.video_upload_widget.update_progress)
            self.video_analyzer.frame_processed.connect(self._on_frame_processed)
            self.video_analyzer.analysis_completed.connect(self._on_analysis_results_ready)
            self.video_analyzer.error_occurred.connect(self._on_analysis_error)
        
        # Start analysis
        success = self.video_analyzer.analyze_video(video_path)
        
        if not success:
            self.update_status(f"Video analizi başlatılamadı: {os.path.basename(video_path)}")
    
    def _on_frame_processed(self, frame_data, processed_frame):
        """
        Handle processed frame from video analyzer.
        
        Args:
            frame_data: FrameData object with analysis results
            processed_frame: Processed frame with visualizations
        """
        # Update video panel if available
        if hasattr(self, 'video_panel'):
            self.video_panel.update_frame(processed_frame)
        
        # Store results for later use
        if not hasattr(self, 'video_analysis_results'):
            self.video_analysis_results = {
                'ear_values': [],
                'mar_values': [],
                'perclos_values': [],
                'drowsiness_alerts': [],
                'timestamps': []
            }
        
        # Add data to results
        if frame_data.face_detected:
            self.video_analysis_results['timestamps'].append(frame_data.timestamp)
            
            if frame_data.ear is not None:
                self.video_analysis_results['ear_values'].append(frame_data.ear)
            
            if frame_data.mar is not None:
                self.video_analysis_results['mar_values'].append(frame_data.mar)
            
            if frame_data.perclos is not None:
                self.video_analysis_results['perclos_values'].append(frame_data.perclos)
            
            # Check for drowsiness alert
            if frame_data.is_drowsy:
                self.video_analysis_results['drowsiness_alerts'].append({
                    'timestamp': frame_data.timestamp,
                    'perclos': frame_data.perclos,
                    'ear': frame_data.ear,
                    'mar': frame_data.mar
                })
    
    def _on_analysis_results_ready(self, statistics):
        """
        Handle analysis results.
        
        Args:
            statistics: Dictionary with analysis statistics
        """
        self.logger.info("Video analysis statistics received")
        
        # Store statistics
        self.video_analysis_statistics = statistics
        
        # Generate report
        if hasattr(self, 'video_analyzer'):
            report = self.video_analyzer.generate_report(statistics)
            
            # Show report in a dialog
            self._show_analysis_report(report)
    
    def _on_analysis_error(self, error_message):
        """
        Handle analysis error.
        
        Args:
            error_message: Error message
        """
        self.logger.error(f"Video analysis error: {error_message}")
        self.update_status(f"Video analizi hatası: {error_message}")
    
    def _show_analysis_report(self, report):
        """
        Show analysis report in a dialog.
        
        Args:
            report: Report text
        """
        # Create dialog
        report_dialog = QDialog(self)
        report_dialog.setWindowTitle("Video Analiz Raporu")
        report_dialog.setMinimumSize(600, 400)
        
        # Create layout
        layout = QVBoxLayout(report_dialog)
        
        # Create text edit for report
        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setPlainText(report)
        layout.addWidget(report_text)
        
        # Create button to close dialog
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(report_dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        # Show dialog
        report_dialog.exec()
    
    def _on_video_analysis_completed(self):
        """Handle video analysis completion."""
        self.logger.info("Video analysis completed")
        self.update_status("Video analizi tamamlandı")
        
        # Stop the analyzer if it's still running
        if hasattr(self, 'video_analyzer') and self.video_analyzer.is_analyzing:
            self.video_analyzer.stop_analysis()


def main():
    """
    Main function to start the application.
    
    This function creates the application and main window, and starts
    the event loop.
    """
    app = QApplication(sys.argv)
    window = DriverDrowsinessMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()