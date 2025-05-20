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

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QStatusBar, QLabel, QSizePolicy
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

# Import logger for application
from src.utils.logger import DrowsinessLogger

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
        
        # Camera settings
        self.camera_id = 0
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 30
        
        # Load configuration
        self.config = load_ui_config()
        
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
        self.setWindowTitle(self.config['window']['title'])
        self.resize(
            self.config['window']['width'],
            self.config['window']['height']
        )
        self.setMinimumSize(
            self.config['window']['min_width'],
            self.config['window']['min_height']
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
        self.menu_manager = MenuManager(self.config, self)
        self.setMenuBar(self.menu_manager)
        
        # Connect menu signals
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
            self.config['layout']['margin'],
            self.config['layout']['margin'],
            self.config['layout']['margin'],
            self.config['layout']['margin']
        )
        main_layout.setSpacing(self.config['layout']['spacing'])
        
        # Top area (video + stats + 3D model)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)  # Panel arasında yeterli boşluk bırak
        
        # Video panel layoutu - başlık olmadan direkt paneli ekle
        video_container = QVBoxLayout()
        
        # Video panel
        self.video_panel = VideoPanel(self.config)
        video_container.addWidget(self.video_panel)
        
        # Sol tarafta video paneli için bir container widget oluştur ve hizalamayı ayarla
        video_widget = QWidget()
        video_widget.setLayout(video_container)
        video_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        top_layout.addWidget(video_widget, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        # 3D Head Pose modeli ekleme
        from src.ui.head_pose_model import Head3DPanel
        self.head_pose_panel = Head3DPanel(self.config)
        top_layout.addWidget(self.head_pose_panel, 0, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        
        # Metrikler ve video paneli arasında az bir esnek alan ekle
        top_layout.addStretch(1)  # Float değil integer kullan
        
        # Metrics panel layoutu - başlık olmadan direkt paneli ekle
        metrics_container = QVBoxLayout()
        
        # Metrics panel
        self.metrics_panel = MetricsPanel(self.config)
        metrics_container.addWidget(self.metrics_panel)
        
        # Sağ tarafta metrikler paneli için bir container widget oluştur ve hizalamayı ayarla
        metrics_widget = QWidget()
        metrics_widget.setLayout(metrics_container)
        metrics_widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        top_layout.addWidget(metrics_widget, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        
        main_layout.addLayout(top_layout)
        
        # Kontrol paneli - başlık olmadan direkt panel ekle
        self.control_panel = ControlPanel(self.config)
        main_layout.addWidget(self.control_panel)
        
        # Connect control panel signals
        self.control_panel.start_clicked.connect(self.on_start)
        self.control_panel.stop_clicked.connect(self.on_stop)
        self.control_panel.settings_clicked.connect(self.on_settings)
        self.control_panel.landmarks_toggled.connect(self._toggle_landmarks)
        self.control_panel.head_pose_toggled.connect(self._toggle_head_pose)
        self.control_panel.gaze_toggled.connect(self._toggle_gaze)
        self.control_panel.expand_charts_clicked.connect(self._show_expanded_charts)
        
        # Grafik paneli - başlık olmadan direkt paneli ekle
        self.chart_panel = ChartPanel(self.config)
        main_layout.addWidget(self.chart_panel)
    
    def showEvent(self, event):
        """Handle window show event - maximize after UI is fully initialized."""
        super().showEvent(event)
        
        # UI tam olarak yüklendikten sonra tam ekran yap
        if self.config['window'].get('start_maximized', True):
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
                    drowsiness_level = getattr(self.drowsiness_detector, 'drowsiness_level', 0)
                
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
            
            # İşlenmiş kareyi kullan
            frame = processed_frame
        
        # FPS hesapla
        frame_processing_time = time.time() - frame_start_time
        current_fps = 1.0 / frame_processing_time if frame_processing_time > 0 else 0
        
        # FPS'i göster
        if self.config.get('visualization', {}).get('show_fps', True):
            # FPS yazısını ekle (sol alt köşe)
            cv2.putText(
                frame, 
                f"FPS: {current_fps:.1f}", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
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
        dialog = SettingsDialog(self.config, self)
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
            self.expanded_charts_window = ExpandedChartsWindow(self.config, parent=self)
            
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