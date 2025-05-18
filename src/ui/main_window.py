#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main window for the driver drowsiness detection application.

This module implements the main window GUI for the driver drowsiness
detection system, including all UI components and layouts.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import logging

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QListWidget, QMenuBar, QToolBar,
    QStatusBar, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGridLayout, QFormLayout, QDialog, QDoubleSpinBox, QComboBox,
    QDialogButtonBox, QMessageBox, QFrame, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSlot, QMargins
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QFont, QPainter, QColor, QPen
from PyQt6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis

# Import custom widgets
from src.ui.widgets import IndicatorWidget

# Import utils
from src.utils.mediapipe_utils import load_ui_config, MediaPipeUtils

# No gaze or head pose imports


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
        self.mediapipe_utils = None
        self.is_capturing = False
        self.show_landmarks = False
        self.show_head_pose = False
        self.show_gaze = False  # Bakış yönü gösterme durumu
        
        # Camera settings
        self.camera_id = 0
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fps = 30
        
        # Load configuration
        self.config = load_ui_config()
        
        # Initialize UI
        self._init_ui()
        
        # Initialize timers
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.update_interval_ms = 33  # ~30 FPS
        
        # Initialize chart timer - yüksek sıklıkta örnekleme (bilimsel analiz için)
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self._update_chart_data)
        self.update_interval_sec = 0.05  # 50ms update for chart - yüksek çözünürlüklü örnekleme
        self.time_counter = 0.0  # Time counter for chart (seconds)
        
        # Initialize drowsiness metrics
        self.eye_closure_history = []
        self.max_history_frames = int(self.config.get('detection', {}).get('perclos_window_sec', 60) * self.camera_fps)
        
        # Kalibrasyon için gerekli değişkenler
        self.is_calibrating = False
        self.calibration_duration = 30.0  # saniye cinsinden kalibrasyon süresi
        self.calibration_start_time = 0.0
        self.ear_values_during_calibration = []  # Kalibrasyon sırasında toplanan EAR değerleri
        self.ear_min = 0.2  # Varsayılan minimum EAR değeri
        self.ear_max = 0.4  # Varsayılan maksimum EAR değeri
        self.ear_range = 0.2  # Varsayılan EAR aralığı (ear_max - ear_min)
        self.is_calibrated = False
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Main window initialized")
    
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
        menu_bar = QMenuBar()
        self.setMenuBar(menu_bar)
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: white;
                border-bottom: 1px solid #e1e1e1;
            }
            QMenuBar::item {
                background-color: white;
                padding: 6px 10px;
            }
            QMenuBar::item:selected {
                background-color: #f0f0f0;
            }
            QMenu {
                background-color: white;
                border: 1px solid #e1e1e1;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #f0f0f0;
            }
        """)
        
        # File menu
        file_menu = menu_bar.addMenu("Dosya")
        
        # File menu actions
        start_action = QAction("Başlat", self)
        start_action.triggered.connect(self.on_start)
        file_menu.addAction(start_action)
        
        stop_action = QAction("Durdur", self)
        stop_action.triggered.connect(self.on_stop)
        file_menu.addAction(stop_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("Ayarlar", self)
        settings_action.triggered.connect(self.on_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Çıkış", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menu_bar.addMenu("Görünüm")
        
        # View menu actions        
        toggle_stats_action = QAction("İstatistikleri Göster", self)
        toggle_stats_action.setCheckable(True)
        toggle_stats_action.setChecked(True)
        toggle_stats_action.triggered.connect(self._toggle_stats_panel)
        view_menu.addAction(toggle_stats_action)
        
        toggle_chart_action = QAction("Grafik Göster", self)
        toggle_chart_action.setCheckable(True)
        toggle_chart_action.setChecked(True)
        toggle_chart_action.triggered.connect(self._toggle_chart)
        view_menu.addAction(toggle_chart_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Yardım")
        
        # Help menu actions
        about_action = QAction("Hakkında", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
    
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
        
        # Top area (video + stats)
        top_layout = QHBoxLayout()
        
        # Video frame
        self.video_frame = QLabel("Kamera görüntüsü burada gösterilecek")
        self.video_frame.setMinimumSize(
            self.config['video_frame']['width'],
            self.config['video_frame']['height']
        )
        self.video_frame.setFrameShape(QFrame.Shape.Box)  # Box frame
        self.video_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_frame.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
        """)
        top_layout.addWidget(self.video_frame)
        
        # Stats panel
        stats_widget = QWidget()
        stats_layout = QGridLayout(stats_widget)
        stats_layout.setContentsMargins(
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding']
        )
        stats_layout.setSpacing(self.config['layout']['spacing'])
        
        # Add indicator widgets
        self.ear_indicator = IndicatorWidget("EAR", self.config)
        stats_layout.addWidget(self.ear_indicator, 0, 0)
        
        self.mar_indicator = IndicatorWidget("MAR", self.config)
        stats_layout.addWidget(self.mar_indicator, 1, 0)
        
        self.perclos_indicator = IndicatorWidget("PERCLOS", self.config)
        stats_layout.addWidget(self.perclos_indicator, 2, 0)
        
        # Stats panel'i layout'a ekle
        self.stats_widget = stats_widget
        top_layout.addWidget(stats_widget)
        
        main_layout.addLayout(top_layout)
        
        # Control area
        control_layout = self._create_controls()
        main_layout.addLayout(control_layout)
        
        # Time series chart
        self._create_chart()
        main_layout.addWidget(self.chart_view)
    
    def _create_controls(self):
        """Create control buttons for the application."""
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 20, 0, 20)
        
        # Control buttons
        self.start_button = QPushButton("Başlat")
        self.start_button.setObjectName("start_button")
        self.start_button.setFixedSize(
            self.config['controls']['button_width'],
            self.config['controls']['button_height']
        )
        self.start_button.clicked.connect(self.on_start)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setFixedSize(
            self.config['controls']['button_width'],
            self.config['controls']['button_height']
        )
        self.stop_button.clicked.connect(self.on_stop)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.settings_button = QPushButton("Ayarlar")
        self.settings_button.setFixedSize(
            self.config['controls']['button_width'],
            self.config['controls']['button_height']
        )
        self.settings_button.clicked.connect(self.on_settings)
        control_layout.addWidget(self.settings_button)
        
        # Toggle face landmarks button
        self.show_landmarks_button = QPushButton("Yüz İşaretlerini Göster")
        self.show_landmarks_button.setFixedSize(
            self.config['controls']['button_width'] + 80,  # Increase width to fit text
            self.config['controls']['button_height']
        )
        self.show_landmarks_button.setCheckable(True)  # Make it toggleable
        self.show_landmarks_button.setChecked(False)  # Off by default
        self.show_landmarks_button.clicked.connect(self._toggle_landmarks)
        control_layout.addWidget(self.show_landmarks_button)
        
        # Toggle head pose button
        self.show_head_pose_button = QPushButton("Baş Duruşunu Göster")
        self.show_head_pose_button.setFixedSize(
            self.config['controls']['button_width'] + 80,  # Increase width to fit text
            self.config['controls']['button_height']
        )
        self.show_head_pose_button.setCheckable(True)  # Make it toggleable
        self.show_head_pose_button.setChecked(False)  # Off by default
        self.show_head_pose_button.clicked.connect(self._toggle_head_pose)
        control_layout.addWidget(self.show_head_pose_button)
        
        # Toggle gaze button
        self.gaze_button = QPushButton("Bakış Yönünü Göster")
        self.gaze_button.setCheckable(True)
        self.gaze_button.setChecked(False)
        self.gaze_button.clicked.connect(self._toggle_gaze)
        control_layout.addWidget(self.gaze_button)
        
        # Add expanded charts button
        self.expand_charts_button = QPushButton("Grafikleri Genişlet")
        self.expand_charts_button.setFixedSize(
            self.config['controls']['button_width'] + 50,  # Increase width to fit text
            self.config['controls']['button_height']
        )
        self.expand_charts_button.clicked.connect(self._show_expanded_charts)
        control_layout.addWidget(self.expand_charts_button)
        
        control_layout.addStretch()
        
        return control_layout
    
    def _create_chart(self):
        """Create the time series chart for EAR, MAR, and PERCLOS data."""
        # Create chart
        chart = QChart()
        chart.setTitle("Metrikler Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)  # Animasyonları kapat
        chart.setBackgroundVisible(False)
        chart.setBackgroundRoundness(0)
        chart.setMargins(QMargins(0, 0, 0, 0))
        chart.layout().setContentsMargins(0, 0, 0, 0)
        chart.setTitleFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['title_size'],
            QFont.Weight.Medium
        ))
        chart.setTitleBrush(QColor("#333333"))
        
        # Create series for EAR, MAR, and PERCLOS - bilimsel görünüm için optimize et
        self.ear_series = QLineSeries()
        self.ear_series.setName("EAR")
        self.ear_series.setPen(QPen(QColor("#007aff"), self.config['chart']['line_width'], Qt.PenStyle.SolidLine))
        self.ear_series.setUseOpenGL(True)  # OpenGL hızlandırma kullan
        
        # Normalize edilmiş EAR serisi ekle
        self.normalized_ear_series = QLineSeries()
        self.normalized_ear_series.setName("Normalize EAR")
        self.normalized_ear_series.setPen(QPen(QColor("#ff5733"), self.config['chart']['line_width'], Qt.PenStyle.SolidLine))
        self.normalized_ear_series.setUseOpenGL(True)  # OpenGL hızlandırma kullan
        
        self.mar_series = QLineSeries()
        self.mar_series.setName("MAR")
        self.mar_series.setPen(QPen(QColor("#5ac8fa"), self.config['chart']['line_width'], Qt.PenStyle.SolidLine))
        self.mar_series.setUseOpenGL(True)  # OpenGL hızlandırma kullan
        
        self.perclos_series = QLineSeries()
        self.perclos_series.setName("PERCLOS")
        self.perclos_series.setPen(QPen(QColor("#ff9500"), self.config['chart']['line_width'], Qt.PenStyle.SolidLine))
        self.perclos_series.setUseOpenGL(True)  # OpenGL hızlandırma kullan
        
        # Add series to chart
        chart.addSeries(self.ear_series)
        chart.addSeries(self.normalized_ear_series)
        chart.addSeries(self.mar_series)
        chart.addSeries(self.perclos_series)
        
        # Create X axis (time)
        self.time_axis = QValueAxis()
        self.time_axis.setRange(0, self.config['chart']['history_duration'])
        self.time_axis.setTitleText("Zaman (sn)")
        self.time_axis.setTitleFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.time_axis.setTitleBrush(QColor("#666666"))
        self.time_axis.setLabelsFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'] - 2
        ))
        self.time_axis.setLabelFormat("%.1f")
        self.time_axis.setGridLineVisible(True)
        self.time_axis.setGridLineColor(QColor("#e1e1e1"))
        self.time_axis.setMinorGridLineVisible(False)
        chart.addAxis(self.time_axis, Qt.AlignmentFlag.AlignBottom)
        
        # Create Y axis for EAR
        self.ear_axis = QValueAxis()
        self.ear_axis.setRange(
            self.config['chart']['y_range_ear'][0],
            self.config['chart']['y_range_ear'][1]
        )
        self.ear_axis.setTitleText("EAR")
        self.ear_axis.setTitleFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.ear_axis.setTitleBrush(QColor("#007aff"))
        self.ear_axis.setLabelsFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'] - 2
        ))
        self.ear_axis.setLabelFormat("%.2f")
        self.ear_axis.setGridLineVisible(True)
        self.ear_axis.setGridLineColor(QColor("#e1e1e1"))
        self.ear_axis.setMinorGridLineVisible(False)
        chart.addAxis(self.ear_axis, Qt.AlignmentFlag.AlignLeft)
        
        # Normalize edilmiş EAR için Y ekseni ekle
        self.normalized_ear_axis = QValueAxis()
        self.normalized_ear_axis.setRange(0.0, 1.0)
        self.normalized_ear_axis.setTitleText("Normalize EAR")
        self.normalized_ear_axis.setTitleFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.normalized_ear_axis.setTitleBrush(QColor("#ff5733"))
        self.normalized_ear_axis.setLabelsFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'] - 2
        ))
        self.normalized_ear_axis.setLabelFormat("%.1f")
        self.normalized_ear_axis.setGridLineVisible(False)
        self.normalized_ear_axis.setMinorGridLineVisible(False)
        chart.addAxis(self.normalized_ear_axis, Qt.AlignmentFlag.AlignRight)
        
        # Create Y axis for MAR
        self.mar_axis = QValueAxis()
        self.mar_axis.setRange(
            self.config['chart']['y_range_mar'][0],
            self.config['chart']['y_range_mar'][1]
        )
        self.mar_axis.setTitleText("MAR")
        self.mar_axis.setTitleFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.mar_axis.setTitleBrush(QColor("#5ac8fa"))
        self.mar_axis.setLabelsFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'] - 2
        ))
        self.mar_axis.setLabelFormat("%.2f")
        self.mar_axis.setGridLineVisible(False)
        self.mar_axis.setMinorGridLineVisible(False)
        chart.addAxis(self.mar_axis, Qt.AlignmentFlag.AlignRight)
        
        # Create Y axis for PERCLOS
        self.perclos_axis = QValueAxis()
        self.perclos_axis.setRange(
            self.config['chart']['y_range_perclos'][0],
            self.config['chart']['y_range_perclos'][1]
        )
        self.perclos_axis.setTitleText("PERCLOS")
        self.perclos_axis.setTitleFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.perclos_axis.setTitleBrush(QColor("#ff9500"))
        self.perclos_axis.setLabelsFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'] - 2
        ))
        self.perclos_axis.setLabelFormat("%.1f")
        self.perclos_axis.setGridLineVisible(False)
        self.perclos_axis.setMinorGridLineVisible(False)
        chart.addAxis(self.perclos_axis, Qt.AlignmentFlag.AlignRight)
        
        # Attach series to axes
        self.ear_series.attachAxis(self.time_axis)
        self.ear_series.attachAxis(self.ear_axis)
        
        # Normalize edilmiş EAR serisini eksenlere bağla
        self.normalized_ear_series.attachAxis(self.time_axis)
        self.normalized_ear_series.attachAxis(self.normalized_ear_axis)
        
        self.mar_series.attachAxis(self.time_axis)
        self.mar_series.attachAxis(self.mar_axis)
        
        self.perclos_series.attachAxis(self.time_axis)
        self.perclos_series.attachAxis(self.perclos_axis)
        
        # Create chart view
        self.chart_view = QChartView(chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.chart_view.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.chart_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.chart_view.setViewportUpdateMode(QChartView.ViewportUpdateMode.FullViewportUpdate)
        self.chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)  # Yakınlaştırma için alan seçimini etkinleştir
        self.chart_view.setStyleSheet("""
            background-color: white;
            border: 1px solid #e1e1e1;
            border-radius: 8px;
        """)
    
    def update_status(self, message):
        """
        Update the status bar message.
        
        Args:
            message: The message to display in the status bar
        """
        self.status_bar.showMessage(message)
    
    def update_video_frame(self, frame):
        """
        Update the video frame with a new image.
        
        Args:
            frame: OpenCV BGR image
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert to QPixmap and set to label
        pixmap = QPixmap.fromImage(q_img)
        self.video_frame.setPixmap(pixmap)
    
    def update_metrics(self, ear, mar, perclos, normalized_ear=None):
        """
        Update all metrics displayed in the UI.
        
        Args:
            ear: Eye Aspect Ratio value
            mar: Mouth Aspect Ratio value
            perclos: PERCLOS value
            normalized_ear: Normalize edilmiş EAR değeri (opsiyonel)
        """
        # Update indicator widgets
        self.ear_indicator.update_value(
            ear, 
            min_val=0.0, 
            max_val=self.config['chart']['y_range_ear'][1]
        )
        
        self.mar_indicator.update_value(
            mar, 
            min_val=0.0, 
            max_val=self.config['chart']['y_range_mar'][1]
        )
        
        self.perclos_indicator.update_value(
            perclos, 
            min_val=0.0, 
            max_val=self.config['chart']['y_range_perclos'][1]
        )
        
        # Normalize edilmiş EAR değerini kaydet - sadece kalibrasyon tamamlandıktan sonra
        if normalized_ear is not None and self.is_calibrated:
            self.ear_indicator.normalized_value = normalized_ear
    
    def _toggle_stats_panel(self, checked):
        """Toggle the visibility of the stats panel."""
        self.stats_widget.setVisible(checked)
    
    def _toggle_chart(self, checked):
        """Toggle the visibility of the chart."""
        self.chart_view.setVisible(checked)
    
    def _toggle_landmarks(self):
        """Toggle visibility of facial landmarks."""
        self.show_landmarks = self.show_landmarks_button.isChecked()
        status = "gösteriliyor" if self.show_landmarks else "gizleniyor"
        self.update_status(f"Yüz işaretleri {status}")
        
        # Update button style to show toggle state more clearly
        if self.show_landmarks:
            self.show_landmarks_button.setStyleSheet("""
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            """)
        else:
            self.show_landmarks_button.setStyleSheet("")  # Reset to default style
    
    def _toggle_head_pose(self):
        """Toggle the display of head pose."""
        self.show_head_pose = self.show_head_pose_button.isChecked()
        
        # Update button appearance
        if self.show_head_pose:
            self.show_head_pose_button.setStyleSheet("background-color: #2196F3; color: white;")
            self.update_status("Baş duruşu gösteriliyor")
        else:
            self.show_head_pose_button.setStyleSheet("")
            self.update_status("Baş duruşu gizlendi")
    
    def _toggle_gaze(self):
        """Toggle the display of gaze direction."""
        self.show_gaze = self.gaze_button.isChecked()
        
        # Update button appearance
        if self.show_gaze:
            self.gaze_button.setStyleSheet("background-color: #9C27B0; color: white;")
            self.update_status("Bakış yönü gösteriliyor")
        else:
            self.gaze_button.setStyleSheet("")
            self.update_status("Bakış yönü gizlendi")
    
    def on_start(self):
        """Handle start button click."""
        if self.is_capturing:
            return
            
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Try to open the camera
        try:
            print(f"INFO: Attempting to open camera ID: {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties from config
            camera_width = self.config.get('camera', {}).get('width', 640)
            camera_height = self.config.get('camera', {}).get('height', 480)
            camera_fps = self.config.get('camera', {}).get('fps', 30)
            
            print(f"INFO: Setting camera properties: width={camera_width}, height={camera_height}, fps={camera_fps}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, camera_fps)
            
            if not self.cap.isOpened():
                error_msg = "Kamera açılamadı! Kamera bağlantısını kontrol edin."
                print(f"ERROR: {error_msg}")
                self.update_status(error_msg)
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                return
                
            # Initialize MediaPipe
            self.mediapipe_utils = MediaPipeUtils()
            
            # Get actual camera properties after setting
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"INFO: Actual camera properties: width={actual_width}, height={actual_height}, fps={actual_fps}")
                
            # Start the camera timer
            self.is_capturing = True
            self.camera_timer.start(self.update_interval_ms)
            
            # Start chart update timer (demo)
            self.time_counter = 0.0
            self.chart_timer.start(int(self.update_interval_sec * 1000))  # Milisaniyeye çevir
            
            # Reset PERCLOS calculation
            self.eye_closure_history = []
            # Başlangıçta göz açık kabul edilerek history'yi doldur
            # Böylece ilk PERCLOS değerleri düşük (0) olarak başlar
            for _ in range(min(20, self.max_history_frames)):
                self.eye_closure_history.append(0)  # Göz açık (0) olarak ekle
            
            # Kalibrasyon sürecini başlat
            self.is_calibrating = True
            self.calibration_start_time = 0.0  # time_counter ile birlikte başlayacak
            self.ear_values_during_calibration = []
            self.is_calibrated = False
            
            self.update_status("Kalibrasyon başladı. Lütfen 30 saniye bekleyin...")
            print("INFO: Calibration started, camera capture successfully initialized")
        except Exception as e:
            error_msg = f"Kamera başlatılamadı: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()  # Print detailed exception
            self.update_status(error_msg)
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def on_stop(self):
        """Handle stop button click."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Stop the camera capture
        self.is_capturing = False
        self.camera_timer.stop()
        
        # Stop chart update timer
        if hasattr(self, 'chart_timer'):
            self.chart_timer.stop()
        
        # Release the camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Release MediaPipe resources
        if self.mediapipe_utils is not None:
            self.mediapipe_utils.release()
            self.mediapipe_utils = None
        
        # Clear the video frame
        self.video_frame.setText("Kamera görüntüsü burada gösterilecek")
        
        self.update_status("Algılama durduruldu.")
    
    def _update_camera_frame(self):
        """Update the video frame with current camera image."""
        if not self.cap or not self.is_capturing:
            return
            
        # Read a frame from the camera
        ret, frame = self.cap.read()
        
        if not ret:
            error_msg = "Kameradan görüntü alınamadı!"
            print(f"ERROR: {error_msg}")
            self.update_status(error_msg)
            self.on_stop()
            return
            
        # Print frame info occasionally (every 30 frames ~ 1 second)
        frame_counter = getattr(self, '_frame_counter', 0)
        if frame_counter % 30 == 0:
            print(f"INFO: Frame received - shape: {frame.shape}")
        self._frame_counter = frame_counter + 1
            
        # Mirror the frame horizontally (selfie view)
        frame = cv2.flip(frame, 1)
        
        # Process the frame with MediaPipe
        landmarks, face_detected = self.mediapipe_utils.detect_face_landmarks(frame)
        
        # Initialize metrics with default values
        ear = 0.0
        mar = 0.0
        perclos = 0.0
        normalized_ear = 0.0
        
        # Variables for drowsiness detection
        left_ear = 0.0
        right_ear = 0.0
        
        if face_detected:
            # Get eye landmarks
            left_eye_landmarks = self.mediapipe_utils.get_eye_landmarks(landmarks, left_eye=True)
            right_eye_landmarks = self.mediapipe_utils.get_eye_landmarks(landmarks, left_eye=False)
            
            # Get mouth landmarks - only inner lip
            inner_lip_landmarks = self.mediapipe_utils.get_inner_lip_landmarks(landmarks)
            
            # Calculate metrics
            left_ear = self.mediapipe_utils.get_eye_aspect_ratio(left_eye_landmarks)
            right_ear = self.mediapipe_utils.get_eye_aspect_ratio(right_eye_landmarks)
            ear = (left_ear + right_ear) / 2.0  # Average EAR
            
            # Kalibrasyon veya normal çalışma modunu kontrol et
            if self.is_calibrating:
                # Kalibrasyon için EAR değerlerini topla
                self.ear_values_during_calibration.append(ear)
                
                # Kalibrasyon süresini kontrol et
                if self.time_counter >= self.calibration_duration:
                    # Kalibrasyon tamamlandı, min ve max EAR değerlerini hesapla
                    if len(self.ear_values_during_calibration) > 0:
                        self.ear_min = max(0.1, min(self.ear_values_during_calibration))  # En az 0.1 olsun
                        self.ear_max = max(self.ear_values_during_calibration)
                        self.ear_range = max(0.1, self.ear_max - self.ear_min)  # En az 0.1 aralık olsun
                        
                        # Kalibrasyon sonuçları hakkında bilgi ver
                        calibration_msg = f"Kalibrasyon tamamlandı! Min EAR: {self.ear_min:.3f}, Max EAR: {self.ear_max:.3f}"
                        print(f"INFO: {calibration_msg}")
                        self.update_status(calibration_msg)
                        
                        # Kalibrasyon modunu kapat
                        self.is_calibrating = False
                        self.is_calibrated = True
                    else:
                        print("WARNING: No valid EAR values collected during calibration")
                        self.is_calibrating = False
                        self.update_status("Kalibrasyon başarısız. Varsayılan değerler kullanılıyor.")
                else:
                    # Kalibrasyonun durumunu ekranda göster
                    remaining_time = int(self.calibration_duration - self.time_counter)
                    self.update_status(f"Kalibrasyon sürüyor... {remaining_time} saniye kaldı. Lütfen normal bakışla ekrana bakın.")
                    
                    # Kalibrasyon metni ekle
                    cv2.putText(
                        frame,
                        f"CALIBRATION: {remaining_time}s",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
            
            # Normalize EAR değerini hesapla (kalibre edilmiş veya varsayılan değerlerle)
            # Sadece kalibrasyon tamamlandıktan sonra normalize edilmiş EAR değerini göster
            if self.is_calibrated:
                # Daha sağlam normalizasyon hesaplaması - ara değişkenler kullanarak
                diff = ear - self.ear_min
                # Bölme işleminden önce sıfırdan küçük değerleri engelle
                safe_diff = max(0.0, diff)
                # Sıfıra bölünmeyi engelle
                if self.ear_range > 0.001:  # Çok küçük değerlerden kaçın
                    raw_normalized = safe_diff / self.ear_range
                else:
                    raw_normalized = 0.0
                # Son güvenlik kontrolü - 0 ile 1 arasında olduğundan emin ol
                normalized_ear = min(1.0, max(0.0, raw_normalized))
                
                # Her saniye bir (yaklaşık her 30 frame) EAR ve normalize edilmiş EAR değerlerini terminale yazdır
                if frame_counter % 30 == 0:
                    print(f"DEBUG: EAR={ear:.4f}, Min={self.ear_min:.4f}, Max={self.ear_max:.4f}, Range={self.ear_range:.4f}")
                    print(f"       diff={diff:.4f}, safe_diff={safe_diff:.4f}, raw_normalized={raw_normalized:.4f}, final_normalized={normalized_ear:.4f}")
                
                # Ekrana normalize edilmiş EAR değerini ekle
                cv2.putText(
                    frame,
                    f"Normalized EAR: {normalized_ear:.2f}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )
            elif frame_counter % 30 == 0:
                # Kalibrasyon sırasında sadece raw EAR değerlerini yazdır
                print(f"DEBUG (Calibration): EAR={ear:.4f}")
            
            # Use landmarks directly for MAR calculation
            mar = self.mediapipe_utils.get_mouth_aspect_ratio(landmarks)
            
            # Update PERCLOS
            is_eye_closed = ear < self.config.get('detection', {}).get('ear_threshold', 0.21)
            self.eye_closure_history.append(1 if is_eye_closed else 0)
            
            # Keep history within window size
            if len(self.eye_closure_history) > self.max_history_frames:
                self.eye_closure_history = self.eye_closure_history[-self.max_history_frames:]
            
            # Calculate PERCLOS as percentage of closed eyes in the window
            if self.eye_closure_history:
                # Minimum veri miktarı kontrolü - az veri ile yanlış yüksek değerlerden kaçınmak için
                min_history_frames = min(10, self.max_history_frames // 10)  # En az 10 kare veya max karenin 1/10'u
                if len(self.eye_closure_history) < min_history_frames:
                    # Yeterli veri yoksa düşük bir değer kullan
                    perclos = 0.0
                else:
                    # Yeterli veri toplanmışsa normal hesaplamayı yap
                    perclos = (sum(self.eye_closure_history) / len(self.eye_closure_history)) * 100.0
            
            # Draw landmarks if enabled
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
                frame = self.mediapipe_utils.draw_facial_landmarks(
                    frame, left_eye_landmarks, 
                    connections=left_eye_connections,
                    landmark_color=(0, 255, 0), 
                    connection_color=(0, 255, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
                frame = self.mediapipe_utils.draw_facial_landmarks(
                    frame, right_eye_landmarks, 
                    connections=right_eye_connections,
                    landmark_color=(0, 255, 0), 
                    connection_color=(0, 255, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
                
                # Draw only inner lip landmarks and connections
                frame = self.mediapipe_utils.draw_facial_landmarks(
                    frame, inner_lip_landmarks, 
                    connections=inner_lip_connections,
                    landmark_color=(255, 0, 0), 
                    connection_color=(255, 0, 0),
                    landmark_radius=2,
                    connection_thickness=1
                )
            
            # Visualize head pose if enabled
            if self.show_head_pose:
                frame = self.mediapipe_utils.visualize_head_pose(
                    frame, 
                    landmarks,
                    visualization_type='cube'  # Her zaman küp görünümünü kullan
                )
            
            # Visualize gaze direction if enabled
            if self.show_gaze:
                # EAR değerini ve eşik değerini visualize_gaze'e ilet
                # Böylece göz kapalıyken bakış vektörü çizilmeyecek
                ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
                frame, normalized_face = self.mediapipe_utils.visualize_gaze(
                    frame, 
                    landmarks,
                    ear_value=ear,  # Ortalama EAR değerini ilet
                    ear_threshold=ear_threshold  # Eşik değerini ilet
                )
                
                # Eğer normalize edilmiş yüz görüntüsü varsa, küçük bir pencerede göster
                if normalized_face is not None:
                    # Normalize edilmiş yüz görüntüsünü yeniden boyutlandır
                    norm_face_display = cv2.resize(normalized_face, (112, 112))
                    
                    # Görüntüyü ana kareye yerleştir (sağ üst köşe)
                    h, w = norm_face_display.shape[:2]
                    frame[10:10+h, frame.shape[1]-w-10:frame.shape[1]-10] = norm_face_display
            
            # Update drowsiness detection
            from src.detection.drowsiness_detector import DrowsinessDetector
            drowsiness_detector = getattr(self, 'drowsiness_detector', None)
            if drowsiness_detector is None:
                self.drowsiness_detector = DrowsinessDetector()
                drowsiness_detector = self.drowsiness_detector
                
            # Update drowsiness state with eye metrics only
            drowsiness_result = drowsiness_detector.update(
                ear_left=left_ear,
                ear_right=right_ear
            )
            
            # Visualize drowsiness detection results
            frame = drowsiness_detector.visualize(
                frame, 
                ear_left=left_ear, 
                ear_right=right_ear,
                show_metrics=True
            )
        
        # Display the frame
        self.update_video_frame(frame)
        
        # Update metrics UI - Normalize edilmiş EAR değeriyle de güncelleyelim
        self.update_metrics(ear, mar, perclos, normalized_ear=normalized_ear)
    
    def _update_chart_data(self):
        """Update the chart with new data points."""
        if not self.is_capturing:
            return
            
        # Increment time counter
        self.time_counter += self.update_interval_sec
        
        # Add data points to series
        self.ear_series.append(self.time_counter, self.ear_indicator.last_value)
        
        # Sadece kalibrasyon tamamlandıktan sonra normalize edilmiş EAR değerlerini ekle
        if self.is_calibrated:
            self.normalized_ear_series.append(self.time_counter, self.ear_indicator.normalized_value)
            
        self.mar_series.append(self.time_counter, self.mar_indicator.last_value)
        self.perclos_series.append(self.time_counter, self.perclos_indicator.last_value)
        
        # Remove old data points if we exceed the chart duration
        history_duration = self.config['chart']['history_duration']
        if self.time_counter > history_duration:
            # Kayan pencere yaklaşımı: Tüm seriyi silip yeniden yüklemek yerine,
            # sadece pencere dışına çıkan noktaları kaldır
            cutoff_time = self.time_counter - history_duration
            
            # Verimli bir şekilde eski noktaları kaldır
            # Tüm seriyi temizlemek yerine, sadece zaman aralığı dışındaki noktaları kaldır
            while self.ear_series.count() > 0 and self.ear_series.at(0).x() < cutoff_time:
                self.ear_series.remove(0)
                
            while self.normalized_ear_series.count() > 0 and self.normalized_ear_series.at(0).x() < cutoff_time:
                self.normalized_ear_series.remove(0)
                
            while self.mar_series.count() > 0 and self.mar_series.at(0).x() < cutoff_time:
                self.mar_series.remove(0)
                
            while self.perclos_series.count() > 0 and self.perclos_series.at(0).x() < cutoff_time:
                self.perclos_series.remove(0)
                
            # X ekseni aralığını yumuşak bir şekilde güncelle
            # Mevcut aralığı al
            current_min = self.time_axis.min()
            current_max = self.time_axis.max()
            
            # Hedef aralık
            target_min = self.time_counter - history_duration
            target_max = self.time_counter
            
            # Aralığı daha yumuşak bir şekilde kaydır - daha düşük damping faktörü
            # Düşük damping faktörü daha yumuşak hareket sağlar
            damping_factor = 0.03  # Çok düşük bir değer, daha yumuşak geçiş
            new_min = current_min + (target_min - current_min) * damping_factor
            new_max = current_max + (target_max - current_max) * damping_factor
            
            # Zaman eksenini güncelle
            self.time_axis.setRange(new_min, new_max)
            
            # Performans optimizasyonu: 
            # Çok yüksek sıklıkta örnekleme yaptığımız için, performans optimizasyonu yapabiliriz
            # Eğer çok fazla nokta birikirse, seyreltme işlemi uygulayabiliriz
            # Bu kod, serinin fazla büyümesini engeller ama bilimsel çalışma için veri kaybı oluşmaz
            max_points_per_series = 1000  # Makul bir limit
            if self.ear_series.count() > max_points_per_series:
                # Seyreltme işlemi - her iki noktadan birini tut
                # NOT: Noktaları silerken, ilk nokta sürekli silinir, endeks kayar!
                i = 1  # Her zaman 1. indeksten başla (0. değil, her bir noktayı koru)
                while i < self.ear_series.count():
                    self.ear_series.remove(i)
                    i += 1  # İndeksler kaydığı için i'yi 2 değil 1 artır
                    
            if self.normalized_ear_series.count() > max_points_per_series:
                i = 1
                while i < self.normalized_ear_series.count():
                    self.normalized_ear_series.remove(i)
                    i += 1
                    
            if self.mar_series.count() > max_points_per_series:
                i = 1
                while i < self.mar_series.count():
                    self.mar_series.remove(i)
                    i += 1
            
            if self.perclos_series.count() > max_points_per_series:
                i = 1
                while i < self.perclos_series.count():
                    self.perclos_series.remove(i)
                    i += 1
    
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
        # Stop the camera if it's running
        if self.is_capturing:
            self.on_stop()
        event.accept()

    def _show_expanded_charts(self):
        """Genişletilmiş grafikleri ayrı bir pencerede göster."""
        # Eğer pencere zaten açıksa odaklan, yoksa yeni pencere oluştur
        if hasattr(self, 'expanded_charts_window') and self.expanded_charts_window.isVisible():
            self.expanded_charts_window.activateWindow()
        else:
            self.expanded_charts_window = ExpandedChartsWindow(self.config, parent=self)
            self.expanded_charts_window.show()
            
            # Eğer veri toplanıyorsa, genişletilmiş grafiklere de veri gönder
            if self.is_capturing:
                # Mevcut verileri genişletilmiş grafiklere aktar
                ear_data = [(point.x(), point.y()) for i in range(self.ear_series.count()) 
                           if (point := self.ear_series.at(i)) is not None]
                mar_data = [(point.x(), point.y()) for i in range(self.mar_series.count()) 
                           if (point := self.mar_series.at(i)) is not None]
                perclos_data = [(point.x(), point.y()) for i in range(self.perclos_series.count()) 
                               if (point := self.perclos_series.at(i)) is not None]
                
                self.expanded_charts_window.initialize_with_data(ear_data, mar_data, perclos_data)
                
                # Timer'ı genişletilmiş grafiklere bağla
                self.chart_timer.timeout.connect(self.expanded_charts_window.update_charts)


class SettingsDialog(QDialog):
    """
    Settings dialog for the driver drowsiness detection application.
    
    This dialog allows the user to configure detection thresholds and
    camera settings.
    """
    
    def __init__(self, config, parent=None):
        """
        Initialize the settings dialog.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        
        self.setWindowTitle("Ayarlar")
        self.setMinimumWidth(400)
        
        # Apply minimalist style
        self.setStyleSheet("""
            QDialog {
                background-color: white;
                border-radius: 10px;
            }
            QLabel {
                color: #333333;
                font-weight: normal;
            }
            QDoubleSpinBox, QComboBox {
                border: 1px solid #e1e1e1;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
                min-height: 20px;
            }
            QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #007aff;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #333333;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #e1e1e1;
            }
            QPushButton:pressed {
                background-color: #d1d1d1;
            }
            QPushButton[default="true"] {
                background-color: #007aff;
                color: white;
            }
            QPushButton[default="true"]:hover {
                background-color: #0066cc;
            }
            QPushButton[default="true"]:pressed {
                background-color: #0055b3;
            }
        """)
        
        # Set layout
        layout = QFormLayout(self)
        layout.setContentsMargins(
            self.config['layout']['margin'] * 2,
            self.config['layout']['margin'] * 2,
            self.config['layout']['margin'] * 2,
            self.config['layout']['margin'] * 2
        )
        layout.setSpacing(self.config['layout']['spacing'] * 2)
        
        # Add a title or header
        title_label = QLabel("Algılama Parametreleri")
        title_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['title_size'],
            QFont.Weight.Medium
        ))
        title_label.setStyleSheet("margin-bottom: 10px;")
        layout.addRow(title_label)
        
        # EAR threshold
        ear_label = QLabel("EAR Eşiği:")
        ear_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.ear_threshold = QDoubleSpinBox()
        self.ear_threshold.setRange(0.1, 0.5)
        self.ear_threshold.setSingleStep(0.01)
        self.ear_threshold.setValue(self.config['indicators']['ear']['critical_threshold'])
        layout.addRow(ear_label, self.ear_threshold)
        
        # MAR threshold
        mar_label = QLabel("MAR Eşiği:")
        mar_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.mar_threshold = QDoubleSpinBox()
        self.mar_threshold.setRange(0.2, 1.0)
        self.mar_threshold.setSingleStep(0.01)
        self.mar_threshold.setValue(self.config['indicators']['mar']['critical_threshold'])
        layout.addRow(mar_label, self.mar_threshold)
        
        # PERCLOS threshold
        perclos_label = QLabel("PERCLOS Eşiği (%):")
        perclos_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.perclos_threshold = QDoubleSpinBox()
        self.perclos_threshold.setRange(0.0, 100.0)
        self.perclos_threshold.setSingleStep(1.0)
        self.perclos_threshold.setValue(self.config['indicators']['perclos']['critical_threshold'])
        layout.addRow(perclos_label, self.perclos_threshold)
        
        # Add another title
        camera_title = QLabel("Kamera Ayarları")
        camera_title.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['title_size'],
            QFont.Weight.Medium
        ))
        camera_title.setStyleSheet("margin-top: 10px; margin-bottom: 10px;")
        layout.addRow(camera_title)
        
        # Camera selection
        camera_label = QLabel("Kamera:")
        camera_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        self.camera_selection = QComboBox()
        self.camera_selection.addItems(["0 - Varsayılan Kamera", "1 - İkinci Kamera"])
        self.camera_selection.setCurrentIndex(0)  # Assuming 0 is the default camera
        layout.addRow(camera_label, self.camera_selection)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setProperty("default", "true")
        ok_button.setText("Kaydet")
        
        cancel_button = button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setText("İptal")
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addRow("", button_box)
    
    def accept(self):
        """Handle OK button click."""
        # In a real implementation, you would save the settings to the config file here
        # For now, just close the dialog
        super().accept()


class AboutDialog(QMessageBox):
    """
    About dialog for the driver drowsiness detection application.
    
    This dialog displays information about the application.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the about dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.setWindowTitle("Hakkında")
        self.setIcon(QMessageBox.Icon.Information)
        self.setText("Sürücü Uykululuk Tespit Sistemi")
        self.setInformativeText(
            "Bu uygulama, sürücülerin uykululuk durumunu tespit etmek için " +
            "bilgisayarlı görü ve yapay zeka teknolojilerini kullanır.\n\n" +
            "Geliştirici: Samet"
        )
        self.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # Customize style
        self.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #007aff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
            QPushButton:pressed {
                background-color: #0055b3;
            }
        """)
        
        # Customize font for title
        title_font = QFont(self.font())
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.setFont(title_font)


class ExpandedChartsWindow(QMainWindow):
    """
    Genişletilmiş grafikleri gösteren ayrı pencere.
    
    Bu pencere, EAR, MAR ve PERCLOS değerlerini ayrı ayrı,
    daha büyük grafikler halinde gösterir.
    """
    
    def __init__(self, config, parent=None):
        """
        Genişletilmiş grafikler penceresini başlat.
        
        Args:
            config: Yapılandırma sözlüğü
            parent: Ebeveyn pencere
        """
        super().__init__(parent)
        
        # Sabit grafik zaman aralığı
        self.fixed_history_duration = 60.0  # Sabit 60 saniye
        
        self.config = config
        self.parent_window = parent
        
        # Pencere özelliklerini ayarla
        self.setWindowTitle("Genişletilmiş Grafikler")
        self.resize(800, 600)
        
        # Ana widget ve layout oluştur
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Her bir metrik için ayrı bir grafik oluştur
        self._create_ear_chart(main_layout)
        self._create_mar_chart(main_layout)
        self._create_perclos_chart(main_layout)
        
        # Bilimsel analiz için ek özellikler
        control_layout = QHBoxLayout()
        
        # X-ekseni her zaman 60 saniye sabit olacak
        # Bu nedenle combobox'ı kaldırıp sabit değer bilgisi ekleyelim
        x_range_label = QLabel("X Ekseni Süresi: 60 saniye (sabit)")
        control_layout.addWidget(x_range_label)
        
        control_layout.addStretch()
        
        # Grafikleri kaydetme butonu
        save_button = QPushButton("Grafikleri Kaydet")
        save_button.clicked.connect(self._save_charts)
        control_layout.addWidget(save_button)
        
        main_layout.addLayout(control_layout)
    
    def _create_ear_chart(self, parent_layout):
        """EAR grafiğini oluştur."""
        chart = QChart()
        chart.setTitle("EAR (Göz Açıklık Oranı) Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundVisible(False)
        
        # Normal EAR serisi
        self.ear_series = QLineSeries()
        self.ear_series.setName("EAR")
        self.ear_series.setPen(QPen(QColor("#007aff"), 2, Qt.PenStyle.SolidLine))
        self.ear_series.setUseOpenGL(True)
        chart.addSeries(self.ear_series)
        
        # Normalize edilmiş EAR serisi
        self.normalized_ear_series = QLineSeries()
        self.normalized_ear_series.setName("Normalize EAR")
        self.normalized_ear_series.setPen(QPen(QColor("#ff5733"), 2, Qt.PenStyle.SolidLine))
        self.normalized_ear_series.setUseOpenGL(True)
        chart.addSeries(self.normalized_ear_series)
        
        # X ekseni - sabit 60 saniye
        self.ear_time_axis = QValueAxis()
        self.ear_time_axis.setRange(0, self.fixed_history_duration)
        self.ear_time_axis.setTitleText("Zaman (sn)")
        chart.addAxis(self.ear_time_axis, Qt.AlignmentFlag.AlignBottom)
        self.ear_series.attachAxis(self.ear_time_axis)
        self.normalized_ear_series.attachAxis(self.ear_time_axis)
        
        # Y ekseni - Normal EAR için
        self.ear_value_axis = QValueAxis()
        self.ear_value_axis.setRange(
            self.config['chart']['y_range_ear'][0],
            self.config['chart']['y_range_ear'][1]
        )
        self.ear_value_axis.setTitleText("EAR Değeri")
        chart.addAxis(self.ear_value_axis, Qt.AlignmentFlag.AlignLeft)
        self.ear_series.attachAxis(self.ear_value_axis)
        
        # Y ekseni - Normalize EAR için (0-1 arası)
        self.normalized_ear_value_axis = QValueAxis()
        self.normalized_ear_value_axis.setRange(0.0, 1.0)
        self.normalized_ear_value_axis.setTitleText("Normalize EAR Değeri (0-1)")
        chart.addAxis(self.normalized_ear_value_axis, Qt.AlignmentFlag.AlignRight)
        self.normalized_ear_series.attachAxis(self.normalized_ear_value_axis)
        
        # EAR eşiğini gösterme çizgisi ekle
        ear_threshold = self.config.get('detection', {}).get('ear_threshold', 0.21)
        ear_line = QLineSeries()
        ear_line.setName("EAR Eşiği")
        ear_line.setPen(QPen(QColor("#ff3b30"), 1, Qt.PenStyle.DashLine))
        ear_line.append(0, ear_threshold)
        ear_line.append(self.fixed_history_duration, ear_threshold)
        chart.addSeries(ear_line)
        ear_line.attachAxis(self.ear_time_axis)
        ear_line.attachAxis(self.ear_value_axis)
        
        # Grafik görünümü oluştur
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        parent_layout.addWidget(chart_view)
    
    def _create_mar_chart(self, parent_layout):
        """MAR grafiğini oluştur."""
        chart = QChart()
        chart.setTitle("MAR (Ağız Açıklık Oranı) Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundVisible(False)
        
        # Series oluştur
        self.mar_series = QLineSeries()
        self.mar_series.setName("MAR")
        self.mar_series.setPen(QPen(QColor("#5ac8fa"), 2, Qt.PenStyle.SolidLine))
        self.mar_series.setUseOpenGL(True)
        chart.addSeries(self.mar_series)
        
        # X ekseni - sabit 60 saniye
        self.mar_time_axis = QValueAxis()
        self.mar_time_axis.setRange(0, self.fixed_history_duration)
        self.mar_time_axis.setTitleText("Zaman (sn)")
        chart.addAxis(self.mar_time_axis, Qt.AlignmentFlag.AlignBottom)
        self.mar_series.attachAxis(self.mar_time_axis)
        
        # Y ekseni
        self.mar_value_axis = QValueAxis()
        self.mar_value_axis.setRange(
            self.config['chart']['y_range_mar'][0],
            self.config['chart']['y_range_mar'][1]
        )
        self.mar_value_axis.setTitleText("MAR Değeri")
        chart.addAxis(self.mar_value_axis, Qt.AlignmentFlag.AlignLeft)
        self.mar_series.attachAxis(self.mar_value_axis)
        
        # Grafik görünümü oluştur
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        parent_layout.addWidget(chart_view)
    
    def _create_perclos_chart(self, parent_layout):
        """PERCLOS grafiğini oluştur."""
        chart = QChart()
        chart.setTitle("PERCLOS (Göz Kapanma Yüzdesi) Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundVisible(False)
        
        # Series oluştur
        self.perclos_series = QLineSeries()
        self.perclos_series.setName("PERCLOS")
        self.perclos_series.setPen(QPen(QColor("#ff9500"), 2, Qt.PenStyle.SolidLine))
        self.perclos_series.setUseOpenGL(True)
        chart.addSeries(self.perclos_series)
        
        # X ekseni - sabit 60 saniye
        self.perclos_time_axis = QValueAxis()
        self.perclos_time_axis.setRange(0, self.fixed_history_duration)
        self.perclos_time_axis.setTitleText("Zaman (sn)")
        chart.addAxis(self.perclos_time_axis, Qt.AlignmentFlag.AlignBottom)
        self.perclos_series.attachAxis(self.perclos_time_axis)
        
        # Y ekseni
        self.perclos_value_axis = QValueAxis()
        self.perclos_value_axis.setRange(
            self.config['chart']['y_range_perclos'][0],
            self.config['chart']['y_range_perclos'][1]
        )
        self.perclos_value_axis.setTitleText("PERCLOS Değeri (%)")
        
        # PERCLOS eşiğini gösterme çizgisi ekle
        perclos_threshold = self.config.get('indicators', {}).get('perclos', {}).get('critical_threshold', 20.0)
        perclos_line = QLineSeries()
        perclos_line.setName("PERCLOS Eşiği")
        perclos_line.setPen(QPen(QColor("#ff3b30"), 1, Qt.PenStyle.DashLine))
        perclos_line.append(0, perclos_threshold)
        perclos_line.append(self.fixed_history_duration, perclos_threshold)
        chart.addSeries(perclos_line)
        chart.addAxis(self.perclos_value_axis, Qt.AlignmentFlag.AlignLeft)
        self.perclos_series.attachAxis(self.perclos_value_axis)
        perclos_line.attachAxis(self.perclos_time_axis)
        perclos_line.attachAxis(self.perclos_value_axis)
        
        # Grafik görünümü oluştur
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        parent_layout.addWidget(chart_view)
    
    def initialize_with_data(self, ear_data, mar_data, perclos_data):
        """Grafikleri mevcut verilerle başlat."""
        # Mevcut serileri temizle
        self.ear_series.clear()
        self.normalized_ear_series.clear()
        self.mar_series.clear()
        self.perclos_series.clear()
        
        # EAR değerlerini ekle
        for x, y in ear_data:
            self.ear_series.append(x, y)
            
        # Eğer kalibrasyon tamamlanmışsa normalize edilmiş EAR değerlerini hesapla ve ekle
        if self.parent_window.is_calibrated:
            # Normalize edilmiş değerleri hesapla ve ekle
            ear_min = self.parent_window.ear_min
            ear_max = self.parent_window.ear_max
            ear_range = self.parent_window.ear_range
            
            for x, y in ear_data:
                # Normalize edilmiş EAR değerini hesapla ve ekle
                # Daha sağlam normalizasyon hesaplaması - ara değişkenler kullanarak
                diff = y - ear_min
                # Bölme işleminden önce sıfırdan küçük değerleri engelle
                safe_diff = max(0.0, diff)
                # Sıfıra bölünmeyi engelle
                if ear_range > 0.001:  # Çok küçük değerlerden kaçın
                    raw_normalized = safe_diff / ear_range
                else:
                    raw_normalized = 0.0
                # Son güvenlik kontrolü - 0 ile 1 arasında olduğundan emin ol
                normalized_ear = min(1.0, max(0.0, raw_normalized))
                
                self.normalized_ear_series.append(x, normalized_ear)
        
        # MAR verisini ekle
        for x, y in mar_data:
            self.mar_series.append(x, y)
        
        # PERCLOS verisini ekle
        for x, y in perclos_data:
            self.perclos_series.append(x, y)
    
    def update_charts(self):
        """Ana penceredeki verilerle grafikleri güncelle."""
        if not self.parent_window.is_capturing:
            return
        
        # Ana pencereden yeni veri noktaları al
        time = self.parent_window.time_counter
        ear = self.parent_window.ear_indicator.last_value
        mar = self.parent_window.mar_indicator.last_value
        perclos = self.parent_window.perclos_indicator.last_value
        
        # Verileri grafiklere ekle
        self.ear_series.append(time, ear)
        
        # Sadece kalibrasyon tamamlandıktan sonra normalize edilmiş EAR değerini hesapla ve ekle
        if self.parent_window.is_calibrated:
            # Normalize edilmiş EAR değerini hesapla
            ear_min = self.parent_window.ear_min
            ear_max = self.parent_window.ear_max
            ear_range = self.parent_window.ear_range
            
            # Daha sağlam normalizasyon hesaplaması - ara değişkenler kullanarak
            diff = ear - ear_min
            # Bölme işleminden önce sıfırdan küçük değerleri engelle
            safe_diff = max(0.0, diff)
            # Sıfıra bölünmeyi engelle
            if ear_range > 0.001:  # Çok küçük değerlerden kaçın
                raw_normalized = safe_diff / ear_range
            else:
                raw_normalized = 0.0
            # Son güvenlik kontrolü - 0 ile 1 arasında olduğundan emin ol
            normalized_ear = min(1.0, max(0.0, raw_normalized))
            
            self.normalized_ear_series.append(time, normalized_ear)
            
        self.mar_series.append(time, mar)
        self.perclos_series.append(time, perclos)
        
        # X ekseni aralığını güncelle - sabit 60 saniyelik pencere kullan
        history_duration = self.fixed_history_duration
        if time > history_duration:
            # Eğer zaman tarih aralığını aştıysa, eski noktaları kaldır
            cutoff_time = time - history_duration
            
            # EAR serilerinden eski noktaları kaldır
            while self.ear_series.count() > 0 and self.ear_series.at(0).x() < cutoff_time:
                self.ear_series.remove(0)
                
            while self.normalized_ear_series.count() > 0 and self.normalized_ear_series.at(0).x() < cutoff_time:
                self.normalized_ear_series.remove(0)
            
            # MAR serisinden eski noktaları kaldır
            while self.mar_series.count() > 0 and self.mar_series.at(0).x() < cutoff_time:
                self.mar_series.remove(0)
            
            # PERCLOS serisinden eski noktaları kaldır
            while self.perclos_series.count() > 0 and self.perclos_series.at(0).x() < cutoff_time:
                self.perclos_series.remove(0)
            
            # Zaman eksenlerini güncelle - zamana bağlı kaydırma yap
            self.ear_time_axis.setRange(time - history_duration, time)
            self.mar_time_axis.setRange(time - history_duration, time)
            self.perclos_time_axis.setRange(time - history_duration, time)
    
    def _save_charts(self):
        """Grafikleri görüntü dosyaları olarak kaydet."""
        from datetime import datetime
        
        # Kayıt klasörünü oluştur
        import os
        save_dir = os.path.join(os.path.expanduser("~"), "DriversMetrics")
        os.makedirs(save_dir, exist_ok=True)
        
        # Zaman damgası oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Her bir grafiği kaydet
        try:
            # Ana widget'taki tüm grafikler
            chart_views = self.findChildren(QChartView)
            
            for i, chart_view in enumerate(chart_views):
                metric_name = ["EAR", "MAR", "PERCLOS"][i]
                filename = os.path.join(save_dir, f"{metric_name}_{timestamp}.png")
                
                # Grafiği yüksek çözünürlüklü görüntü olarak kaydet
                pixmap = QPixmap(chart_view.size())
                pixmap.fill(Qt.GlobalColor.white)
                
                painter = QPainter(pixmap)
                chart_view.render(painter)
                painter.end()
                
                pixmap.save(filename)
            
            # Bilgi mesajı göster
            QMessageBox.information(
                self,
                "Grafikler Kaydedildi",
                f"Grafikler {save_dir} dizinine kaydedildi."
            )
            
        except Exception as e:
            QMessageBox.warning(
                self,
                "Kayıt Hatası",
                f"Grafikler kaydedilirken bir hata oluştu: {str(e)}"
            )
    
    def closeEvent(self, event):
        """Pencere kapatıldığında timer bağlantısını kaldır."""
        if hasattr(self.parent_window, "chart_timer"):
            try:
                self.parent_window.chart_timer.timeout.disconnect(self.update_charts)
            except TypeError:
                # Bağlantı zaten yoksa hata oluşabilir, yoksay
                pass
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