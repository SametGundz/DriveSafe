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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QListWidget, QMenuBar, QToolBar,
    QStatusBar, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGridLayout, QFormLayout, QDialog, QDoubleSpinBox, QComboBox,
    QDialogButtonBox, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSlot, QMargins
from PyQt6.QtGui import QPixmap, QImage, QAction, QIcon, QFont, QPainter, QColor, QPen
from PyQt6.QtCharts import QChartView, QChart, QLineSeries, QValueAxis

# Import custom widgets
from src.ui.widgets import IndicatorWidget

# Import utils
from src.utils.mediapipe_utils import load_ui_config, MediaPipeUtils

# Import GazeEstimator
from src.detection.gaze_estimator import GazeEstimator


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
        
        # Load configuration
        self.config = load_ui_config()
        
        # Initialize video capture variables
        self.cap = None
        self.camera_id = self.config.get('camera', {}).get('device', 0)
        self.is_capturing = False
        
        # Initialize MediaPipe
        self.mediapipe_utils = None
        
        # Initialize GazeEstimator
        self.gaze_estimator = None
        
        # Flag to control landmark and gaze visibility
        self.show_landmarks = False
        
        # Create camera timer for video updates
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.camera_timer.setInterval(33)  # ~30 fps
        
        # Sample data for demonstration
        self.demo_data = {
            'ear': 0.25,
            'mar': 0.5,
            'perclos': 5.0,
            'gaze_dir': (0.1, 0.2, 0.8)
        }
        
        # Initialize time counter for demo data
        self.time_counter = 0.0
        self.update_interval_sec = 0.1  # 100ms in seconds
        
        # Initialize PERCLOS calculation variables
        self.eye_closure_history = []
        self.max_history_frames = int(self.config.get('detection', {})
                                    .get('perclos', {}).get('window_size', 150))
        
        # Initialize UI
        self._init_ui()
    
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
        
        # Gaze direction indicator
        gaze_widget = QWidget()
        gaze_layout = QVBoxLayout(gaze_widget)
        gaze_layout.setContentsMargins(
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding']
        )
        gaze_layout.setSpacing(5)  # Azaltılmış boşluk
        
        gaze_title = QLabel("Göz Bakış Yönü")
        gaze_title.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        gaze_layout.addWidget(gaze_title)
        
        # Create form layout for gaze direction values
        form_widget = QWidget()
        self.gaze_group_layout = QFormLayout(form_widget)
        self.gaze_group_layout.setContentsMargins(0, 0, 0, 0)
        self.gaze_group_layout.setSpacing(3)  # Reduced spacing
        
        # Create labels for each coordinate
        self.gaze_x_label = QLabel("0.00")
        self.gaze_y_label = QLabel("0.00")
        self.gaze_z_label = QLabel("0.00")
        
        # Smaller font
        smaller_font = QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'] - 1
        )
        self.gaze_x_label.setFont(smaller_font)
        self.gaze_y_label.setFont(smaller_font)
        self.gaze_z_label.setFont(smaller_font)
        
        # Style settings
        self.gaze_x_label.setStyleSheet("color: #ff3b30; font-weight: bold;")  # Red
        self.gaze_y_label.setStyleSheet("color: #34c759; font-weight: bold;")  # Green
        self.gaze_z_label.setStyleSheet("color: #5856d6; font-weight: bold;")  # Purple
        
        # Add labels to form layout
        self.gaze_group_layout.addRow("X:", self.gaze_x_label)
        self.gaze_group_layout.addRow("Y:", self.gaze_y_label)
        self.gaze_group_layout.addRow("Z:", self.gaze_z_label)
        
        # Add form widget to gaze layout
        gaze_layout.addWidget(form_widget)
        
        # Create status and focus labels (will be populated in update_metrics)
        self.gaze_status_label = QLabel("Merkez")
        self.gaze_status_label.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.gaze_group_layout.addRow("Durum:", self.gaze_status_label)
        
        # Add angle labels
        self.gaze_yaw_label = QLabel("0.0°")
        self.gaze_yaw_label.setStyleSheet("font-weight: bold; color: #FF0000;")
        self.gaze_group_layout.addRow("Yaw:", self.gaze_yaw_label)
        
        self.gaze_pitch_label = QLabel("0.0°")
        self.gaze_pitch_label.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.gaze_group_layout.addRow("Pitch:", self.gaze_pitch_label)
        
        self.gaze_focus_label = QLabel("Yüksek")
        self.gaze_focus_label.setStyleSheet("font-weight: bold; color: #00FF00;")
        self.gaze_group_layout.addRow("Odak:", self.gaze_focus_label)
        
        # Add widget to stats_layout
        stats_layout.addWidget(gaze_widget, 3, 0)
        
        self.stats_widget = stats_widget
        top_layout.addWidget(stats_widget)
        
        main_layout.addLayout(top_layout)
        
        # Control area
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
        
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        # Time series chart
        self._create_chart()
        main_layout.addWidget(self.chart_view)
    
    def _create_chart(self):
        """Create the time series chart for EAR, MAR, and PERCLOS data."""
        # Create chart
        chart = QChart()
        chart.setTitle("Metrikler Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
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
        
        # Create series for EAR, MAR, and PERCLOS
        self.ear_series = QLineSeries()
        self.ear_series.setName("EAR")
        self.ear_series.setPen(QPen(QColor("#007aff"), self.config['chart']['line_width']))
        
        self.mar_series = QLineSeries()
        self.mar_series.setName("MAR")
        self.mar_series.setPen(QPen(QColor("#5ac8fa"), self.config['chart']['line_width']))
        
        self.perclos_series = QLineSeries()
        self.perclos_series.setName("PERCLOS")
        self.perclos_series.setPen(QPen(QColor("#ff9500"), self.config['chart']['line_width']))
        
        # Add series to chart
        chart.addSeries(self.ear_series)
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
        
        self.mar_series.attachAxis(self.time_axis)
        self.mar_series.attachAxis(self.mar_axis)
        
        self.perclos_series.attachAxis(self.time_axis)
        self.perclos_series.attachAxis(self.perclos_axis)
        
        # Create chart view
        self.chart_view = QChartView(chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
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
    
    def update_metrics(self, ear, mar, perclos, gaze_dir):
        """
        Update all metrics displayed in the UI.
        
        Args:
            ear: Eye Aspect Ratio value
            mar: Mouth Aspect Ratio value
            perclos: PERCLOS value
            gaze_dir: List or tuple of [x, y, z] gaze direction vector
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
        
        # Update gaze direction with more descriptive information
        x, y, z = gaze_dir
        
        # Convert normalized directions to angles (in degrees)
        yaw_angle = np.arcsin(np.clip(x, -1.0, 1.0)) * 180.0 / np.pi
        pitch_angle = np.arcsin(np.clip(y, -1.0, 1.0)) * 180.0 / np.pi
        
        # Determine gaze direction in words with HTML color codes for PyQt
        if abs(x) < 0.2 and abs(y) < 0.2:
            gaze_status = "MERKEZE BAKIYOR"
            status_color = "#00FF00"  # Green for center (HTML format for PyQt)
        elif x < -0.3:
            gaze_status = "SOLA BAKIYOR"
            status_color = "#FFA500"  # Orange for left (HTML format for PyQt)
        elif x > 0.3:
            gaze_status = "SAĞA BAKIYOR"
            status_color = "#FFA500"  # Orange for right (HTML format for PyQt)
        elif y < -0.3:
            gaze_status = "YUKARI BAKIYOR"
            status_color = "#FFA500"  # Orange for up (HTML format for PyQt)
        elif y > 0.3:
            gaze_status = "AŞAĞI BAKIYOR"
            status_color = "#FFA500"  # Orange for down (HTML format for PyQt)
        else:
            gaze_status = "MERKEZ CIVARINDA"
            status_color = "#00FF00"  # Green for near center (HTML format for PyQt)
            
        # Format coordinate values
        self.gaze_x_label.setText(f"{x:.2f}")
        self.gaze_y_label.setText(f"{y:.2f}")
        self.gaze_z_label.setText(f"{z:.2f}")
        
        # Update angle labels
        self.gaze_yaw_label.setText(f"{yaw_angle:.1f}°")
        self.gaze_pitch_label.setText(f"{pitch_angle:.1f}°")
        
        # Update the status label with colorful text
        self.gaze_status_label.setText(gaze_status)
        self.gaze_status_label.setStyleSheet(f"font-weight: bold; color: {status_color};")
        
        # Update focus assessment based on z-value (depth)
        if z > 0.7:
            focus_status = "Yüksek"
            focus_color = "#00FF00"  # Green for high focus
        elif z > 0.4:
            focus_status = "Orta"
            focus_color = "#FFFF00"  # Yellow for medium focus
        else:
            focus_status = "Düşük"
            focus_color = "#FF0000"  # Red for low focus
            
        self.gaze_focus_label.setText(focus_status)
        self.gaze_focus_label.setStyleSheet(f"font-weight: bold; color: {focus_color};")
    
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
            
            # Initialize GazeEstimator
            try:
                self.gaze_estimator = GazeEstimator()
                print("INFO: GazeEstimator başarıyla başlatıldı")
            except Exception as e:
                print(f"WARNING: GazeEstimator başlatılamadı: {e}")
                self.gaze_estimator = None
            
            # Get actual camera properties after setting
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"INFO: Actual camera properties: width={actual_width}, height={actual_height}, fps={actual_fps}")
                
            # Start the camera timer
            self.is_capturing = True
            self.camera_timer.start()
            
            # Start chart update timer (demo)
            self.time_counter = 0.0
            self.chart_timer = QTimer(self)
            self.chart_timer.timeout.connect(self._update_chart_data)
            self.chart_timer.setInterval(100)  # 100ms
            self.chart_timer.start()
            
            # Reset PERCLOS calculation
            self.eye_closure_history = []
            
            self.update_status("Algılama başlatıldı.")
            print("INFO: Camera capture started successfully")
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
        
        # GazeEstimator'ı temizle (herhangi bir release metodu yok, Python GC ile temizlenir)
        self.gaze_estimator = None
        
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
        if getattr(self, '_frame_counter', 0) % 30 == 0:
            print(f"INFO: Frame received - shape: {frame.shape}")
        self._frame_counter = getattr(self, '_frame_counter', 0) + 1
            
        # Mirror the frame horizontally (selfie view)
        frame = cv2.flip(frame, 1)
        
        # Process the frame with MediaPipe
        landmarks, face_detected = self.mediapipe_utils.detect_face_landmarks(frame)
        
        # Initialize metrics with default values
        ear = 0.0
        mar = 0.0
        perclos = 0.0
        gaze_dir = (0.0, 0.0, 0.0)
        gaze_angles = (0.0, 0.0)  # Yaw and pitch açıları
        
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
            
            # Use landmarks directly for MAR calculation
            mar = self.mediapipe_utils.get_mouth_aspect_ratio(landmarks)
            
            # Calculate gaze direction
            gaze_dir = self.mediapipe_utils.get_eye_gaze_direction(landmarks)
            
            # Update PERCLOS
            is_eye_closed = ear < self.config.get('detection', {}).get('ear_threshold', 0.21)
            self.eye_closure_history.append(1 if is_eye_closed else 0)
            
            # Keep history within window size
            if len(self.eye_closure_history) > self.max_history_frames:
                self.eye_closure_history = self.eye_closure_history[-self.max_history_frames:]
            
            # Calculate PERCLOS as percentage of closed eyes in the window
            if self.eye_closure_history:
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
                
                # Eğer GazeEstimator etkinse daha doğru bakış tahmini yap
                if self.gaze_estimator is not None:
                    # GazeEstimator için landmarks_dict hazırla
                    h, w, _ = frame.shape
                    
                    # MediaPipe yüz işaret noktalarını GazeEstimator için uygun formata dönüştür
                    left_eye_indices = self.mediapipe_utils.LEFT_EYE_INDICES
                    right_eye_indices = self.mediapipe_utils.RIGHT_EYE_INDICES
                    
                    # GazeEstimator'ın istediği formatta landmarks_dict oluştur
                    landmarks_dict = {
                        'all_landmarks': landmarks,
                        'left_eye': np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in left_eye_indices]),
                        'right_eye': np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in right_eye_indices])
                    }
                    
                    # Bakış tahmini yap
                    success, gaze_angles, normalized_face = self.gaze_estimator.detect(frame, landmarks_dict)
                    
                    if success:
                        # Gaze vektörünü çiz
                        frame = self.gaze_estimator.draw_gaze_vector(frame, landmarks_dict, gaze_angles, length=100, thickness=2, color=(0, 0, 255))
                        
                        # GazeEstimator'dan alınan açıları kullanarak gaze_dir güncelle
                        # Bu, update_metrics fonksiyonunda kullanılacak
                        yaw_rad = np.radians(gaze_angles[0])
                        pitch_rad = np.radians(gaze_angles[1])
                        
                        # 3D gaze vektörü oluştur
                        x = -np.sin(yaw_rad) * np.cos(pitch_rad)
                        y = -np.sin(pitch_rad)
                        z = -np.cos(yaw_rad) * np.cos(pitch_rad)
                        gaze_dir = (x, y, z)
                else:
                    # GazeEstimator yoksa, sadece MediaPipe ile tahmin edilen gaze'i kullan
                    frame = self.mediapipe_utils.draw_gaze_direction_v2(
                        frame, landmarks, gaze_dir, 
                        arrow_color=(0, 0, 255),
                        arrow_length=150,
                        arrow_thickness=3
                    )
        
        # Display the frame
        self.update_video_frame(frame)
        
        # Update metrics UI
        self.update_metrics(ear, mar, perclos, gaze_dir)
    
    def _update_chart_data(self):
        """Update the chart with new data points."""
        if not self.is_capturing:
            return
            
        # Increment time counter
        self.time_counter += self.update_interval_sec
        
        # Get metrics from latest frame processing, should already have real values
        # Just add the current values to the chart
        
        # Add data points to series
        self.ear_series.append(self.time_counter, self.ear_indicator.last_value)
        self.mar_series.append(self.time_counter, self.mar_indicator.last_value)
        self.perclos_series.append(self.time_counter, self.perclos_indicator.last_value)
        
        # Remove old data points if we exceed the chart duration
        history_duration = self.config['chart']['history_duration']
        if self.time_counter > history_duration:
            # Adjust the X axis to show a sliding window
            self.time_axis.setRange(self.time_counter - history_duration, self.time_counter)
            
            # Optional: Remove old points to save memory
            # This would need to track the points being added
    
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