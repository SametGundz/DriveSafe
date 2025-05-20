#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Control panel for the driver drowsiness detection application.

This module implements the control panel with buttons for
starting, stopping and configuring the application.
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ControlPanel(QWidget):
    """
    Panel for controlling the application.
    
    This class implements a panel with buttons for starting,
    stopping and configuring the application.
    """
    
    # Signals
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()
    landmarks_toggled = pyqtSignal(bool)
    head_pose_toggled = pyqtSignal(bool)
    gaze_toggled = pyqtSignal(bool)
    gaze_zone_toggled = pyqtSignal(bool)
    expand_charts_clicked = pyqtSignal()
    
    def __init__(self, config, parent=None):
        """
        Initialize the control panel.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        
        # Sabit yükseklik ve genişlemesini sadece yatayda yapması için boyut politikası ayarla
        self.setFixedHeight(80)  # Sabit kontrol paneli yüksekliği
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,  # Genişlik için expanding
            QSizePolicy.Policy.Fixed       # Yükseklik için fixed
        )
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 20)
        
        # Create control buttons
        self.start_button = QPushButton("Başlat")
        self.start_button.setObjectName("start_button")
        self.start_button.setFixedSize(
            self.config['controls']['button_width'],
            self.config['controls']['button_height']
        )
        self.start_button.clicked.connect(self.start_clicked)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Durdur")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setFixedSize(
            self.config['controls']['button_width'],
            self.config['controls']['button_height']
        )
        self.stop_button.clicked.connect(self.stop_clicked)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        self.settings_button = QPushButton("Ayarlar")
        self.settings_button.setFixedSize(
            self.config['controls']['button_width'],
            self.config['controls']['button_height']
        )
        self.settings_button.clicked.connect(self.settings_clicked)
        layout.addWidget(self.settings_button)
        
        # Toggle face landmarks button
        self.landmarks_button = QPushButton("Yüz İşaretlerini Göster")
        self.landmarks_button.setFixedSize(
            self.config['controls']['button_width'] + 80,  # Increase width to fit text
            self.config['controls']['button_height']
        )
        self.landmarks_button.setCheckable(True)
        self.landmarks_button.setChecked(False)
        self.landmarks_button.clicked.connect(self._on_landmarks_toggled)
        layout.addWidget(self.landmarks_button)
        
        # Toggle head pose button
        self.head_pose_button = QPushButton("Baş Duruşunu Göster")
        self.head_pose_button.setFixedSize(
            self.config['controls']['button_width'] + 80,
            self.config['controls']['button_height']
        )
        self.head_pose_button.setCheckable(True)
        self.head_pose_button.setChecked(False)
        self.head_pose_button.clicked.connect(self._on_head_pose_toggled)
        layout.addWidget(self.head_pose_button)
        
        # Toggle gaze button
        self.gaze_button = QPushButton("Bakış Yönünü Göster")
        self.gaze_button.setCheckable(True)
        self.gaze_button.setChecked(False)
        self.gaze_button.clicked.connect(self._on_gaze_toggled)
        layout.addWidget(self.gaze_button)
        
        # Toggle gaze zone button
        self.gaze_zone_button = QPushButton("Bakış Bölgesini Göster")
        self.gaze_zone_button.setCheckable(True)
        self.gaze_zone_button.setChecked(False)
        self.gaze_zone_button.clicked.connect(self._on_gaze_zone_toggled)
        layout.addWidget(self.gaze_zone_button)
        
        # Add expanded charts button
        self.expand_charts_button = QPushButton("Grafikleri Genişlet")
        self.expand_charts_button.setFixedSize(
            self.config['controls']['button_width'] + 50,
            self.config['controls']['button_height']
        )
        self.expand_charts_button.clicked.connect(self.expand_charts_clicked)
        layout.addWidget(self.expand_charts_button)
        
        layout.addStretch()
    
    def _on_landmarks_toggled(self, checked):
        """Handle landmarks button toggle."""
        self.landmarks_toggled.emit(checked)
        if checked:
            self.landmarks_button.setStyleSheet("""
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            """)
        else:
            self.landmarks_button.setStyleSheet("")
    
    def _on_head_pose_toggled(self, checked):
        """Handle head pose button toggle."""
        self.head_pose_toggled.emit(checked)
        if checked:
            self.head_pose_button.setStyleSheet("background-color: #2196F3; color: white;")
        else:
            self.head_pose_button.setStyleSheet("")
    
    def _on_gaze_toggled(self, checked):
        """Handle gaze button toggle."""
        self.gaze_toggled.emit(checked)
        if checked:
            self.gaze_button.setStyleSheet("background-color: #9C27B0; color: white;")
        else:
            self.gaze_button.setStyleSheet("")
    
    def _on_gaze_zone_toggled(self, checked):
        """Handle gaze zone button toggle."""
        self.gaze_zone_toggled.emit(checked)
        if checked:
            self.gaze_zone_button.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.gaze_zone_button.setStyleSheet("")
    
    def update_start_stop_state(self, is_capturing):
        """Update the state of start/stop buttons."""
        self.start_button.setEnabled(not is_capturing)
        self.stop_button.setEnabled(is_capturing) 