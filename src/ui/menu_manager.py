#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Menu manager for the driver drowsiness detection application.

This module implements the menu system for the application.
"""

from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction


class MenuManager(QMenuBar):
    """
    Menu manager for the application.
    
    This class implements the menu system with File, View, and Help menus.
    """
    
    # Signals
    upload_video_triggered = pyqtSignal()
    start_triggered = pyqtSignal()
    stop_triggered = pyqtSignal()
    settings_triggered = pyqtSignal()
    exit_triggered = pyqtSignal()
    stats_toggle_triggered = pyqtSignal(bool)
    chart_toggle_triggered = pyqtSignal(bool)
    about_triggered = pyqtSignal()
    
    def __init__(self, config, parent=None):
        """
        Initialize the menu manager.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        
        # Set style
        self.setStyleSheet("""
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
        
        # Create menus
        self._create_file_menu()
        self._create_view_menu()
        self._create_help_menu()
    
    def _create_file_menu(self):
        """Create the File menu."""
        file_menu = self.addMenu("Dosya")
        
        # File menu actions
        upload_video_action = QAction("Video Yükle", self)
        upload_video_action.triggered.connect(self.upload_video_triggered)
        file_menu.addAction(upload_video_action)
        
        file_menu.addSeparator()
        
        start_action = QAction("Başlat", self)
        start_action.triggered.connect(self.start_triggered)
        file_menu.addAction(start_action)
        
        stop_action = QAction("Durdur", self)
        stop_action.triggered.connect(self.stop_triggered)
        file_menu.addAction(stop_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("Ayarlar", self)
        settings_action.triggered.connect(self.settings_triggered)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Çıkış", self)
        exit_action.triggered.connect(self.exit_triggered)
        file_menu.addAction(exit_action)
    
    def _create_view_menu(self):
        """Create the View menu."""
        view_menu = self.addMenu("Görünüm")
        
        # View menu actions
        self.toggle_stats_action = QAction("İstatistikleri Göster", self)
        self.toggle_stats_action.setCheckable(True)
        self.toggle_stats_action.setChecked(True)
        self.toggle_stats_action.triggered.connect(self.stats_toggle_triggered)
        view_menu.addAction(self.toggle_stats_action)
        
        self.toggle_chart_action = QAction("Grafik Göster", self)
        self.toggle_chart_action.setCheckable(True)
        self.toggle_chart_action.setChecked(True)
        self.toggle_chart_action.triggered.connect(self.chart_toggle_triggered)
        view_menu.addAction(self.toggle_chart_action)
    
    def _create_help_menu(self):
        """Create the Help menu."""
        help_menu = self.addMenu("Yardım")
        
        # Help menu actions
        about_action = QAction("Hakkında", self)
        about_action.triggered.connect(self.about_triggered)
        help_menu.addAction(about_action) 