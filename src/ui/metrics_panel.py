#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics panel for the driver drowsiness detection application.

This module implements a widget for displaying real-time metrics like EAR, MAR, and PERCLOS.
"""

from PyQt6.QtWidgets import QWidget, QGridLayout, QSizePolicy, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Import custom widget
from src.ui.widgets import IndicatorWidget


class MetricsPanel(QWidget):
    """
    Panel for displaying drowsiness detection metrics.
    
    This class implements a widget that displays various metrics relevant to
    drowsiness detection, such as Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
    and PERCLOS.
    """
    
    def __init__(self, config, parent=None):
        """
        Initialize the metrics panel.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        
        # Set appearance - daha belirgin kenarlar ve arka plan rengi
        self.setStyleSheet("""
            background-color: white;
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            padding: 10px;
        """)
        
        # Sabit genişlik yerine genişleyen boyut politikası kullan
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Fixed
        )
        
        # Create layout
        self._create_layout()
    
    def _create_layout(self):
        """Create the panel layout with indicator widgets."""
        # Create grid layout - daha kompakt bir görünüm için
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)  # Daha az iç boşluk
        layout.setSpacing(10)  # Göstergeler arası boşluk
        
        # "Uykululuk Metrikleri" başlığı kaldırıldı
        
        # Indicator widget'ları ekle - her biri için bir satırda
        self.ear_indicator = IndicatorWidget("EAR", self.config)
        layout.addWidget(self.ear_indicator)
        
        self.mar_indicator = IndicatorWidget("MAR", self.config)
        layout.addWidget(self.mar_indicator)
        
        self.perclos_indicator = IndicatorWidget("PERCLOS", self.config)
        layout.addWidget(self.perclos_indicator)
    
    def update_metrics(self, ear, mar, perclos):
        """
        Update all metrics displayed in the UI.
        
        Args:
            ear: Eye Aspect Ratio value
            mar: Mouth Aspect Ratio value
            perclos: PERCLOS value
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