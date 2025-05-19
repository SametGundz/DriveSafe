#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics panel for the driver drowsiness detection application.

This module implements a widget for displaying real-time metrics like EAR, MAR, and PERCLOS.
"""

from PyQt6.QtWidgets import QWidget, QGridLayout, QSizePolicy
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
        
        # Set appearance
        self.setStyleSheet("""
            background-color: white;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
            padding: 8px;
        """)
        
        # Sabit boyutları ayarla - Video panel ile aynı boyutu kullan
        self.setFixedSize(
            self.config['layout'].get('dock_width', 400),  # Video panel ile aynı genişlik
            self.config['video_frame']['height']
        )
        
        # Boyut politikasını sabit olarak ayarla ve sağa hizalamayı belirt
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed, 
            QSizePolicy.Policy.Fixed
        )
        
        # Create layout
        self._create_layout()
    
    def _create_layout(self):
        """Create the panel layout with indicator widgets."""
        # Create grid layout
        layout = QGridLayout(self)
        layout.setContentsMargins(
            self.config['layout']['padding'] * 2,  # Daha fazla padding ekle
            self.config['layout']['padding'] * 2,  # Daha fazla padding ekle
            self.config['layout']['padding'] * 2,  # Daha fazla padding ekle
            self.config['layout']['padding'] * 2   # Daha fazla padding ekle
        )
        layout.setSpacing(self.config['layout']['spacing'] * 2)  # İndikatörler arası daha fazla boşluk
        
        # Indicator widget'ları ekle - daha fazla boşluk ile her biri için bir satırda
        self.ear_indicator = IndicatorWidget("EAR", self.config)
        layout.addWidget(self.ear_indicator, 0, 0, 1, 1)  # İlk satır
        
        self.mar_indicator = IndicatorWidget("MAR", self.config)
        layout.addWidget(self.mar_indicator, 1, 0, 1, 1)  # İkinci satır
        
        self.perclos_indicator = IndicatorWidget("PERCLOS", self.config)
        layout.addWidget(self.perclos_indicator, 2, 0, 1, 1)  # Üçüncü satır
        
        # Boş bir widget ekleyerek alt kısımda boşluk bırak
        empty_widget = QWidget()
        empty_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(empty_widget, 3, 0, 1, 1)  # Dördüncü satır
    
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