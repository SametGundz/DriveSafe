#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom widgets for the driver drowsiness detection application.

This module implements custom widgets used in the application, such as
the indicator widget for displaying metric values.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, QMargins
from PyQt6.QtGui import QFont, QColor

class IndicatorWidget(QWidget):
    """
    Widget for displaying a metric with name, value, and progress bar.
    
    This class implements a widget that displays a metric name, its current value,
    and a progress bar indicating the value's position in a range.
    """
    
    def __init__(self, name, config, parent=None):
        """
        Initialize the indicator widget.
        
        Args:
            name: The name of the metric (e.g., 'EAR', 'MAR', 'PERCLOS')
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.name = name
        self.config = config
        self.last_value = 0.0
        
        # Set up styling based on metric type
        if name == "EAR":
            # Varsayılan renk değeri - configde yoksa varsayılan değer kullan
            self.color = self.config['indicators']['ear'].get('color', "#007aff")
            self.warning_threshold = self.config['indicators']['ear']['warning_threshold']
            self.critical_threshold = self.config['indicators']['ear']['critical_threshold']
            self.value_format = "{:.2f}"
            self.inverted = True  # Lower is worse for EAR
        elif name == "MAR":
            self.color = self.config['indicators']['mar'].get('color', "#5ac8fa")
            self.warning_threshold = self.config['indicators']['mar']['warning_threshold']
            self.critical_threshold = self.config['indicators']['mar']['critical_threshold']
            self.value_format = "{:.2f}"
            self.inverted = False  # Higher is worse for MAR
        elif name == "PERCLOS":
            self.color = self.config['indicators']['perclos'].get('color', "#ff9500")
            self.warning_threshold = self.config['indicators']['perclos']['warning_threshold']
            self.critical_threshold = self.config['indicators']['perclos']['critical_threshold']
            self.value_format = "{:.1f}%"
            self.inverted = False  # Higher is worse for PERCLOS
        else:
            self.color = "#333333"
            self.warning_threshold = 0.5
            self.critical_threshold = 0.7
            self.value_format = "{:.2f}"
            self.inverted = False
        
        # Initialize UI elements
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI elements."""
        # Set size policy
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding']
        )
        layout.setSpacing(self.config['layout']['spacing'])
        
        # Header layout for name and value
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(self.config['layout']['spacing'])
        
        # Name label
        self.name_label = QLabel(self.name)
        self.name_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'] + 2,  # Biraz daha büyük font
            QFont.Weight.Bold
        ))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.name_label.setStyleSheet(f"color: {self.color};")
        self.name_label.setMinimumWidth(120)  # Label için minimum genişlik
        header_layout.addWidget(self.name_label)
        
        # Value label
        self.value_label = QLabel(self.value_format.format(self.last_value))
        self.value_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['value_size'] + 2,  # Biraz daha büyük font
            QFont.Weight.Medium
        ))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumWidth(80)  # Değer için minimum genişlik
        header_layout.addWidget(self.value_label)
        
        # Add header layout to main layout
        layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # Hide the text on the progress bar
        self.progress_bar.setFixedHeight(self.config['indicators'].get('bar_height', 15))  # Varsayılan yükseklik
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #f8f8f8;
                border: 1px solid #e1e1e1;
                border-radius: 4px;
                padding: 1px;
                min-height: 20px;  /* İlerleme çubuğu yüksekliği */
            }}
            QProgressBar::chunk {{
                background-color: {self.color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)
    
    def update_value(self, value, min_val=0.0, max_val=1.0):
        """
        Update the displayed value and progress bar.
        
        Args:
            value: The current value of the metric
            min_val: Minimum value for the progress bar range
            max_val: Maximum value for the progress bar range
        """
        # Store the last value
        self.last_value = value
        
        # Update the value label
        if self.name == "PERCLOS":
            display_value = value  # PERCLOS is already a percentage
        else:
            display_value = value
        
        self.value_label.setText(self.value_format.format(display_value))
        
        # Calculate the percentage for the progress bar
        percentage = max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))
        
        # Update the progress bar - değer direkt olarak kullanılır
        # EAR için, değer düştükçe çubuk da küçülür
        # MAR ve PERCLOS için, değer arttıkça çubuk da büyür
        self.progress_bar.setValue(int(percentage))
        
        # Update the color based on thresholds
        self._update_color(value)
    
    def _update_color(self, value):
        """
        Update the color of the indicator based on the current value.
        
        Args:
            value: The current value of the metric
        """
        if self.inverted:
            # For metrics where lower values are worse (e.g., EAR)
            if value <= self.critical_threshold:
                color = "#ff3b30"  # Red for critical
            elif value <= self.warning_threshold:
                color = "#ff9500"  # Orange for warning
            else:
                color = self.color  # Default color for normal
        else:
            # For metrics where higher values are worse (e.g., MAR, PERCLOS)
            if value >= self.critical_threshold:
                color = "#ff3b30"  # Red for critical
            elif value >= self.warning_threshold:
                color = "#ff9500"  # Orange for warning
            else:
                color = self.color  # Default color for normal
        
        # Update the progress bar color
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #f8f8f8;
                border: 1px solid #e1e1e1;
                border-radius: 4px;
                padding: 1px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        
        # Update the name label color
        self.name_label.setStyleSheet(f"color: {color};") 