#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom widgets for the driver drowsiness detection GUI.

This module contains custom widget classes that are used in the 
driver drowsiness detection application GUI.
"""

from PyQt6.QtWidgets import (
    QWidget, QLabel, QProgressBar, QVBoxLayout, QHBoxLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont


class IndicatorWidget(QWidget):
    """
    A widget for displaying a metric with a label, value, progress bar, and status icon.
    
    This widget is used for displaying metrics like EAR, MAR, and PERCLOS.
    """
    
    def __init__(self, label_text, config):
        """
        Initialize the indicator widget.
        
        Args:
            label_text: Text label for the indicator
            config: Configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self.indicator_config = config['indicators'].get(label_text.lower(), {})
        self.label_text = label_text  # Store the label text for later use
        self.last_value = 0.0
        self.normalized_value = 0.0  # Normalize edilmiş değeri saklamak için
        
        # Set up the layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Create label
        label = QLabel(label_text)
        label.setFixedWidth(
            self.config['indicators'].get('label_width', 80)
        )
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        layout.addWidget(label)
        
        # Create value display
        self.value_label = QLabel("0.00")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setFixedWidth(50)
        self.value_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['value_size'],
            QFont.Weight.Bold
        ))
        layout.addWidget(self.value_label)
        
        # Create normalized value label for EAR
        self.normalized_label = QLabel("")
        self.normalized_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.normalized_label.setFixedWidth(50)
        self.normalized_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['value_size'] - 1,
            QFont.Weight.Normal
        ))
        # Sadece EAR göstergesi için normalize edilmiş değeri göster
        if label_text.lower() == "ear":
            layout.addWidget(self.normalized_label)
        else:
            self.normalized_label.setVisible(False)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(
            self.config['indicators'].get('progress_bar_height', 15)
        )
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e1e1e1;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #34c759;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Create status icon
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(16, 16)
        self.status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_icon)
        
        # Set initial state
        self.update_value(0.0)
    
    def update_value(self, value, min_val=0.0, max_val=1.0):
        """
        Update the indicator with a new value.
        
        Args:
            value: The current value to display
            min_val: Minimum possible value (for scaling)
            max_val: Maximum possible value (for scaling)
        """
        # Store the current value
        self.last_value = value
        
        # Update the value label with appropriate formatting
        if self.label_text.lower() == "perclos":
            # Use 1 decimal place for PERCLOS (percentage)
            self.value_label.setText(f"{value:.1f}%")
        else:
            # Use 3 decimal places for EAR and MAR
            self.value_label.setText(f"{value:.3f}")
        
        # Normalize edilmiş değeri göster (sadece EAR için) - Değer varsa
        if self.label_text.lower() == "ear" and hasattr(self, 'normalized_value') and self.normalized_value is not None:
            self.normalized_label.setText(f"[{self.normalized_value:.2f}]")
            self.normalized_label.setStyleSheet("color: #007aff;")  # Apple blue
        else:
            if self.label_text.lower() == "ear":
                self.normalized_label.setText("")
        
        # Map the value to a percentage (0-100) for the progress bar
        percentage = int(((value - min_val) / (max_val - min_val)) * 100)
        percentage = max(0, min(100, percentage))  # Clamp to 0-100 range
        self.progress_bar.setValue(percentage)
        
        # Update progress bar color based on thresholds
        critical_threshold = self.indicator_config.get('critical_threshold', 0.2)
        warning_threshold = self.indicator_config.get('warning_threshold', 0.25)
        
        # Determine color based on thresholds
        if self.label_text.lower() == "ear":
            # For EAR, lower values are critical
            if value <= critical_threshold:
                color = "#ff3b30"  # Apple red
                self.value_label.setStyleSheet("color: #ff3b30; font-weight: bold;")
            elif value <= warning_threshold:
                color = "#ff9500"  # Apple orange
                self.value_label.setStyleSheet("color: #ff9500; font-weight: bold;")
            else:
                color = "#34c759"  # Apple green
                self.value_label.setStyleSheet("color: #34c759; font-weight: bold;")
        elif self.label_text.lower() == "mar":
            # For MAR, higher values are critical
            if value >= critical_threshold:
                color = "#ff3b30"  # Apple red
                self.value_label.setStyleSheet("color: #ff3b30; font-weight: bold;")
            elif value >= warning_threshold:
                color = "#ff9500"  # Apple orange
                self.value_label.setStyleSheet("color: #ff9500; font-weight: bold;")
            else:
                color = "#34c759"  # Apple green
                self.value_label.setStyleSheet("color: #34c759; font-weight: bold;")
        elif self.label_text.lower() == "perclos":
            # For PERCLOS, higher values are critical
            if value >= critical_threshold:
                color = "#ff3b30"  # Apple red
                self.value_label.setStyleSheet("color: #ff3b30; font-weight: bold;")
            elif value >= warning_threshold:
                color = "#ff9500"  # Apple orange
                self.value_label.setStyleSheet("color: #ff9500; font-weight: bold;")
            else:
                color = "#34c759"  # Apple green
                self.value_label.setStyleSheet("color: #34c759; font-weight: bold;")
        else:
            # Default behavior (higher values are better)
            if value >= critical_threshold:
                color = "#34c759"  # Apple green
                self.value_label.setStyleSheet("color: #34c759; font-weight: bold;")
            elif value >= warning_threshold:
                color = "#ff9500"  # Apple orange
                self.value_label.setStyleSheet("color: #ff9500; font-weight: bold;")
            else:
                color = "#ff3b30"  # Apple red
                self.value_label.setStyleSheet("color: #ff3b30; font-weight: bold;")
        
        # Update the progress bar style
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: #f0f0f0;
                border: none;
                border-radius: 4px;
                text-align: center;
                margin-top: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 4px;
            }}
            """
        )
    
    def sizeHint(self):
        """Suggested size for the widget."""
        return QSize(200, 75) 