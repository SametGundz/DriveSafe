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
    Custom widget that combines a label and a progress bar for displaying
    metrics with thresholds (EAR, MAR, PERCLOS).
    
    Attributes:
        title_label: Label displaying the indicator name
        value_label: Label displaying the current value
        progress_bar: Progress bar visualizing the value
        last_value: Last value set for this indicator
    """
    
    def __init__(self, title, config, parent=None):
        """
        Initialize the indicator widget.
        
        Args:
            title: Title of the indicator (e.g., "EAR", "MAR", "PERCLOS")
            config: Configuration dictionary containing threshold values and colors
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Store configuration
        self.title = title
        self.config = config
        self.last_value = 0.0
        
        # Determine which indicator configuration to use
        if title.lower() == "ear":
            self.indicator_config = config['indicators']['ear']
        elif title.lower() == "mar":
            self.indicator_config = config['indicators']['mar']
        elif title.lower() == "perclos":
            self.indicator_config = config['indicators']['perclos']
        else:
            # Default to EAR configuration if title is not recognized
            self.indicator_config = config['indicators']['ear']
        
        # UI setup
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget's UI components."""
        # Set the layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding'],
            self.config['layout']['padding']
        )
        layout.setSpacing(self.config['layout']['spacing'])
        
        # Create header with title and value
        header_layout = QHBoxLayout()
        
        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size'],
            QFont.Weight.Medium
        ))
        # Use .get() method with a default value of 80 if 'label_width' doesn't exist
        label_width = self.config.get('indicators', {}).get('label_width', 80)
        self.title_label.setMinimumWidth(label_width)
        header_layout.addWidget(self.title_label)
        
        # Value label (right-aligned)
        self.value_label = QLabel("0.00")
        self.value_label.setFont(QFont(
            self.config['fonts']['family'],
            self.config['fonts']['label_size']
        ))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.value_label)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        # Use default value for progress_bar_height if it doesn't exist
        progress_bar_height = self.config.get('indicators', {}).get('progress_bar_height', 15)
        self.progress_bar.setFixedHeight(progress_bar_height)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        
        # Apply stylesheet for a modern, minimalist progress bar
        # Use default color if normal_color doesn't exist
        normal_color = self.indicator_config.get('normal_color', "#34c759")  # Default to green
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
                background-color: {normal_color};
                border-radius: 4px;
            }}
            """
        )
        
        layout.addWidget(self.progress_bar)
        
        # Add subtle border around the widget
        self.setStyleSheet("""
            IndicatorWidget {
                background-color: white;
                border: 1px solid #e1e1e1;
                border-radius: 8px;
                padding: 4px;
            }
        """)
    
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
        if self.title.lower() == "perclos":
            # Use 1 decimal place for PERCLOS (percentage)
            self.value_label.setText(f"{value:.1f}%")
        else:
            # Use 3 decimal places for EAR and MAR
            self.value_label.setText(f"{value:.3f}")
        
        # Map the value to a percentage (0-100) for the progress bar
        percentage = int(((value - min_val) / (max_val - min_val)) * 100)
        percentage = max(0, min(100, percentage))  # Clamp to 0-100 range
        self.progress_bar.setValue(percentage)
        
        # Update progress bar color based on thresholds
        critical_threshold = self.indicator_config['critical_threshold']
        warning_threshold = self.indicator_config['warning_threshold']
        
        # Determine color based on thresholds
        if self.title.lower() == "ear":
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
        elif self.title.lower() == "mar":
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
        elif self.title.lower() == "perclos":
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