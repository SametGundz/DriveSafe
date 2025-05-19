#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chart handler for the driver drowsiness detection system.

This module provides the ChartHandler class that manages chart data.
"""

import logging
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from src.ui.graph_utils import create_chart_data_from_series


class ChartHandler(QObject):
    """
    Handler for chart operations and data management.
    
    This class manages chart data updates.
    """
    
    def __init__(self, chart_panel, charts_info_label, config):
        """
        Initialize the chart handler.
        
        Args:
            chart_panel: Reference to the chart panel
            charts_info_label: Reference to the charts info label
            config: Configuration dictionary
        """
        super().__init__()
        
        # Store references
        self.chart_panel = chart_panel
        self.charts_info_label = charts_info_label
        self.config = config
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Update interval
        self.update_interval_sec = 0.05  # 50ms update for chart
        
        # Initial info label
        if self.charts_info_label:
            self.charts_info_label.setText("Grafikler hazır...")
    
    def update_chart_data(self, ear_value, mar_value, perclos_value):
        """
        Update chart data with new values.
        
        Args:
            ear_value: Eye aspect ratio value
            mar_value: Mouth aspect ratio value
            perclos_value: PERCLOS value
        """
        # Update main chart panel
        if self.chart_panel.isVisible():
            self.chart_panel.update_chart_data(
                ear_value, 
                mar_value, 
                perclos_value, 
                self.update_interval_sec
            )
    
    def update_charts(self, data_dict):
        """
        Update charts from the processed frame data dictionary.
        
        Args:
            data_dict: Dictionary containing metrics from processed frame
        """
        # Extract values from data dictionary
        ear = data_dict.get('ear', 0.0)
        mar = data_dict.get('mar', 0.0)
        perclos = data_dict.get('perclos', 0.0)
        
        # Update chart data
        self.update_chart_data(ear, mar, perclos)
        
        # Update info label with additional data if available
        if self.charts_info_label:
            face_detected = data_dict.get('face_detected', False)
            status = "Yüz Algılandı" if face_detected else "Yüz Bulunamadı"
            self.charts_info_label.setText(f"Durum: {status} | EAR: {ear:.3f} | MAR: {mar:.3f} | PERCLOS: {perclos:.1f}%")
    
    def cleanup(self):
        """Clean up resources."""
        pass