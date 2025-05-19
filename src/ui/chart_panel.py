#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chart panel for the driver drowsiness detection application.

This module implements the chart panel for displaying EAR, MAR, and PERCLOS
time series data.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QMargins
from PyQt6.QtGui import QFont, QPen, QColor, QPainter
from PyQt6.QtCharts import QChart, QLineSeries, QValueAxis, QChartView


class ChartPanel(QWidget):
    """
    Panel for displaying time series charts.
    
    This class implements a chart panel with time series data for
    EAR, MAR, and PERCLOS.
    """
    
    def __init__(self, config, parent=None):
        """
        Initialize the chart panel.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        
        # Sabit yükseklik ayarla, genişlik olarak horizontal sizePolicy kullan
        self.setFixedHeight(250)  # Sabit yükseklik
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,  # Genişlik için expanding
            QSizePolicy.Policy.Fixed       # Yükseklik için fixed
        )
        
        # Setup layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create chart
        self._create_chart()
        layout.addWidget(self.chart_view)
        
        # Initialize data counter
        self.time_counter = 0.0
    
    def _create_chart(self):
        """Create the time series chart for EAR, MAR, and PERCLOS data."""
        # Create chart
        chart = QChart()
        chart.setTitle("Metrikler Zaman Grafiği")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
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
        self.ear_series.setPen(QPen(QColor("#007aff"), self.config['chart']['line_width'], Qt.PenStyle.SolidLine))
        self.ear_series.setUseOpenGL(True)
        
        self.mar_series = QLineSeries()
        self.mar_series.setName("MAR")
        self.mar_series.setPen(QPen(QColor("#5ac8fa"), self.config['chart']['line_width'], Qt.PenStyle.SolidLine))
        self.mar_series.setUseOpenGL(True)
        
        self.perclos_series = QLineSeries()
        self.perclos_series.setName("PERCLOS")
        self.perclos_series.setPen(QPen(QColor("#ff9500"), self.config['chart']['line_width'], Qt.PenStyle.SolidLine))
        self.perclos_series.setUseOpenGL(True)
        
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
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.chart_view.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.chart_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.chart_view.setViewportUpdateMode(QChartView.ViewportUpdateMode.FullViewportUpdate)
        self.chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
        self.chart_view.setStyleSheet("""
            background-color: white;
            border: 1px solid #e1e1e1;
            border-radius: 8px;
        """)
    
    def update_chart_data(self, ear_value, mar_value, perclos_value, time_increment):
        """
        Update the chart with new data points.
        
        Args:
            ear_value: Current EAR value
            mar_value: Current MAR value
            perclos_value: Current PERCLOS value
            time_increment: Time increment in seconds
        """
        # Update time counter
        self.time_counter += time_increment
        
        # Add new data points
        self.ear_series.append(self.time_counter, ear_value)
        self.mar_series.append(self.time_counter, mar_value)
        self.perclos_series.append(self.time_counter, perclos_value)
        
        # Adjust X axis range if needed
        if self.time_counter > self.time_axis.max():
            self.time_axis.setRange(
                self.time_counter - self.config['chart']['history_duration'],
                self.time_counter
            )
            
    def reset(self):
        """Reset the chart, clearing all data series and resetting time counter."""
        # Clear all series data
        self.ear_series.clear()
        self.mar_series.clear()
        self.perclos_series.clear()
        
        # Reset time counter
        self.time_counter = 0.0
        
        # Reset time axis
        self.time_axis.setRange(0, self.config['chart']['history_duration']) 