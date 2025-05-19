#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog boxes for the driver drowsiness detection application.

This module implements various dialog boxes used in the application,
such as Settings dialog and About dialog.
"""

from PyQt6.QtWidgets import (
    QDialog, QMessageBox, QVBoxLayout, QFormLayout, QLabel, 
    QDoubleSpinBox, QComboBox, QDialogButtonBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


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