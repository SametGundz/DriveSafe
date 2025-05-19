#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog handler for the driver drowsiness detection system.

This module provides the DialogHandler class that manages dialog windows
and handles their interactions.
"""

import logging
from typing import Optional, Dict, Any, Union
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtWidgets import QDialog

from src.ui.dialogs import SettingsDialog, AboutDialog, CalibrationDialog


class DialogHandler(QObject):
    """
    Handler for dialog operations.
    
    This class manages the creation, display, and lifecycles of various dialog windows
    including settings, about, and calibration dialogs.
    """
    
    # Signals
    settings_applied = pyqtSignal(dict)  # Signal emitted when settings are applied
    calibration_completed = pyqtSignal(bool, dict)  # Signal emitted when calibration completes (success, results)
    
    def __init__(self, config, parent=None):
        """
        Initialize the dialog handler.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Store references
        self.config = config
        self.parent = parent
        
        # Track active dialogs
        self._active_dialogs = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def show_settings(self, modal: bool = True) -> Optional[SettingsDialog]:
        """
        Show settings dialog.
        
        Args:
            modal: Whether the dialog should be modal (blocking)
            
        Returns:
            Optional[SettingsDialog]: The dialog instance or None if already shown
        """
        # Check if dialog is already open
        if 'settings' in self._active_dialogs and self._active_dialogs['settings'].isVisible():
            # Bring to front if already open
            self._active_dialogs['settings'].raise_()
            self._active_dialogs['settings'].activateWindow()
            return None
        
        # Create dialog
        dialog = SettingsDialog(self.config, self.parent)
        
        # Connect signals
        dialog.settings_changed.connect(self._on_settings_changed)
        
        # Set up dialog behavior
        if modal:
            # Run as modal dialog (blocking)
            result = dialog.exec()
            self.logger.info(f"Settings dialog closed with result: {result}")
            return None
        else:
            # Show as non-modal dialog
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.finished.connect(lambda: self._cleanup_dialog('settings'))
            
            # Track active dialog
            self._active_dialogs['settings'] = dialog
            
            # Show dialog
            dialog.show()
            self.logger.info("Settings dialog shown (non-modal)")
            return dialog
    
    def show_about(self, modal: bool = True) -> Optional[AboutDialog]:
        """
        Show about dialog.
        
        Args:
            modal: Whether the dialog should be modal (blocking)
            
        Returns:
            Optional[AboutDialog]: The dialog instance or None if already shown
        """
        # Check if dialog is already open
        if 'about' in self._active_dialogs and self._active_dialogs['about'].isVisible():
            # Bring to front if already open
            self._active_dialogs['about'].raise_()
            self._active_dialogs['about'].activateWindow()
            return None
        
        # Create dialog
        dialog = AboutDialog(self.parent)
        
        # Set up dialog behavior
        if modal:
            # Run as modal dialog (blocking)
            result = dialog.exec()
            self.logger.info(f"About dialog closed with result: {result}")
            return None
        else:
            # Show as non-modal dialog
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.finished.connect(lambda: self._cleanup_dialog('about'))
            
            # Track active dialog
            self._active_dialogs['about'] = dialog
            
            # Show dialog
            dialog.show()
            self.logger.info("About dialog shown (non-modal)")
            return dialog
    
    def show_calibration_dialog(self, ear_calibrator, modal: bool = False) -> Optional[CalibrationDialog]:
        """
        Show calibration dialog.
        
        Args:
            ear_calibrator: The EAR calibrator instance
            modal: Whether the dialog should be modal (blocking)
            
        Returns:
            Optional[CalibrationDialog]: The dialog instance or None if already shown
        """
        # Check if dialog is already open
        if 'calibration' in self._active_dialogs and self._active_dialogs['calibration'].isVisible():
            # Bring to front if already open
            self._active_dialogs['calibration'].raise_()
            self._active_dialogs['calibration'].activateWindow()
            return None
        
        # Create dialog
        dialog = CalibrationDialog(ear_calibrator, self.parent)
        
        # Connect signals
        dialog.calibration_finished.connect(self._on_calibration_finished)
        
        # Set up dialog behavior
        if modal:
            # Run as modal dialog (blocking)
            result = dialog.exec()
            self.logger.info(f"Calibration dialog closed with result: {result}")
            return None
        else:
            # Show as non-modal dialog
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.finished.connect(lambda: self._cleanup_dialog('calibration'))
            
            # Track active dialog
            self._active_dialogs['calibration'] = dialog
            
            # Show dialog
            dialog.show()
            self.logger.info("Calibration dialog shown (non-modal)")
            return dialog
    
    def show_help(self, modal: bool = True) -> Optional[QDialog]:
        """
        Show help dialog.
        
        Args:
            modal: Whether the dialog should be modal (blocking)
            
        Returns:
            Optional[QDialog]: The dialog instance or None if already shown
        """
        # Check if dialog is already open
        if 'help' in self._active_dialogs and self._active_dialogs['help'].isVisible():
            # Bring to front if already open
            self._active_dialogs['help'].raise_()
            self._active_dialogs['help'].activateWindow()
            return None
        
        # For now, just use a simple QDialog for help
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextBrowser
        
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Yardım")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Create help content
        help_text = """
        <h2>Sürücü Uykululuk Tespit Sistemi - Yardım</h2>
        
        <h3>Başlarken</h3>
        <ol>
            <li>Uygulamayı başlatmak için 'Başlat' düğmesine tıklayın.</li>
            <li>Kamera görüntünüzün doğru göründüğünden emin olun.</li>
            <li>Daha doğru sonuçlar için 'Kalibre Et' düğmesini kullanarak EAR değerlerini kalibre edin.</li>
        </ol>
        
        <h3>Göstergeler</h3>
        <ul>
            <li><b>EAR (Eye Aspect Ratio):</b> Göz açıklık oranı</li>
            <li><b>MAR (Mouth Aspect Ratio):</b> Ağız açıklık oranı</li>
            <li><b>PERCLOS:</b> Belli bir zaman diliminde gözlerin kapalı olduğu sürenin yüzdesi</li>
        </ul>
        
        <h3>Özellikler</h3>
        <ul>
            <li><b>3D Model:</b> Baş pozisyonunu görselleştirmek için bir 3D model kullanılır</li>
            <li><b>Metrikler:</b> EAR, MAR ve PERCLOS değerleri gerçek zamanlı olarak gösterilir</li>
            <li><b>Grafikler:</b> Metriklerin zaman içindeki değişimini gösterir</li>
        </ul>
        """
        
        help_browser = QTextBrowser()
        help_browser.setHtml(help_text)
        layout.addWidget(help_browser)
        
        # Add close button
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        # Set up dialog behavior
        if modal:
            # Run as modal dialog (blocking)
            result = dialog.exec()
            self.logger.info(f"Help dialog closed with result: {result}")
            return None
        else:
            # Show as non-modal dialog
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.finished.connect(lambda: self._cleanup_dialog('help'))
            
            # Track active dialog
            self._active_dialogs['help'] = dialog
            
            # Show dialog
            dialog.show()
            self.logger.info("Help dialog shown (non-modal)")
            return dialog
    
    def _on_settings_changed(self, new_settings: Dict[str, Any]):
        """
        Handle settings changes.
        
        Args:
            new_settings: Dictionary of changed settings
        """
        # Update config
        self._update_config(new_settings)
        
        # Emit signal
        self.settings_applied.emit(new_settings)
        self.logger.debug(f"Settings applied: {new_settings}")
    
    def _on_calibration_finished(self, success: bool, results: Optional[Dict[str, Any]] = None):
        """
        Handle calibration completion.
        
        Args:
            success: Whether calibration was successful
            results: Calibration results dictionary
        """
        # Emit signal
        self.calibration_completed.emit(success, results or {})
        self.logger.info(f"Calibration completed: success={success}")
    
    def _update_config(self, new_settings: Dict[str, Any]):
        """
        Update configuration with new settings.
        
        Args:
            new_settings: Dictionary of new settings values
        """
        # Update config
        for section, values in new_settings.items():
            if section not in self.config:
                self.config[section] = {}
            
            if isinstance(values, dict):
                # Update section values
                for key, value in values.items():
                    self.config[section][key] = value
            else:
                # Direct value
                self.config[section] = values
    
    def _cleanup_dialog(self, dialog_key: str):
        """
        Clean up dialog reference when closed.
        
        Args:
            dialog_key: Key of the dialog to clean up
        """
        if dialog_key in self._active_dialogs:
            self._active_dialogs.pop(dialog_key)
            self.logger.debug(f"Dialog '{dialog_key}' cleaned up")
    
    def close_all_dialogs(self):
        """Close all active dialogs."""
        for key, dialog in list(self._active_dialogs.items()):
            if dialog and dialog.isVisible():
                dialog.close()
                self.logger.debug(f"Dialog '{key}' closed")
        
        # Clear dialog list
        self._active_dialogs.clear()
    
    def is_dialog_active(self, dialog_type: str) -> bool:
        """
        Check if a specific dialog is active.
        
        Args:
            dialog_type: Type of dialog to check
            
        Returns:
            bool: True if dialog is active, False otherwise
        """
        return dialog_type in self._active_dialogs and self._active_dialogs[dialog_type].isVisible()
    
    def get_active_dialogs(self) -> Dict[str, QDialog]:
        """
        Get all active dialogs.
        
        Returns:
            Dict[str, QDialog]: Dictionary of active dialogs (key: dialog type, value: dialog instance)
        """
        return self._active_dialogs.copy()