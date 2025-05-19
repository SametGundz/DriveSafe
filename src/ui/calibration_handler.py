#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration handler for the driver drowsiness detection system.

This module provides the CalibrationHandler class that manages 
the EAR calibration process.
"""

import logging
from PyQt6.QtCore import QObject, pyqtSignal

from src.utils import get_ear_calibrator
from src.ui.dialogs import CalibrationDialog


class CalibrationHandler(QObject):
    """
    Handler for EAR calibration operations.
    
    This class manages the eye aspect ratio calibration process
    to adapt the system to individual users.
    """
    
    # Define signals
    calibration_started = pyqtSignal()
    calibration_finished = pyqtSignal(bool)  # Bool parameter indicates success
    
    def __init__(self, camera_handler, config):
        """
        Initialize the calibration handler.
        
        Args:
            camera_handler: Reference to the camera handler
            config: Configuration dictionary
        """
        super().__init__()
        
        # Store references
        self.camera_handler = camera_handler
        self.config = config
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Get EAR calibrator
        self.ear_calibrator = get_ear_calibrator()
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_overlay_active = False
        
        # Dialog reference
        self.calibration_dialog = None
    
    def start_calibration(self):
        """Start the EAR calibration process."""
        # If camera is not capturing, start it first
        if not self.camera_handler.is_capturing:
            self.camera_handler.start_capture()
            # If camera still not running, exit
            if not self.camera_handler.is_capturing:
                return
        
        # Enable calibration mode
        self.is_calibrating = True
        self.calibration_overlay_active = True
        
        # Reset and start calibrator
        self.ear_calibrator.reset()
        self.ear_calibrator.start_calibration()
        
        # Emit signal
        self.calibration_started.emit()
        
        # Show calibration dialog
        self.calibration_dialog = CalibrationDialog(self.ear_calibrator)
        self.calibration_dialog.calibration_finished.connect(self._on_calibration_finished)
        self.calibration_dialog.show()
    
    def _on_calibration_finished(self, success):
        """
        Handle calibration dialog completion.
        
        Args:
            success: Whether calibration was successful
        """
        self.is_calibrating = False
        self.calibration_overlay_active = False
        
        if success:
            # If calibration was successful, set new threshold
            calibration_results = self.ear_calibrator.calibration_results
            
            # Calculate absolute EAR threshold from normalized value 0.3
            normalized_threshold = 0.3
            new_ear_threshold = self.ear_calibrator.get_threshold_ear(normalized_threshold)
            
            # Update configuration
            if 'detection' not in self.config:
                self.config['detection'] = {}
            
            self.config['detection']['ear_threshold'] = new_ear_threshold
            
            # Update thresholds for indicators too
            if 'indicators' in self.config and 'ear' in self.config['indicators']:
                self.config['indicators']['ear']['critical_threshold'] = new_ear_threshold
            
            self.logger.info(f"Calibration completed successfully. New EAR threshold: {new_ear_threshold:.3f}")
        else:
            self.logger.info("Calibration cancelled or failed")
        
        # Emit completion signal
        self.calibration_finished.emit(success)
    
    def is_active(self):
        """Check if calibration is active."""
        return self.is_calibrating and self.calibration_overlay_active
    
    def cleanup(self):
        """Clean up resources."""
        if self.calibration_dialog and self.calibration_dialog.isVisible():
            self.calibration_dialog.close()
        
        self.is_calibrating = False
        self.calibration_overlay_active = False