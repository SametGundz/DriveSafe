#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging utility for the drowsiness detection system.

This module provides a configurable logger for the application,
with options for console and file output, as well as log level control.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any

class DrowsinessLogger:
    """
    A class that provides logging functionality for the drowsiness detection system.
    
    Attributes:
        logger: The logging.Logger instance
        log_file: Path to the log file
        console_level: Logging level for console output
        file_level: Logging level for file output
    """
    
    # Log levels mapping
    LOG_LEVELS = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    def __init__(self, 
                name: str = 'drowsiness_detection',
                log_dir: Optional[str] = None,
                console_level: str = 'info',
                file_level: str = 'debug',
                enable_console: bool = True,
                enable_file: bool = True,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logger with configuration.
        
        Args:
            name: Logger name
            log_dir: Directory for log files (default: logs/ in project root)
            console_level: Logging level for console output
            file_level: Logging level for file output
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
            config: Optional configuration dictionary to override settings
        """
        # Initialize the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all logs
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Override settings from config if provided
        if config and 'logging' in config:
            log_config = config['logging']
            console_level = log_config.get('console_level', console_level)
            file_level = log_config.get('file_level', file_level)
            enable_console = log_config.get('enable_console', enable_console)
            enable_file = log_config.get('enable_file', enable_file)
            log_dir = log_config.get('log_dir', log_dir)
        
        # Set up log directory
        if log_dir is None:
            # Default to 'logs' in the project root
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up log file path with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        # Set log levels
        self.console_level = self._get_log_level(console_level)
        self.file_level = self._get_log_level(file_level)
        
        # Add console handler if enabled
        if enable_console:
            self._add_console_handler()
        
        # Add file handler if enabled
        if enable_file:
            self._add_file_handler()
            
        self.logger.info(f"Logger initialized: {name}")
    
    def _get_log_level(self, level: str) -> int:
        """
        Convert string log level to logging module level.
        
        Args:
            level: String log level (debug, info, warning, error, critical)
            
        Returns:
            int: Logging module level constant
        """
        return self.LOG_LEVELS.get(level.lower(), logging.INFO)
    
    def _add_console_handler(self):
        """Add a console handler to the logger."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add a file handler to the logger."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.file_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """
        Get the logger instance.
        
        Returns:
            logging.Logger: Logger instance
        """
        return self.logger
    
    def log_drowsiness_data(self, 
                           ear: Optional[float], 
                           mar: Optional[float], 
                           perclos: Optional[float], 
                           kss_score: Optional[int],
                           alert_status: bool):
        """
        Log drowsiness detection data.
        
        Args:
            ear: Eye Aspect Ratio value
            mar: Mouth Aspect Ratio value
            perclos: PERCLOS value
            kss_score: Karolinska Sleepiness Scale score
            alert_status: Whether drowsiness alert is active
        """
        # Create a structured log message
        data = {
            'time': time.time(),
            'ear': ear,
            'mar': mar,
            'perclos': perclos,
            'kss_score': kss_score,
            'alert': alert_status
        }
        
        # Log with appropriate level based on alert status
        if alert_status:
            self.logger.warning(f"DROWSINESS ALERT! Data: {data}")
        else:
            self.logger.debug(f"Drowsiness data: {data}")
    
    def log_system_status(self, fps: float, frame_count: int, status: str):
        """
        Log system status information.
        
        Args:
            fps: Current frames per second
            frame_count: Total frames processed
            status: System status description
        """
        self.logger.info(f"System status: FPS={fps:.2f}, Frames={frame_count}, Status={status}")
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """
        Log an error.
        
        Args:
            error_message: Error message
            exception: Optional exception object
        """
        if exception:
            self.logger.error(f"{error_message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_message)
    
    def log_startup(self, config: Dict[str, Any]):
        """
        Log application startup with configuration.
        
        Args:
            config: Application configuration
        """
        self.logger.info(f"Application starting with configuration: {config}")
    
    def log_shutdown(self):
        """Log application shutdown."""
        self.logger.info("Application shutting down")

# Create a default logger instance for easy import
default_logger = DrowsinessLogger().get_logger()
