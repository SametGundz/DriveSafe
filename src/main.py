#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the driver drowsiness detection application.

This script initializes and runs the driver drowsiness detection GUI.
"""

import sys
import os
import logging
from pathlib import Path
import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the main window
from src.ui.main_window import DriverDrowsinessMainWindow
from PyQt6.QtWidgets import QApplication


def setup_logging():
    """
    Set up logging configuration for the application.
    
    This function creates log directory if it doesn't exist and configures
    the logging system based on settings in the config file.
    """
    # Create logs directory if it doesn't exist
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Get logging settings
    log_config = config.get('logging', {})
    console_level = getattr(logging, log_config.get('console_level', 'INFO').upper())
    file_level = getattr(logging, log_config.get('file_level', 'DEBUG').upper())
    enable_console = log_config.get('enable_console', True)
    enable_file = log_config.get('enable_file', True)
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger to lowest level
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        log_file = log_dir / "driver_drowsiness.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def main():
    """
    Main function to start the application.
    
    This function sets up logging, initializes the application,
    and starts the main event loop.
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Driver Drowsiness Detection Application")
    
    # Initialize Qt application
    app = QApplication(sys.argv)
    
    try:
        # Create and show the main window
        window = DriverDrowsinessMainWindow()
        window.show()
        
        # Start the event loop
        logger.info("Application started successfully")
        exit_code = app.exec()
        logger.info(f"Application exited with code {exit_code}")
        return exit_code
    
    except Exception as e:
        logger.exception(f"Error in main application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
