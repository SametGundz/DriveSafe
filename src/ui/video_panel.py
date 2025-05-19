#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video panel widget for the driver drowsiness detection application.

This module implements a widget for displaying the video feed from the camera.
"""

import cv2
import logging
from PyQt6.QtWidgets import QLabel, QFrame, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QSize

# Get module-specific logger
logger = logging.getLogger(__name__)

class VideoPanel(QLabel):
    """
    Widget for displaying video feed from camera.
    
    This class inherits from QLabel and provides functionality to display 
    and update video frames.
    """
    
    def __init__(self, config, parent=None):
        """
        Initialize the video panel.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        
        # Set initial text and appearance
        self.setText("Kamera görüntüsü burada gösterilecek")
        
        # Kamera görüntüsünü büyütmek için video_frame boyutlarını direkt kullan
        panel_width = self.config['video_frame']['width']
        panel_height = self.config['video_frame']['height']
        
        self.setFixedSize(
            panel_width,
            panel_height
        )
        
        # Boyut politikasını sabit olarak ayarla
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed
        )
        
        self.setFrameShape(QFrame.Shape.Box)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Görüntüyü merkeze al
        self.setStyleSheet("""
            background-color: #f8f8f8;
            border: 1px solid #e1e1e1;
            border-radius: 4px;
        """)
        
        logger.debug(f"VideoPanel initialized with size: {panel_width}x{panel_height}")
    
    def update_frame(self, frame):
        """
        Update the video frame with a new image.
        
        Args:
            frame: OpenCV BGR image
        """
        if frame is None:
            logger.warning("Received None frame in update_frame")
            self.setText("Kamera görüntüsü alınamadı")
            return
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Convert to QPixmap and set to label
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale pixmap to fit label while maintaining aspect ratio
            self.setPixmap(pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            logger.debug(f"Frame updated: {w}x{h}")
        except Exception as e:
            logger.error(f"Error updating video frame: {str(e)}")
            self.setText(f"Görüntü işleme hatası: {str(e)}")
    
    def resizeEvent(self, event):
        """
        Handle resize event to resize the image if present.
        
        Args:
            event: Resize event
        """
        super().resizeEvent(event)
        
        # If we have a pixmap, rescale it when the widget is resized
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull():
            self.setPixmap(pixmap.scaled(
                event.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            logger.debug(f"VideoPanel resized to: {event.size().width()}x{event.size().height()}") 