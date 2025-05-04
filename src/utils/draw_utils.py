#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drawing utilities for the drowsiness detection system.

This module contains functions for drawing visualizations on frames,
such as facial landmarks, gaze vectors, and status information.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

def draw_landmarks(frame: np.ndarray, landmarks: List[List[float]], 
                  color: Tuple[int, int, int] = (0, 255, 0),
                  radius: int = 1) -> np.ndarray:
    """
    Draw facial landmarks on the frame.
    
    Args:
        frame: Input frame
        landmarks: List of [x, y, z] landmark coordinates
        color: Color for the landmarks (BGR)
        radius: Radius of landmark points
        
    Returns:
        np.ndarray: Frame with visualized landmarks
    """
    vis_frame = frame.copy()
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(vis_frame, (x, y), radius, color, -1)
    return vis_frame

def draw_gaze_vector(frame: np.ndarray, 
                    gaze_vector: np.ndarray, 
                    origin: Tuple[int, int],
                    length: float = 50.0, 
                    color: Tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
    """
    Draw a gaze direction vector on the frame.
    
    Args:
        frame: Input frame
        gaze_vector: 3D gaze direction vector [x, y, z]
        origin: Origin point (x, y) for the vector
        length: Length of the vector to draw
        color: Color for the vector (BGR)
        
    Returns:
        np.ndarray: Frame with visualized gaze vector
    """
    vis_frame = frame.copy()
    
    if gaze_vector is None:
        return vis_frame
    
    # Scale the vector to the desired length
    norm = np.linalg.norm(gaze_vector)
    if norm > 0:
        gaze_vector = gaze_vector / norm * length
    
    # Calculate the end point of the vector
    # For 2D visualization, we'll use x and y components
    end_point = (
        int(origin[0] + gaze_vector[0]),
        int(origin[1] + gaze_vector[1])
    )
    
    # Draw the vector
    cv2.arrowedLine(vis_frame, origin, end_point, color, 2)
    
    return vis_frame

def draw_status_box(frame: np.ndarray, 
                   kss_score: int, 
                   alert_status: bool,
                   status_text: str = None) -> np.ndarray:
    """
    Draw a status box with drowsiness information.
    
    Args:
        frame: Input frame
        kss_score: Karolinska Sleepiness Scale score (1-9)
        alert_status: Whether drowsiness alert is active
        status_text: Optional status text to display
        
    Returns:
        np.ndarray: Frame with status box
    """
    vis_frame = frame.copy()
    h, w, _ = vis_frame.shape
    
    # Define status box position and size
    box_height = 60
    box_y = h - box_height
    
    # Create semi-transparent overlay
    overlay = vis_frame.copy()
    
    # Determine color based on alert status
    if alert_status:
        bg_color = (0, 0, 200)  # Red for alert
    elif kss_score > 5:
        bg_color = (0, 140, 255)  # Orange for warning
    else:
        bg_color = (0, 200, 0)  # Green for normal
    
    # Draw background box
    cv2.rectangle(overlay, (0, box_y), (w, h), bg_color, -1)
    
    # Apply transparency
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
    
    # Add status text
    if status_text is None:
        if alert_status:
            status_text = "DROWSINESS ALERT!"
        elif kss_score > 5:
            status_text = f"Warning: Showing signs of sleepiness (KSS: {kss_score})"
        else:
            status_text = f"Normal: Alert (KSS: {kss_score})"
    
    # Position text in middle of box
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = box_y + (box_height + text_size[1]) // 2
    
    # Draw text
    cv2.putText(vis_frame, status_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return vis_frame

def draw_metrics_panel(frame: np.ndarray, 
                      metrics: Dict[str, Any],
                      position: Tuple[int, int] = (10, 30),
                      line_height: int = 30,
                      font_scale: float = 0.7,
                      thickness: int = 2) -> np.ndarray:
    """
    Draw a panel of metrics on the frame.
    
    Args:
        frame: Input frame
        metrics: Dictionary of metrics to display with their values and colors
        position: Starting position (x, y) for the panel
        line_height: Vertical spacing between lines
        font_scale: Font scale for text
        thickness: Thickness of text
        
    Returns:
        np.ndarray: Frame with metrics panel
    """
    vis_frame = frame.copy()
    x, y = position
    
    for label, value_info in metrics.items():
        if isinstance(value_info, tuple):
            value, color = value_info
        else:
            value = value_info
            color = (255, 255, 255)  # Default white
        
        text = f"{label}: {value}"
        cv2.putText(vis_frame, text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += line_height
    
    return vis_frame
