#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GazeZoneDetector test script.
This script demonstrates how to use the GazeZoneDetector class with the existing GazeDetector.
"""

import os
import sys
import time
import cv2
import numpy as np
import logging
import argparse
from typing import Tuple

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.mediapipe_helper import MediaPipeHelper
from src.utils.gaze_detector import get_gaze_detector
from src.detection.gaze_zone_detector import get_gaze_zone_detector

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GazeZoneTest")

ZONE_COLORS = [
    (255, 0, 255),  # 0: Road Center - Magenta
    (0, 255, 255),  # 1: Dashboard - Yellow
    (0, 128, 255),  # 2: Left Side - Brown
    (255, 0, 0),    # 3: Right Side - Blue
    (0, 0, 255),    # 4: Rear Mirror - Red
]

def rad2deg(rad: float) -> float:
    """Convert radians to degrees"""
    return rad * 180.0 / np.pi

def draw_zone_info(frame: np.ndarray, zone_id: int, zone_name: str) -> np.ndarray:
    """Draw zone information on the frame"""
    color = ZONE_COLORS[zone_id] if zone_id is not None and 0 <= zone_id < len(ZONE_COLORS) else (200, 200, 200)
    
    # Draw zone id and name
    cv2.putText(frame, f"Zone: {zone_id} - {zone_name}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test GazeZoneDetector with webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--show_landmarks", action="store_true", help="Show facial landmarks")
    parser.add_argument("--history", type=int, default=10, help="History size for gaze zone detection")
    parser.add_argument("--stability", type=float, default=0.6, help="Stability threshold for gaze zone detection")
    args = parser.parse_args()
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Could not open camera {args.camera}")
        return
    
    # Initialize MediaPipe Helper with GazeDetector
    gaze_detector = get_gaze_detector()
    mp_helper = MediaPipeHelper(gaze_detector=gaze_detector)
    
    # Initialize GazeZoneDetector
    gaze_zone_detector = get_gaze_zone_detector(
        history_size=args.history,
        stability_threshold=args.stability
    )
    
    logger.info("Press 'q' to quit, 'r' to reset zone statistics")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break
        
        # Process the frame with MediaPipe
        results, landmarks = mp_helper.process_frame(frame)
        
        if results and landmarks:
            # Draw face mesh if requested
            if args.show_landmarks:
                mp_helper.draw_landmarks(frame, landmarks)
            
            # Process gaze with the GazeDetector
            vis_frame, normalized_face = gaze_detector.visualize_gaze(frame, landmarks)
            
            # Get the gaze vector (pitch, yaw) in radians
            gaze_vector = gaze_detector._last_gaze_vector
            
            if gaze_vector is not None:
                # Convert to degrees for the GazeZoneDetector
                pitch_deg = rad2deg(gaze_vector[0])
                yaw_deg = rad2deg(gaze_vector[1])
                
                # Update the GazeZoneDetector with the current gaze angles
                current_zone = gaze_zone_detector.update(pitch_deg, yaw_deg)
                
                # Get zone name
                zone_name = gaze_zone_detector.get_zone_name(current_zone)
                
                # Draw zone information
                if current_zone is not None:
                    vis_frame = draw_zone_info(vis_frame, current_zone, zone_name)
            
            # Show zone durations
            zone_stats = gaze_zone_detector.get_zone_statistics()
            y_pos = 120
            for zone_id, duration in sorted(zone_stats.items()):
                if duration > 0:
                    name = gaze_zone_detector.get_zone_name(zone_id)
                    color = ZONE_COLORS[zone_id] if 0 <= zone_id < len(ZONE_COLORS) else (200, 200, 200)
                    cv2.putText(vis_frame, f"{name}: {duration:.1f}s", (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_pos += 20
            
            # Display the frame
            cv2.imshow("Gaze Zone Detection", vis_frame)
            
            # Show normalized face if available
            if normalized_face is not None:
                cv2.imshow("Normalized Face", normalized_face)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            logger.info("Resetting zone statistics")
            gaze_zone_detector.reset()
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    mp_helper.release()
    gaze_detector.release()

if __name__ == "__main__":
    main() 