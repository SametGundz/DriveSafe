#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gaze Estimation Demo

This script demonstrates the gaze estimation functionality using the ETH-XGaze model.
It can process input from either a webcam or a video file.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import mediapipe as mp
import time
import logging

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the GazeEstimator class
from src.detection.gaze_estimator import GazeEstimator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gaze_estimation_demo')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gaze Estimation Demo')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--input', type=str, default='webcam',
                        help='Input source (webcam or path to video file)')
    parser.add_argument('--show_normalized', action='store_true', 
                        help='Show the normalized face image')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the ETH-XGaze model file (optional)')
    return parser.parse_args()

def process_landmarks(landmarks, image_width, image_height):
    """
    Convert MediaPipe landmarks to the format expected by GazeEstimator.
    
    Args:
        landmarks: MediaPipe face mesh landmarks
        image_width: Width of the input image
        image_height: Height of the input image
        
    Returns:
        Dictionary containing facial landmarks
    """
    # Convert landmarks to pixel coordinates
    points = []
    for lm in landmarks.landmark:
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        z = lm.z  # This is a relative depth value
        points.append([x, y, z])
    
    # Extract specific landmarks for eyes
    left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    # Make sure we have all the required landmarks for eye tracking and gaze estimation
    # This is to verify necessary landmarks are present and valid
    required_landmarks = [33, 133, 362, 263, 4, 5]
    for idx in required_landmarks:
        if idx >= len(points):
            logger.warning(f"Required landmark index {idx} is out of range (max: {len(points)-1})")
            # In this case we won't be able to do gaze estimation
    
    # Extract eye landmarks
    left_eye = []
    for idx in left_eye_indices:
        if idx < len(points):
            left_eye.append(points[idx][:2])  # Keep only x, y
    
    right_eye = []
    for idx in right_eye_indices:
        if idx < len(points):
            right_eye.append(points[idx][:2])  # Keep only x, y
    
    left_eye = np.array(left_eye) if left_eye else np.array([[0, 0]])
    right_eye = np.array(right_eye) if right_eye else np.array([[0, 0]])
    
    # Return landmarks dictionary
    return {
        'all_landmarks': points,
        'left_eye': left_eye,
        'right_eye': right_eye
    }

def main():
    """Main function to run the gaze estimation demo."""
    args = parse_args()
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize GazeEstimator
    try:
        gaze_estimator = GazeEstimator(model_path=args.model_path)
        logger.info("Gaze estimator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize gaze estimator: {e}")
        return
    
    # Open video source
    if args.input.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
        logger.info("Using webcam as input")
    else:
        cap = cv2.VideoCapture(args.input)
        logger.info(f"Using video file: {args.input}")
    
    if not cap.isOpened():
        logger.error("Could not open video source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video properties: {width}x{height}, {fps} FPS")
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream")
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        results = face_mesh.process(rgb_frame)
        
        # Process frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to the format expected by GazeEstimator
                landmarks_dict = process_landmarks(face_landmarks, width, height)
                
                # Estimate gaze
                success, gaze_angles, face_img = gaze_estimator.detect(frame, landmarks_dict)
                
                if success:
                    # Draw gaze vector
                    frame = gaze_estimator.draw_gaze_vector(frame, landmarks_dict, gaze_angles, length=100)
                    
                    # Display gaze angles
                    yaw, pitch = gaze_angles
                    cv2.putText(frame, f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show normalized face if requested
                if args.show_normalized and face_img is not None:
                    # Resize for better visibility
                    face_display = cv2.resize(face_img, (224, 224))
                    # Place in the corner of the frame
                    h, w = face_display.shape[:2]
                    frame[10:10+h, width-w-10:width-10] = face_display
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            fps_display = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Gaze Estimation Demo', frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo finished")

if __name__ == "__main__":
    main() 