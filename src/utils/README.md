# MediaPipe Utilities

This directory contains a modular implementation of MediaPipe-based utilities for the driver drowsiness detection system.

## Structure

The utilities have been refactored into a modular, object-oriented architecture with the following components:

- `face_landmark_detector.py`: Face detection and landmark extraction
- `facial_metrics.py`: Calculation of facial metrics (EAR, MAR)
- `head_pose_estimator.py`: Head pose estimation from facial landmarks
- `gaze_detector.py`: Gaze direction detection and visualization
- `mediapipe_helper.py`: Unified interface that integrates all components

This modular approach offers several advantages:
- Clear separation of concerns
- Better code organization and maintainability
- Easier debugging and testing
- More flexible architecture

## Usage

You can use either the unified `MediaPipeHelper` interface or the individual components:

```python
# Using the unified interface
from src.utils import MediaPipeHelper, get_mediapipe_helper

# Get the helper instance
mp_helper = get_mediapipe_helper()

# Detect face landmarks
landmarks, face_detected = mp_helper.detect_face_landmarks(frame)

# Get EAR value
left_eye_landmarks = mp_helper.get_eye_landmarks(landmarks, left_eye=True)
ear = mp_helper.get_eye_aspect_ratio(left_eye_landmarks)

# Or using individual components
from src.utils import FaceLandmarkDetector, facial_metrics

# Get the detector instance
detector = FaceLandmarkDetector()

# Detect face landmarks
landmarks, face_detected = detector.detect_face_landmarks(frame)

# Get EAR value
left_eye_landmarks = detector.get_eye_landmarks(landmarks, left_eye=True)
ear = facial_metrics.get_eye_aspect_ratio(left_eye_landmarks)
```

## Backward Compatibility

For backward compatibility, the old `MediaPipeUtils` class is still available, but it's recommended to migrate to the new modular approach.

```python
# Legacy usage (deprecated)
from src.utils import MediaPipeUtils, get_mediapipe_face_mesh

# Get the helper instance
mp_utils = get_mediapipe_face_mesh()

# Use as before
landmarks, face_detected = mp_utils.detect_face_landmarks(frame)
``` 