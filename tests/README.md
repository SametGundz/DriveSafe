# Driver Drowsiness Detection Tests

This directory contains tests for the driver drowsiness detection system.

## Running Tests

You can run all tests using the provided script:

```bash
python tests/run_tests.py
```

Or run individual test files:

```bash
python -m unittest tests/utils/test_face_landmark_detector.py
python -m unittest tests/utils/test_facial_metrics.py
python -m unittest tests/utils/test_head_pose_estimator.py
python -m unittest tests/utils/test_gaze_detector.py
python -m unittest tests/utils/test_mediapipe_helper.py
```

## Test Structure

The tests are organized by module:

- `utils/`: Tests for utility modules
  - `test_face_landmark_detector.py`: Tests for the `FaceLandmarkDetector` class
  - `test_facial_metrics.py`: Tests for facial metrics calculation functions
  - `test_head_pose_estimator.py`: Tests for the `HeadPoseEstimator` class
  - `test_gaze_detector.py`: Tests for the `GazeDetector` class
  - `test_mediapipe_helper.py`: Tests for the `MediaPipeHelper` class
- `detection/`: Tests for detection modules
  - `test_gaze_zone_detector.py`: Tests for the `GazeZoneDetector` class

## Interactive Demos

Some test files also serve as interactive demonstrations:

- `test_gaze_zone_detector.py`: Demonstrates the gaze zone detection system with a webcam
  ```bash
  python tests/test_gaze_zone_detector.py --camera 0 --history 10 --stability 0.7
  ```
  This demo visualizes which zone in the car interior the driver is looking at (dashboard, mirrors, windshield, etc.) and tracks viewing duration for each zone.

## Writing New Tests

When writing new tests:

1. Create a new test file in the appropriate directory
2. Use the `unittest` framework
3. Run `python -m unittest discover` to ensure all tests are discovered and run properly

Example:

```python
import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import module to test
from src.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.your_object = YourClass()
    
    def test_some_function(self):
        """Test description."""
        result = self.your_object.some_function()
        self.assertEqual(result, expected_value)

if __name__ == '__main__':
    unittest.main() 