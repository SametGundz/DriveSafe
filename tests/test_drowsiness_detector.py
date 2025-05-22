#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uykululuk Tespiti Test Modülü.
Bu modül, DrowsinessDetector sınıfının test betiklerini içerir.
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path
from collections import deque

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the class to test
from src.detection.drowsiness_detector import DrowsinessDetector

class TestDrowsinessDetector(unittest.TestCase):
    """
    DrowsinessDetector sınıfı için test sınıfı.
    """
    
    def setUp(self):
        """
        Her test öncesi çalışan hazırlık metodu.
        """
        # Create a detector with default configuration
        self.detector = DrowsinessDetector()
        
        # Test landmarks for a fully open eye
        self.open_eye = [
            [0, 0],    # P1
            [1, -1],   # P2
            [2, -1],   # P3
            [3, 0],    # P4
            [2, 1],    # P5
            [1, 1]     # P6
        ]
        
        # Test landmarks for a nearly closed eye
        self.closed_eye = [
            [0, 0],    # P1
            [1, -0.1], # P2
            [2, -0.1], # P3
            [3, 0],    # P4
            [2, 0.1],  # P5
            [1, 0.1]   # P6
        ]
    
    def test_calculate_ear(self):
        """
        calculate_ear() metodu için test.
        """
        # Test with open eye
        ear_open = self.detector.calculate_ear(self.open_eye)
        self.assertGreater(ear_open, 0.2)
        
        # Test with closed eye
        ear_closed = self.detector.calculate_ear(self.closed_eye)
        self.assertLess(ear_closed, 0.2)
        
        # Test with invalid landmarks
        ear_invalid = self.detector.calculate_ear([])
        self.assertEqual(ear_invalid, 0.0)
    
    def test_calculate_perclos(self):
        """
        calculate_perclos() metodu için test.
        """
        # Empty ear_values
        self.detector.ear_values.clear()
        perclos = self.detector.calculate_perclos()
        self.assertEqual(perclos, 0.0)
        
        # All eyes open
        self.detector.ear_values.clear()
        for _ in range(100):
            self.detector.ear_values.append(0.3)  # Open eye
        perclos = self.detector.calculate_perclos()
        self.assertEqual(perclos, 0.0)
        
        # All eyes closed
        self.detector.ear_values.clear()
        for _ in range(100):
            self.detector.ear_values.append(0.15)  # Closed eye
        perclos = self.detector.calculate_perclos()
        self.assertEqual(perclos, 1.0)
        
        # 50% eyes closed
        self.detector.ear_values.clear()
        for i in range(100):
            if i % 2 == 0:
                self.detector.ear_values.append(0.3)  # Open eye
            else:
                self.detector.ear_values.append(0.15)  # Closed eye
        perclos = self.detector.calculate_perclos()
        self.assertAlmostEqual(perclos, 0.5, places=2)
    
    def test_update_with_ear_only(self):
        """
        update() metodunun sadece EAR değerleriyle test edilmesi.
        """
        # Update with open eyes
        result = self.detector.update(ear_left=0.3, ear_right=0.3)
        
        # Check results
        self.assertFalse(result["is_eyes_closed"])
        self.assertEqual(result["drowsiness_state"], "Alert")
        self.assertLess(result["drowsiness_level"], 0.3)
        
        # Update with closed eyes
        result = self.detector.update(ear_left=0.15, ear_right=0.15)
        
        # Check results
        self.assertTrue(result["is_eyes_closed"])
        
        # Eyes just closed, so drowsiness level should still be low
        self.assertEqual(result["drowsiness_state"], "Alert")
        
        # Simulate eyes closed for a longer period
        self.detector.eyes_closed_start_time = self.detector.last_update_time - 1.5  # 1.5 seconds
        
        # Update again
        result = self.detector.update(ear_left=0.15, ear_right=0.15)
        
        # Check drowsiness level increased
        self.assertGreater(result["drowsiness_level"], 0.0)
    
    def test_multiple_updates(self):
        """
        Bir dizi güncelleme sonrasında detector davranışının testi.
        """
        # Simulate a sequence of updates to build history
        
        # Start with normal state
        for _ in range(30):  # 1 second assuming 30 FPS
            self.detector.update(ear_left=0.3, ear_right=0.3)
        
        # Transition to drowsy state with eyes closing
        for _ in range(30):  # 1 second
            self.detector.update(ear_left=0.15, ear_right=0.15)
        
        # Get result after some drowsy time
        result = self.detector.update(ear_left=0.15, ear_right=0.15)
        
        # PERCLOS should be elevated (about 50% for last 2 seconds)
        self.assertGreater(result["perclos"], 0.3)
        
        # Drowsiness level should be elevated
        self.assertGreater(result["drowsiness_level"], 0.3)
        
        # Return to normal state
        for _ in range(90):  # 3 seconds
            self.detector.update(ear_left=0.3, ear_right=0.3)
        
        # Get final result
        result = self.detector.update(ear_left=0.3, ear_right=0.3)
        
        # PERCLOS should be reduced
        self.assertLess(result["perclos"], 0.2)
        
        # Drowsiness level should be reduced
        self.assertLess(result["drowsiness_level"], 0.3)
    
    def test_drowsiness_levels(self):
        """
        Farklı uykululuk seviyeleri için tespit durumunu test eder.
        """
        # Helper function to mock drowsiness levels
        def mock_drowsiness(level):
            # Patch the _calculate_drowsiness_level method to return our desired level
            original_method = self.detector._calculate_drowsiness_level
            self.detector._calculate_drowsiness_level = lambda: level
            
            # Call update to get the result
            result = self.detector.update(ear_value=0.3)
            
            # Restore the original method
            self.detector._calculate_drowsiness_level = original_method
            
            return result
        
        # Test "Uyanık" level
        result = mock_drowsiness(0.2)
        self.assertEqual(result["drowsiness_state"], "Alert")
        
        # Test "Yorgun" level
        result = mock_drowsiness(0.4)
        self.assertEqual(result["drowsiness_state"], "Tired")
        
        # Test "Uykulu" level
        result = mock_drowsiness(0.7)
        self.assertEqual(result["drowsiness_state"], "Drowsy")
        
        # Test "Tehlikeli" level
        result = mock_drowsiness(0.9)
        self.assertEqual(result["drowsiness_state"], "Danger")

if __name__ == "__main__":
    unittest.main() 