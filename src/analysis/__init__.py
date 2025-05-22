"""
Video analysis package for the driver drowsiness detection application.

This package provides functionality for analyzing videos for driver drowsiness,
including frame-by-frame processing, gaze detection, and metrics calculation.
"""

from src.analysis.video_analyzer import VideoAnalyzer
from src.analysis.gaze_processor import GazeProcessor

__all__ = ['VideoAnalyzer', 'GazeProcessor'] 