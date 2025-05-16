import numpy as np
from collections import deque
import yaml
import os
from typing import Dict, Optional, List, Tuple, Any, Union

class PERCLOSDetector:
    """
    A class to detect PERCLOS (PERcentage of eyelid CLOSure) over time.
    
    PERCLOS is a drowsiness measure defined as the percentage of time the eyes are closed
    over a specific time window. This implementation tracks EAR (Eye Aspect Ratio) values
    over N frames and determines the percentage of frames where the eyes were closed
    based on an EAR threshold.
    
    Attributes:
        window_size: Number of frames to consider for PERCLOS calculation
        ear_threshold: Threshold for determining if eyes are closed
        ear_history: Deque to store EAR values for the last N frames
        closed_history: Deque to store boolean values indicating if eyes were closed
        config: Configuration dictionary loaded from config.yaml
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        'perclos_window_size': 150,  # 5 seconds at 30fps
        'ear_threshold': 0.2,        # EAR threshold for eye closure
        'perclos_threshold': 20.0,   # PERCLOS percentage for drowsiness alert
        'fps': 30.0,                 # Frames per second assumption
        'min_blink_frames': 2,       # Minimum frames for a valid blink
        'max_blink_frames': 7,       # Maximum frames for a normal blink
    }
    
    # Karolinska Sleepiness Scale (KSS) mapping based on PERCLOS
    KSS_MAPPING = [
        (5.0, 1),   # 1: Extremely alert
        (10.0, 2),  # 2: Very alert
        (15.0, 3),  # 3: Alert
        (20.0, 4),  # 4: Rather alert
        (25.0, 5),  # 5: Neither alert nor sleepy
        (30.0, 6),  # 6: Some signs of sleepiness
        (35.0, 7),  # 7: Sleepy, no effort to stay awake
        (40.0, 8),  # 8: Sleepy, some effort to stay awake
        (float('inf'), 9)  # 9: Very sleepy, great effort to stay awake
    ]
    
    def __init__(self, window_size: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PERCLOSDetector with configuration.
        
        Args:
            window_size: Number of frames to consider for PERCLOS calculation
                (default: from config or 150 frames, approximately 5 seconds at 30 fps)
            config: Optional configuration dictionary to override loading from file
        """
        # Load configuration
        self.config = config if config is not None else self._load_config()
        
        # Set window size (from explicit parameter, config, or default)
        self.window_size = window_size if window_size is not None else self.config.get(
            'perclos_window_size', self.DEFAULT_CONFIG['perclos_window_size'])
        
        # Set parameters from config or use defaults
        self.ear_threshold = self.config.get('ear_threshold', self.DEFAULT_CONFIG['ear_threshold'])
        self.perclos_threshold = self.config.get('perclos_threshold', self.DEFAULT_CONFIG['perclos_threshold'])
        self.fps = self.config.get('fps', self.DEFAULT_CONFIG['fps'])
        self.min_blink_frames = self.config.get('min_blink_frames', self.DEFAULT_CONFIG['min_blink_frames'])
        self.max_blink_frames = self.config.get('max_blink_frames', self.DEFAULT_CONFIG['max_blink_frames'])
        
        # Initialize history deques
        self.ear_history = deque(maxlen=self.window_size)
        self.closed_history = deque(maxlen=self.window_size)
        
        # Initialize PERCLOS value and KSS score
        self.current_perclos = 0.0
        self.current_kss = 1  # Start with "Extremely alert"
        
        # For testing and debugging
        self.debug_info = {}

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        # Start with default configuration
        config = self.DEFAULT_CONFIG.copy()
        
        # Try to load from file and update defaults
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                  'config', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Update only keys that exist in file_config
                    config.update(file_config)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not load config file at {config_path}. Using default values. Error: {e}")
            
        return config

    def update(self, ear: Optional[float]) -> Tuple[float, int]:
        """
        Update the PERCLOS calculation with a new EAR value.
        
        Args:
            ear: The current Eye Aspect Ratio value (None if no face/eyes detected)
            
        Returns:
            Tuple containing:
            - Current PERCLOS value (percentage)
            - Current Karolinska Sleepiness Scale (KSS) score (1-9)
        """
        # If EAR is None (no face detected), consider the eyes as open
        # This is a conservative approach to avoid false positives
        if ear is None:
            is_closed = False
            # Use the last known EAR value or a default open eye value
            ear_value = max(self.ear_history[-1] if self.ear_history else 0.3, 0.3)
        else:
            # Determine if eyes are closed based on threshold
            is_closed = ear < self.ear_threshold
            ear_value = ear
        
        # Add to history
        self.ear_history.append(ear_value)
        self.closed_history.append(is_closed)
        
        # Calculate PERCLOS only if we have enough history
        if len(self.closed_history) >= self.window_size // 2:  # At least half the window to start calculating
            # PERCLOS is the percentage of frames where eyes were closed
            closed_count = sum(self.closed_history)
            self.current_perclos = (closed_count / len(self.closed_history)) * 100.0
            
            # Update Karolinska Sleepiness Scale score
            self.current_kss = self._calculate_kss_score(self.current_perclos)
        
        return self.current_perclos, self.current_kss

    def _calculate_kss_score(self, perclos: float) -> int:
        """
        Calculate Karolinska Sleepiness Scale (KSS) score based on PERCLOS value.
        
        Args:
            perclos: The current PERCLOS value (percentage)
            
        Returns:
            int: KSS score from 1 (extremely alert) to 9 (very sleepy)
        """
        for threshold, score in self.KSS_MAPPING:
            if perclos <= threshold:
                return score
        return 9  # Maximum sleepiness if above all thresholds

    def get_perclos(self) -> float:
        """
        Get the current PERCLOS value.
        
        Returns:
            float: The current PERCLOS value (percentage)
        """
        return self.current_perclos
    
    def get_kss_score(self) -> int:
        """
        Get the current Karolinska Sleepiness Scale score.
        
        Returns:
            int: KSS score from 1 (extremely alert) to 9 (very sleepy)
        """
        return self.current_kss

    def get_status(self) -> Tuple[float, int, bool, str]:
        """
        Get the current drowsiness status information.
        
        Returns:
            Tuple containing:
            - Current PERCLOS value (percentage)
            - Current KSS score (1-9)
            - Boolean indicating if PERCLOS is above threshold (alert needed)
            - Status message
        """
        is_drowsy = self.current_perclos > self.perclos_threshold
        
        if len(self.closed_history) < self.window_size // 2:
            status = "Collecting data..."
        elif is_drowsy:
            status = f"DROWSINESS ALERT (KSS: {self.current_kss})"
        else:
            status = f"Normal (KSS: {self.current_kss})"
        
        return self.current_perclos, self.current_kss, is_drowsy, status

    def get_history_stats(self) -> Dict:
        """
        Get statistics about the EAR history for analysis.
        
        Returns:
            Dict: Dictionary containing statistics
        """
        if not self.ear_history:
            return {
                "min_ear": None,
                "max_ear": None,
                "avg_ear": None,
                "blink_count": 0,
                "blink_rate": 0.0,
                "avg_blink_duration": 0.0,
                "kss_score": 1
            }
        
        # Calculate blink count and duration
        blink_count = 0
        current_blink_start = None
        blink_durations = []
        
        for i, is_closed in enumerate(self.closed_history):
            if is_closed and current_blink_start is None:
                # Blink started
                current_blink_start = i
            elif not is_closed and current_blink_start is not None:
                # Blink ended
                blink_duration = i - current_blink_start
                # Only count as a blink if duration is within range
                if self.min_blink_frames <= blink_duration <= self.max_blink_frames:
                    blink_count += 1
                    blink_durations.append(blink_duration)
                current_blink_start = None
        
        # Handle case where last frames are part of a blink
        if current_blink_start is not None:
            blink_duration = len(self.closed_history) - current_blink_start
            if self.min_blink_frames <= blink_duration <= self.max_blink_frames:
                blink_count += 1
                blink_durations.append(blink_duration)
        
        # Calculate average blink duration
        avg_blink_duration = np.mean(blink_durations) if blink_durations else 0.0
        
        # Calculate blink rate (blinks per minute)
        window_duration_minutes = (self.window_size / self.fps) / 60.0
        blink_rate = blink_count / window_duration_minutes if window_duration_minutes > 0 else 0.0
        
        # Get EAR statistics
        ear_array = np.array(self.ear_history)
        
        return {
            "min_ear": np.min(ear_array),
            "max_ear": np.max(ear_array),
            "avg_ear": np.mean(ear_array),
            "blink_count": blink_count,
            "blink_rate": blink_rate,
            "avg_blink_duration": avg_blink_duration,
            "kss_score": self.current_kss
        }

    def visualize(self, frame, font_scale: float = 0.7) -> np.ndarray:
        """
        Visualize PERCLOS information on the frame.
        
        Args:
            frame: Input frame/image
            font_scale: Font scale for visualization text
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        # Return original frame without any text overlays
        return frame.copy()
    
    def reset(self) -> None:
        """
        Reset the detector state (useful when changing video sources).
        """
        self.ear_history.clear()
        self.closed_history.clear()
        self.current_perclos = 0.0
        self.current_kss = 1
