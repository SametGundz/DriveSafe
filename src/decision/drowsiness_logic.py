import os
import time
import yaml
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

class DrowsinessLogic:
    """
    A class to evaluate driver drowsiness based on multiple indicators:
    - EAR (Eye Aspect Ratio)
    - MAR (Mouth Aspect Ratio)
    - PERCLOS (Percentage of Eye Closure)
    - Gaze Direction

    The class applies configurable rules to determine drowsiness level and outputs
    a Karolinska Sleepiness Scale (KSS) rating from 1-9.

    Attributes:
        config: Configuration dictionary loaded from config.yaml
        ear_history: List to track EAR values and timestamps
        mar_history: List to track MAR values and timestamps 
        gaze_history: List to track gaze direction values
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        'ear_threshold': 0.21,
        'mar_threshold': 0.65,
        'ear_closed_duration': 2.0,       # Duration in seconds for closed eyes to indicate drowsiness
        'mar_open_duration': 1.0,         # Duration in seconds for open mouth to indicate yawning
        'forward_pitch_threshold': 30.0,  # Head pitch forward threshold (degrees)
        'perclos_threshold': 40.0,        # PERCLOS percentage threshold for drowsiness
        'gaze_deviation_threshold': 20.0, # Gaze deviation threshold (degrees)
        'history_duration': 5.0,          # Duration to keep history (seconds)
    }
    
    # Karolinska Sleepiness Scale (KSS) mapping
    KSS_DESCRIPTION = {
        1: "Extremely alert",
        2: "Very alert",
        3: "Alert",
        4: "Rather alert",
        5: "Neither alert nor sleepy",
        6: "Some signs of sleepiness",
        7: "Sleepy, no effort to stay awake",
        8: "Sleepy, some effort to stay awake",
        9: "Very sleepy, great effort to stay awake"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DrowsinessLogic with configuration.
        
        Args:
            config: Optional configuration dictionary to override loading from file
        """
        # Load configuration
        self.config = config if config is not None else self._load_config()
        
        # Set thresholds from config or use defaults
        self.ear_threshold = self.config.get('ear_threshold', self.DEFAULT_CONFIG['ear_threshold'])
        self.mar_threshold = self.config.get('mar_threshold', self.DEFAULT_CONFIG['mar_threshold'])
        
        # Get drowsiness-specific config (with defaults if not present)
        drowsiness_config = self.config.get('drowsiness', {})
        self.ear_closed_duration = drowsiness_config.get('ear_closed_duration', 
                                                        self.DEFAULT_CONFIG['ear_closed_duration'])
        self.mar_open_duration = drowsiness_config.get('mar_open_duration', 
                                                      self.DEFAULT_CONFIG['mar_open_duration'])
        self.forward_pitch_threshold = drowsiness_config.get('forward_pitch_threshold', 
                                                           self.DEFAULT_CONFIG['forward_pitch_threshold'])
        self.perclos_threshold = drowsiness_config.get('perclos_threshold', 
                                                      self.DEFAULT_CONFIG['perclos_threshold'])
        self.gaze_deviation_threshold = drowsiness_config.get('gaze_deviation_threshold', 
                                                            self.DEFAULT_CONFIG['gaze_deviation_threshold'])
        
        # History tracking duration
        self.history_duration = drowsiness_config.get('history_duration', 
                                                    self.DEFAULT_CONFIG['history_duration'])
        
        # Initialize history trackers with timestamps
        self.ear_history = []  # [(timestamp, ear_value, is_closed), ...]
        self.mar_history = []  # [(timestamp, mar_value, is_open), ...]
        self.gaze_history = []  # [(timestamp, gaze_x, gaze_y, gaze_z), ...]
        
        # Initialize state indicators
        self.prolonged_eye_closure = False
        self.yawning_detected = False
        self.head_distraction = False
        self.perclos_drowsy = False
        self.gaze_distraction = False
        
        # Current KSS score
        self.current_kss = 1
        
        # Debug information
        self.debug_info = {}

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                   'config', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except (FileNotFoundError, yaml.YAMLError):
            print(f"Warning: Could not load config file at {config_path}. Using default values.")
            return {}
            
    def _clean_history(self, current_time: float):
        """
        Remove entries older than history_duration from all history lists.
        
        Args:
            current_time: Current timestamp to compare against
        """
        # Clean EAR history
        self.ear_history = [entry for entry in self.ear_history 
                           if current_time - entry[0] <= self.history_duration]
        
        # Clean MAR history
        self.mar_history = [entry for entry in self.mar_history 
                           if current_time - entry[0] <= self.history_duration]
        
        # Clean gaze history
        self.gaze_history = [entry for entry in self.gaze_history 
                            if current_time - entry[0] <= self.history_duration]
    
    def _check_prolonged_eye_closure(self, current_time: float) -> bool:
        """
        Check if eyes have been closed for a duration exceeding the threshold.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            bool: True if prolonged eye closure is detected
        """
        if not self.ear_history:
            return False
            
        # Find consecutive closed eye entries
        closure_start = None
        for timestamp, _, is_closed in reversed(self.ear_history):
            if is_closed and closure_start is None:
                closure_start = timestamp
            elif not is_closed and closure_start is not None:
                break
                
        if closure_start is None:
            return False
            
        # Check if current closure exceeds threshold duration
        return (current_time - closure_start) >= self.ear_closed_duration
    
    def _check_yawning(self, current_time: float) -> bool:
        """
        Check if mouth has been open for a duration exceeding the threshold.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            bool: True if yawning is detected
        """
        if not self.mar_history:
            return False
            
        # Find consecutive open mouth entries
        open_start = None
        for timestamp, _, is_open in reversed(self.mar_history):
            if is_open and open_start is None:
                open_start = timestamp
            elif not is_open and open_start is not None:
                break
                
        if open_start is None:
            return False
            
        # Check if current opening exceeds threshold duration
        return (current_time - open_start) >= self.mar_open_duration
    
    def _check_gaze_distraction(self) -> bool:
        """
        Check if gaze direction has been consistently deviating.
        
        Returns:
            bool: True if gaze distraction is detected
        """
        if len(self.gaze_history) < 10:  # Need sufficient history
            return False
            
        # Calculate average gaze deviation from center
        gaze_vectors = np.array([entry[1:] for entry in self.gaze_history])
        
        # Simple method: check if magnitude of average gaze vector exceeds threshold
        avg_gaze = np.mean(gaze_vectors, axis=0)
        gaze_angle = np.degrees(np.arccos(avg_gaze[2] / np.linalg.norm(avg_gaze)))
        
        return gaze_angle > self.gaze_deviation_threshold
    
    def update(self, ear: Optional[float], mar: Optional[float], 
              perclos: Optional[float], 
              gaze_direction: Optional[Tuple[float, float, float]]) -> int:
        """
        Update drowsiness state with new measurements and calculate KSS score.
        
        Args:
            ear: Eye Aspect Ratio value (None if not available)
            mar: Mouth Aspect Ratio value (None if not available)
            perclos: PERCLOS value as percentage (None if not available)
            gaze_direction: Tuple of (x, y, z) gaze direction vector (None if not available)
            
        Returns:
            int: Current Karolinska Sleepiness Scale (KSS) score (1-9)
        """
        current_time = time.time()
        
        # Clean old history entries
        self._clean_history(current_time)
        
        # Update EAR history
        if ear is not None:
            is_closed = ear < self.ear_threshold
            self.ear_history.append((current_time, ear, is_closed))
            
        # Update MAR history
        if mar is not None:
            is_open = mar > self.mar_threshold
            self.mar_history.append((current_time, mar, is_open))
            
        # Update gaze history
        if gaze_direction is not None:
            self.gaze_history.append((current_time, *gaze_direction))
            
        # Check conditions
        self.prolonged_eye_closure = self._check_prolonged_eye_closure(current_time)
        self.yawning_detected = self._check_yawning(current_time)
        
        # Head distraction is no longer detected using head pose
        self.head_distraction = False
        
        # PERCLOS-based drowsiness
        if perclos is not None:
            self.perclos_drowsy = perclos > self.perclos_threshold
            
        # Gaze distraction
        self.gaze_distraction = self._check_gaze_distraction()
        
        # Calculate KSS score
        self.current_kss = self._calculate_kss()
        
        # Update debug info
        self._update_debug_info(ear, mar, perclos, gaze_direction)
        
        return self.current_kss
        
    def _calculate_kss(self) -> int:
        """
        Calculate Karolinska Sleepiness Scale (KSS) score based on drowsiness indicators.
        
        Returns:
            int: KSS score from 1 (extremely alert) to 9 (very sleepy)
        """
        # Base score
        score = 1
        
        # Increment score based on indicators
        if self.perclos_drowsy:
            score += 4  # Strong indicator of drowsiness
            
        if self.prolonged_eye_closure:
            score += 3  # Strong indicator of drowsiness
            
        if self.yawning_detected:
            score += 2  # Moderate indicator of drowsiness
            
        if self.head_distraction:
            score += 2  # Moderate indicator of distraction
            
        if self.gaze_distraction:
            score += 1  # Mild indicator of distraction
            
        # Cap score at 9
        return min(score, 9)
        
    def _update_debug_info(self, ear: Optional[float], mar: Optional[float],
                          perclos: Optional[float],
                          gaze_direction: Optional[Tuple[float, float, float]]):
        """
        Update debug information with current values and state.
        
        Args:
            ear: Current EAR value
            mar: Current MAR value
            perclos: Current PERCLOS value
            gaze_direction: Current gaze direction vector
        """
        self.debug_info = {
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "gaze_direction": gaze_direction,
            "state": {
                "prolonged_eye_closure": self.prolonged_eye_closure,
                "yawning_detected": self.yawning_detected,
                "head_distraction": self.head_distraction,
                "perclos_drowsy": self.perclos_drowsy,
                "gaze_distraction": self.gaze_distraction
            },
            "kss_score": self.current_kss,
            "kss_description": self.KSS_DESCRIPTION.get(self.current_kss, "Unknown")
        }
        
    def get_kss_score(self) -> int:
        """
        Get the current Karolinska Sleepiness Scale score.
        
        Returns:
            int: KSS score from 1 (extremely alert) to 9 (very sleepy)
        """
        return self.current_kss
        
    def get_kss_description(self) -> str:
        """
        Get the description for the current KSS score.
        
        Returns:
            str: Text description of the current drowsiness level
        """
        return self.KSS_DESCRIPTION.get(self.current_kss, "Unknown")
        
    def get_alert_status(self) -> bool:
        """
        Check if the current KSS score exceeds the alert threshold.
        
        Returns:
            bool: True if an alert should be triggered
        """
        # Get alert threshold from config or use default (6)
        alert_threshold = self.config.get('drowsiness', {}).get('kss_alert_threshold', 6)
        return self.current_kss >= alert_threshold
        
    def get_state_summary(self) -> Dict:
        """
        Get a summary of the current drowsiness state.
        
        Returns:
            Dict: Dictionary with current state information
        """
        return {
            "kss_score": self.current_kss,
            "kss_description": self.KSS_DESCRIPTION.get(self.current_kss, "Unknown"),
            "alert_needed": self.get_alert_status(),
            "prolonged_eye_closure": self.prolonged_eye_closure,
            "yawning_detected": self.yawning_detected,
            "head_distraction": self.head_distraction,
            "perclos_drowsy": self.perclos_drowsy,
            "gaze_distraction": self.gaze_distraction
        }
