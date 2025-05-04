import os
import torch
from typing import Dict, Any, Optional
import yaml

class ModelLoader:
    """
    A utility class to load neural network models for the drowsiness detection system.
    """
    
    def __init__(self):
        """Initialize the ModelLoader class."""
        # Base directory for all models
        self.base_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Ensure models directory exists
        os.makedirs(self.base_model_dir, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
    
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
    
    def load_eth_xgaze_model(self, device: str = 'cpu') -> Optional[torch.nn.Module]:
        """
        Load the ETH-XGaze gaze estimation model.
        
        Args:
            device: Device to load the model on ('cpu' or 'cuda')
            
        Returns:
            torch.nn.Module: Loaded PyTorch model or None if loading fails
        """
        model_path = os.path.join(self.base_model_dir, 'eth_xgaze_model.pth')
        
        try:
            # Check if model file exists
            if not os.path.isfile(model_path):
                print(f"Error: ETH-XGaze model not found at {model_path}")
                print("Please download the model and place it in the models directory.")
                return None
            
            # Load the model
            model = torch.load(model_path, map_location=device)
            
            # Set the model to evaluation mode
            model.eval()
            
            print(f"ETH-XGaze model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading ETH-XGaze model: {str(e)}")
            return None
