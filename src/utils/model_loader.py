import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
import yaml

class GazeResNet(nn.Module):
    """
    ResNet-based gaze estimation model from ETH-XGaze.
    
    This is a simplified version of the model architecture used in ETH-XGaze.
    It consists of a ResNet50 backbone followed by a fully connected layer
    that outputs the gaze direction as pitch and yaw angles.
    """
    def __init__(self):
        super(GazeResNet, self).__init__()
        # Import torchvision only if needed to reduce dependencies
        import torchvision.models as models
        
        # Load a pre-trained ResNet50 model
        self.gaze_network = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer for gaze estimation
        # Output is 2 values: pitch and yaw - same as ETH-XGaze
        self.gaze_fc = nn.Sequential(nn.Linear(2048, 2))
    
    def forward(self, x):
        """Forward pass through the network."""
        # Feature extraction using ResNet
        x = self.gaze_network.conv1(x)
        x = self.gaze_network.bn1(x)
        x = self.gaze_network.relu(x)
        x = self.gaze_network.maxpool(x)
        
        x = self.gaze_network.layer1(x)
        x = self.gaze_network.layer2(x)
        x = self.gaze_network.layer3(x)
        x = self.gaze_network.layer4(x)
        
        x = self.gaze_network.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Gaze direction estimation
        gaze = self.gaze_fc(x)
        
        return gaze

class ONNXGazeModel:
    """
    Wrapper class for ONNX model to provide an interface similar to PyTorch models.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        # Import onnxruntime only when needed
        import onnxruntime as ort
        
        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set up providers based on device
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create inference session
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Store device
        self.device = device
        
        print(f"ONNX model loaded successfully from {model_path}")
        print(f"Input shape: {self.input_shape}, Input name: {self.input_name}, Output name: {self.output_name}")
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference with the ONNX model.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            torch.Tensor: Output tensor (gaze prediction)
        """
        # Convert PyTorch tensor to numpy
        if isinstance(x, torch.Tensor):
            x_numpy = x.cpu().numpy()
        else:
            x_numpy = x
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: x_numpy})
        
        # Convert output back to PyTorch tensor for consistency
        return torch.tensor(outputs[0])
    
    def eval(self):
        """
        Set the model to evaluation mode (no-op for ONNX models, for compatibility).
        """
        # ONNX models are always in inference mode, so this is a no-op
        return self

class ModelLoader:
    """
    A utility class to load neural network models for the drowsiness detection system.
    Specifically designed to handle ETH-XGaze models in various formats.
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
    
    def load_eth_xgaze_model(self, device: str = 'cpu') -> Optional[Union[torch.nn.Module, ONNXGazeModel]]:
        """
        Load the ETH-XGaze gaze estimation model.
        
        This method will first try to load the ONNX model (preferred for inference).
        If ONNX model is not found, it will fall back to the PyTorch model.
        
        Args:
            device: Device to load the model on ('cpu' or 'cuda')
            
        Returns:
            Union[torch.nn.Module, ONNXGazeModel]: Loaded model or None if loading fails
        """
        # Check for ONNX model first (preferred for inference)
        onnx_model_path = os.path.join(self.base_model_dir, 'eth_xgaze_model.onnx')
        pth_model_path = os.path.join(self.base_model_dir, 'eth_xgaze_model.pth')
        
        # Try loading ONNX model first
        if os.path.isfile(onnx_model_path):
            try:
                # Check if onnxruntime is installed
                try:
                    import onnxruntime
                except ImportError:
                    print("Warning: onnxruntime not installed. Please install it: pip install onnxruntime")
                    print("Falling back to PyTorch model.")
                    return self._load_pytorch_model(pth_model_path, device)
                
                print(f"Loading ETH-XGaze ONNX model from {onnx_model_path}")
                return ONNXGazeModel(onnx_model_path, device)
                
            except Exception as e:
                print(f"Error loading ONNX model: {str(e)}")
                print("Falling back to PyTorch model.")
                return self._load_pytorch_model(pth_model_path, device)
        else:
            print(f"ONNX model not found at {onnx_model_path}. Checking for PyTorch model.")
            return self._load_pytorch_model(pth_model_path, device)
    
    def _load_pytorch_model(self, model_path: str, device: str) -> Optional[torch.nn.Module]:
        """
        Load the PyTorch model.
        
        This method tries to match the ETH-XGaze model format as closely as possible.
        ETH-XGaze model is typically saved as a checkpoint dictionary with
        'model_state' key containing the model state_dict.
        
        Args:
            model_path: Path to the PyTorch model file
            device: Device to load the model on ('cpu' or 'cuda')
            
        Returns:
            torch.nn.Module: Loaded PyTorch model or None if loading fails
        """
        try:
            # Check if model file exists
            if not os.path.isfile(model_path):
                print(f"Warning: ETH-XGaze pre-trained weights not found at {model_path}")
                print("Creating a model with default initialization.")
                print("Please download the pre-trained weights for better performance.")
                
                # Create a new model instance
                model = GazeResNet()
                model = model.to(device)
                model.eval()
                return model
            
            print(f"Loading ETH-XGaze PyTorch model from {model_path}")
            
            # ETH-XGaze specific loading approach
            try:
                # First try - Load as ETH-XGaze format (checkpoint with 'model_state' key)
                checkpoint = torch.load(model_path, map_location=device)
                model = GazeResNet()
                
                # Check if it's an ETH-XGaze style checkpoint
                if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                    print("Loading ETH-XGaze checkpoint format with 'model_state' key")
                    model.load_state_dict(checkpoint['model_state'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    print("Loading checkpoint format with 'state_dict' key")
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Try loading directly as a state dictionary
                    print("Loading as direct state dictionary")
                    model.load_state_dict(checkpoint)
                
                model = model.to(device)
                model.eval()  # Always set to evaluation mode for inference
                return model
                
            except Exception as e:
                print(f"Initial loading attempt failed: {str(e)}")
                
                # Second try - Load the entire model if saved that way
                try:
                    print("Attempting to load as full model")
                    model = torch.load(model_path, map_location=device)
                    if hasattr(model, 'eval'):
                        model.eval()  # Set to evaluation mode
                        return model
                    else:
                        raise ValueError("Loaded object does not have eval() method")
                        
                except Exception as e2:
                    print(f"Could not load as full model: {str(e2)}")
                    
                    # Last resort - create a new model
                    print("Creating a new default model as fallback")
                    model = GazeResNet()
                    model = model.to(device)
                    model.eval()
                    return model
            
        except Exception as e:
            print(f"Error loading ETH-XGaze PyTorch model: {str(e)}")
            print("Creating a model with default initialization as fallback.")
            
            # Create a new model instance as fallback
            try:
                model = GazeResNet()
                model = model.to(device)
                model.eval()
                return model
            except Exception as e2:
                print(f"Failed to create fallback model: {str(e2)}")
                return None
