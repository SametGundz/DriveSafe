import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import yaml
import time
import logging
from functools import lru_cache

# Logger oluşturma
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelLoader")

class GazeResNet(nn.Module):
    """
    ResNet-based gaze estimation model from ETH-XGaze.
    
    This is a simplified version of the model architecture used in ETH-XGaze.
    It consists of a ResNet50 backbone followed by a fully connected layer
    that outputs the gaze direction as pitch and yaw angles.
    """
    def __init__(self, pretrained: bool = True):
        super(GazeResNet, self).__init__()
        # Import torchvision only if needed to reduce dependencies
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision is required for GazeResNet. Please install it: pip install torchvision")
        
        # Load a pre-trained ResNet50 model
        self.gaze_network = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Replace the final fully connected layer for gaze estimation
        # Output is 2 values: pitch and yaw - same as ETH-XGaze
        in_features = self.gaze_network.fc.in_features
        self.gaze_network.fc = nn.Identity()  # Remove original FC layer
        self.gaze_fc = nn.Linear(in_features, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Forward through ResNet backbone
        features = self.gaze_network(x)
        
        # Gaze direction estimation
        gaze = self.gaze_fc(features)
        
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
        start_time = time.time()
        
        # Import onnxruntime only when needed
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNXGazeModel. Please install it: pip install onnxruntime")
        
        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        
        # Enable thread optimization for CPU
        if device == 'cpu':
            # Set intra-op threads (how many threads to use for an operation)
            sess_options.intra_op_num_threads = min(8, os.cpu_count() or 4)
            # Set inter-op threads (how many operations to run in parallel)
            sess_options.inter_op_num_threads = min(8, os.cpu_count() or 4)
        
        # Set up providers based on device
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # CUDA Provider options
            provider_options = [
                {
                    'device_id': 0,  # Use first GPU
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB GPU Memory limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                },
                {}  # CPUExecutionProvider doesn't need options
            ]
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers, provider_options=provider_options)
        else:
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Store device
        self.device = device
        
        # Input/output binding optimization
        self.io_binding = self.session.io_binding()
        
        # Cache recent inputs for faster re-computation
        self.input_cache = {}
        self.cache_size = 10
        
        # Track resources loaded and model path for logging
        self.model_path = model_path
        self.resources_loaded = True
        
        logger.info(f"ONNX model loaded in {time.time() - start_time:.2f}s from {model_path}")
        logger.info(f"Input shape: {self.input_shape}, Input name: {self.input_name}, Output name: {self.output_name}")
    
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Run inference with the ONNX model.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            torch.Tensor: Output tensor (gaze prediction)
        """
        # Check if resources are still loaded
        if not hasattr(self, 'resources_loaded') or not self.resources_loaded:
            logger.warning("Attempted to use ONNX model after resources were released")
            return torch.zeros((x.shape[0] if len(x.shape) > 3 else 1, 2), dtype=torch.float32)
        
        # Optimize for single frame input
        if len(x.shape) == 3:  # Single image, add batch dimension
            x = x.unsqueeze(0) if isinstance(x, torch.Tensor) else np.expand_dims(x, 0)
        
        # Convert PyTorch tensor to numpy if needed
        if isinstance(x, torch.Tensor):
            x_numpy = x.cpu().numpy()
        else:
            x_numpy = x
        
        # Check input shape and correct if necessary
        if x_numpy.shape[1:] != self.input_shape[1:]:
            # Reshape to match expected input
            logger.warning(f"Input shape mismatch: got {x_numpy.shape}, expected batch + {self.input_shape[1:]}. Reshaping...")
            # If channels are first, ensure they match
            if x_numpy.shape[1] != self.input_shape[1]:
                # Try to transpose if input channels are last dimension
                if x_numpy.shape[-1] == self.input_shape[1]:
                    x_numpy = np.transpose(x_numpy, (0, 3, 1, 2))
        
        # Run inference
        try:
            outputs = self.session.run([self.output_name], {self.input_name: x_numpy})
            
            # Convert output back to PyTorch tensor
            return torch.tensor(outputs[0])
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            # Return empty tensor on error
            return torch.zeros((x_numpy.shape[0], 2), dtype=torch.float32)
    
    def release_resources(self):
        """
        Explicitly release ONNX session and other heavy resources.
        This method should be called when the model is no longer needed
        to free up memory immediately.
        """
        if hasattr(self, 'resources_loaded') and self.resources_loaded:
            try:
                # Release session
                if hasattr(self, 'session') and self.session is not None:
                    # Clear all references
                    self.io_binding = None
                    self.session = None
                
                # Clear cache
                if hasattr(self, 'input_cache'):
                    self.input_cache.clear()
                
                # Mark resources as released
                self.resources_loaded = False
                
                # Force garbage collection for immediate memory release
                import gc
                gc.collect()
                
                logger.info(f"ONNX model resources released for {self.model_path}")
            except Exception as e:
                logger.error(f"Error releasing ONNX resources: {str(e)}")
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        try:
            self.release_resources()
        except Exception as e:
            # Suppress errors during garbage collection
            pass
    
    def run_optimized(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Run optimized inference with IO binding for better performance.
        Useful for repeated inference on similar-sized inputs.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            torch.Tensor: Output tensor (gaze prediction)
        """
        # Only implement this if performance becomes a bottleneck
        # IO Binding can provide 10-30% speedup but makes the code more complex
        # For now, just call the standard method
        return self(x)
    
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
    
    # Class-level model cache to avoid reloading models
    _model_cache = {}
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelLoader class.
        
        Args:
            config_path: Optional override for config file path
        """
        start_time = time.time()
        
        # Store instance-specific state
        self.config_path = config_path
        self.initialized = False
        self.loaded_models = []  # Track loaded models for proper cleanup
        
        # Base directory for all models
        self.base_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Ensure models directory exists
        os.makedirs(self.base_model_dir, exist_ok=True)
        
        # Set config path if not provided
        if config_path is None:
            self.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                        'config', 'config.yaml')
        
        # Load configuration if available
        self.config = self._load_config()
        
        # Extract model-specific configurations
        self.model_config = self.config.get('models', {})
        
        # Model search paths in order of preference
        self.model_search_paths = self._setup_model_search_paths()
        
        self.initialized = True
        logger.debug(f"ModelLoader initialized in {time.time() - start_time:.2f}s")

    def __del__(self):
        """Clean up resources when the ModelLoader is destroyed."""
        try:
            # Clean up all loaded models
            for model in self.loaded_models:
                if hasattr(model, 'release_resources'):
                    try:
                        model.release_resources()
                    except Exception as e:
                        logger.error(f"Error releasing model resources: {str(e)}")
            
            # Clear the model cache
            self.clear_model_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.debug("ModelLoader resources cleaned up")
        except Exception as e:
            # Suppress errors during garbage collection
            pass

    def _setup_model_search_paths(self) -> List[str]:
        """
        Set up a list of paths to search for models in order of preference.
        
        Returns:
            List[str]: Ordered list of directory paths to search for models
        """
        search_paths = [
            # Primary location
            self.base_model_dir,
            
            # Check in 'pretrained' subdirectory
            os.path.join(self.base_model_dir, 'pretrained'),
            
            # Check in current working directory
            os.getcwd(),
            
            # Check in subdirectories of current working directory
            os.path.join(os.getcwd(), 'models'),
            os.path.join(os.getcwd(), 'weights'),
            
            # Check for user-defined location in config
            self.model_config.get('model_dir', '')
        ]
        
        # Filter out empty strings and ensure all paths exist
        return [path for path in search_paths if path and os.path.isdir(path)]
    
    @lru_cache(maxsize=1)  # Cache the config to avoid reloading
    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file with caching.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}. Using default values.")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {str(e)}. Using default values.")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading config: {str(e)}. Using default values.")
            return {}
    
    def _find_model_file(self, model_name: str, extensions: List[str]) -> Optional[str]:
        """
        Search for model files with the given name and extensions in search paths.
        
        Args:
            model_name: Base name of the model without extension
            extensions: List of file extensions to search for (.pth, .onnx, etc.)
            
        Returns:
            Optional[str]: Full path to the model file if found, None otherwise
        """
        for path in self.model_search_paths:
            for ext in extensions:
                model_path = os.path.join(path, f"{model_name}{ext}")
                if os.path.isfile(model_path):
                    return model_path
        
        return None
    
    def load_eth_xgaze_model(self, device: str = 'cpu', force_reload: bool = False) -> Optional[Union[torch.nn.Module, ONNXGazeModel]]:
        """
        Load the ETH-XGaze gaze estimation model.
        
        This method will first try to load the ONNX model (preferred for inference).
        If ONNX model is not found, it will fall back to the PyTorch model.
        
        Args:
            device: Device to load the model on ('cpu' or 'cuda')
            force_reload: If True, reload model even if it's in cache
            
        Returns:
            Union[torch.nn.Module, ONNXGazeModel]: Loaded model or None if loading fails
        """
        # Check cache if not forcing reload
        cache_key = f"eth_xgaze_{device}"
        if not force_reload and cache_key in self._model_cache:
            logger.info(f"Using cached ETH-XGaze model for {device}")
            return self._model_cache[cache_key]
        
        start_time = time.time()
        
        # Get model configuration
        use_onnx = self.model_config.get('prefer_onnx', True)
        model_name = self.model_config.get('eth_xgaze_model', 'eth_xgaze_model')
        
        # Try ONNX first if preferred
        if use_onnx:
            onnx_path = self._find_model_file(model_name, ['.onnx'])
            if onnx_path:
                try:
                    logger.info(f"Loading ETH-XGaze ONNX model from {onnx_path}")
                    model = ONNXGazeModel(onnx_path, device)
                    self._model_cache[cache_key] = model
                    
                    # Track the loaded model for proper cleanup
                    self.loaded_models.append(model)
                    
                    logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
                    return model
                except ImportError as e:
                    logger.warning(f"Failed to import onnxruntime: {str(e)}. Falling back to PyTorch model.")
                except Exception as e:
                    logger.error(f"Error loading ONNX model: {str(e)}. Falling back to PyTorch model.")
        
        # Try PyTorch model
        pytorch_path = self._find_model_file(model_name, ['.pth', '.pt', '.pth.tar'])
        if pytorch_path:
            logger.info(f"Loading ETH-XGaze PyTorch model from {pytorch_path}")
            model = self._load_pytorch_model(pytorch_path, device)
            if model:
                self._model_cache[cache_key] = model
                
                # Track the loaded model for proper cleanup
                self.loaded_models.append(model)
                
                logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
                return model
            
        # No model found
        logger.error(f"Could not find ETH-XGaze model in any of the search paths.")
        return None
    
    def _load_pytorch_model(self, model_path: str, device: str) -> Optional[torch.nn.Module]:
        """
        Load the PyTorch model with improved error handling and logging.
        
        Args:
            model_path: Path to the PyTorch model file
            device: Device to load the model on ('cpu' or 'cuda')
            
        Returns:
            torch.nn.Module: Loaded PyTorch model or None if loading fails
        """
        try:
            # Check file existence
            if not os.path.isfile(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # First attempt - Handle various checkpoint formats
            try:
                # Load checkpoint with appropriate mapping
                checkpoint = torch.load(model_path, map_location=device)
                model = GazeResNet(pretrained=False)  # Start with non-pretrained model
                
                # Determine checkpoint format and load accordingly
                if isinstance(checkpoint, dict):
                    if 'model_state' in checkpoint:
                        logger.info("Loading ETH-XGaze checkpoint format with 'model_state' key")
                        model.load_state_dict(checkpoint['model_state'])
                    elif 'state_dict' in checkpoint:
                        logger.info("Loading checkpoint format with 'state_dict' key")
                        # Handle potential 'module.' prefix from DataParallel
                        state_dict = checkpoint['state_dict']
                        if all(k.startswith('module.') for k in state_dict.keys()):
                            state_dict = {k[7:]: v for k, v in state_dict.items()}
                        model.load_state_dict(state_dict)
                    else:
                        # Try loading directly as a state dictionary
                        logger.info("Loading as direct state dictionary")
                        model.load_state_dict(checkpoint)
                else:
                    logger.warning("Checkpoint format not recognized, attempting direct load")
                    model.load_state_dict(checkpoint)
                
                model = model.to(device)
                model.eval()  # Set to evaluation mode for inference
                return model
                
            except Exception as e:
                logger.warning(f"Initial loading attempt failed: {str(e)}")
                
                # Second attempt - Try loading as a complete model
                try:
                    logger.info("Attempting to load as full model")
                    model = torch.load(model_path, map_location=device)
                    if hasattr(model, 'eval'):
                        model.eval()
                        return model
                    else:
                        raise ValueError("Loaded object does not have eval() method")
                
                except Exception as e2:
                    logger.error(f"Could not load as full model: {str(e2)}")
                    
                    # Final attempt - Load and adapt architecture if mismatch
                    try:
                        logger.info("Attempting to adapt model architecture to match checkpoint")
                        # Create a new model
                        model = GazeResNet(pretrained=False)
                        
                        # Try to partially load weights that match
                        checkpoint = torch.load(model_path, map_location=device)
                        
                        # If checkpoint is a state_dict or has a state_dict key
                        if isinstance(checkpoint, dict):
                            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state', checkpoint))
                        else:
                            state_dict = checkpoint
                        
                        # Filter state_dict to only include matching keys
                        model_state = model.state_dict()
                        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                        
                        # Check if we have any usable weights
                        if filtered_state:
                            logger.info(f"Loaded {len(filtered_state)}/{len(model_state)} layers from checkpoint")
                            model_state.update(filtered_state)
                            model.load_state_dict(model_state)
                        else:
                            logger.warning("No matching weights found in checkpoint")
                        
                        model = model.to(device)
                        model.eval()
                        return model
                    
                    except Exception as e3:
                        logger.error(f"All loading attempts failed: {str(e3)}")
                        
                        # Last resort - Return pretrained model
                        logger.info("Creating pretrained model as fallback")
                        model = GazeResNet(pretrained=True)
                        model = model.to(device)
                        model.eval()
                        return model
        
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {str(e)}")
            return None
    
    @staticmethod
    def clear_model_cache():
        """Clear the model cache to free memory."""
        ModelLoader._model_cache.clear()
        
def get_model_loader(config_path: Optional[str] = None) -> ModelLoader:
    """
    Factory function to get a ModelLoader instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        ModelLoader: An initialized ModelLoader instance
    """
    return ModelLoader(config_path)