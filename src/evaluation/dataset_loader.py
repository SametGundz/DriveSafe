"""
Dataset loader for drowsiness detection evaluation.

This module provides functionality to load and prepare various benchmark datasets
for evaluating drowsiness detection algorithms.
"""

import os
import glob
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for loading drowsiness detection benchmark datasets."""
    
    def __init__(self, dataset_path, cache_dir=None):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the dataset directory
            cache_dir: Path to cache preprocessed data (optional)
        """
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        if self.cache_dir and not self.cache_dir.exists():
            os.makedirs(self.cache_dir, exist_ok=True)
            
    def load_metadata(self):
        """
        Load dataset metadata.
        
        Returns:
            dict: Dataset metadata
        """
        raise NotImplementedError("Subclasses must implement load_metadata")
        
    def load_data(self, split='train', subset=None):
        """
        Load dataset samples.
        
        Args:
            split: Data split ('train', 'validation', 'test')
            subset: Optional subset name or ID
            
        Returns:
            tuple: (X, y) data samples and labels
        """
        raise NotImplementedError("Subclasses must implement load_data")
        
    def get_splits(self):
        """
        Get available dataset splits.
        
        Returns:
            list: Available data splits
        """
        raise NotImplementedError("Subclasses must implement get_splits")


class NTHUDDDDataset(DatasetLoader):
    """
    Loader for the NTHU Driver Drowsiness Detection Dataset.
    
    Reference: https://www.ee.cuhk.edu.hk/~xgwang/lab/dataset.html
    """
    
    def __init__(self, dataset_path, cache_dir=None):
        super().__init__(dataset_path, cache_dir)
        self._metadata = None
        
    def load_metadata(self):
        """Load NTHU-DDD dataset metadata."""
        if self._metadata is not None:
            return self._metadata
            
        metadata_file = self.dataset_path / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self._metadata = json.load(f)
        else:
            # Build metadata by scanning directory structure
            self._metadata = self._build_metadata()
            
        return self._metadata
    
    def _build_metadata(self):
        """Build metadata by scanning the dataset directory structure."""
        metadata = {
            "subjects": [],
            "conditions": ["glasses", "no_glasses"],
            "states": ["drowsy", "alert"],
            "splits": {}
        }
        
        # Find all subject directories
        subject_dirs = [d for d in self.dataset_path.glob("*") if d.is_dir() and not d.name.startswith(".")]
        metadata["subjects"] = [d.name for d in subject_dirs]
        
        # Assign default splits (80% train, 10% validation, 10% test)
        num_subjects = len(metadata["subjects"])
        train_size = int(0.8 * num_subjects)
        val_size = int(0.1 * num_subjects)
        
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(num_subjects)
        
        metadata["splits"]["train"] = [metadata["subjects"][i] for i in indices[:train_size]]
        metadata["splits"]["validation"] = [metadata["subjects"][i] for i in indices[train_size:train_size+val_size]]
        metadata["splits"]["test"] = [metadata["subjects"][i] for i in indices[train_size+val_size:]]
        
        return metadata
    
    def get_splits(self):
        """Get available dataset splits."""
        metadata = self.load_metadata()
        return list(metadata["splits"].keys())
    
    def load_data(self, split='train', subset=None):
        """
        Load NTHU-DDD dataset samples.
        
        Args:
            split: Data split ('train', 'validation', 'test')
            subset: Optional subset (condition/state combination)
            
        Returns:
            tuple: (X, y, metadata) data samples, labels, and sample metadata
        """
        metadata = self.load_metadata()
        
        if split not in metadata["splits"]:
            raise ValueError(f"Invalid split: {split}. Available splits: {list(metadata['splits'].keys())}")
        
        subjects = metadata["splits"][split]
        
        # Parse subset if provided (e.g., "glasses_drowsy")
        condition = None
        state = None
        
        if subset:
            parts = subset.split("_")
            if len(parts) == 2:
                condition, state = parts
                
                if condition not in metadata["conditions"]:
                    raise ValueError(f"Invalid condition: {condition}. Available: {metadata['conditions']}")
                    
                if state not in metadata["states"]:
                    raise ValueError(f"Invalid state: {state}. Available: {metadata['states']}")
        
        # Collect image paths and labels
        X = []  # Image paths
        y = []  # Labels (0: alert, 1: drowsy)
        sample_metadata = []
        
        for subject in subjects:
            subject_dir = self.dataset_path / subject
            
            for cond in [condition] if condition else metadata["conditions"]:
                for st in [state] if state else metadata["states"]:
                    data_dir = subject_dir / cond / st
                    
                    if not data_dir.exists():
                        continue
                        
                    # Get all image files
                    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
                    
                    for img_path in image_files:
                        X.append(str(img_path))
                        y.append(1 if st == "drowsy" else 0)
                        sample_metadata.append({
                            "subject": subject,
                            "condition": cond,
                            "state": st,
                            "file": img_path.name
                        })
        
        return X, np.array(y), sample_metadata


class UTA_REALDataset(DatasetLoader):
    """
    Loader for the UTA-RLDD (Real-Life Drowsiness Dataset).
    
    Reference: https://sites.google.com/view/utarldd/home
    """
    
    def __init__(self, dataset_path, cache_dir=None):
        super().__init__(dataset_path, cache_dir)
        self._metadata = None
        
    def load_metadata(self):
        """Load UTA-RLDD dataset metadata."""
        if self._metadata is not None:
            return self._metadata
            
        metadata_file = self.dataset_path / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self._metadata = json.load(f)
        else:
            # Build metadata by scanning directory structure
            self._metadata = self._build_metadata()
            
        return self._metadata
    
    def _build_metadata(self):
        """Build metadata by scanning the dataset directory structure."""
        metadata = {
            "subjects": [],
            "states": ["alert", "low_vigilance", "drowsy"],
            "splits": {}
        }
        
        # The dataset is organized as dataset_path/state/subject_id.mp4
        for state in metadata["states"]:
            state_dir = self.dataset_path / state
            
            if not state_dir.exists():
                logger.warning(f"State directory not found: {state_dir}")
                continue
                
            video_files = list(state_dir.glob("*.mp4"))
            
            for video_file in video_files:
                subject_id = video_file.stem
                if subject_id not in metadata["subjects"]:
                    metadata["subjects"].append(subject_id)
        
        # Assign default splits (70% train, 15% validation, 15% test)
        num_subjects = len(metadata["subjects"])
        train_size = int(0.7 * num_subjects)
        val_size = int(0.15 * num_subjects)
        
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(num_subjects)
        
        metadata["splits"]["train"] = [metadata["subjects"][i] for i in indices[:train_size]]
        metadata["splits"]["validation"] = [metadata["subjects"][i] for i in indices[train_size:train_size+val_size]]
        metadata["splits"]["test"] = [metadata["subjects"][i] for i in indices[train_size+val_size:]]
        
        return metadata
    
    def get_splits(self):
        """Get available dataset splits."""
        metadata = self.load_metadata()
        return list(metadata["splits"].keys())
    
    def load_data(self, split='train', subset=None, frame_interval=30):
        """
        Load UTA-RLDD dataset samples.
        
        Args:
            split: Data split ('train', 'validation', 'test')
            subset: Optional state subset ('alert', 'low_vigilance', 'drowsy')
            frame_interval: Extract one frame every N frames
            
        Returns:
            tuple: (X, y, metadata) data samples, labels, and sample metadata
        """
        metadata = self.load_metadata()
        
        if split not in metadata["splits"]:
            raise ValueError(f"Invalid split: {split}. Available splits: {list(metadata['splits'].keys())}")
        
        subjects = metadata["splits"][split]
        
        # Check if subset is valid
        if subset and subset not in metadata["states"]:
            raise ValueError(f"Invalid state: {subset}. Available: {metadata['states']}")
        
        # Use cache if available
        cache_key = f"uta_rldd_{split}_{subset}_{frame_interval}"
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.npz"
            if cache_file.exists():
                data = np.load(cache_file, allow_pickle=True)
                return list(data['X']), data['y'], data['metadata'].tolist()
        
        # Collect frames and labels
        X = []  # Frame data
        y = []  # Labels (0: alert, 1: low vigilance, 2: drowsy)
        sample_metadata = []
        
        states_to_load = [subset] if subset else metadata["states"]
        state_to_label = {
            "alert": 0,
            "low_vigilance": 1,
            "drowsy": 2
        }
        
        for state in states_to_load:
            state_dir = self.dataset_path / state
            
            if not state_dir.exists():
                continue
                
            for subject in subjects:
                video_path = state_dir / f"{subject}.mp4"
                
                if not video_path.exists():
                    continue
                    
                # Extract frames from video
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                frame_idx = 0
                while cap.isOpened() and frame_idx < frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                        
                    # Save frame data
                    X.append(frame)
                    y.append(state_to_label[state])
                    sample_metadata.append({
                        "subject": subject,
                        "state": state,
                        "video": video_path.name,
                        "frame": frame_idx
                    })
                    
                    frame_idx += frame_interval
                
                cap.release()
        
        # Save to cache if enabled
        if self.cache_dir:
            np.savez(
                self.cache_dir / f"{cache_key}.npz",
                X=np.array(X, dtype=object),
                y=np.array(y),
                metadata=np.array(sample_metadata, dtype=object)
            )
        
        return X, np.array(y), sample_metadata


class CustomDataset(DatasetLoader):
    """
    Loader for custom datasets with predefined structure.
    
    Expected directory structure:
    dataset_path/
        train/
            alert/
                *.jpg
            drowsy/
                *.jpg
        validation/
            alert/
                *.jpg
            drowsy/
                *.jpg
        test/
            alert/
                *.jpg
            drowsy/
                *.jpg
    """
    
    def __init__(self, dataset_path, cache_dir=None):
        super().__init__(dataset_path, cache_dir)
        self._metadata = None
        
    def load_metadata(self):
        """Load custom dataset metadata."""
        if self._metadata is not None:
            return self._metadata
            
        metadata = {
            "splits": [],
            "states": []
        }
        
        # Find all splits
        for item in self.dataset_path.glob("*"):
            if item.is_dir() and not item.name.startswith("."):
                metadata["splits"].append(item.name)
        
        # Find all states (assuming they're consistent across splits)
        for item in (self.dataset_path / metadata["splits"][0]).glob("*"):
            if item.is_dir() and not item.name.startswith("."):
                metadata["states"].append(item.name)
        
        self._metadata = metadata
        return metadata
    
    def get_splits(self):
        """Get available dataset splits."""
        metadata = self.load_metadata()
        return metadata["splits"]
    
    def load_data(self, split='train', subset=None):
        """
        Load custom dataset samples.
        
        Args:
            split: Data split (directory name)
            subset: Optional state subset (subdirectory name)
            
        Returns:
            tuple: (X, y, metadata) data samples, labels, and sample metadata
        """
        metadata = self.load_metadata()
        
        if split not in metadata["splits"]:
            raise ValueError(f"Invalid split: {split}. Available splits: {metadata['splits']}")
        
        # Check if subset is valid
        if subset and subset not in metadata["states"]:
            raise ValueError(f"Invalid state: {subset}. Available: {metadata['states']}")
        
        # Collect image paths and labels
        X = []  # Image paths
        y = []  # Labels
        sample_metadata = []
        
        states_to_load = [subset] if subset else metadata["states"]
        
        for i, state in enumerate(states_to_load):
            state_dir = self.dataset_path / split / state
            
            if not state_dir.exists():
                continue
                
            # Get all image files
            image_files = list(state_dir.glob("*.jpg")) + list(state_dir.glob("*.png"))
            
            for img_path in image_files:
                X.append(str(img_path))
                y.append(i)  # Use index as label
                sample_metadata.append({
                    "split": split,
                    "state": state,
                    "file": img_path.name
                })
        
        return X, np.array(y), sample_metadata


def get_dataset_loader(dataset_name, dataset_path, cache_dir=None):
    """
    Factory function to get the appropriate dataset loader.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset directory
        cache_dir: Path to cache preprocessed data (optional)
        
    Returns:
        DatasetLoader: Dataset loader instance
    """
    dataset_loaders = {
        "nthu-ddd": NTHUDDDDataset,
        "uta-rldd": UTA_REALDataset,
        "custom": CustomDataset
    }
    
    if dataset_name.lower() not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_loaders.keys())}")
    
    return dataset_loaders[dataset_name.lower()](dataset_path, cache_dir) 


def load_benchmark_dataset(dataset_path, split='test', dataset_type=None, subset=None, cache_dir=None, load_images=True, preprocess_fn=None):
    """
    Load image-label pairs from a dataset for benchmarking.
    
    This function provides a convenient interface to load data from different 
    dataset types in a consistent format suitable for benchmarking algorithms.
    
    Args:
        dataset_path (str or Path): Path to the dataset directory
        split (str): Data split to load ('train', 'validation', 'test')
        dataset_type (str, optional): Type of dataset ('nthu-ddd', 'uta-rldd', 'custom', etc.)
            If None, will try to detect the dataset type from directory structure
        subset (str, optional): Subset of the dataset to load
        cache_dir (str or Path, optional): Directory to cache preprocessed data
        load_images (bool): Whether to load image data or just return paths
        preprocess_fn (callable, optional): Function to preprocess loaded images
            Should take an image (numpy array) as input and return processed image
    
    Returns:
        tuple: (images, labels, metadata)
            - images: List of image arrays (if load_images=True) or image paths
            - labels: Array of corresponding labels
            - metadata: List of metadata dictionaries for each sample
    
    Raises:
        ValueError: If dataset_type is not recognized or cannot be detected
        FileNotFoundError: If dataset_path does not exist
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Detect dataset type if not provided
    if dataset_type is None:
        dataset_type = _detect_dataset_type(dataset_path)
        logger.info(f"Detected dataset type: {dataset_type}")
    
    # Get appropriate dataset loader
    loader = get_dataset_loader(dataset_type, dataset_path, cache_dir)
    
    # Load data
    data, labels, metadata = loader.load_data(split=split, subset=subset)
    
    # Process images if requested
    if load_images:
        images = []
        for item in data:
            if isinstance(item, str):  # Item is a file path
                try:
                    img = cv2.imread(item)
                    if img is None:
                        logger.warning(f"Failed to load image: {item}")
                        continue
                        
                    # Convert from BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Apply preprocessing if provided
                    if preprocess_fn is not None:
                        img = preprocess_fn(img)
                        
                    images.append(img)
                except Exception as e:
                    logger.error(f"Error loading image {item}: {str(e)}")
            else:  # Item is already an image
                if preprocess_fn is not None:
                    item = preprocess_fn(item)
                images.append(item)
                
        return images, labels, metadata
    else:
        # Return paths instead of loaded images
        return data, labels, metadata


def _detect_dataset_type(dataset_path):
    """
    Attempt to detect the dataset type based on directory structure.
    
    Args:
        dataset_path (Path): Path to the dataset directory
        
    Returns:
        str: Detected dataset type
        
    Raises:
        ValueError: If dataset type cannot be determined
    """
    # Check for NTHU-DDD structure
    has_subjects = False
    has_glasses_folder = False
    
    for item in dataset_path.glob("*"):
        if item.is_dir() and not item.name.startswith("."):
            has_subjects = True
            # Check for glasses/no_glasses subdirectories
            for subdir in item.glob("*"):
                if subdir.is_dir() and subdir.name in ["glasses", "no_glasses"]:
                    has_glasses_folder = True
                    break
    
    if has_subjects and has_glasses_folder:
        return "nthu-ddd"
    
    # Check for UTA-RLDD structure
    has_alert = dataset_path.joinpath("alert").is_dir()
    has_drowsy = dataset_path.joinpath("drowsy").is_dir()
    has_low_vigilance = dataset_path.joinpath("low_vigilance").is_dir()
    
    if has_alert and has_drowsy and has_low_vigilance:
        return "uta-rldd"
    
    # Check for custom dataset structure (train/test/val splits with class subdirectories)
    has_train = dataset_path.joinpath("train").is_dir()
    has_test = dataset_path.joinpath("test").is_dir()
    
    if has_train and has_test:
        return "custom"
    
    # If we get here, we couldn't determine the dataset type
    raise ValueError(
        "Could not auto-detect dataset type. Please specify the dataset_type parameter."
    ) 