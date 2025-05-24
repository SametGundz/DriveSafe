"""
UI utility functions for handling icons and resources.
"""
import os
from typing import Optional


def get_icon_path(icon_name: str) -> str:
    """
    Get the path to an icon file.
    
    Args:
        icon_name: Name of the icon file (e.g., "upload.png")
        
    Returns:
        Path to the icon file, or empty string if not found
    """
    # Common icon directories to check
    icon_dirs = [
        os.path.join("src", "ui", "icons"),
        os.path.join("assets", "icons"),
        os.path.join("resources", "icons"),
        "icons"
    ]
    
    for icon_dir in icon_dirs:
        icon_path = os.path.join(icon_dir, icon_name)
        if os.path.exists(icon_path):
            return icon_path
    
    # If icon not found, return empty string to avoid crashes
    # The QIcon constructor will handle empty strings gracefully
    return ""


def get_resource_path(resource_name: str) -> str:
    """
    Get the path to a resource file.
    
    Args:
        resource_name: Name of the resource file
        
    Returns:
        Path to the resource file, or empty string if not found
    """
    # Common resource directories to check
    resource_dirs = [
        os.path.join("src", "ui", "resources"),
        os.path.join("assets"),
        os.path.join("resources"),
        "data"
    ]
    
    for resource_dir in resource_dirs:
        resource_path = os.path.join(resource_dir, resource_name)
        if os.path.exists(resource_path):
            return resource_path
    
    # If resource not found, return empty string
    return ""


def create_icon_directory() -> str:
    """
    Create the default icon directory if it doesn't exist.
    
    Returns:
        Path to the created icon directory
    """
    icon_dir = os.path.join("src", "ui", "icons")
    os.makedirs(icon_dir, exist_ok=True)
    return icon_dir


def create_resource_directory() -> str:
    """
    Create the default resource directory if it doesn't exist.
    
    Returns:
        Path to the created resource directory
    """
    resource_dir = os.path.join("src", "ui", "resources")
    os.makedirs(resource_dir, exist_ok=True)
    return resource_dir 