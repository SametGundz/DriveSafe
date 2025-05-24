#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Result exporter module for the driver drowsiness detection application.

This module provides functionality to export video analysis results to JSON format.
"""

import os
import json
import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Get module-specific logger
logger = logging.getLogger(__name__)

class ResultExporter:
    """
    Class for exporting video analysis results to JSON format.
    
    This class provides methods to:
    - Export analysis results to JSON file
    - Create summary reports from frame-by-frame analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ResultExporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        logger.debug("ResultExporter initialized")
    
    def export_results(self, analysis_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Export analysis results to a JSON file.
        
        Args:
            analysis_data: Dictionary containing analysis data
            output_path: Path to save the JSON file (optional)
            
        Returns:
            str: Path to the exported JSON file
        """
        # Create output directory if it doesn't exist
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        else:
            # Create default output path if not provided
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = analysis_data.get("video_info", {}).get("filename", "unknown")
            base_filename = os.path.splitext(os.path.basename(video_filename))[0]
            
            # Create reports directory in the project root
            reports_dir = Path("reports") / "json"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = str(reports_dir / f"{base_filename}_analysis_{timestamp}.json")
        
        try:
            # Write JSON data to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Analysis results exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting analysis results: {str(e)}")
            raise
    
    def create_summary_report(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary report from frame-by-frame analysis results.
        
        Args:
            frame_results: List of dictionaries containing frame analysis results
            
        Returns:
            Dict: Summary report
        """
        if not frame_results:
            return {
                "zone_distribution": {},
                "average_drowsiness": 0.0,
                "total_analysis_time": 0.0
            }
        
        # Calculate zone distribution
        zone_counts = {}
        for frame in frame_results:
            zone_id = frame.get("zone_id")
            if zone_id:
                zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
        
        total_frames = len(frame_results)
        zone_distribution = {zone: (count / total_frames) * 100 for zone, count in zone_counts.items()}
        
        # Calculate average drowsiness
        drowsiness_values = [frame.get("drowsiness_level", 0.0) for frame in frame_results if "drowsiness_level" in frame]
        average_drowsiness = sum(drowsiness_values) / len(drowsiness_values) if drowsiness_values else 0.0
        
        # Calculate total analysis time
        start_time = frame_results[0].get("timestamp", 0.0) if frame_results else 0.0
        end_time = frame_results[-1].get("timestamp", 0.0) if frame_results else 0.0
        total_analysis_time = end_time - start_time
        
        # Create additional statistics
        face_detected_frames = sum(1 for frame in frame_results if frame.get("face_detected", False))
        face_detection_rate = (face_detected_frames / total_frames) * 100 if total_frames > 0 else 0.0
        
        # Calculate EAR statistics
        ear_values = [frame.get("ear", 0.0) for frame in frame_results if "ear" in frame]
        avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0.0
        min_ear = min(ear_values) if ear_values else 0.0
        max_ear = max(ear_values) if ear_values else 0.0
        
        # Calculate MAR statistics
        mar_values = [frame.get("mar", 0.0) for frame in frame_results if "mar" in frame]
        avg_mar = sum(mar_values) / len(mar_values) if mar_values else 0.0
        min_mar = min(mar_values) if mar_values else 0.0
        max_mar = max(mar_values) if mar_values else 0.0
        
        return {
            "zone_distribution": zone_distribution,
            "average_drowsiness": average_drowsiness,
            "total_analysis_time": total_analysis_time,
            "face_detection_rate": face_detection_rate,
            "ear_statistics": {
                "average": avg_ear,
                "minimum": min_ear,
                "maximum": max_ear
            },
            "mar_statistics": {
                "average": avg_mar,
                "minimum": min_mar,
                "maximum": max_mar
            }
        }
    
    def format_analysis_data(self, video_path: str, frame_results: List[Dict[str, Any]], statistics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format analysis data into the required JSON structure.
        
        Args:
            video_path: Path to the analyzed video
            frame_results: List of dictionaries containing frame analysis results
            statistics: Dictionary containing analysis statistics
            
        Returns:
            Dict: Formatted analysis data
        """
        # Get video information
        import cv2
        video_info = {}
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0.0
                
                video_info = {
                    "filename": os.path.basename(video_path),
                    "total_frames": frame_count,
                    "fps": fps,
                    "duration": duration,
                    "analysis_date": datetime.datetime.now().isoformat()
                }
                
                cap.release()
        except Exception as e:
            logger.error(f"Error getting video information: {str(e)}")
            # Use default values if video info cannot be obtained
            video_info = {
                "filename": os.path.basename(video_path),
                "total_frames": len(frame_results),
                "fps": 30.0,  # Default assumption
                "duration": 0.0,
                "analysis_date": datetime.datetime.now().isoformat()
            }
        
        # Create summary if not provided
        if not statistics:
            statistics = self.create_summary_report(frame_results)
        
        # Format the data
        analysis_data = {
            "video_info": video_info,
            "frame_analysis": frame_results,
            "summary": statistics
        }
        
        return analysis_data


def get_result_exporter(config: Dict[str, Any] = None) -> ResultExporter:
    """
    Factory function to get a ResultExporter instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ResultExporter: An instance of ResultExporter
    """
    return ResultExporter(config) 