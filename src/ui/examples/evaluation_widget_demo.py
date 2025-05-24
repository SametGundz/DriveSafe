#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script for the EvaluationWidget.

This script demonstrates how to use the EvaluationWidget in a standalone application.
It creates sample ground truth and system output files if they don't exist,
and showcases the advanced visualization features of the evaluation module.
"""

import sys
import os
import json
import random
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
    QLabel, QHBoxLayout, QTabWidget, QSplitter, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.ui.evaluation_widget import EvaluationWidget


class DemoWindow(QMainWindow):
    """Demo window for the EvaluationWidget."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Gaze Zone Evaluation Demo")
        self.setGeometry(100, 100, 1200, 900)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Add header
        header_layout = QHBoxLayout()
        title_label = QLabel("Driver Drowsiness Detection - Gaze Zone Evaluation Demo")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        main_layout.addLayout(header_layout)
        
        # Add description
        description_label = QLabel(
            "This demo showcases the EvaluationWidget with advanced visualization features. "
            "It automatically generates sample data for evaluation and displays various metrics "
            "and visualizations, including confusion matrices, per-zone performance charts, and error analysis."
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet("font-size: 12px; margin-bottom: 10px;")
        main_layout.addWidget(description_label)
        
        # Create sample files if they don't exist
        sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
        os.makedirs(sample_dir, exist_ok=True)
        
        gt_path = os.path.join(sample_dir, 'ground_truth.json')
        sys_path = os.path.join(sample_dir, 'system_output.json')
        
        if not os.path.exists(gt_path) or not os.path.exists(sys_path):
            self.create_sample_files(gt_path, sys_path)
        
        # Create configuration for the evaluation widget
        config = {
            "title": "Gaze Zone Evaluation",
            "fonts": {"family": "Arial", "size": 10},
            "controls": {"button_width": 150, "button_height": 40},
            "colors": {"primary": "#007aff", "secondary": "#5ac8fa"}
        }
        
        # Create evaluation widget
        self.eval_widget = EvaluationWidget(config)
        main_layout.addWidget(self.eval_widget)
        
        # Connect signals
        self.eval_widget.evaluation_completed.connect(self.on_evaluation_complete)
        
        # Add action buttons at the bottom
        action_layout = QHBoxLayout()
        
        # Create new sample data button
        new_data_btn = QPushButton("Generate New Sample Data")
        new_data_btn.setMinimumHeight(40)
        new_data_btn.clicked.connect(self.generate_new_data)
        action_layout.addWidget(new_data_btn)
        
        # Add export visualizations button
        export_btn = QPushButton("Export All Visualizations")
        export_btn.setMinimumHeight(40)
        export_btn.clicked.connect(self.export_visualizations)
        action_layout.addWidget(export_btn)
        
        main_layout.addLayout(action_layout)
        
        # Pre-fill file paths for demo
        self.eval_widget.system_path_label.setText(sys_path)
        self.eval_widget.system_output_path = sys_path
        
        self.eval_widget.ground_truth_path_label.setText(gt_path)
        self.eval_widget.ground_truth_path = gt_path
        
        # Enable evaluation button
        self.eval_widget.evaluate_button.setEnabled(True)
        
        # Automatically start evaluation
        self.eval_widget.evaluate_button.click()
    
    def create_sample_files(self, gt_path, sys_path, error_rate=0.2):
        """Create sample ground truth and system output files."""
        print("Creating sample files...")
        
        # Parameters for sample data
        num_frames = 1000
        num_zones = 6
        
        # Create ground truth data
        # Format: {"0": 3, "1": 3, "2": 3, ...}
        ground_truth = {}
        
        # Generate sequence of zones with realistic transitions
        current_zone = random.randint(0, num_zones - 1)
        for i in range(num_frames):
            # 90% chance to stay in the same zone, 10% chance to change
            if random.random() < 0.9 and i > 0:
                ground_truth[str(i)] = current_zone
            else:
                # Change zone with preference for adjacent zones
                new_zone = current_zone
                while new_zone == current_zone:
                    # Prefer adjacent zones (with wrapping)
                    if random.random() < 0.7:
                        new_zone = (current_zone + random.choice([-1, 1])) % num_zones
                    else:
                        new_zone = random.randint(0, num_zones - 1)
                current_zone = new_zone
                ground_truth[str(i)] = current_zone
        
        # Create system output data
        # Format: {"frame_analysis": [{"frame_number": 0, "gaze": {"zone_id": 0}}, ...]}
        system_output = {"frame_analysis": []}
        
        for frame_num, zone_id in ground_truth.items():
            # Introduce errors based on error rate
            if random.random() < error_rate:
                # Make error, but with bias towards adjacent zones
                if random.random() < 0.8:
                    # Adjacent zone error
                    predicted_zone = (zone_id + random.choice([-1, 1])) % num_zones
                else:
                    # Random zone error
                    predicted_zone = random.randint(0, num_zones - 1)
                    while predicted_zone == zone_id:
                        predicted_zone = random.randint(0, num_zones - 1)
            else:
                predicted_zone = zone_id
            
            # Add to system output
            system_output["frame_analysis"].append({
                "frame_number": int(frame_num),
                "gaze": {"zone_id": predicted_zone}
            })
        
        # Save files
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        
        with open(gt_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        with open(sys_path, 'w') as f:
            json.dump(system_output, f, indent=2)
        
        print(f"Sample files created at {os.path.dirname(gt_path)}")
        print(f"- Ground truth: {os.path.basename(gt_path)}")
        print(f"- System output: {os.path.basename(sys_path)}")
        print(f"- Number of frames: {num_frames}")
        print(f"- Number of zones: {num_zones}")
        print(f"- Error rate: {error_rate:.2f}")
        
        return gt_path, sys_path
    
    def generate_new_data(self):
        """Generate new sample data with random error rate."""
        # Select a random error rate between 0.1 and 0.4
        error_rate = random.uniform(0.1, 0.4)
        
        # Get existing file paths
        sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
        gt_path = os.path.join(sample_dir, 'ground_truth.json')
        sys_path = os.path.join(sample_dir, 'system_output.json')
        
        # Create new sample files
        self.create_sample_files(gt_path, sys_path, error_rate)
        
        # Refresh the evaluation widget
        self.eval_widget.system_path_label.setText(sys_path)
        self.eval_widget.system_output_path = sys_path
        
        self.eval_widget.ground_truth_path_label.setText(gt_path)
        self.eval_widget.ground_truth_path = gt_path
        
        # Run evaluation again
        self.eval_widget.evaluate_button.click()
        
        # Show message
        QMessageBox.information(
            self, "New Data Generated", 
            f"New sample data generated with error rate: {error_rate:.2f}\n"
            f"Evaluation is running with the new data."
        )
    
    def export_visualizations(self):
        """Export all visualizations to a directory."""
        if not self.eval_widget.evaluation_results:
            QMessageBox.warning(self, "No Data", "Please run evaluation first.")
            return
        
        # Ask user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Visualizations"
        )
        
        if not output_dir:
            return
        
        # Generate and save visualizations
        self.eval_widget.generate_detailed_report()
        
        # Show message
        QMessageBox.information(
            self, "Export Complete", 
            f"Visualizations have been exported to:\n{output_dir}"
        )
    
    def on_evaluation_complete(self, results):
        """Handle evaluation completion."""
        print("\n=== Evaluation Results ===")
        print(f"Accuracy: {results['summary']['accuracy']:.4f}")
        
        # Print advanced metrics if available
        if "macro_f1" in results["summary"]:
            print(f"Macro F1: {results['summary']['macro_f1']:.4f}")
        if "weighted_f1" in results["summary"]:
            print(f"Weighted F1: {results['summary']['weighted_f1']:.4f}")
        if "cohen_kappa" in results["summary"]:
            print(f"Cohen's Kappa: {results['summary']['cohen_kappa']:.4f}")
        
        # Print error analysis
        if "error_analysis" in results:
            error = results["error_analysis"]
            print(f"\nTotal frames: {error.get('total_samples', 0)}")
            print(f"Correct predictions: {error.get('correct_predictions', 0)}")
            print(f"Total errors: {error.get('total_errors', 0)}")
            print(f"Error rate: {error.get('error_rate', 0):.4f}")
            
            # Print most confused pairs
            if "confused_pairs" in error and error["confused_pairs"]:
                print("\nTop 3 most confused zone pairs:")
                for i, pair in enumerate(error["confused_pairs"][:3]):
                    print(f"  {i+1}. True zone {pair['true_zone']} → Predicted zone {pair['predicted_zone']}: {pair['count']} times ({pair['percentage']:.2f}%)")
        
        # Print visualization information
        print("\nVisualization Features Available:")
        print("1. Confusion Matrix - Shows the counts/percentages of true vs. predicted zones")
        print("2. Zone Performance - Displays F1-Score, Precision, and Recall for each zone")
        print("3. Error Analysis - Lists the most confused zone pairs and error statistics")
        print("4. Statistical Tests - Shows results from McNemar's test for significance")
        print("\nUse the 'Generate Detailed Report' button to export all visualizations")
        print("===========================")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec()) 