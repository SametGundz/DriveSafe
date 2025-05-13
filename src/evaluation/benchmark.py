"""
Benchmark module for evaluating drowsiness detection methods.

This module provides functionality to evaluate and compare multiple drowsiness
detection methods on standard benchmark datasets.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import csv

from src.evaluation.metrics import (
    calculate_basic_metrics,
    calculate_confusion_matrix,
    plot_confusion_matrix,
    calculate_roc_auc,
    plot_roc_curve,
    calculate_detection_latency,
    calculate_false_alarm_rate,
    calculate_metrics
)
from src.evaluation.dataset_loader import get_dataset_loader, load_benchmark_dataset
from src.evaluation.visualization import (
    plot_metric_comparison, 
    plot_metrics_by_dataset, 
    plot_roc_curves,
    plot_processing_time_comparison,
    plot_metric_vs_time,
    plot_false_alarm_comparison,
    plot_detection_latency_comparison,
    generate_visualization_set
)

# Configure logging
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark runner for evaluating drowsiness detection methods."""
    
    def __init__(self, output_dir="results/benchmarks", cache_dir="results/cache"):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            cache_dir: Directory to cache intermediate results
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.results = {}
        
    def register_method(self, method_name, method_fn, method_params=None):
        """
        Register a drowsiness detection method for benchmarking.
        
        Args:
            method_name: Name of the method
            method_fn: Method function that takes input data and returns predictions
            method_params: Optional parameters for the method
            
        Returns:
            self: For method chaining
        """
        if method_name in self.results:
            logger.warning(f"Method '{method_name}' already registered. Overwriting.")
            
        self.results[method_name] = {
            "function": method_fn,
            "params": method_params or {},
            "metrics": {},
            "datasets": {}
        }
        
        return self
    
    def evaluate_on_dataset(self, dataset_name, dataset_path, methods=None, splits=None, timeout=None):
        """
        Evaluate registered methods on a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset
            methods: List of method names to evaluate (default: all registered methods)
            splits: List of data splits to evaluate on (default: all available splits)
            timeout: Maximum execution time per method in seconds (default: no limit)
            
        Returns:
            dict: Benchmark results
        """
        logger.info(f"Evaluating methods on dataset: {dataset_name}")
        
        # Load dataset
        dataset = get_dataset_loader(dataset_name, dataset_path, self.cache_dir)
        
        # If no methods specified, use all registered methods
        if methods is None:
            methods = list(self.results.keys())
            
        # If no splits specified, use all available splits
        if splits is None:
            splits = dataset.get_splits()
            
        # Initialize results for this dataset if not already done
        for method_name in methods:
            if method_name not in self.results:
                raise ValueError(f"Method '{method_name}' not registered")
                
            if dataset_name not in self.results[method_name]["datasets"]:
                self.results[method_name]["datasets"][dataset_name] = {
                    "splits": {},
                    "aggregate_metrics": {}
                }
        
        # Evaluate each method on each split
        for split in splits:
            logger.info(f"Evaluating on split: {split}")
            
            # Load data for this split
            X, y_true, metadata = dataset.load_data(split=split)
            
            for method_name in methods:
                logger.info(f"Evaluating method: {method_name}")
                
                method_fn = self.results[method_name]["function"]
                method_params = self.results[method_name]["params"]
                
                # Measure execution time and handle timeout
                start_time = time.time()
                
                try:
                    # Run method with timeout if specified
                    if timeout:
                        with ProcessPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(method_fn, X, **method_params)
                            y_pred = future.result(timeout=timeout)
                    else:
                        y_pred = method_fn(X, **method_params)
                        
                    execution_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = calculate_basic_metrics(y_true, y_pred)
                    
                    # Add execution time
                    metrics["execution_time"] = execution_time
                    metrics["samples_per_second"] = len(X) / execution_time
                    
                    # Calculate confusion matrix
                    cm = calculate_confusion_matrix(y_true, y_pred)
                    
                    # Save results for this method and split
                    self.results[method_name]["datasets"][dataset_name]["splits"][split] = {
                        "metrics": metrics,
                        "confusion_matrix": cm.tolist(),
                        "predictions": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
                        "ground_truth": y_true.tolist()
                    }
                    
                except Exception as e:
                    logger.error(f"Error evaluating method '{method_name}' on split '{split}': {str(e)}")
                    self.results[method_name]["datasets"][dataset_name]["splits"][split] = {
                        "metrics": {"error": str(e)},
                        "status": "failed"
                    }
        
        # Calculate aggregate metrics across all splits
        for method_name in methods:
            aggregate_metrics = self._calculate_aggregate_metrics(
                self.results[method_name]["datasets"][dataset_name]["splits"]
            )
            self.results[method_name]["datasets"][dataset_name]["aggregate_metrics"] = aggregate_metrics
        
        return self.results
    
    def evaluate_all(self, datasets, methods=None, splits=None, timeout=None):
        """
        Evaluate all registered methods on multiple datasets.
        
        Args:
            datasets: Dict mapping dataset names to paths
            methods: List of method names to evaluate (default: all registered methods)
            splits: List of data splits to evaluate on (default: all available splits)
            timeout: Maximum execution time per method in seconds
            
        Returns:
            dict: Benchmark results
        """
        for dataset_name, dataset_path in datasets.items():
            self.evaluate_on_dataset(dataset_name, dataset_path, methods, splits, timeout)
            
        # Calculate overall aggregate metrics across all datasets
        for method_name in self.results:
            overall_metrics = {}
            
            for metric_name in ["accuracy", "precision", "recall", "f1_score", "execution_time", "samples_per_second"]:
                values = []
                
                for dataset_name in self.results[method_name]["datasets"]:
                    if metric_name in self.results[method_name]["datasets"][dataset_name]["aggregate_metrics"]:
                        values.append(self.results[method_name]["datasets"][dataset_name]["aggregate_metrics"][metric_name])
                
                if values:
                    overall_metrics[metric_name] = sum(values) / len(values)
            
            self.results[method_name]["metrics"] = overall_metrics
            
        return self.results
    
    def _calculate_aggregate_metrics(self, split_results):
        """
        Calculate aggregate metrics across multiple splits.
        
        Args:
            split_results: Results for each split
            
        Returns:
            dict: Aggregate metrics
        """
        aggregate_metrics = {}
        
        # Metrics to aggregate
        metrics_to_aggregate = ["accuracy", "precision", "recall", "f1_score", "execution_time", "samples_per_second"]
        
        for metric_name in metrics_to_aggregate:
            values = []
            
            for split_name, split_data in split_results.items():
                if "metrics" in split_data and metric_name in split_data["metrics"]:
                    values.append(split_data["metrics"][metric_name])
            
            if values:
                aggregate_metrics[metric_name] = sum(values) / len(values)
        
        return aggregate_metrics
    
    def save_results(self, filename=None):
        """
        Save benchmark results to a file.
        
        Args:
            filename: Name of the file to save results to
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Convert to serializable format
        serializable_results = {}
        
        for method_name, method_data in self.results.items():
            serializable_results[method_name] = {
                "params": method_data["params"],
                "metrics": method_data["metrics"],
                "datasets": method_data["datasets"]
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Benchmark results saved to: {output_path}")
        
        return output_path
    
    def load_results(self, filepath):
        """
        Load benchmark results from a file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            self: For method chaining
        """
        with open(filepath, 'r') as f:
            loaded_results = json.load(f)
            
        # Update results, preserving method functions
        for method_name, method_data in loaded_results.items():
            if method_name in self.results:
                method_fn = self.results[method_name]["function"]
            else:
                method_fn = None
                
            self.results[method_name] = {
                "function": method_fn,
                "params": method_data["params"],
                "metrics": method_data["metrics"],
                "datasets": method_data["datasets"]
            }
            
        return self
    
    def generate_report(self, output_file=None):
        """
        Generate a comprehensive benchmarking report.
        
        Args:
            output_file: Path to save the report (default: auto-generated name)
            
        Returns:
            str: Path to the generated report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"benchmark_report_{timestamp}.html"
        else:
            output_file = Path(output_file)
            
        # Create basic HTML report
        html_content = []
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html><head>")
        html_content.append("<title>Drowsiness Detection Benchmark Report</title>")
        html_content.append("<style>")
        html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_content.append("table { border-collapse: collapse; width: 100%; }")
        html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_content.append("th { background-color: #f2f2f2; }")
        html_content.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        html_content.append("h1, h2, h3 { color: #333; }")
        html_content.append(".method { margin-bottom: 30px; }")
        html_content.append(".dataset { margin-bottom: 20px; }")
        html_content.append("</style>")
        html_content.append("</head><body>")
        
        # Header
        html_content.append("<h1>Drowsiness Detection Benchmark Report</h1>")
        html_content.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary table
        html_content.append("<h2>Performance Summary</h2>")
        html_content.append("<table>")
        html_content.append("<tr><th>Method</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Exec. Time (s)</th><th>Samples/sec</th></tr>")
        
        for method_name, method_data in self.results.items():
            metrics = method_data["metrics"]
            html_content.append("<tr>")
            html_content.append(f"<td>{method_name}</td>")
            html_content.append(f"<td>{metrics.get('accuracy', 'N/A'):.4f}</td>")
            html_content.append(f"<td>{metrics.get('precision', 'N/A'):.4f}</td>")
            html_content.append(f"<td>{metrics.get('recall', 'N/A'):.4f}</td>")
            html_content.append(f"<td>{metrics.get('f1_score', 'N/A'):.4f}</td>")
            html_content.append(f"<td>{metrics.get('execution_time', 'N/A'):.2f}</td>")
            html_content.append(f"<td>{metrics.get('samples_per_second', 'N/A'):.2f}</td>")
            html_content.append("</tr>")
            
        html_content.append("</table>")
        
        # Detailed results for each method
        html_content.append("<h2>Detailed Results</h2>")
        
        for method_name, method_data in self.results.items():
            html_content.append(f"<div class='method'>")
            html_content.append(f"<h3>Method: {method_name}</h3>")
            
            # Method parameters
            html_content.append("<h4>Parameters:</h4>")
            html_content.append("<ul>")
            for param_name, param_value in method_data["params"].items():
                html_content.append(f"<li>{param_name}: {param_value}</li>")
            html_content.append("</ul>")
            
            # Results for each dataset
            for dataset_name, dataset_data in method_data["datasets"].items():
                html_content.append(f"<div class='dataset'>")
                html_content.append(f"<h4>Dataset: {dataset_name}</h4>")
                
                # Aggregate metrics for this dataset
                agg_metrics = dataset_data["aggregate_metrics"]
                html_content.append("<h5>Aggregate Metrics:</h5>")
                html_content.append("<table>")
                html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
                
                for metric_name, metric_value in agg_metrics.items():
                    html_content.append("<tr>")
                    html_content.append(f"<td>{metric_name}</td>")
                    html_content.append(f"<td>{metric_value:.4f}</td>")
                    html_content.append("</tr>")
                    
                html_content.append("</table>")
                
                # Results for each split
                html_content.append("<h5>Split Results:</h5>")
                
                for split_name, split_data in dataset_data["splits"].items():
                    html_content.append(f"<h6>Split: {split_name}</h6>")
                    
                    # Check if evaluation failed
                    if split_data.get("status") == "failed":
                        html_content.append(f"<p>Evaluation failed: {split_data['metrics'].get('error', 'Unknown error')}</p>")
                        continue
                        
                    # Metrics for this split
                    metrics = split_data["metrics"]
                    html_content.append("<table>")
                    html_content.append("<tr><th>Metric</th><th>Value</th></tr>")
                    
                    for metric_name, metric_value in metrics.items():
                        html_content.append("<tr>")
                        html_content.append(f"<td>{metric_name}</td>")
                        html_content.append(f"<td>{metric_value:.4f if isinstance(metric_value, float) else metric_value}</td>")
                        html_content.append("</tr>")
                        
                    html_content.append("</table>")
                
                html_content.append("</div>")  # End of dataset
            
            html_content.append("</div>")  # End of method
        
        # Footer
        html_content.append("<hr>")
        html_content.append("<p><em>Generated using the Drowsiness Detection Benchmark framework.</em></p>")
        html_content.append("</body></html>")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(html_content))
            
        logger.info(f"Benchmark report generated: {output_file}")
        
        return output_file
    
    def plot_comparison(self, metric='accuracy', output_file=None):
        """
        Generate a comparative plot of methods based on a specific metric.
        
        Args:
            metric: Metric to compare ('accuracy', 'precision', etc.)
            output_file: Path to save the plot (default: display only)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        method_names = list(self.results.keys())
        dataset_names = set()
        
        for method_data in self.results.values():
            dataset_names.update(method_data["datasets"].keys())
            
        dataset_names = list(dataset_names)
        
        # Prepare data for plotting
        data = []
        
        for method_name in method_names:
            method_values = []
            
            for dataset_name in dataset_names:
                if dataset_name in self.results[method_name]["datasets"]:
                    agg_metrics = self.results[method_name]["datasets"][dataset_name]["aggregate_metrics"]
                    if metric in agg_metrics:
                        method_values.append(agg_metrics[metric])
                    else:
                        method_values.append(0)
                else:
                    method_values.append(0)
                    
            data.append(method_values)
            
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(dataset_names))
        width = 0.8 / len(method_names)
        
        for i, (method_name, method_values) in enumerate(zip(method_names, data)):
            ax.bar(x + i * width - 0.4 + width / 2, method_values, width, label=method_name)
            
        ax.set_xlabel('Dataset')
        ax.set_ylabel(f'{metric.capitalize()}')
        ax.set_title(f'Comparison of Methods by {metric.capitalize()}')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.legend()
        
        fig.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Comparison plot saved to: {output_file}")
            
        return fig


def run_benchmark(methods, datasets, output_dir="results/benchmarks", timeout=None):
    """
    Run a benchmark evaluation for multiple methods on multiple datasets.
    
    Args:
        methods: Dict mapping method names to functions and parameters
        datasets: Dict mapping dataset names to paths
        output_dir: Directory to save benchmark results
        timeout: Maximum execution time per method in seconds
        
    Returns:
        BenchmarkRunner: Benchmark runner instance with results
    """
    benchmark = BenchmarkRunner(output_dir=output_dir)
    
    # Register all methods
    for method_name, method_data in methods.items():
        method_fn = method_data["function"]
        method_params = method_data.get("params", {})
        benchmark.register_method(method_name, method_fn, method_params)
    
    # Run evaluation
    benchmark.evaluate_all(datasets, timeout=timeout)
    
    # Save results
    benchmark.save_results()
    
    # Generate report
    benchmark.generate_report()
    
    return benchmark 


def generate_comparison_table(results_dict, export_csv=False, output_dir="results/benchmarks", 
                             metrics=None, include_platform_info=False):
    """
    Generate a markdown table comparing different drowsiness detection methods.
    
    This function creates a well-formatted markdown table that compares the performance
    of different methods across key metrics. It can also export the results to a CSV
    file for inclusion in research papers or thesis documents.
    
    Args:
        results_dict (dict): Dictionary of benchmark results (from BenchmarkRunner or DriversystemBenchmark)
        export_csv (bool): Whether to export results to a CSV file
        output_dir (str): Directory to save output files
        metrics (list): List of metrics to include in the table 
                      (default: accuracy, recall, f1_score, avg_processing_time)
        include_platform_info (bool): Whether to include platform requirement info
        
    Returns:
        str: Markdown table comparing methods
    """
    if metrics is None:
        metrics = ['accuracy', 'recall', 'f1_score', 'avg_processing_time', 'false_alarm_rate']
    
    # Prepare output directory
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize table rows
    table_rows = []
    
    # Create header row with metrics
    header_row = ["Method"]
    metric_display_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'avg_processing_time': 'Latency (s)',
        'average_response_time': 'Response Time (s)',
        'false_alarm_rate': 'False Alarms/h',
        'samples_per_second': 'FPS',
        'execution_time': 'Exec. Time (s)'
    }
    
    # Add selected metrics to header
    for metric in metrics:
        if metric in metric_display_names:
            header_row.append(metric_display_names[metric])
        else:
            header_row.append(metric.replace('_', ' ').capitalize())
    
    # Add platform info column if requested
    if include_platform_info:
        header_row.append("Platform Requirements")
    
    table_rows.append(header_row)
    
    # Process different result formats (handle both BenchmarkRunner and DriversystemBenchmark)
    method_metrics = {}
    
    # Try to handle results from DriversystemBenchmark
    if isinstance(results_dict, dict) and all(isinstance(v, dict) and 'accuracy' in v for v in results_dict.values()):
        # Direct metrics dictionary format from DriversystemBenchmark
        method_metrics = results_dict
    elif isinstance(results_dict, dict) and "results" in results_dict:
        # JSON loaded format with "results" key
        method_metrics = results_dict["results"] 
    elif hasattr(results_dict, "results"):
        # BenchmarkRunner object
        runner_results = results_dict.results
        for method_name, method_data in runner_results.items():
            method_metrics[method_name] = method_data.get("metrics", {})
    else:
        logger.warning("Unrecognized results format. Table may be incomplete.")
    
    # Add data rows for each method
    method_names = sorted(method_metrics.keys())
    
    for method_name in method_names:
        row = [method_name]
        metrics_data = method_metrics[method_name]
        
        for metric in metrics:
            # Get metric value if available, otherwise N/A
            if metric in metrics_data:
                value = metrics_data[metric]
                if isinstance(value, float):
                    # Format based on metric type
                    if metric in ['avg_processing_time', 'average_response_time', 'execution_time']:
                        row.append(f"{value:.3f}")  # 3 decimal places for time measurements
                    elif metric in ['false_alarm_rate']:
                        row.append(f"{value:.2f}")  # 2 decimal places for rates
                    else:
                        row.append(f"{value:.4f}")  # 4 decimal places for accuracy metrics
                else:
                    row.append(str(value))
            else:
                row.append("N/A")
        
        # Add platform info placeholder (user would need to update this manually)
        if include_platform_info:
            row.append("")
            
        table_rows.append(row)
    
    # Generate markdown table
    markdown_lines = []
    markdown_lines.append(" | ".join(table_rows[0]))
    markdown_lines.append(" | ".join(["---"] * len(table_rows[0])))
    
    for row in table_rows[1:]:
        markdown_lines.append(" | ".join(row))
    
    markdown_table = "\n".join(markdown_lines)
    
    # Print the table to console
    print("Comparison Table:\n")
    print(markdown_table)
    print("\n")
    
    # Export to CSV if requested
    if export_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = output_path / f"method_comparison_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in table_rows:
                writer.writerow(row)
        
        logger.info(f"Comparison table exported to CSV: {csv_file}")
    
    # Also save markdown to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file = output_path / f"method_comparison_{timestamp}.md"
    
    with open(md_file, 'w') as f:
        f.write(markdown_table)
        
    logger.info(f"Comparison table saved as markdown: {md_file}")
    
    return markdown_table


class DriversystemBenchmark:
    """
    Specialized benchmark system for driver drowsiness detection methods.
    
    This class provides functionality to load datasets, evaluate various detection
    methods, and visualize comparative results with a focus on drowsiness detection metrics.
    """
    
    def __init__(self, output_dir="results/driver_benchmarks", cache_dir="results/cache"):
        """
        Initialize the driver system benchmark.
        
        Args:
            output_dir (str): Directory to save benchmark results
            cache_dir (str): Directory to cache intermediate results
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.load_benchmark_dataset = load_benchmark_dataset
        self.calculate_metrics = calculate_metrics
        
        # Store results for different methods
        self.dataset = None
        self.labels = None
        self.metadata = None
        self.timestamps = None
        self.results = {}
        
    def load_dataset(self, dataset_path, split='test', dataset_type=None, subset=None, 
                     preprocess_fn=None, create_timestamps=True):
        """
        Load a dataset for benchmarking.
        
        Args:
            dataset_path (str): Path to the dataset
            split (str): Data split to use ('train', 'validation', 'test')
            dataset_type (str, optional): Type of dataset (e.g., 'nthu-ddd')
            subset (str, optional): Subset of the dataset to load
            preprocess_fn (callable, optional): Function to preprocess images
            create_timestamps (bool): Whether to create synthetic timestamps
                for evaluating time-based metrics
                
        Returns:
            tuple: (images, labels, metadata)
        """
        logger.info(f"Loading dataset from {dataset_path}, split={split}")
        
        # Load the dataset
        self.dataset, self.labels, self.metadata = self.load_benchmark_dataset(
            dataset_path, 
            split=split, 
            dataset_type=dataset_type,
            subset=subset,
            cache_dir=self.cache_dir,
            load_images=True,
            preprocess_fn=preprocess_fn
        )
        
        # Create synthetic timestamps if requested (for time-based metrics)
        if create_timestamps:
            # Create timestamps with 1/30 second intervals (assuming 30 fps)
            self.timestamps = np.arange(len(self.dataset)) / 30.0
        else:
            self.timestamps = None
            
        logger.info(f"Loaded {len(self.dataset)} samples with {sum(self.labels)} positive instances")
        
        return self.dataset, self.labels, self.metadata
    
    def evaluate_method(self, detector, method_name, batch_size=1):
        """
        Evaluate a drowsiness detection method on the loaded dataset.
        
        Args:
            detector: Detection method function or object with predict() method
            method_name (str): Name of the method for results tracking
            batch_size (int): Batch size for processing (default: process one image at a time)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.dataset is None or self.labels is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        logger.info(f"Evaluating method: {method_name}")
        
        predictions = []
        processing_times = []
        
        # Process dataset
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i+batch_size]
            
            # Measure execution time
            start_time = time.time()
            
            try:
                # Handle different detector interfaces
                if hasattr(detector, 'predict'):
                    # Scikit-learn like interface
                    if batch_size > 1:
                        batch_pred = detector.predict(batch)
                    else:
                        batch_pred = [detector.predict(batch[0])]
                else:
                    # Function interface
                    if batch_size > 1:
                        batch_pred = detector(batch)
                    else:
                        batch_pred = [detector(batch[0])]
                
                # Record predictions
                predictions.extend(batch_pred)
                
            except Exception as e:
                logger.error(f"Error processing batch at index {i}: {str(e)}")
                # In case of error, assume non-drowsy (negative prediction)
                predictions.extend([0] * len(batch))
            
            # Record processing time
            execution_time = time.time() - start_time
            processing_times.append(execution_time)
        
        # Ensure predictions have the same length as labels
        predictions = predictions[:len(self.labels)]
        if len(predictions) < len(self.labels):
            predictions.extend([0] * (len(self.labels) - len(predictions)))
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(
            predictions, 
            self.labels,
            timestamps=self.timestamps
        )
        
        # Add average processing time
        metrics['avg_processing_time'] = np.mean(processing_times)
        metrics['predictions'] = predictions
        
        # Save results
        self.results[method_name] = metrics
        
        logger.info(f"Method {method_name} evaluation complete: "
                   f"Accuracy={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}")
        
        return metrics
    
    def evaluate_baseline_methods(self):
        """
        Evaluate standard baseline methods for drowsiness detection.
        
        This includes:
        - PERCLOS (Percentage of eye closure)
        - Eye Aspect Ratio (EAR)
        - Simple thresholding approaches
        
        Returns:
            dict: Dictionary of method names mapped to their evaluation metrics
        """
        if self.dataset is None or self.labels is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        logger.info("Evaluating baseline methods")
        
        # 1. Simple PERCLOS threshold detector
        def perclos_detector(image):
            """Simple simulated PERCLOS detector"""
            # In a real implementation, this would detect eye closure
            # Here we'll use a simple proxy based on image darkness
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            darkness = 1.0 - gray.mean() / 255.0
            # Assume drowsy if darkness > 0.6 (simulating eye closure)
            return 1 if darkness > 0.6 else 0
        
        self.evaluate_method(perclos_detector, "PERCLOS")
        
        # 2. Simulated EAR detector
        def ear_detector(image):
            """Simple simulated Eye Aspect Ratio detector"""
            # In a real implementation, this would calculate the eye aspect ratio
            # Here we'll use image variance as a proxy 
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            variance = np.var(gray) / (255.0 * 255.0)  # Normalized variance
            # Assume drowsy if variance is low (simulating closed or droopy eyes)
            return 1 if variance < 0.03 else 0
        
        self.evaluate_method(ear_detector, "EAR")
        
        # 3. Combined EAR + head pose
        def combined_detector(image):
            """Simple combined detector using EAR and head pose"""
            # EAR component
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            variance = np.var(gray) / (255.0 * 255.0)
            
            # Head pose component (simulated with horizontal variance)
            h_variance = np.var(np.mean(image, axis=0)) / (255.0 * 255.0) if len(image.shape) == 3 else 0
            
            # Combined decision
            return 1 if (variance < 0.03 or h_variance < 0.01) else 0
        
        self.evaluate_method(combined_detector, "Combined_EAR_Pose")
        
        # 4. Random baseline (for comparison)
        def random_detector(image):
            """Random detection baseline"""
            return np.random.choice([0, 1], p=[0.7, 0.3])  # Biased toward non-drowsy
        
        self.evaluate_method(random_detector, "Random_Baseline")
        
        logger.info("Baseline method evaluation complete")
        return {k: v for k, v in self.results.items() 
                if k in ["PERCLOS", "EAR", "Combined_EAR_Pose", "Random_Baseline"]}
    
    def evaluate_your_method(self, custom_detector, method_name="Custom_Method", batch_size=1):
        """
        Evaluate your custom drowsiness detection method.
        
        Args:
            custom_detector: Your detection method function or object
            method_name (str): Name for your method
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating custom method: {method_name}")
        return self.evaluate_method(custom_detector, method_name, batch_size)
    
    def compare_and_visualize(self, metrics=None, output_prefix=None):
        """
        Compare and visualize the performance of all evaluated methods.
        
        Args:
            metrics (list): List of metric names to visualize
            output_prefix (str): Prefix for output files
            
        Returns:
            dict: Dictionary of figure objects for each visualization
        """
        if not self.results:
            raise ValueError("No methods evaluated yet")
            
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                       'average_response_time', 'false_alarm_rate']
        
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"drowsiness_benchmark_{timestamp}"
        
        # Create a results directory if it doesn't exist
        results_dir = self.output_dir / "visualizations"
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Generating performance visualizations")
        
        figures = {}
        
        # 1. Bar chart comparing basic metrics
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in basic_metrics if m in metrics]
        
        if available_metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            method_names = list(self.results.keys())
            x = np.arange(len(method_names))
            width = 0.8 / len(available_metrics)
            
            for i, metric in enumerate(available_metrics):
                values = [self.results[method].get(metric, 0) for method in method_names]
                ax.bar(x + i * width - 0.4 + width / 2, values, width, label=metric.capitalize())
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Score')
            ax.set_title('Basic Performance Metrics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.legend()
            fig.tight_layout()
            
            # Save figure
            output_path = results_dir / f"{output_prefix}_basic_metrics.png"
            fig.savefig(output_path)
            figures['basic_metrics'] = fig
            
            logger.info(f"Basic metrics visualization saved to: {output_path}")
            
        # 2. Response time and false alarm rate (specific to drowsiness detection)
        time_metrics = ['average_response_time', 'false_alarm_rate']
        available_time_metrics = [m for m in time_metrics if m in metrics]
        
        if available_time_metrics:
            fig, axes = plt.subplots(len(available_time_metrics), 1, figsize=(10, 4*len(available_time_metrics)))
            if len(available_time_metrics) == 1:
                axes = [axes]
                
            method_names = list(self.results.keys())
            
            for i, metric in enumerate(available_time_metrics):
                values = [self.results[method].get(metric, 0) for method in method_names]
                
                # Color bars based on whether lower is better (response time) or higher is better
                colors = ['#3498db' if metric == 'false_alarm_rate' else '#e74c3c' for _ in method_names]
                
                axes[i].bar(method_names, values, color=colors)
                axes[i].set_xlabel('Method')
                axes[i].set_ylabel(metric.replace('_', ' ').capitalize())
                axes[i].set_title(f'{metric.replace("_", " ").capitalize()} Comparison')
                axes[i].set_xticklabels(method_names, rotation=45, ha='right')
            
            fig.tight_layout()
            
            # Save figure
            output_path = results_dir / f"{output_prefix}_time_metrics.png"
            fig.savefig(output_path)
            figures['time_metrics'] = fig
            
            logger.info(f"Time-based metrics visualization saved to: {output_path}")
        
        # 3. Confusion matrices for each method
        for method_name, results in self.results.items():
            if 'confusion_matrix' not in results:
                continue
                
            cm = results['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Set labels and ticks
            class_names = ['Alert', 'Drowsy']
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_names, yticklabels=class_names,
                   title=f'Confusion Matrix - {method_name}',
                   ylabel='True label',
                   xlabel='Predicted label')
            
            # Rotate tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Loop over data dimensions and create text annotations
            fmt = 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            fig.tight_layout()
            
            # Save figure
            output_path = results_dir / f"{output_prefix}_cm_{method_name}.png"
            fig.savefig(output_path)
            figures[f'cm_{method_name}'] = fig
            
            logger.info(f"Confusion matrix for {method_name} saved to: {output_path}")
        
        # 4. Summary plot with all key metrics for best methods
        fig, ax = plt.subplots(figsize=(12, 8))
        
        available_metrics = [m for m in metrics if any(m in results for results in self.results.values())]
        method_names = list(self.results.keys())
        
        # Create a radar chart (polar plot)
        # Convert metrics to a 0-1 scale for comparability
        scaled_metrics = {}
        
        for metric in available_metrics:
            values = [self.results[method].get(metric, 0) for method in method_names]
            
            # For metrics where lower is better, invert the scale
            if metric in ['average_response_time', 'false_alarm_rate']:
                if max(values) > 0:
                    scaled_metrics[metric] = [1 - (v / max(values)) for v in values]
                else:
                    scaled_metrics[metric] = [1 for _ in values]
            else:
                if max(values) > 0:
                    scaled_metrics[metric] = [v / max(values) for v in values]
                else:
                    scaled_metrics[metric] = [0 for _ in values]
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        for i, method in enumerate(method_names):
            values = [scaled_metrics[metric][i] for metric in available_metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.1)
        
        # Set the labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').capitalize() for m in available_metrics])
        ax.set_title('Method Comparison Across Metrics')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save figure
        output_path = results_dir / f"{output_prefix}_radar_chart.png"
        fig.savefig(output_path)
        figures['radar_chart'] = fig
        
        logger.info(f"Radar chart saved to: {output_path}")
        
        # Generate HTML report
        self._generate_report(output_prefix)
        
        return figures
    
    def _generate_report(self, output_prefix=None):
        """
        Generate a comprehensive HTML report of benchmark results.
        
        Args:
            output_prefix (str): Prefix for the output report file
            
        Returns:
            str: Path to the generated report
        """
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"drowsiness_benchmark_{timestamp}"
        
        output_file = self.output_dir / f"{output_prefix}_report.html"
        
        # Create HTML content
        html_content = []
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html><head>")
        html_content.append("<title>Driver Drowsiness Detection Benchmark Report</title>")
        html_content.append("<style>")
        html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_content.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_content.append("th { background-color: #f2f2f2; }")
        html_content.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        html_content.append("h1, h2, h3 { color: #333; }")
        html_content.append(".method { margin-bottom: 30px; }")
        html_content.append(".img-container { text-align: center; margin: 20px 0; }")
        html_content.append(".img-container img { max-width: 100%; height: auto; }")
        html_content.append("</style>")
        html_content.append("</head><body>")
        
        # Header
        html_content.append("<h1>Driver Drowsiness Detection Benchmark Report</h1>")
        html_content.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Dataset information
        html_content.append("<h2>Dataset Information</h2>")
        html_content.append(f"<p>Total samples: {len(self.dataset) if self.dataset is not None else 'N/A'}</p>")
        html_content.append(f"<p>Positive samples (drowsy): {sum(self.labels) if self.labels is not None else 'N/A'}</p>")
        
        # Performance summary table
        html_content.append("<h2>Performance Summary</h2>")
        html_content.append("<table>")
        
        # Table header
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 
                          'average_response_time', 'false_alarm_rate', 'avg_processing_time']
        html_content.append("<tr><th>Method</th>" + 
                           "".join([f"<th>{m.replace('_', ' ').capitalize()}</th>" for m in metrics_to_show]) + 
                           "</tr>")
        
        # Table rows
        for method_name, metrics in self.results.items():
            html_content.append("<tr>")
            html_content.append(f"<td>{method_name}</td>")
            
            for metric in metrics_to_show:
                value = metrics.get(metric, "N/A")
                if isinstance(value, (int, float)):
                    html_content.append(f"<td>{value:.4f}</td>")
                else:
                    html_content.append(f"<td>{value}</td>")
                    
            html_content.append("</tr>")
            
        html_content.append("</table>")
        
        # Visualizations
        html_content.append("<h2>Visualizations</h2>")
        
        # Path to visualizations (relative to the report)
        vis_dir = "visualizations"
        
        # Add visualization images
        vis_files = {
            "Basic Metrics": f"{output_prefix}_basic_metrics.png",
            "Time-Based Metrics": f"{output_prefix}_time_metrics.png",
            "Overall Comparison": f"{output_prefix}_radar_chart.png"
        }
        
        for title, filename in vis_files.items():
            html_content.append(f"<h3>{title}</h3>")
            html_content.append("<div class='img-container'>")
            html_content.append(f"<img src='{vis_dir}/{filename}' alt='{title}'>")
            html_content.append("</div>")
        
        # Confusion matrices
        html_content.append("<h3>Confusion Matrices</h3>")
        
        for method_name in self.results.keys():
            html_content.append(f"<h4>{method_name}</h4>")
            html_content.append("<div class='img-container'>")
            html_content.append(f"<img src='{vis_dir}/{output_prefix}_cm_{method_name}.png' alt='Confusion Matrix - {method_name}'>")
            html_content.append("</div>")
        
        # Footer
        html_content.append("<hr>")
        html_content.append("<p><em>Generated using the Driver Drowsiness Detection Benchmark framework.</em></p>")
        html_content.append("</body></html>")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(html_content))
            
        logger.info(f"Benchmark report generated: {output_file}")
        
        return output_file
    
    def save_results(self, filename=None):
        """
        Save benchmark results to a file.
        
        Args:
            filename (str, optional): Output filename
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"driver_benchmark_results_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Prepare results for serialization (convert numpy arrays to lists)
        serializable_results = {}
        
        for method_name, method_results in self.results.items():
            serializable_results[method_name] = {}
            
            for k, v in method_results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[method_name][k] = v.tolist()
                elif k == 'predictions' and isinstance(v, list):
                    # Only keep the first 10 predictions to save space
                    serializable_results[method_name][k] = v[:10]
                else:
                    serializable_results[method_name][k] = v
        
        # Add dataset metadata
        result_data = {
            "dataset_info": {
                "num_samples": len(self.dataset) if self.dataset is not None else 0,
                "num_positive": sum(self.labels) if self.labels is not None else 0,
                "timestamp": datetime.now().isoformat()
            },
            "results": serializable_results
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"Benchmark results saved to: {output_path}")
        
        return output_path
    
    def load_results(self, filepath):
        """
        Load previously saved benchmark results.
        
        Args:
            filepath (str): Path to the results file
            
        Returns:
            self: For method chaining
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if "results" in data:
            self.results = data["results"]
            
        logger.info(f"Loaded benchmark results from: {filepath}")
        
        return self
    
    def generate_comparison_table(self, export_csv=False, metrics=None, include_platform_info=False):
        """
        Generate a markdown table comparing all evaluated methods.
        
        This method wraps the standalone generate_comparison_table function to provide
        a convenient way to generate comparison tables directly from the benchmark object.
        
        Args:
            export_csv (bool): Whether to export results to a CSV file
            metrics (list): List of metrics to include in the table
            include_platform_info (bool): Whether to include platform requirement info
            
        Returns:
            str: Markdown table comparing methods
        """
        return generate_comparison_table(
            self.results,
            export_csv=export_csv,
            output_dir=str(self.output_dir),
            metrics=metrics,
            include_platform_info=include_platform_info
        )
    
    def visualize_results(self, metrics=None, output_dir=None, prefix=None, include_interactive=False):
        """
        Generate comprehensive visualizations for benchmark results.
        
        This method creates a variety of visualizations including bar charts,
        line plots, ROC curves, and performance comparisons between methods.
        
        Args:
            metrics (list, optional): List of metrics to visualize. Default is
                ['accuracy', 'precision', 'recall', 'f1_score', 'avg_processing_time', 'false_alarm_rate']
            output_dir (str, optional): Directory to save visualizations. Default is a 
                'visualizations' subdirectory in the benchmark output directory.
            prefix (str, optional): Prefix for visualization filenames.
            include_interactive (bool): Whether to include interactive visualizations (HTML)
                
        Returns:
            dict: Dictionary mapping visualization types to file paths
        """
        if not self.results:
            raise ValueError("No benchmark results available. Run evaluation methods first.")
            
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                      'avg_processing_time', 'false_alarm_rate']
            
        if output_dir is None:
            output_dir = self.output_dir / "visualizations"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Use timestamp if no prefix provided
        if prefix is None:
            prefix = datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
            
        # Generate comprehensive visualization set
        logger.info(f"Generating visualizations in {output_dir}")
        vis_paths = generate_visualization_set(self.results, output_dir=output_dir, prefix=prefix)
        
        # Add individual visualizations for important metrics
        vis_paths.update(self._create_additional_visualizations(output_dir, prefix, metrics))
        
        # Generate interactive visualizations if requested
        if include_interactive:
            vis_paths.update(self._create_interactive_visualizations(output_dir, prefix))
            
        logger.info(f"Generated {len(vis_paths)} visualizations")
        
        return vis_paths
    
    def _create_additional_visualizations(self, output_dir, prefix, metrics):
        """
        Create additional specialized visualizations.
        
        Args:
            output_dir (Path): Directory to save visualizations
            prefix (str): Filename prefix
            metrics (list): List of metrics to visualize
            
        Returns:
            dict: Dictionary mapping visualization types to file paths
        """
        additional_paths = {}
        
        # Create a performance vs. time scatter plot
        if 'accuracy' in metrics and 'avg_processing_time' in metrics:
            scatter_path = os.path.join(output_dir, f"{prefix}_perf_vs_time_scatter.png")
            fig = plot_metric_vs_time(
                self.results,
                perf_metric='accuracy',
                time_metric='avg_processing_time',
                output_file=scatter_path,
                figsize=(10, 8)
            )
            plt.close(fig)
            additional_paths['perf_vs_time_scatter'] = scatter_path
        
        # Create a bar chart comparing false alarm rates
        if 'false_alarm_rate' in metrics:
            far_path = os.path.join(output_dir, f"{prefix}_false_alarm_rates.png")
            fig = plot_false_alarm_comparison(
                self.results,
                output_file=far_path,
                figsize=(12, 6)
            )
            plt.close(fig)
            additional_paths['false_alarm_rates'] = far_path
            
        return additional_paths
    
    def _create_interactive_visualizations(self, output_dir, prefix):
        """
        Create interactive HTML visualizations.
        
        Args:
            output_dir (Path): Directory to save visualizations
            prefix (str): Filename prefix
            
        Returns:
            dict: Dictionary mapping visualization types to file paths
        """
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd
        
        interactive_paths = {}
        
        try:
            # Convert results to DataFrame for plotting
            data = []
            
            for method_name, metrics in self.results.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    row = {'method': method_name}
                    row.update(metrics)
                    data.append(row)
            
            if not data:
                return interactive_paths
                
            df = pd.DataFrame(data)
            
            # Create interactive scatter plot
            if 'accuracy' in df.columns and 'avg_processing_time' in df.columns:
                fig = px.scatter(
                    df, 
                    x='avg_processing_time', 
                    y='accuracy',
                    text='method',
                    title='Performance vs. Processing Time',
                    labels={
                        'avg_processing_time': 'Average Processing Time (s)',
                        'accuracy': 'Accuracy'
                    },
                    color='method'
                )
                
                # Add annotations
                fig.update_traces(
                    textposition='top center',
                    marker=dict(size=12)
                )
                
                # Save as HTML
                scatter_path = os.path.join(output_dir, f"{prefix}_interactive_scatter.html")
                fig.write_html(scatter_path)
                interactive_paths['interactive_scatter'] = scatter_path
                
        except (ImportError, Exception) as e:
            logger.warning(f"Error creating interactive visualizations: {str(e)}")
            
        return interactive_paths 