"""
Result visualization module for drowsiness detection evaluation.

This module provides functions to create various visualizations for benchmark and
evaluation results, including bar charts, line plots, and ROC curves.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Configure visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper")


def plot_metric_comparison(results, metric='accuracy', output_file=None, title=None, figsize=(10, 6)):
    """
    Create a bar chart comparing methods based on a specific metric.
    
    Args:
        results (dict): Dictionary containing benchmark results
        metric (str): Metric to plot
        output_file (str, optional): Path to save the figure
        title (str, optional): Custom title for the plot
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Extract method names and values
    method_names = []
    metric_values = []
    
    # Handle different results formats
    if isinstance(results, dict):
        if "results" in results:
            # Handle JSON loaded format
            results_data = results["results"]
        elif all(isinstance(v, dict) and "metrics" in v for k, v in results.items()):
            # DirectSystem benchmark format
            results_data = results
        else:
            # BenchmarkRunner format
            results_data = {}
            for method_name, method_data in results.items():
                if "metrics" in method_data:
                    results_data[method_name] = method_data
    else:
        # Assume it's a BenchmarkRunner object
        results_data = {}
        for method_name, method_data in results.results.items():
            if "metrics" in method_data:
                results_data[method_name] = method_data
    
    # Extract values
    for method_name, method_data in results_data.items():
        if "metrics" in method_data and metric in method_data["metrics"]:
            method_names.append(method_name)
            metric_values.append(method_data["metrics"][metric])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate bars with nice colors
    colors = get_color_palette(len(method_names))
    y_pos = np.arange(len(method_names))
    
    # Create horizontal bar chart
    bars = ax.barh(y_pos, metric_values, color=colors)
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if isinstance(metric_values[i], float):
            # Format based on metric type
            if metric in ['avg_processing_time', 'average_response_time', 'execution_time']:
                value_str = f"{width:.3f}"  # 3 decimals for time
            elif metric in ['false_alarm_rate']:
                value_str = f"{width:.2f}"  # 2 decimals for rates
            else:
                value_str = f"{width:.4f}"  # 4 decimals for accuracy metrics
        else:
            value_str = str(width)
            
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, value_str,
                ha='left', va='center')
    
    # Add metric labels and formatting
    if "accuracy" in metric or "precision" in metric or "recall" in metric or "f1" in metric:
        # Format as percentage for accuracy-related metrics
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        if ax.get_xlim()[1] < 1.0:
            ax.set_xlim(0, 1.0)
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names)
    ax.set_xlabel(metric.replace("_", " ").title())
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Comparison of Methods by {metric.replace("_", " ").title()}')
    
    # Add grid lines for readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_by_dataset(results, metrics=None, output_file=None, figsize=(12, 8)):
    """
    Create a grouped bar chart showing method performance across different datasets.
    
    Args:
        results (dict): Dictionary containing benchmark results
        metrics (list, optional): List of metrics to include
        output_file (str, optional): Path to save the figure
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    if metrics is None:
        metrics = ['accuracy', 'f1_score']
    
    # Extract dataset and method information
    datasets = set()
    methods = set()
    
    # Handle different results formats
    if isinstance(results, dict):
        if "results" in results:
            # JSON loaded format
            results_data = results["results"]
        else:
            # Assume BenchmarkRunner format
            results_data = results
    else:
        # Assume BenchmarkRunner object
        results_data = results.results
    
    # Collect datasets and methods
    for method_name, method_data in results_data.items():
        methods.add(method_name)
        if "datasets" in method_data:
            for dataset_name in method_data["datasets"].keys():
                datasets.add(dataset_name)
    
    datasets = sorted(list(datasets))
    methods = sorted(list(methods))
    
    # Prepare data for plotting
    data = []
    for method in methods:
        for dataset in datasets:
            for metric in metrics:
                value = None
                try:
                    method_data = results_data[method]
                    if "datasets" in method_data and dataset in method_data["datasets"]:
                        dataset_data = method_data["datasets"][dataset]
                        if "aggregate_metrics" in dataset_data:
                            value = dataset_data["aggregate_metrics"].get(metric)
                except (KeyError, TypeError):
                    pass
                
                if value is not None:
                    data.append({
                        'method': method,
                        'dataset': dataset,
                        'metric': metric.replace("_", " ").title(),
                        'value': value
                    })
    
    # Check if we have data
    if not data:
        raise ValueError("No data available for plotting. Check the results format or metrics.")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create figure with subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    # Handle case of single metric
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        metric_display = metric.replace("_", " ").title()
        
        # Filter data for this metric
        metric_data = df[df['metric'] == metric_display]
        
        # Get colors
        colors = get_color_palette(len(methods))
        
        # Create grouped bar chart
        sns.barplot(
            x='dataset', 
            y='value', 
            hue='method', 
            data=metric_data, 
            ax=axes[i],
            palette=colors
        )
        
        # Set labels and title
        axes[i].set_title(f'{metric_display} by Dataset')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(metric_display)
        
        # Format y-axis for accuracy metrics
        if "accuracy" in metric.lower() or "precision" in metric.lower() or "recall" in metric.lower() or "f1" in metric.lower():
            axes[i].yaxis.set_major_formatter(PercentFormatter(1.0))
            if axes[i].get_ylim()[1] < 1.0:
                axes[i].set_ylim(0, 1.0)
        
        # Improve legend
        if i == 0:  # Only show legend on first subplot
            axes[i].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[i].legend().set_visible(False)
    
    # Final x-axis label
    axes[-1].set_xlabel('Dataset')
    
    # Rotate x-tick labels for better readability
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(results, output_file=None, figsize=(10, 8)):
    """
    Plot ROC curves for multiple methods.
    
    Args:
        results (dict): Dictionary containing benchmark results with ROC curve data
        output_file (str, optional): Path to save the figure
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Track methods added to the plot
    methods_added = 0
    
    # Handle different results formats
    if isinstance(results, dict):
        if "results" in results:
            # JSON loaded format
            results_data = results["results"]
        else:
            # Assume BenchmarkRunner format
            results_data = results
    else:
        # Assume BenchmarkRunner object
        results_data = results.results
    
    # Get colors
    colors = get_color_palette(len(results_data))
    
    # Plot ROC curve for each method
    for i, (method_name, method_data) in enumerate(results_data.items()):
        # Look for ROC curve data in the results
        fpr, tpr, roc_auc = None, None, None
        
        # Check in metrics
        if "roc_curve" in method_data.get("metrics", {}):
            roc_data = method_data["metrics"]["roc_curve"]
            if isinstance(roc_data, dict):
                fpr = roc_data.get("fpr")
                tpr = roc_data.get("tpr")
                roc_auc = roc_data.get("auc")
        
        # If not found, check datasets
        if fpr is None and "datasets" in method_data:
            for dataset_name, dataset_data in method_data["datasets"].items():
                if "aggregate_metrics" in dataset_data and "roc_curve" in dataset_data["aggregate_metrics"]:
                    roc_data = dataset_data["aggregate_metrics"]["roc_curve"]
                    if isinstance(roc_data, dict):
                        fpr = roc_data.get("fpr")
                        tpr = roc_data.get("tpr")
                        roc_auc = roc_data.get("auc")
                        break
        
        # If we have data, plot it
        if fpr is not None and tpr is not None:
            # Convert to lists if they're still in JSON format
            if isinstance(fpr, list) and isinstance(tpr, list):
                fpr = np.array(fpr)
                tpr = np.array(tpr)
                
                # Plot this method's ROC curve
                label = f'{method_name} (AUC = {roc_auc:.3f})' if roc_auc is not None else method_name
                ax.plot(fpr, tpr, lw=2, color=colors[i], label=label)
                methods_added += 1
    
    # Add the random guess line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2, label='Random Guess')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid and legend
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if requested
    if output_file and methods_added > 0:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig if methods_added > 0 else None


def plot_processing_time_comparison(results, output_file=None, figsize=(10, 6), metric='avg_processing_time'):
    """
    Create a bar chart comparing processing times across methods.
    
    Args:
        results (dict): Dictionary containing benchmark results
        output_file (str, optional): Path to save the figure
        figsize (tuple): Figure size
        metric (str): Which time metric to use
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Prepare data
    method_names = []
    proc_times = []
    
    # Handle different results formats
    if isinstance(results, dict):
        if "results" in results:
            # JSON loaded format
            results_data = results["results"]
        elif all(isinstance(v, dict) and metric in v for k, v in results.items()):
            # DirectSystem benchmark format
            results_data = results
        else:
            # BenchmarkRunner format
            results_data = {}
            for method_name, method_data in results.items():
                if "metrics" in method_data:
                    results_data[method_name] = method_data
    else:
        # Assume it's a BenchmarkRunner object
        results_data = {}
        for method_name, method_data in results.results.items():
            if "metrics" in method_data:
                results_data[method_name] = method_data
    
    # Extract values
    for method_name, method_data in results_data.items():
        if "metrics" in method_data and metric in method_data["metrics"]:
            method_names.append(method_name)
            proc_times.append(method_data["metrics"][metric])
    
    # Sort by processing time
    sorted_indices = np.argsort(proc_times)
    method_names = [method_names[i] for i in sorted_indices]
    proc_times = [proc_times[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    colors = get_color_palette(len(method_names))
    y_pos = np.arange(len(method_names))
    bars = ax.barh(y_pos, proc_times, color=colors)
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names)
    
    title_metric = metric.replace("_", " ").title()
    ax.set_xlabel(f"{title_metric} (seconds)")
    ax.set_title(f"{title_metric} Comparison")
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f"{width:.4f}s", va='center')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metric_vs_time(results, output_file=None, figsize=(10, 6), 
                      perf_metric='accuracy', time_metric='avg_processing_time'):
    """
    Create a scatter plot showing the trade-off between performance and processing time.
    
    Args:
        results (dict): Dictionary containing benchmark results
        output_file (str, optional): Path to save the figure
        figsize (tuple): Figure size
        perf_metric (str): Performance metric to use
        time_metric (str): Time metric to use
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Prepare data
    method_names = []
    perf_values = []
    time_values = []
    
    # Handle different results formats
    if isinstance(results, dict):
        if "results" in results:
            # JSON loaded format
            results_data = results["results"]
        elif all(isinstance(v, dict) and "metrics" in v for k, v in results.items()):
            # DirectSystem benchmark format
            results_data = results
        else:
            # BenchmarkRunner format
            results_data = {}
            for method_name, method_data in results.items():
                if "metrics" in method_data:
                    results_data[method_name] = method_data
    else:
        # Assume it's a BenchmarkRunner object
        results_data = {}
        for method_name, method_data in results.results.items():
            if "metrics" in method_data:
                results_data[method_name] = method_data
    
    # Extract values
    for method_name, method_data in results_data.items():
        if "metrics" in method_data:
            metrics = method_data["metrics"]
            if perf_metric in metrics and time_metric in metrics:
                method_names.append(method_name)
                perf_values.append(metrics[perf_metric])
                time_values.append(metrics[time_metric])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    colors = get_color_palette(len(method_names))
    scatter = ax.scatter(time_values, perf_values, s=100, c=colors, alpha=0.7)
    
    # Add method labels
    for i, method in enumerate(method_names):
        ax.annotate(method, (time_values[i], perf_values[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    # Set labels and title
    ax.set_xlabel(f"{time_metric.replace('_', ' ').title()} (seconds)")
    ax.set_ylabel(f"{perf_metric.replace('_', ' ').title()}")
    ax.set_title(f"Performance vs. Processing Time")
    
    # Format y-axis for accuracy metrics
    if "accuracy" in perf_metric.lower() or "precision" in perf_metric.lower() or "recall" in perf_metric.lower() or "f1" in perf_metric.lower():
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if ax.get_ylim()[1] < 1.0:
            ax.set_ylim(0, 1.0)
    
    # Add grid
    ax.grid(linestyle='--', alpha=0.7)
    
    # Add "better" region indicator
    # Top-left is better (higher performance, lower time)
    ax.text(0.05, 0.95, "Better", transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.5))
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def plot_condition_comparison(results, condition_type, metric='detection_rate', 
                           output_file=None, figsize=(12, 6)):
    """
    Plot performance across different conditions (e.g., lighting, glasses).
    
    Args:
        results (dict): Dictionary containing real-world evaluation results
        condition_type (str): Type of condition to compare (e.g., 'lighting', 'glasses')
        metric (str): Metric to compare
        output_file (str, optional): Path to save the figure
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Handle results format
    if "results_by_condition" not in results:
        raise ValueError("Results don't contain condition-specific data")
        
    if condition_type not in results["results_by_condition"]:
        raise ValueError(f"Condition type '{condition_type}' not found in results")
    
    # Extract data
    conditions = []
    values = []
    
    condition_data = results["results_by_condition"][condition_type]
    
    for condition, metrics in condition_data.items():
        if metric in metrics:
            conditions.append(condition)
            values.append(metrics[metric])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    colors = get_color_palette(len(conditions))
    bars = ax.bar(conditions, values, color=colors)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if isinstance(values[i], float):
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f"{height:.3f}", ha='center')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   str(height), ha='center')
    
    # Set labels and title
    ax.set_xlabel(condition_type.capitalize())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by {condition_type.capitalize()}")
    
    # Format y-axis for certain metrics
    if metric in ['detection_rate', 'accuracy', 'precision', 'recall', 'f1_score']:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1.0)
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def plot_false_alarm_comparison(results, output_file=None, figsize=(10, 6)):
    """
    Plot false alarm rates for different methods.
    
    Args:
        results (dict): Dictionary containing benchmark results
        output_file (str, optional): Path to save the figure
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    return plot_metric_comparison(results, metric='false_alarm_rate', 
                                 output_file=output_file, figsize=figsize,
                                 title='False Alarm Rate Comparison')


def plot_detection_latency_comparison(results, output_file=None, figsize=(10, 6)):
    """
    Plot detection latency for different methods.
    
    Args:
        results (dict): Dictionary containing benchmark results
        output_file (str, optional): Path to save the figure
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    return plot_metric_comparison(results, 
                                 metric='detection_latency', 
                                 output_file=output_file, 
                                 figsize=figsize,
                                 title='Detection Latency Comparison (lower is better)')


def get_color_palette(n_colors):
    """
    Get a nice color palette for plotting.
    
    Args:
        n_colors (int): Number of colors needed
        
    Returns:
        list: List of colors
    """
    if n_colors <= 10:
        # Use Tableau 10 palette for smaller sets
        return sns.color_palette("tab10", n_colors)
    else:
        # Generate a larger palette
        return sns.color_palette("husl", n_colors)


def generate_visualization_set(results, output_dir="visualizations", prefix=None):
    """
    Generate a complete set of visualizations for benchmark results.
    
    Args:
        results (dict): Dictionary containing benchmark results
        output_dir (str): Directory to save visualizations
        prefix (str, optional): Prefix for filenames
        
    Returns:
        dict: Dictionary mapping visualization types to file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp-based prefix if none provided
    if prefix is None:
        from datetime import datetime
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths for each visualization
    paths = {}
    
    # Generate visualizations
    try:
        # Basic metrics comparison
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in basic_metrics:
            file_path = output_path / f"{prefix}_{metric}.png"
            fig = plot_metric_comparison(results, metric=metric, output_file=file_path)
            plt.close(fig)
            paths[f"{metric}_comparison"] = file_path
        
        # Processing time comparison
        file_path = output_path / f"{prefix}_processing_time.png"
        fig = plot_processing_time_comparison(results, output_file=file_path)
        plt.close(fig)
        paths["processing_time"] = file_path
        
        # Performance vs. time plot
        file_path = output_path / f"{prefix}_perf_vs_time.png"
        fig = plot_metric_vs_time(results, output_file=file_path)
        plt.close(fig)
        paths["perf_vs_time"] = file_path
        
        # ROC curves
        file_path = output_path / f"{prefix}_roc_curves.png"
        fig = plot_roc_curves(results, output_file=file_path)
        if fig:
            plt.close(fig)
            paths["roc_curves"] = file_path
        
        # False alarm comparison
        file_path = output_path / f"{prefix}_false_alarm.png"
        fig = plot_false_alarm_comparison(results, output_file=file_path)
        plt.close(fig)
        paths["false_alarm"] = file_path
        
        # Detection latency comparison
        file_path = output_path / f"{prefix}_detection_latency.png"
        fig = plot_detection_latency_comparison(results, output_file=file_path)
        plt.close(fig)
        paths["detection_latency"] = file_path
        
        # Dataset comparison (if available)
        try:
            file_path = output_path / f"{prefix}_dataset_comparison.png"
            fig = plot_metrics_by_dataset(results, output_file=file_path)
            plt.close(fig)
            paths["dataset_comparison"] = file_path
        except (ValueError, KeyError):
            # Might not have dataset-specific data
            pass
            
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    return paths 