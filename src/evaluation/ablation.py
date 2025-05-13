"""
Ablation study module for drowsiness detection system.

This module provides functionality to perform ablation studies by systematically
removing or modifying components of the drowsiness detection system to evaluate
their contribution to overall performance.
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
from collections import OrderedDict
import itertools

from src.evaluation.metrics import calculate_basic_metrics
from src.evaluation.dataset_loader import get_dataset_loader

# Configure logging
logger = logging.getLogger(__name__)


class AblationStudy:
    """Ablation study for drowsiness detection system components."""
    
    def __init__(self, base_model_fn, output_dir="results/ablation", cache_dir="results/cache"):
        """
        Initialize the ablation study.
        
        Args:
            base_model_fn: Function for the complete model (baseline)
            output_dir: Directory to save ablation results
            cache_dir: Directory to cache intermediate results
        """
        self.base_model_fn = base_model_fn
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.components = OrderedDict()
        self.variants = {}
        self.results = {}
        
    def register_component(self, component_name, enabled=True):
        """
        Register a system component for ablation.
        
        Args:
            component_name: Name of the component
            enabled: Whether the component is enabled by default
            
        Returns:
            self: For method chaining
        """
        self.components[component_name] = enabled
        return self
    
    def register_variant(self, variant_name, component_configs, variant_fn):
        """
        Register a system variant for ablation.
        
        Args:
            variant_name: Name of the variant
            component_configs: Dict of component configurations for this variant
            variant_fn: Function that implements this variant
            
        Returns:
            self: For method chaining
        """
        self.variants[variant_name] = {
            "components": component_configs,
            "function": variant_fn
        }
        return self
    
    def generate_configurations(self):
        """
        Generate all possible component configurations for ablation.
        
        Returns:
            list: List of configurations to test
        """
        configurations = []
        
        # Add base configuration (all components enabled)
        base_config = {comp: True for comp in self.components}
        configurations.append(("base", base_config))
        
        # Add single-component ablations
        for component in self.components:
            config = base_config.copy()
            config[component] = False
            configurations.append((f"no_{component}", config))
        
        # Add registered variants
        for variant_name, variant_data in self.variants.items():
            config = base_config.copy()
            for comp_name, comp_value in variant_data["components"].items():
                if comp_name in config:
                    config[comp_name] = comp_value
            configurations.append((variant_name, config))
        
        return configurations
    
    def evaluate_on_dataset(self, dataset_name, dataset_path, split="test", subset=None):
        """
        Run ablation study on a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset
            split: Data split to evaluate on
            subset: Optional dataset subset
            
        Returns:
            dict: Ablation study results
        """
        logger.info(f"Running ablation study on dataset: {dataset_name}")
        
        # Load dataset
        dataset = get_dataset_loader(dataset_name, dataset_path, self.cache_dir)
        
        # Load data
        X, y_true, metadata = dataset.load_data(split=split, subset=subset)
        
        # Generate configurations to test
        configurations = self.generate_configurations()
        
        # Initialize results for this dataset
        dataset_results = {
            "metadata": {
                "dataset": dataset_name,
                "split": split,
                "subset": subset,
                "samples": len(X),
                "timestamp": datetime.now().isoformat()
            },
            "configurations": {},
            "comparative_metrics": {}
        }
        
        # Evaluate each configuration
        for config_name, config in configurations:
            logger.info(f"Evaluating configuration: {config_name}")
            
            start_time = time.time()
            
            try:
                # Determine which function to use
                if config_name in self.variants:
                    predict_fn = self.variants[config_name]["function"]
                else:
                    # For ablation configurations, use base model with component flags
                    predict_fn = lambda x, **kwargs: self.base_model_fn(x, component_flags=config, **kwargs)
                
                # Run prediction
                y_pred = predict_fn(X)
                
                # Calculate metrics
                execution_time = time.time() - start_time
                metrics = calculate_basic_metrics(y_true, y_pred)
                metrics["execution_time"] = execution_time
                metrics["samples_per_second"] = len(X) / execution_time
                
                # Save configuration results
                dataset_results["configurations"][config_name] = {
                    "components": config,
                    "metrics": metrics,
                    "predictions": y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
                }
                
            except Exception as e:
                logger.error(f"Error evaluating configuration '{config_name}': {str(e)}")
                dataset_results["configurations"][config_name] = {
                    "components": config,
                    "metrics": {"error": str(e)},
                    "status": "failed"
                }
        
        # Calculate comparative metrics
        base_metrics = dataset_results["configurations"]["base"]["metrics"]
        
        for config_name, config_data in dataset_results["configurations"].items():
            if config_name == "base" or "error" in config_data["metrics"]:
                continue
                
            config_metrics = config_data["metrics"]
            changes = {}
            
            for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
                if metric_name in base_metrics and metric_name in config_metrics:
                    abs_change = config_metrics[metric_name] - base_metrics[metric_name]
                    rel_change = abs_change / base_metrics[metric_name] if base_metrics[metric_name] != 0 else float('inf')
                    
                    changes[metric_name] = {
                        "absolute": abs_change,
                        "relative": rel_change,
                        "base_value": base_metrics[metric_name],
                        "config_value": config_metrics[metric_name]
                    }
            
            dataset_results["comparative_metrics"][config_name] = changes
        
        # Save results for this dataset
        self.results[dataset_name] = dataset_results
        
        return dataset_results
    
    def evaluate_on_multiple_datasets(self, datasets):
        """
        Run ablation study on multiple datasets.
        
        Args:
            datasets: Dict mapping dataset names to paths
            
        Returns:
            dict: Aggregated ablation study results
        """
        for dataset_name, dataset_path in datasets.items():
            self.evaluate_on_dataset(dataset_name, dataset_path)
            
        # Calculate aggregate results across all datasets
        self._calculate_aggregate_results()
            
        return self.results
    
    def _calculate_aggregate_results(self):
        """
        Calculate aggregate results across all datasets.
        
        Returns:
            dict: Aggregate results
        """
        if not self.results:
            return {}
            
        # Initialize aggregate results
        aggregate_results = {
            "metadata": {
                "datasets": list(self.results.keys()),
                "timestamp": datetime.now().isoformat()
            },
            "configurations": {},
            "comparative_metrics": {}
        }
        
        # Get all configuration names
        all_configs = set()
        for dataset_results in self.results.values():
            all_configs.update(dataset_results["configurations"].keys())
        
        # Aggregate metrics for each configuration
        for config_name in all_configs:
            config_values = []
            metrics_values = {}
            
            for dataset_name, dataset_results in self.results.items():
                if config_name in dataset_results["configurations"]:
                    config_data = dataset_results["configurations"][config_name]
                    
                    if "error" not in config_data["metrics"]:
                        config_values.append(config_data["components"])
                        
                        for metric_name, metric_value in config_data["metrics"].items():
                            if isinstance(metric_value, (int, float)):
                                if metric_name not in metrics_values:
                                    metrics_values[metric_name] = []
                                metrics_values[metric_name].append(metric_value)
            
            # Compute average metrics
            avg_metrics = {}
            for metric_name, values in metrics_values.items():
                avg_metrics[metric_name] = sum(values) / len(values)
            
            # Merge component configurations (use most common value for each component)
            merged_config = {}
            if config_values:
                for component in config_values[0]:
                    values = [config[component] for config in config_values]
                    merged_config[component] = max(set(values), key=values.count)
            
            aggregate_results["configurations"][config_name] = {
                "components": merged_config,
                "metrics": avg_metrics
            }
        
        # Calculate comparative metrics
        if "base" in aggregate_results["configurations"]:
            base_metrics = aggregate_results["configurations"]["base"]["metrics"]
            
            for config_name, config_data in aggregate_results["configurations"].items():
                if config_name == "base":
                    continue
                    
                config_metrics = config_data["metrics"]
                changes = {}
                
                for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
                    if metric_name in base_metrics and metric_name in config_metrics:
                        abs_change = config_metrics[metric_name] - base_metrics[metric_name]
                        rel_change = abs_change / base_metrics[metric_name] if base_metrics[metric_name] != 0 else float('inf')
                        
                        changes[metric_name] = {
                            "absolute": abs_change,
                            "relative": rel_change,
                            "base_value": base_metrics[metric_name],
                            "config_value": config_metrics[metric_name]
                        }
                
                aggregate_results["comparative_metrics"][config_name] = changes
        
        self.aggregate_results = aggregate_results
        return aggregate_results
    
    def save_results(self, filename=None):
        """
        Save ablation study results to a file.
        
        Args:
            filename: Name of the file to save results to
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ablation_results_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Ensure aggregate results are calculated
        if not hasattr(self, "aggregate_results"):
            self._calculate_aggregate_results()
        
        # Prepare serializable results
        serializable_results = {
            "aggregate": self.aggregate_results,
            "datasets": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Ablation study results saved to: {output_path}")
        
        return output_path
    
    def load_results(self, filepath):
        """
        Load ablation study results from a file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            self: For method chaining
        """
        with open(filepath, 'r') as f:
            loaded_results = json.load(f)
            
        if "datasets" in loaded_results:
            self.results = loaded_results["datasets"]
            
        if "aggregate" in loaded_results:
            self.aggregate_results = loaded_results["aggregate"]
            
        return self
    
    def plot_component_impact(self, metric="accuracy", output_file=None):
        """
        Plot the impact of ablating each component on a specific metric.
        
        Args:
            metric: Metric to analyze ('accuracy', 'precision', etc.)
            output_file: Path to save the plot (default: display only)
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if not hasattr(self, "aggregate_results"):
            self._calculate_aggregate_results()
        
        # Extract component impact data
        components = []
        impacts = []
        
        base_value = self.aggregate_results["configurations"]["base"]["metrics"].get(metric, 0)
        
        for config_name, config_data in self.aggregate_results["configurations"].items():
            if config_name.startswith("no_"):
                component_name = config_name[3:]  # Remove "no_" prefix
                config_value = config_data["metrics"].get(metric, 0)
                impact = base_value - config_value
                
                components.append(component_name)
                impacts.append(impact)
        
        # Sort by impact
        sorted_indices = np.argsort(impacts)
        components = [components[i] for i in sorted_indices]
        impacts = [impacts[i] for i in sorted_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(components))
        ax.barh(y_pos, impacts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(components)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel(f'Impact on {metric.capitalize()} (higher = more important)')
        ax.set_title(f'Component Impact on {metric.capitalize()}')
        
        # Add values on bars
        for i, v in enumerate(impacts):
            ax.text(v + 0.001, i, f'{v:.4f}', va='center')
        
        fig.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Component impact plot saved to: {output_file}")
            
        return fig
    
    def generate_report(self, output_file=None):
        """
        Generate a comprehensive ablation study report.
        
        Args:
            output_file: Path to save the report (default: auto-generated name)
            
        Returns:
            str: Path to the generated report
        """
        if not hasattr(self, "aggregate_results"):
            self._calculate_aggregate_results()
            
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"ablation_report_{timestamp}.html"
        else:
            output_file = Path(output_file)
            
        # Create basic HTML report
        html_content = []
        html_content.append("<!DOCTYPE html>")
        html_content.append("<html><head>")
        html_content.append("<title>Drowsiness Detection Ablation Study Report</title>")
        html_content.append("<style>")
        html_content.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_content.append("table { border-collapse: collapse; width: 100%; }")
        html_content.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_content.append("th { background-color: #f2f2f2; }")
        html_content.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        html_content.append("h1, h2, h3 { color: #333; }")
        html_content.append(".positive { color: green; }")
        html_content.append(".negative { color: red; }")
        html_content.append("</style>")
        html_content.append("</head><body>")
        
        # Header
        html_content.append("<h1>Drowsiness Detection Ablation Study Report</h1>")
        html_content.append(f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary
        html_content.append("<h2>Component Impact Summary</h2>")
        html_content.append("<p>The table below shows the impact of removing each component from the system.</p>")
        
        # Component impact table
        html_content.append("<table>")
        html_content.append("<tr><th>Component</th><th>Impact on Accuracy</th><th>Impact on F1 Score</th></tr>")
        
        base_metrics = self.aggregate_results["configurations"]["base"]["metrics"]
        
        for config_name, config_data in self.aggregate_results["configurations"].items():
            if config_name.startswith("no_"):
                component_name = config_name[3:]  # Remove "no_" prefix
                metrics = config_data["metrics"]
                
                acc_impact = base_metrics.get("accuracy", 0) - metrics.get("accuracy", 0)
                f1_impact = base_metrics.get("f1_score", 0) - metrics.get("f1_score", 0)
                
                html_content.append("<tr>")
                html_content.append(f"<td>{component_name}</td>")
                html_content.append(f"<td class=\"{'negative' if acc_impact > 0 else 'positive'}\">{acc_impact:.4f}</td>")
                html_content.append(f"<td class=\"{'negative' if f1_impact > 0 else 'positive'}\">{f1_impact:.4f}</td>")
                html_content.append("</tr>")
                
        html_content.append("</table>")
        
        # Variants comparison
        html_content.append("<h2>Variant Comparison</h2>")
        html_content.append("<p>The table below compares different system variants against the baseline.</p>")
        
        # Variant comparison table
        html_content.append("<table>")
        html_content.append("<tr><th>Variant</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Change from Base</th></tr>")
        
        for config_name, config_data in self.aggregate_results["configurations"].items():
            if not config_name.startswith("no_") or config_name == "base":
                metrics = config_data["metrics"]
                
                html_content.append("<tr>")
                html_content.append(f"<td>{config_name}</td>")
                html_content.append(f"<td>{metrics.get('accuracy', 'N/A'):.4f}</td>")
                html_content.append(f"<td>{metrics.get('precision', 'N/A'):.4f}</td>")
                html_content.append(f"<td>{metrics.get('recall', 'N/A'):.4f}</td>")
                html_content.append(f"<td>{metrics.get('f1_score', 'N/A'):.4f}</td>")
                
                if config_name != "base" and "comparative_metrics" in self.aggregate_results:
                    comp_metrics = self.aggregate_results["comparative_metrics"].get(config_name, {}).get("f1_score", {})
                    rel_change = comp_metrics.get("relative", 0) * 100  # Convert to percentage
                    
                    if rel_change > 0:
                        html_content.append(f"<td class=\"positive\">+{rel_change:.2f}%</td>")
                    else:
                        html_content.append(f"<td class=\"negative\">{rel_change:.2f}%</td>")
                else:
                    html_content.append("<td>Baseline</td>")
                    
                html_content.append("</tr>")
                
        html_content.append("</table>")
        
        # Dataset specific results
        html_content.append("<h2>Dataset-Specific Results</h2>")
        
        for dataset_name, dataset_results in self.results.items():
            html_content.append(f"<h3>Dataset: {dataset_name}</h3>")
            
            # Metadata
            metadata = dataset_results["metadata"]
            html_content.append("<p><strong>Split:</strong> " + metadata.get("split", "N/A") + "</p>")
            html_content.append("<p><strong>Samples:</strong> " + str(metadata.get("samples", "N/A")) + "</p>")
            
            # Configuration results table
            html_content.append("<table>")
            html_content.append("<tr><th>Configuration</th><th>Accuracy</th><th>F1 Score</th><th>Execution Time (s)</th></tr>")
            
            for config_name, config_data in dataset_results["configurations"].items():
                metrics = config_data["metrics"]
                
                if "error" in metrics:
                    html_content.append("<tr>")
                    html_content.append(f"<td>{config_name}</td>")
                    html_content.append(f"<td colspan=\"3\">Error: {metrics['error']}</td>")
                    html_content.append("</tr>")
                else:
                    html_content.append("<tr>")
                    html_content.append(f"<td>{config_name}</td>")
                    html_content.append(f"<td>{metrics.get('accuracy', 'N/A'):.4f}</td>")
                    html_content.append(f"<td>{metrics.get('f1_score', 'N/A'):.4f}</td>")
                    html_content.append(f"<td>{metrics.get('execution_time', 'N/A'):.2f}</td>")
                    html_content.append("</tr>")
                    
            html_content.append("</table>")
        
        # Footer
        html_content.append("<hr>")
        html_content.append("<p><em>Generated using the Drowsiness Detection Ablation Study framework.</em></p>")
        html_content.append("</body></html>")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(html_content))
            
        logger.info(f"Ablation study report generated: {output_file}")
        
        return output_file


def run_ablation_study(base_model_fn, components, variants=None, datasets=None):
    """
    Run a comprehensive ablation study.
    
    Args:
        base_model_fn: Function for the complete model (baseline)
        components: List of component names to ablate
        variants: Dict mapping variant names to (component_configs, variant_fn) pairs
        datasets: Dict mapping dataset names to paths
        
    Returns:
        AblationStudy: Ablation study instance with results
    """
    ablation = AblationStudy(base_model_fn)
    
    # Register components
    for component in components:
        ablation.register_component(component)
    
    # Register variants
    if variants:
        for variant_name, (component_configs, variant_fn) in variants.items():
            ablation.register_variant(variant_name, component_configs, variant_fn)
    
    # Run study
    if datasets:
        ablation.evaluate_on_multiple_datasets(datasets)
    
    # Save results
    ablation.save_results()
    
    # Generate report
    ablation.generate_report()
    
    # Plot component impact
    for metric in ["accuracy", "f1_score"]:
        output_file = ablation.output_dir / f"component_impact_{metric}.png"
        ablation.plot_component_impact(metric=metric, output_file=output_file)
    
    return ablation 


def perform_ablation_study(config_path, dataset_path, output_dir="results/ablation", cache_dir="results/cache"):
    """
    Perform ablation study based on a configuration file.
    
    This function loads model configuration from a file, creates multiple variants of the model
    with different components selectively disabled, and evaluates them on a dataset.
    
    Args:
        config_path (str): Path to the configuration file (JSON format)
        dataset_path (str): Path to the dataset for evaluation
        output_dir (str): Directory to save ablation results
        cache_dir (str): Directory to cache intermediate results
        
    Returns:
        dict: Dictionary of results for comparison
    
    Example config file format:
    {
        "base_model": {
            "name": "DrowsinessDetector",
            "params": {
                "threshold": 0.5,
                "use_gpu": true
            }
        },
        "components": [
            "gaze_detector",
            "head_pose_estimator",
            "perclos_calculator",
            "blink_frequency_analyzer"
        ],
        "dataset": {
            "name": "nthu-ddd",
            "split": "test"
        },
        "output": {
            "save_results": true,
            "generate_report": true,
            "plot_metrics": ["accuracy", "f1_score"]
        }
    }
    """
    import json
    import importlib
    import sys
    from pathlib import Path
    
    logger.info(f"Starting ablation study using config from: {config_path}")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise
    
    # Extract configuration values
    base_model_config = config.get("base_model", {})
    model_name = base_model_config.get("name")
    model_params = base_model_config.get("params", {})
    components = config.get("components", [])
    
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name", Path(dataset_path).name)
    dataset_split = dataset_config.get("split", "test")
    dataset_subset = dataset_config.get("subset")
    
    output_config = config.get("output", {})
    save_results = output_config.get("save_results", True)
    generate_report = output_config.get("generate_report", True)
    plot_metrics = output_config.get("plot_metrics", ["accuracy", "f1_score"])
    
    # Create a function to load and initialize the model
    def create_model_with_flags(component_flags=None):
        """Create model with specified component flags."""
        # Find the model class in the system modules
        try:
            # Try importing from common module paths
            for module_path in ["src.models", "models", "src"]:
                try:
                    module = importlib.import_module(f"{module_path}.{model_name.lower()}")
                    model_class = getattr(module, model_name)
                    break
                except (ImportError, AttributeError):
                    continue
            else:
                # If not found in common paths, try direct import
                module = importlib.import_module(model_name.lower())
                model_class = getattr(module, model_name)
        except Exception as e:
            logger.error(f"Error importing model {model_name}: {str(e)}")
            raise ImportError(f"Could not import model class {model_name}")
        
        # Create model instance with base parameters
        model_instance = model_class(**model_params)
        
        # Apply component flags if provided
        if component_flags:
            for component, enabled in component_flags.items():
                # Convert to standard format if component is in the original list
                if component in components and not enabled:
                    component_attr = f"use_{component}"
                    # Try to disable the component
                    if hasattr(model_instance, component_attr):
                        setattr(model_instance, component_attr, False)
                        logger.info(f"Disabled component: {component}")
                    elif hasattr(model_instance, "disable_component"):
                        model_instance.disable_component(component)
                        logger.info(f"Disabled component: {component}")
                    else:
                        logger.warning(f"Could not disable component: {component} (attribute not found)")
        
        return model_instance
    
    # Function to perform prediction
    def predict_with_model(X, component_flags=None, **kwargs):
        """Wrapper to create model with flags and run prediction."""
        model = create_model_with_flags(component_flags)
        return model.predict(X)
    
    # Set up ablation study
    ablation = AblationStudy(predict_with_model, output_dir=output_dir, cache_dir=cache_dir)
    
    # Register components for ablation
    for component in components:
        ablation.register_component(component)
    
    # Perform evaluation
    results = ablation.evaluate_on_dataset(dataset_name, dataset_path, split=dataset_split, subset=dataset_subset)
    
    # Save results if requested
    if save_results:
        ablation.save_results()
    
    # Generate report if requested
    if generate_report:
        ablation.generate_report()
    
    # Plot component impact for specified metrics
    if plot_metrics:
        for metric in plot_metrics:
            output_file = Path(output_dir) / f"component_impact_{metric}.png"
            ablation.plot_component_impact(metric=metric, output_file=output_file)
    
    return {
        "summary": ablation.aggregate_results if hasattr(ablation, "aggregate_results") else {},
        "detailed": ablation.results,
        "components": components,
        "model_name": model_name
    }


def evaluate_on_dataset(model, dataset_path, dataset_type=None, split="test", subset=None, 
                      batch_size=32, preprocess_fn=None, cache_dir=None):
    """
    Evaluate a drowsiness detection model on a dataset.
    
    This is a utility function that loads a dataset and evaluates a model on it,
    calculating performance metrics.
    
    Args:
        model: Model instance with a predict() method
        dataset_path (str): Path to the dataset
        dataset_type (str, optional): Type of dataset
        split (str): Data split to use
        subset (str, optional): Subset of the dataset
        batch_size (int): Batch size for evaluation
        preprocess_fn (callable, optional): Function to preprocess images
        cache_dir (str, optional): Directory to cache dataset
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from src.evaluation.dataset_loader import load_benchmark_dataset
    from src.evaluation.metrics import calculate_metrics
    
    logger.info(f"Evaluating model on dataset: {dataset_path}")
    
    # Load dataset
    images, labels, metadata = load_benchmark_dataset(
        dataset_path,
        split=split,
        dataset_type=dataset_type,
        subset=subset,
        cache_dir=cache_dir,
        load_images=True,
        preprocess_fn=preprocess_fn
    )
    
    # Prepare batches
    total_samples = len(images)
    predictions = []
    
    # Process in batches
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch_images = images[i:batch_end]
        
        # Get predictions
        try:
            batch_predictions = model.predict(batch_images)
            predictions.extend(batch_predictions)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Fill with default predictions (0)
            predictions.extend([0] * (batch_end - i))
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    
    logger.info(f"Evaluation completed with accuracy: {metrics['accuracy']:.4f}")
    
    return {
        "metrics": metrics,
        "predictions": predictions,
        "ground_truth": labels,
        "metadata": metadata
    } 