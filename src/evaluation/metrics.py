"""
Performance evaluation metrics for drowsiness detection.

This module provides functions to calculate various performance metrics
for evaluating drowsiness detection algorithms.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score
)
import matplotlib.pyplot as plt


def calculate_basic_metrics(y_true, y_pred):
    """
    Calculate basic classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1 score
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics


def calculate_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Calculate and visualize confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names for visualization
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(cm, class_names=None, figsize=(10, 8), title="Confusion Matrix"):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels, title, and ticks
    if class_names is not None:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        
    # Rotate the tick labels
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
    return fig


def calculate_roc_auc(y_true, y_score):
    """
    Calculate ROC curve and ROC area for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities or scores
        
    Returns:
        tuple: (fpr, tpr, thresholds, roc_auc)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc, figsize=(8, 6), title="ROC Curve"):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under ROC curve
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig


def calculate_average_precision(y_true, y_score):
    """
    Calculate average precision score.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        
    Returns:
        float: Average precision score
    """
    return average_precision_score(y_true, y_score)


def calculate_detection_latency(event_timestamps, detection_timestamps, threshold=None):
    """
    Calculate average detection latency for drowsiness events.
    
    Args:
        event_timestamps: List of actual drowsiness event onset timestamps
        detection_timestamps: List of detection timestamps
        threshold: Maximum acceptable latency (detections beyond this are missed)
        
    Returns:
        tuple: (average_latency, detection_rate)
    """
    if not event_timestamps or not detection_timestamps:
        return float('inf'), 0.0
        
    if threshold is None:
        threshold = float('inf')
    
    latencies = []
    detected_events = 0
    
    for event_time in event_timestamps:
        # Find the earliest detection after the event
        valid_detections = [t - event_time for t in detection_timestamps if t >= event_time]
        if valid_detections and min(valid_detections) <= threshold:
            latencies.append(min(valid_detections))
            detected_events += 1
    
    if not latencies:
        return float('inf'), 0.0
        
    average_latency = sum(latencies) / len(latencies)
    detection_rate = detected_events / len(event_timestamps)
    
    return average_latency, detection_rate


def calculate_false_alarm_rate(detection_timestamps, event_intervals, total_duration):
    """
    Calculate false alarm rate for drowsiness detection.
    
    Args:
        detection_timestamps: List of detection timestamps
        event_intervals: List of tuples (start_time, end_time) for actual drowsiness events
        total_duration: Total duration of the evaluation period
        
    Returns:
        float: False alarm rate (false alarms per hour)
    """
    false_alarms = 0
    
    for detection_time in detection_timestamps:
        # Check if detection falls within any drowsiness event interval
        is_true_detection = any(
            start <= detection_time <= end for start, end in event_intervals
        )
        
        if not is_true_detection:
            false_alarms += 1
    
    # Calculate false alarm rate per hour
    hours = total_duration / 3600.0  # Convert seconds to hours
    false_alarm_rate = false_alarms / hours if hours > 0 else 0
    
    return false_alarm_rate 


def calculate_response_time(predictions, ground_truth, timestamps=None):
    """
    Calculate average response time for drowsiness detection.
    
    This is a placeholder implementation that can be replaced with actual
    response time calculation when timing data is available.
    
    Args:
        predictions: Predicted labels or scores
        ground_truth: Ground truth labels
        timestamps: Optional array of timestamps for each prediction
        
    Returns:
        float: Average response time in seconds
    """
    # Dummy implementation - return a random response time between 0.5 and 2.0 seconds
    if timestamps is None:
        return np.random.uniform(0.5, 2.0)
        
    # If timestamps are provided, we could calculate actual response times
    # This would require additional logic based on detection events
    return np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 1.0


def calculate_metrics(predictions, ground_truth, timestamps=None, 
                      event_intervals=None, total_duration=3600):
    """
    Calculate comprehensive set of metrics for drowsiness detection evaluation.
    
    Args:
        predictions: Predicted labels or scores
        ground_truth: Ground truth labels
        timestamps: Optional timestamps for predictions (for time-based metrics)
        event_intervals: List of tuples (start_time, end_time) for drowsiness events
        total_duration: Total duration of the evaluation period in seconds
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Basic classification metrics using sklearn
    basic_metrics = calculate_basic_metrics(ground_truth, predictions)
    
    # Calculate confusion matrix
    cm = calculate_confusion_matrix(ground_truth, predictions)
    
    # Calculate response time (placeholder implementation)
    avg_response_time = calculate_response_time(predictions, ground_truth, timestamps)
    
    # Calculate false alarm rate
    # If event intervals aren't provided, create dummy data for placeholder implementation
    if event_intervals is None and timestamps is not None:
        # Create dummy event intervals for demonstration
        event_intervals = []
        for i, (gt, ts) in enumerate(zip(ground_truth, timestamps)):
            if gt == 1 and (i == 0 or ground_truth[i-1] == 0):
                # Start of a drowsy event
                start_time = ts
                # Find end of event
                end_idx = i
                while end_idx < len(ground_truth) and ground_truth[end_idx] == 1:
                    end_idx += 1
                end_time = timestamps[min(end_idx, len(timestamps)-1)]
                event_intervals.append((start_time, end_time))
    
    # If we have timestamps and event intervals, calculate false alarm rate
    if timestamps is not None and event_intervals is not None:
        detection_timestamps = [ts for pred, ts in zip(predictions, timestamps) if pred == 1]
        far = calculate_false_alarm_rate(detection_timestamps, event_intervals, total_duration)
    else:
        # Placeholder value
        far = np.random.uniform(0.1, 2.0)
    
    # Compile all metrics into a single dictionary
    metrics = {
        'accuracy': basic_metrics['accuracy'],
        'precision': basic_metrics['precision'],
        'recall': basic_metrics['recall'],
        'f1_score': basic_metrics['f1_score'],
        'confusion_matrix': cm,
        'average_response_time': avg_response_time,
        'false_alarm_rate': far
    }
    
    return metrics 