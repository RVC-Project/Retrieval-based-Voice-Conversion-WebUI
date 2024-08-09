import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import logging

# Suppress TensorBoard event processing logs
logging.getLogger('tensorboard').setLevel(logging.ERROR)

# Import TensorBoard modules after setting logging level
from tensorboard.backend.event_processing import event_accumulator

def generate_loss_graphs(log_dir):
    """
    Generates and saves plots for the given TensorBoard logs.
    Args:
        log_dir (str): Directory with TensorBoard logs.
    """
    scalar_data = extract_scalar_data(log_dir)
    plot_scalar_data(scalar_data, log_dir)

def smooth(scalars: List[float], weight: float) -> List[float]:
    """
    Smooths the given list of scalars using exponential smoothing.

    Args:
        scalars (List[float]): List of scalar values to be smoothed.
        weight (float): Weight for smoothing, between 0 and 1.

    Returns:
        List[float]: Smoothed scalar values.
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def extract_scalar_data(log_dir):
    """
    Extracts specific scalar data from TensorBoard logs.

    Args:
        log_dir (str): Directory with TensorBoard logs.

    Returns:
        dict: A dictionary where keys are scalar names and values are lists of scalar events.
    """

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    scalar_data = {}
    desired_tags = ["loss/d/total", "loss/g/total", "loss/g/fm", "loss/g/mel", "loss/g/kl"]

    for tag in desired_tags:
        if tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(tag)
            scalar_data[tag] = {}
            previous_value = 0.0  # Initialize fallback value

            for event in scalar_events:
                value = event.value
                # Check if value is NaN, use previous value or fallback to 0.0
                if np.isnan(value):
                    value = previous_value

                if event.step not in scalar_data[tag]:
                    scalar_data[tag][event.step] = [value]
                else:
                    scalar_data[tag][event.step].append(value)

                previous_value = value  # Update previous value for the next iteration

            # Calculate the average for each step. Restarting training can cause multiple events for the same step.
            scalar_data[tag] = {step: sum(values) / len(values) for step, values in scalar_data[tag].items()}
        else:
            print(f"Tag: {tag} not found in the TensorBoard logs.")

    return scalar_data

def sanitize_filename(filename):
    """
    Sanitize the filename by replacing invalid characters with underscores.
    
    Args:
        filename (str): The original filename or tag to sanitize.
        
    Returns:
        str: Sanitized filename.
    """
    # Replace slashes and other invalid characters with underscores
    return filename.replace('/', '_').replace('\\', '_')

def plot_scalar_data(scalar_data, log_dir, output_dir="loss_graphs", smooth_weight=0.75):
    """
    Generates and saves plots for the given scalar data, with optional smoothing.

    Args:
        scalar_data (dict): A dictionary where keys are scalar names and values are lists of scalar events.
        log_dir (str): Base directory where the generated JPEG files will be saved.
        output_dir (str): Subdirectory under `log_dir` where the JPEG files will be saved.
        smooth_weight (float): Weight for smoothing, between 0 and 1.
    """
    loss_graph_dir = os.path.join(log_dir, output_dir)
    if not os.path.exists(loss_graph_dir):
        os.makedirs(loss_graph_dir)
    
    for tag, events in scalar_data.items():
        # Sanitize the tag for use in the filename
        sanitized_tag = sanitize_filename(tag)
        file_path = os.path.join(loss_graph_dir, f'{sanitized_tag}.jpeg')

        # Extract steps and values from the scalar events
        steps = list(events.keys())
        values = list(events.values())

        # Print the last tag, step, and value
        if steps and values:  # Ensure that the list is not empty
            last_step = steps[-1]
            last_value = values[-1]
            print(f'Last entry - Tag: {tag}, Step: {last_step}, Value: {last_value}')

        # Apply smoothing
        smoothed_values = smooth(values, smooth_weight)

        plt.figure(figsize=(20, 12))
        plt.plot(steps, values, label=f'{tag} (original)')
        plt.plot(steps, smoothed_values, label=f'{tag} (smoothed)', linestyle='--')
        plt.xlabel('Steps')
        plt.ylabel(tag)
        plt.title(f'{tag} over time')
        plt.yscale('log')  # Set y-axis to logarithmic scale for better visualization, reminder another approach could be identify a cutoff point and disregard a few datapoints at the beginning of the training
        plt.grid(True)
        plt.legend()
        plt.savefig(file_path)
        plt.close()
