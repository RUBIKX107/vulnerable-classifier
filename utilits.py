#!/usr/bin/env python3
"""
Utility functions for AI Security MNIST Exercise
================================================

This module contains helper functions used throughout the project.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from config import Config

def create_directories():
    """Create necessary project directories if they don't exist."""
    directories = [
        Config.DATA_DIR,
        Config.MODELS_DIR,
        Config.RESULTS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("Project directories created/verified.")

def set_random_seeds(seed=None):
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = Config.RANDOM_SEED
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print(f"Random seed set to: {seed}")

def save_training_history(history, filepath=None):
    """
    Save training history to JSON file.
    
    Args:
        history: Keras training history object
        filepath: Path to save the history (optional)
    """
    if filepath is None:
        filepath = Config.TRAINING_HISTORY_PATH
    
    # Convert history to serializable format
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'epochs': len(history.history['loss'])
    }
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {filepath}")

def load_training_history(filepath=None):
    """Load training history from JSON file."""
    if filepath is None:
        filepath = Config.TRAINING_HISTORY_PATH
    
    with open(filepath, 'r') as f:
        history_dict = json.load(f)
    
    return history_dict

def visualize_training_history(history, save_path=None):
    """
    Visualize training history with loss and accuracy plots.
    
    Args:
        history: Keras training history object or loaded history dict
        save_path: Path to save the plot (optional)
    """
    # Handle both history object and dict
    if hasattr(history, 'history'):
        hist = history.history
    else:
        hist = history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation loss
    ax1.plot(hist['loss'], label='Training Loss', color='blue')
    ax1.plot(hist['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training & validation accuracy
    ax2.plot(hist['accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(hist['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(Config.RESULTS_DIR, 'training_history.png')
    
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    plt.show()
    
    print(f"Training history plot saved to: {save_path}")

def load_model(model_path=None):
    """Load a trained model."""
    if model_path is None:
        model_path = Config.MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    return model

def preprocess_image(image):
    """
    Preprocess a single image for model input.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure image is float32 and normalized
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    if image.max() > 1.0:
        image = image / 255.0
    
    # Ensure correct shape for model input
    if len(image.shape) == 2:  # Grayscale image
        image = image.reshape(1, 28, 28, 1)
    elif len(image.shape) == 3 and image.shape[-1] == 1:  # Single image
        image = image.reshape(1, 28, 28, 1)
    
    return image

def calculate_model_confidence(predictions):
    """
    Calculate confidence metrics for model predictions.
    
    Args:
        predictions: Model prediction probabilities
        
    Returns:
        Dictionary with confidence metrics
    """
    max_probs = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate entropy as uncertainty measure
    entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)
    
    return {
        'max_probability': max_probs,
        'predicted_class': predicted_classes,
        'entropy': entropy,
        'mean_confidence': np.mean(max_probs),
        'std_confidence': np.std(max_probs)
    }

def visualize_images_grid(images, labels, predictions=None, title="Images", 
                         rows=2, cols=5, figsize=(15, 6)):
    """
    Visualize a grid of images with labels and optional predictions.
    
    Args:
        images: Array of images to display
        labels: True labels
        predictions: Optional predicted labels
        title: Plot title
        rows, cols: Grid dimensions
        figsize: Figure size
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Display image
            ax.imshow(images[i].reshape(28, 28), cmap='gray')
            
            # Create label text
            label_text = f'True: {labels[i]}'
            if predictions is not None:
                color = 'green' if labels[i] == predictions[i] else 'red'
                label_text += f'\nPred: {predictions[i]}'
            else:
                color = 'black'
            
            ax.set_title(label_text, color=color, fontsize=10)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_adversarial_examples(original_images, adversarial_images, 
                            original_labels, adversarial_predictions, 
                            epsilon, save_dir=None):
    """
    Save adversarial examples and their metadata.
    
    Args:
        original_images: Original clean images
        adversarial_images: Generated adversarial images
        original_labels: True labels
        adversarial_predictions: Predictions on adversarial images
        epsilon: Attack strength parameter
        save_dir: Directory to save results
    """
    if save_dir is None:
        save_dir = Config.RESULTS_DIR
    
    # Create filename with epsilon value
    filename = f'adversarial_examples_eps_{epsilon:.3f}'
    
    # Save images and metadata
    np.savez(os.path.join(save_dir, f'{filename}.npz'),
             original_images=original_images,
             adversarial_images=adversarial_images,
             original_labels=original_labels,
             adversarial_predictions=adversarial_predictions,
             epsilon=epsilon)
    
    print(f"Adversarial examples saved to: {save_dir}/{filename}.npz")

def compute_attack_success_rate(original_labels, adversarial_predictions):
    """
    Compute the success rate of an adversarial attack.
    
    Args:
        original_labels: True labels
        adversarial_predictions: Predictions on adversarial examples
        
    Returns:
        Success rate as percentage
    """
    successful_attacks = np.sum(original_labels != adversarial_predictions)
    total_examples = len(original_labels)
    success_rate = (successful_attacks / total_examples) * 100
    
    return success_rate

def print_attack_summary(epsilon, success_rate, num_examples):
    """Print a summary of attack results."""
    print(f"\n=== Attack Summary ===")
    print(f"Epsilon: {epsilon}")
    print(f"Total examples: {num_examples}")
    print(f"Successful attacks: {int(success_rate * num_examples / 100)}")
    print(f"Success rate: {success_rate:.2f}%")
    print("=====================")

def create_experiment_log(experiment_name, parameters, results, 
                         log_path=None):
    """
    Create a log entry for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        parameters: Dictionary of experiment parameters
        results: Dictionary of experiment results
        log_path: Path to save the log
    """
    if log_path is None:
        log_path = os.path.join(Config.RESULTS_DIR, 'experiment_log.json')
    
    # Load existing log or create new one
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []
    
    # Add new experiment entry
    experiment_entry = {
