#!/usr/bin/env python3
"""
Configuration file for AI Security MNIST Exercise
=================================================

This file contains all configuration parameters for the project.
Modify these settings to experiment with different parameters.
"""

import os

class Config:
    """Configuration class containing all project parameters."""
    
    # Project directories
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    
    # Model parameters
    MODEL_NAME = 'mnist_classifier'
    MODEL_PATH = os.path.join(MODELS_DIR, f'{MODEL_NAME}.h5')
    
    # Training parameters
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'sparse_categorical_crossentropy'
    
    # Data parameters
    IMAGE_SIZE = (28, 28)
    NUM_CLASSES = 10
    VALIDATION_SPLIT = 0.2
    
    # Adversarial attack parameters
    EPSILON_VALUES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    DEFAULT_EPSILON = 0.1
    
    # Results and logging
    RESULTS_PATH = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.json')
    
    # Visualization parameters
    FIGURE_SIZE = (12, 8)
    DPI = 150
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Attack configurations
    ATTACK_METHODS = {
        'FGSM': {
            'name': 'Fast Gradient Sign Method',
            'epsilon_range': [0.01, 0.1, 0.3],
            'description': 'Single-step attack using gradient sign'
        },
        'PGD': {
            'name': 'Projected Gradient Descent',
            'epsilon_range': [0.01, 0.1, 0.3],
            'iterations': 10,
            'step_size': 0.01,
            'description': 'Multi-step iterative attack'
        }
    }
    
    # Defense configurations
    DEFENSE_METHODS = {
        'ADVERSARIAL_TRAINING': {
            'name': 'Adversarial Training',
            'epsilon': 0.1,
            'description': 'Train on adversarial examples'
        },
        'INPUT_PREPROCESSING': {
            'name': 'Input Preprocessing',
            'methods': ['gaussian_blur', 'median_filter', 'bit_depth_reduction'],
            'description': 'Preprocess inputs to remove adversarial noise'
        }
    }
    
    @classmethod
    def get_attack_config(cls, attack_name):
        """Get configuration for a specific attack method."""
        return cls.ATTACK_METHODS.get(attack_name.upper(), {})
    
    @classmethod
    def get_defense_config(cls, defense_name):
        """Get configuration for a specific defense method."""
        return cls.DEFENSE_METHODS.get(defense_name.upper(), {})
    
    @classmethod
    def print_config(cls):
        """Print current configuration settings."""
        print("=== Current Configuration ===")
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Model Path: {cls.MODEL_PATH}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Default Epsilon: {cls.DEFAULT_EPSILON}")
        print(f"Random Seed: {cls.RANDOM_SEED}")
        print("============================")

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Configuration for development environment."""
    EPOCHS = 5  # Faster training for development
    BATCH_SIZE = 64

class ProductionConfig(Config):
    """Configuration for production environment."""
    EPOCHS = 20  # More thorough training
    BATCH_SIZE = 256

class TestingConfig(Config):
    """Configuration for testing environment."""
    EPOCHS = 2  # Minimal training for tests
    BATCH_SIZE = 32
    MODEL_NAME = 'test_model'

# Factory function to get appropriate config
def get_config(environment='default'):
    """
    Get configuration based on environment.
    
    Args:
        environment (str): 'development', 'production', 'testing', or 'default'
        
    Returns:
        Config: Appropriate configuration class
    """
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
        'default': Config
    }
    
    return configs.get(environment, Config)
