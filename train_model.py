#!/usr/bin/env python3
"""
MNIST Classifier Training Script
================================

This script trains a CNN model on the MNIST dataset for AI security research.
The trained model will be used to demonstrate adversarial vulnerabilities.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from utils import create_directories, save_training_history, visualize_training_history

class MNISTClassifier:
    """
    A CNN classifier for MNIST digit recognition.
    
    This class handles the complete pipeline:
    - Data loading and preprocessing
    - Model architecture definition
    - Training with callbacks
    - Evaluation and saving
    """
    
    def __init__(self, config=None):
        """Initialize the classifier with configuration."""
        self.config = config or Config()
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
        # Create necessary directories
        create_directories()
        
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset."""
        print("Loading MNIST dataset...")
        
        # Load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        self.x_train = self.x_train.reshape(-1, 28, 28, 1)
        self.x_test = self.x_test.reshape(-1, 28, 28, 1)
        
        print(f"Training set shape: {self.x_train.shape}")
        print(f"Test set shape: {self.x_test.shape}")
        print(f"Number of classes: {len(np.unique(self.y_train))}")
        
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def build_model(self):
        """Build the CNN architecture."""
        print("Building CNN model...")
        
        self.model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),  # Add dropout for regularization
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=self.config.OPTIMIZER,
            loss=self.config.LOSS_FUNCTION,
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self):
        """Train the model with callbacks."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        print(f"Starting training for {self.config.EPOCHS} epochs...")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.config.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(self.x_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        print("Evaluating model on test set...")
        
        # Get predictions
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        predictions = self.model.predict(self.x_test, verbose=0)
        
        # Calculate additional metrics
        predicted_classes = np.argmax(predictions, axis=1)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save evaluation results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'evaluation_date': datetime.now().isoformat(),
            'model_path': self.config.MODEL_PATH
        }
        
        with open(self.config.RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_accuracy
    
    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(self.config.MODEL_PATH)
        print(f"Model saved to {self.config.MODEL_PATH}")
    
    def visualize_sample_predictions(self, num_samples=10):
        """Visualize model predictions on sample images."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Get random samples
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        sample_images = self.x_test[indices]
        sample_labels = self.y_test[indices]
        
        # Get predictions
        predictions = self.model.predict(sample_images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        # Create visualization
        plt.figure(figsize=(15, 6))
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
            
            true_label = sample_labels[i]
            pred_label = predicted_labels[i]
            confidence = np.max(predictions[i])
            
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}', 
                     color=color, fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'sample_predictions.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main training pipeline."""
    print("=== MNIST Classifier Training ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Initialize classifier
    classifier = MNISTClassifier()
    
    # Load and preprocess data
    classifier.load_and_preprocess_data()
    
    # Build model
    classifier.build_model()
    
    # Train model
    classifier.train()
    
    # Evaluate model
    accuracy = classifier.evaluate()
    
    # Save training history
    if classifier.history:
        save_training_history(classifier.history)
        visualize_training_history(classifier.history)
    
    # Visualize sample predictions
    classifier.visualize_sample_predictions()
    
    print(f"\n=== Training Complete ===")
    print(f"Final test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {classifier.config.MODEL_PATH}")
    print(f"Results saved to: {classifier.config.RESULTS_PATH}")

if __name__ == "__main__":
    main()
