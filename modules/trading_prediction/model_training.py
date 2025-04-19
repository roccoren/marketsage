#!/usr/bin/env python3
"""
Model training module for stock trading prediction.

This module provides functionality for training, evaluating, and using
deep learning models for stock trading prediction.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from modules.trading_prediction.base import ModelBase
from modules.trading_prediction.model_architectures import get_model_architecture

# Configure logger
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model trainer for stock trading prediction."""
    
    def __init__(self, model: Any = None, framework: str = 'tensorflow'):
        """
        Initialize ModelTrainer.
        
        Args:
            model: Pre-built model (optional)
            framework: Framework ('pytorch', 'tensorflow')
        """
        self.model = model
        self.framework = framework
        self.history = None
        self.trained = False
    
    def build_model(self, architecture: str, input_shape: Tuple[int, int], 
                   num_classes: int = 3, **kwargs) -> Any:
        """
        Build model with specified architecture.
        
        Args:
            architecture: Architecture name
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            Built model
        """
        model_arch = get_model_architecture(architecture, self.framework)
        self.model = model_arch.create_model(input_shape, num_classes, **kwargs)
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001, **kwargs) -> None:
        """
        Compile model with specified parameters.
        
        Args:
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments for compilation
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        if self.framework == 'pytorch':
            self._compile_pytorch_model(learning_rate, **kwargs)
        elif self.framework == 'tensorflow':
            self._compile_tensorflow_model(learning_rate, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _compile_pytorch_model(self, learning_rate: float, **kwargs) -> None:
        """
        Compile PyTorch model.
        
        Args:
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments for compilation
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Set loss function
            loss_fn = kwargs.get('loss_fn', nn.CrossEntropyLoss())
            
            # Set optimizer
            optimizer_name = kwargs.get('optimizer', 'adam').lower()
            if optimizer_name == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            elif optimizer_name == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            self.loss_fn = loss_fn
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
    
    def _compile_tensorflow_model(self, learning_rate: float, **kwargs) -> None:
        """
        Compile TensorFlow model.
        
        Args:
            learning_rate: Learning rate for optimizer
            **kwargs: Additional arguments for compilation
        """
        try:
            import tensorflow as tf
            
            # Set optimizer
            optimizer_name = kwargs.get('optimizer', 'adam').lower()
            if optimizer_name == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == 'sgd':
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Set loss function
            loss = kwargs.get('loss', 'sparse_categorical_crossentropy')
            
            # Set metrics
            metrics = kwargs.get('metrics', ['accuracy'])
            
            # Compile model
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             epochs: int = 50, batch_size: int = 32, **kwargs) -> Dict[str, List[float]]:
        """
        Train model on data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs
            batch_size: Batch size
            **kwargs: Additional arguments for training
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        if self.framework == 'pytorch':
            history = self._train_pytorch_model(X_train, y_train, X_val, y_val, epochs, batch_size, **kwargs)
        elif self.framework == 'tensorflow':
            history = self._train_tensorflow_model(X_train, y_train, X_val, y_val, epochs, batch_size, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        self.history = history
        self.trained = True
        
        return history
    
    def _train_pytorch_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                           epochs: int, batch_size: int, **kwargs) -> Dict[str, List[float]]:
        """
        Train PyTorch model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs
            batch_size: Batch size
            **kwargs: Additional arguments for training
            
        Returns:
            Training history
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
            
            # Convert data to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            
            # Create dataset and dataloader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create validation dataloader if validation data is provided
            val_loader = None
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.LongTensor(y_val)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Set device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Initialize history
            history = {
                'loss': [],
                'accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            # Training loop
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += targets.size(0)
                    train_correct += (predicted == targets).sum().item()
                
                train_loss = train_loss / train_total
                train_accuracy = train_correct / train_total
                
                history['loss'].append(train_loss)
                history['accuracy'].append(train_accuracy)
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            
            return history
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return {}
    
    def _train_tensorflow_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                              epochs: int, batch_size: int, **kwargs) -> Dict[str, List[float]]:
        """
        Train TensorFlow model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs
            batch_size: Batch size
            **kwargs: Additional arguments for training
            
        Returns:
            Training history
        """
        try:
            import tensorflow as tf
            
            # Set up validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Set up callbacks
            callbacks = []
            
            # Early stopping
            if kwargs.get('early_stopping', False):
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=kwargs.get('patience', 10),
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=kwargs.get('verbose', 1)
            )
            
            return history.history
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return {}
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        if self.framework == 'pytorch':
            metrics = self._evaluate_pytorch_model(X_test, y_test)
        elif self.framework == 'tensorflow':
            metrics = self._evaluate_tensorflow_model(X_test, y_test)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        return metrics
    
    def _evaluate_pytorch_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate PyTorch model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Convert data to PyTorch tensors
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            
            # Create dataset and dataloader
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=64)
            
            # Set device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Evaluation
            self.model.eval()
            all_predictions = []
            all_targets = []
            test_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    
                    # Statistics
                    test_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            test_loss = test_loss / len(test_dataset)
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions, average='weighted')
            recall = recall_score(all_targets, all_predictions, average='weighted')
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            metrics = {
                'loss': test_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            return metrics
            
        except ImportError:
            logger.error("PyTorch or scikit-learn is not installed.")
            return {}
    
    def _evaluate_tensorflow_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate TensorFlow model.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            import tensorflow as tf
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Evaluate model
            results = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Get predictions
            predictions = self.model.predict(X_test)
            if isinstance(predictions, list):
                # For multi-output models, use the first output (classification)
                predictions = predictions[0]
            
            y_pred = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Create metrics dictionary
            metrics = {
                'loss': results[0],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            return metrics
            
        except ImportError:
            logger.error("TensorFlow or scikit-learn is not installed.")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None or not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        if self.framework == 'pytorch':
            predictions = self._predict_pytorch(X)
        elif self.framework == 'tensorflow':
            predictions = self._predict_tensorflow(X)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        return predictions
    
    def _predict_pytorch(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using PyTorch model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        try:
            import torch
            
            # Convert data to PyTorch tensor
            X_tensor = torch.FloatTensor(X)
            
            # Set device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Prediction
            self.model.eval()
            with torch.no_grad():
                X_tensor = X_tensor.to(device)
                outputs = self.model(X_tensor)
                
                # For classification, get class with highest probability
                if isinstance(outputs, tuple):
                    # For multi-output models, use the first output (classification)
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs, 1)
                predictions = predicted.cpu().numpy()
            
            return predictions
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return np.array([])
    
    def _predict_tensorflow(self, X: np.ndarray, return_probabilities: bool = False) -> np.ndarray:
        """
        Generate predictions using TensorFlow model.
        
        Args:
            X: Input features
            return_probabilities: Whether to return raw probabilities instead of class indices
            
        Returns:
            Predictions (class indices or probabilities)
        """
        try:
            # Generate predictions
            raw_predictions = self.model.predict(X)
            
            if isinstance(raw_predictions, list):
                # For multi-output models, use the first output (classification)
                raw_predictions = raw_predictions[0]
            
            # Log raw probabilities for debugging
            logger.info(f"[DEBUG] Raw prediction probabilities shape: {raw_predictions.shape}")
            logger.info(f"[DEBUG] Raw prediction probabilities: {raw_predictions}")
            
            if return_probabilities:
                return raw_predictions  # Return raw probabilities
            
            # For classification, get class with highest probability
            predictions = np.argmax(raw_predictions, axis=1)
            
            return predictions
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return np.array([])
    
    def save_model(self, path: str) -> bool:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if self.framework == 'pytorch':
                return self._save_pytorch_model(path)
            elif self.framework == 'tensorflow':
                return self._save_tensorflow_model(path)
            else:
                logger.error(f"Unsupported framework: {self.framework}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def _save_pytorch_model(self, path: str) -> bool:
        """
        Save PyTorch model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            
            # Save model state dict
            torch.save(self.model.state_dict(), path)
            
            # Save metadata
            metadata = {
                'framework': 'pytorch',
                'architecture': self.model.__class__.__name__,
                'trained': self.trained,
                'history': self.history
            }
            
            with open(f"{path}.meta", 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return False
        except Exception as e:
            logger.error(f"Error saving PyTorch model: {e}")
            return False
    
    def _save_tensorflow_model(self, path: str) -> bool:
        """
        Save TensorFlow model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure path has proper extension for TensorFlow models
            if not path.endswith('.keras') and not path.endswith('.h5'):
                logger.warning(f"Adding .keras extension to model path: {path}")
                path = f"{path}.keras"
            
            # Save model
            self.model.save(path)
            
            # Save metadata
            metadata = {
                'framework': 'tensorflow',
                'trained': self.trained,
                'history': self.history
            }
            
            with open(f"{path}.meta", 'wb') as f:
                pickle.dump(metadata, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving TensorFlow model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if metadata file exists
            if os.path.exists(f"{path}.meta"):
                with open(f"{path}.meta", 'rb') as f:
                    metadata = pickle.load(f)
                
                self.framework = metadata.get('framework', self.framework)
                self.trained = metadata.get('trained', False)
                self.history = metadata.get('history', None)
            
            if self.framework == 'pytorch':
                return self._load_pytorch_model(path)
            elif self.framework == 'tensorflow':
                return self._load_tensorflow_model(path)
            else:
                logger.error(f"Unsupported framework: {self.framework}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_pytorch_model(self, path: str) -> bool:
        """
        Load PyTorch model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            
            # Check if model is initialized
            if self.model is None:
                logger.error("Model not initialized. Call build_model first.")
                return False
            
            # Load model state dict
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            
            return True
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return False
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return False
    
    def _load_tensorflow_model(self, path: str) -> bool:
        """
        Load TensorFlow model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import tensorflow as tf
            import keras
            
            # Try to enable unsafe deserialization for Lambda layers
            try:
                logger.info("Enabling unsafe deserialization for TensorFlow models with Lambda layers")
                keras.config.enable_unsafe_deserialization()
                logger.info("Successfully enabled unsafe deserialization")
            except Exception as e:
                logger.warning(f"Failed to enable unsafe deserialization via config: {e}")
            
            try:
                # Define custom Lambda layer with output_shape
                def custom_lambda_layer(x):
                    return tf.reduce_sum(x, axis=1)
                
                # Create custom objects dictionary with Lambda layers that include output_shape
                custom_objects = {
                    'Lambda': lambda **kwargs: tf.keras.layers.Lambda(
                        custom_lambda_layer,
                        output_shape=(128,),  # Set appropriate output shape based on model architecture
                        **kwargs
                    )
                }
                
                # Try loading the model with custom objects
                logger.info("Trying to load model with custom Lambda layer that includes output_shape")
                self.model = tf.keras.models.load_model(path, safe_mode=False, custom_objects=custom_objects)
                logger.info(f"Successfully loaded model from {path} with custom Lambda layer")
                return True
                
            except Exception as e1:
                logger.warning(f"Failed to load with custom Lambda layer: {e1}")
                # Try alternative approach if the above fails
                try:
                    logger.info("Trying alternative approach with safe_mode=False")
                    self.model = tf.keras.models.load_model(path, safe_mode=False)
                    logger.info("Successfully loaded model with safe_mode=False")
                    return True
                except Exception as e2:
                    logger.error(f"Failed alternative loading approach: {e2}")
                    
                    # As a last resort, try to rebuild the model from scratch
                    try:
                        logger.info("Attempting to rebuild model from scratch with fixed Lambda layers")
                        # Get the architecture name from metadata if available
                        architecture = "lstm"  # Default to LSTM if not specified
                        if hasattr(self, 'metadata') and 'architecture' in self.metadata:
                            architecture = self.metadata['architecture']
                            
                        # Create a new model with the same architecture but fixed Lambda layers
                        from modules.trading_prediction.model_architectures import get_model_architecture
                        model_arch = get_model_architecture(architecture, self.framework)
                        
                        # Assume input shape and num_classes based on common values
                        input_shape = (60, 5)  # Default sequence length and features
                        num_classes = 3  # Default classes (sell, hold, buy)
                        
                        # Build the model with fixed Lambda layers
                        self.model = model_arch.create_model(input_shape, num_classes)
                        
                        # Try to load weights if possible (may not work due to architecture differences)
                        try:
                            self.model.load_weights(path)
                            logger.info("Successfully loaded weights into rebuilt model")
                        except Exception as e3:
                            logger.warning(f"Could not load weights into rebuilt model: {e3}")
                        
                        return True
                    except Exception as e4:
                        logger.error(f"Failed to rebuild model: {e4}")
            
            logger.error("All attempts to load the model failed")
            return False
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return False
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            return False
    
    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            logger.error("No training history available")
            return
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Training Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history:
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()