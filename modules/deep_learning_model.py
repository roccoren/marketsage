#!/usr/bin/env python3
"""
Deep Learning Model for stock prediction.
This module provides a PyTorch-based implementation of LSTM/Transformer models
with attention mechanisms for stock price prediction.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from modules.forecasting import ForecastModel

# Configure logging
logger = logging.getLogger(__name__)

# Set global random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SelfAttention(nn.Module):
    """Self-attention mechanism for feature importance."""
    
    def __init__(self, hidden_dim):
        """
        Initialize self-attention module.
        
        Args:
            hidden_dim (int): Dimension of hidden features
        """
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x):
        """
        Apply self-attention to input features.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # x shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        
        # Create query, key, value projections
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # Move scale to device if needed
        device = x.device
        scale = self.scale.to(device)
        
        # Calculate attention scores
        # [batch_size, seq_len, seq_len]
        attention = torch.matmul(Q, K.permute(0, 2, 1)) / scale
        
        # Apply softmax to get attention weights
        # [batch_size, seq_len, seq_len]
        attention_weights = F.softmax(attention, dim=-1)
        
        # Apply attention weights to values
        # [batch_size, seq_len, hidden_dim]
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class LSTMWithAttention(nn.Module):
    """LSTM model with self-attention for stock prediction."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        """
        Initialize LSTM with attention model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (usually 1 for regression)
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Feature embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = SelfAttention(hidden_dim)
        
        # Output layers
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # x shape: [batch_size, seq_len, input_dim]
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM layers
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, hidden_dim]
        
        # We use the last output with attention applied
        out = attn_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Fully connected layers with dropout
        out = self.dropout_layer(F.relu(self.fc(out)))  # [batch_size, hidden_dim//2]
        predictions = self.output(out)  # [batch_size, output_dim]
        
        return predictions, attn_weights


class TransformerWithAttention(nn.Module):
    """Transformer model with self-attention for stock prediction."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2,
                 num_heads=4, dropout=0.5):
        """
        Initialize Transformer with attention model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (usually 1 for regression)
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Feature embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Custom attention for feature importance
        self.attention = SelfAttention(hidden_dim)
        
        # Output layers
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # x shape: [batch_size, seq_len, input_dim]
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        embedded = self.pos_encoder(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # Apply custom attention for feature importance
        attn_out, attn_weights = self.attention(transformer_out)  # [batch_size, seq_len, hidden_dim]
        
        # We use the last output with attention applied
        out = attn_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Fully connected layers with dropout
        out = self.dropout_layer(F.relu(self.fc(out)))  # [batch_size, hidden_dim//2]
        predictions = self.output(out)  # [batch_size, output_dim]
        
        return predictions, attn_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Hidden dimension
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent but not model parameters)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DeepLearningModel(ForecastModel):
    """Deep Learning model for stock prediction using PyTorch."""
    
    def __init__(self, **kwargs):
        """
        Initialize DeepLearningModel.
        
        Args:
            **kwargs: Model parameters
        """
        # Default parameters
        self.params = {
            'model_type': 'lstm',  # 'lstm' or 'transformer'
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'sequence_length': 10,
            'attention_heads': 4,  # For transformer
            'weight_decay': 0.0001,
            'device': 'auto',  # 'auto', 'cpu', or 'cuda'
            'inference_batch_size': 64,
            'early_stopping_patience': 10,
            'optimizer': 'adam'  # 'adam' or 'sgd'
        }
        self.params.update(kwargs)
        
        # Device detection logic
        if self.params['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.params['device'])
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model attributes
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.attention_weights = None
        self.scaler = None
        self.performance_metrics = []
        
        # Temporary storage for large tensors
        self._temp_tensors = []
    
    def _build_model(self, input_dim=None):
        """
        Build the PyTorch model.
        
        Args:
            input_dim (int, optional): Input dimension. If not provided, uses default.
        """
        # If input_dim is not provided, use a default
        if input_dim is None:
            input_dim = self.params.get('input_dim', 10)
        
        # Store input_dim in params
        self.params['input_dim'] = input_dim
        
        # Create the model based on model_type
        if self.params['model_type'].lower() == 'lstm':
            self.model = LSTMWithAttention(
                input_dim=input_dim,
                hidden_dim=self.params['hidden_dim'],
                output_dim=1,  # Regression task
                num_layers=self.params['num_layers'],
                dropout=self.params['dropout']
            )
        elif self.params['model_type'].lower() == 'transformer':
            self.model = TransformerWithAttention(
                input_dim=input_dim,
                hidden_dim=self.params['hidden_dim'],
                output_dim=1,  # Regression task
                num_layers=self.params['num_layers'],
                num_heads=self.params['attention_heads'],
                dropout=self.params['dropout']
            )
        else:
            raise ValueError(f"Unknown model type: {self.params['model_type']}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        return self.model
    
    def _prepare_sequences(self, data: pd.DataFrame, target_series: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM/Transformer models.
        
        Args:
            data: DataFrame with features
            target_series: Optional target Series. If None, assumes data is for prediction only.
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        seq_length = self.params['sequence_length']
        features = []
        targets = []
        
        # Extract feature names if not already stored
        if self.feature_names is None:
            self.feature_names = list(data.columns)
        
        # Check if we have enough data for at least one sequence
        if len(data) < seq_length:
            logger.warning(f"Data length ({len(data)}) is less than sequence length ({seq_length})")
            # For very small datasets, pad with zeros to reach sequence length
            if len(data) > 0:
                # Use available data and pad with zeros
                pad_length = seq_length - len(data)
                padded_data = pd.concat([
                    pd.DataFrame(np.zeros((pad_length, data.shape[1])), columns=data.columns),
                    data
                ])
                features.append(padded_data.values)
                
                if target_series is not None and len(target_series) > 0:
                    targets.append(target_series.iloc[-1])
            
            return np.array(features), np.array(targets)
        
        # Create sequences
        if target_series is not None:
            # For training: create sequences with targets
            for i in range(len(data) - seq_length):
                features.append(data.iloc[i:i+seq_length].values)
                targets.append(target_series.iloc[i+seq_length])
        else:
            # For prediction: create sequences without targets
            for i in range(len(data) - seq_length + 1):
                features.append(data.iloc[i:i+seq_length].values)
            
            # If no sequences were created (edge case), use the last sequence
            if not features and len(data) >= seq_length:
                features.append(data.iloc[-seq_length:].values)
        
        return np.array(features), np.array(targets)
    
    def _prepare_data(self, X, y=None):
        """
        Prepare data efficiently to minimize memory usage.
        
        Args:
            X: Features (DataFrame, numpy array, or tensor)
            y: Targets (Series, numpy array, or tensor), optional
            
        Returns:
            X_tensor, y_tensor (if y is provided) or just X_tensor (if y is None)
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        elif isinstance(X, torch.Tensor):
            X_values = X.cpu().numpy() if X.device != self.device else X.detach().cpu().numpy()
        else:
            X_values = X
        
        # Create tensors directly on the correct device if small enough
        # For larger datasets, create on CPU first then transfer in batches
        data_size_mb = X_values.nbytes / (1024 * 1024)
        
        if data_size_mb < 100 or self.device.type == 'cpu':  # Small data or CPU
            X_tensor = torch.tensor(X_values, dtype=torch.float32, device=self.device)
        else:
            # For large data on GPU, keep on CPU and transfer in batches via DataLoader
            X_tensor = torch.tensor(X_values, dtype=torch.float32)
        
        if y is None:
            return X_tensor
        
        # Process y if provided
        if isinstance(y, pd.Series):
            y_values = y.values
        elif isinstance(y, torch.Tensor):
            y_values = y.cpu().numpy() if y.device != self.device else y.detach().cpu().numpy()
        else:
            y_values = y
        
        if data_size_mb < 100 or self.device.type == 'cpu':  # Small data or CPU
            y_tensor = torch.tensor(y_values, dtype=torch.float32, device=self.device)
        else:
            # For large data on GPU, keep on CPU and transfer in batches via DataLoader
            y_tensor = torch.tensor(y_values, dtype=torch.float32)
        
        return X_tensor, y_tensor
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on historical data.
        
        Args:
            X: Features DataFrame
            y: Target Series
        """
        start_time = time.time()
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Prepare sequence data
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        # Build model if not already built
        if self.model is None:
            input_dim = X.shape[1]  # Number of features
            self._build_model(input_dim)
        
        # Prepare data tensors
        X_tensor, y_tensor = self._prepare_data(X_seq, y_seq)
        
        # Create DataLoader for batch processing
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Ensure batch_size is at least 1
        batch_size = max(1, min(self.params['batch_size'], len(dataset)))
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Initialize optimizer
        if self.params['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.params['learning_rate'],
                weight_decay=self.params['weight_decay']
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.params['learning_rate'],
                momentum=0.9,
                weight_decay=self.params['weight_decay']
            )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        # Use mixed precision training if on GPU
        use_mixed_precision = self.device.type == 'cuda'
        scaler = None
        if use_mixed_precision:
            try:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                logger.info("Using mixed precision training")
            except ImportError:
                use_mixed_precision = False
                logger.warning("Mixed precision training not available")
        
        # Training loop
        epochs = self.params['epochs']
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                # Move batch to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass with mixed precision if available
                optimizer.zero_grad()
                
                if use_mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs, attention = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                    
                    # Scale gradients and optimize
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs, attention = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
            
            # Calculate average loss
            avg_loss = epoch_loss / len(dataset)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model state
                self._best_model_state = {
                    key: value.cpu().clone() for key, value in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.params['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if hasattr(self, '_best_model_state'):
            self.model.load_state_dict(self._best_model_state)
        
        # Store attention weights from the last batch for feature importance
        with torch.no_grad():
            self.model.eval()
            _, self.attention_weights = self.model(batch_X[:1])  # Use first sample for attention
            self.attention_weights = self._ensure_cpu_tensor(self.attention_weights)
        
        self.is_trained = True
        
        # Track performance
        end_time = time.time()
        self._track_performance("train", start_time, end_time, len(X))
        
        logger.info(f"Model training completed on {self.device}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            numpy.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        start_time = time.time()
        
        # Prepare sequence data
        X_seq = self._prepare_sequences(X)[0]  # Only need features for prediction
        
        # Prepare data tensor
        X_tensor = self._prepare_data(X_seq)
        
        # Create DataLoader for batch processing
        dataset = TensorDataset(X_tensor)
        
        # Ensure batch_size is at least 1
        batch_size = max(1, min(self.params.get('inference_batch_size', 64), len(dataset)))
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Prediction
        self.model.eval()
        predictions = []
        attention_weights = []
        
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(self.device)
                outputs, attention = self.model(batch_X)
                
                # Store predictions and attention weights
                predictions.append(self._ensure_cpu_tensor(outputs))
                attention_weights.append(self._ensure_cpu_tensor(attention))
        
        # Check if we have any predictions
        if not predictions:
            logger.warning("No predictions generated - empty dataset or sequence issue")
            # Return a default prediction (last known value or 0)
            return np.array([0.0])
        
        # Combine results
        all_predictions = torch.cat(predictions).numpy()
        
        # Ensure we return a 1D array even if there's only one prediction
        if all_predictions.ndim == 0:  # scalar
            all_predictions = np.array([all_predictions.item()])
        elif all_predictions.ndim > 1:  # multi-dimensional
            all_predictions = all_predictions.squeeze()
            # If squeeze results in a scalar, convert back to 1D array
            if all_predictions.ndim == 0:
                all_predictions = np.array([all_predictions.item()])
        
        # Store attention weights for feature importance (use the last batch if available)
        if attention_weights:
            self.attention_weights = attention_weights[-1]
        
        # Track performance
        end_time = time.time()
        self._track_performance("predict", start_time, end_time, len(X))
        
        return all_predictions
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict with model parameters
        """
        # Basic parameters
        params = self.params.copy()
        
        # Add model architecture info
        if self.model is not None:
            params['model_architecture'] = self.model.__class__.__name__
            
            # Add model state dict if trained
            if self.is_trained:
                model_state = {
                    'state_dict': {
                        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in self.model.state_dict().items()
                    }
                }
                params['model_state'] = model_state
        
        return params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters.
        
        Args:
            params: Dictionary with parameters
        """
        # Update basic parameters
        for key, value in params.items():
            if key != 'model_state':
                self.params[key] = value
        
        # Rebuild model with new parameters if needed
        if self.model is None or params.get('model_architecture') != self.model.__class__.__name__:
            input_dim = self.params.get('input_dim')
            if input_dim is not None:
                self._build_model(input_dim)
        
        # Load model state if provided
        if 'model_state' in params and self.model is not None:
            state_dict = params['model_state'].get('state_dict')
            if state_dict:
                # Convert numpy arrays or strings to tensors
                converted_state_dict = {}
                for k, v in state_dict.items():
                    if isinstance(v, np.ndarray):
                        converted_state_dict[k] = torch.tensor(v)
                    elif isinstance(v, str):
                        # Skip string values, likely serialization issue
                        logger.warning(f"Skipping string value for parameter {k}")
                        continue
                    else:
                        converted_state_dict[k] = v
                
                # If we have a valid state dict, load it
                if converted_state_dict:
                    try:
                        self.model.load_state_dict(converted_state_dict)
                        self.is_trained = True
                    except Exception as e:
                        logger.error(f"Error loading state dict: {e}")
                        # Train from scratch if loading fails
                        self.is_trained = False
                else:
                    logger.warning("No valid parameters found in state dict")
                    self.is_trained = False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract feature importance from attention weights.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if self.attention_weights is None or self.feature_names is None:
            return {}
        
        try:
            # Average attention weights across all samples
            avg_weights = self.attention_weights.mean(dim=0)
            
            # Take the last sequence position (predicting the next value)
            # and get its attention to all input features
            last_position_weights = avg_weights[-1]
            
            # Map weights to feature names, ensuring we don't go out of bounds
            feature_importance = {}
            weight_size = min(len(self.feature_names), len(last_position_weights))
            
            for i in range(weight_size):
                feature = self.feature_names[i]
                feature_importance[feature] = float(last_position_weights[i])
            
            # If we have more features than weights, assign zero importance
            for i in range(weight_size, len(self.feature_names)):
                feature = self.feature_names[i]
                feature_importance[feature] = 0.0
            
            # Normalize to sum to 1.0
            total = sum(feature_importance.values())
            if total > 0:  # Avoid division by zero
                for feature in feature_importance:
                    feature_importance[feature] /= total
            
            return feature_importance
        except Exception as e:
            import logging
            logging.warning(f"Error calculating feature importance: {e}")
            return {}
    
    def _ensure_cpu_tensor(self, tensor):
        """Ensure tensor is on CPU for storage or numpy conversion."""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach()
        return tensor
    
    def _ensure_device_tensor(self, tensor):
        """Ensure tensor is on the correct device."""
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
            return tensor.to(self.device)
        elif isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, device=self.device)
        return tensor
    
    def _track_performance(self, function_name, start_time, end_time, data_size=None):
        """Track performance metrics for operations."""
        duration = end_time - start_time
        
        metrics = {
            'function': function_name,
            'duration_seconds': duration,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        
        if data_size:
            metrics['data_size'] = data_size
            metrics['throughput'] = data_size / duration if duration > 0 else 0
        
        # Log performance metrics
        logger.debug(f"Performance: {metrics}")
        
        # Store metrics for later analysis
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = []
        self.performance_metrics.append(metrics)
        
        return metrics
    
    def visualize_attention(self, save_path=None):
        """
        Visualize attention weights as a heatmap.
        
        Args:
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.attention_weights is None or self.feature_names is None:
            return None
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Convert attention weights to numpy
            if isinstance(self.attention_weights, torch.Tensor):
                attn_weights = self.attention_weights.detach().cpu().numpy()
            else:
                attn_weights = self.attention_weights
            
            # Average across batch dimension if needed
            if len(attn_weights.shape) > 2:
                attn_weights = attn_weights.mean(axis=0)
            
            plt.figure(figsize=(12, 8))
            
            # Ensure dimensions match
            feature_count = min(len(self.feature_names), attn_weights.shape[0], attn_weights.shape[1])
            feature_names = self.feature_names[:feature_count]
            
            # Use the subset of attention weights that match our features
            attn_subset = attn_weights[:feature_count, :feature_count]
            
            sns.heatmap(
                attn_subset,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='viridis'
            )
            plt.title('Feature Attention Weights')
            plt.xlabel('Features')
            plt.ylabel('Attention From')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            
            return plt.gcf()
        except ImportError:
            logger.warning("Matplotlib or seaborn not available for visualization")
            return None
    
    def save_model_state(self, path):
        """
        Save model state to disk.
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: True if successful
        """
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'params': self.params,
                'feature_names': self.feature_names,
            }, path)
            
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model_state(self, path):
        """
        Load model state from disk.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            bool: True if successful
        """
        try:
            # Load checkpoint
            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
            
            # Update parameters
            self.params.update(checkpoint.get('params', {}))
            
            # Build model if needed
            if self.model is None:
                input_dim = self.params.get('input_dim')
                if input_dim is not None:
                    self._build_model(input_dim)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load feature names
            self.feature_names = checkpoint.get('feature_names')
            
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def optimize_for_inference(self):
        """
        Optimize model for inference based on device.
        
        Returns:
            bool: True if optimization was successful
        """
        if not self.is_trained:
            return False
        
        if self.device.type == 'cpu':
            # Quantize model for CPU inference
            try:
                import torch.quantization
                
                # Prepare model for quantization
                self.model.eval()
                self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(self.model, inplace=True)
                
                # Convert to quantized model
                torch.quantization.convert(self.model, inplace=True)
                logger.info("Model quantized for faster CPU inference")
                return True
            except Exception as e:
                logger.warning(f"Quantization failed: {e}. Using original model.")
                return False
        else:
            # For GPU, we'll use TorchScript compilation
            try:
                # Create example input for tracing
                example_input = torch.rand(
                    1, self.params.get('sequence_length', 10),
                    self.params.get('input_dim', 10),
                    device=self.device
                )
                
                # Trace and optimize
                self.model = torch.jit.trace(self.model, example_input)
                self.model = torch.jit.optimize_for_inference(self.model)
                logger.info("Model optimized for GPU inference")
                return True
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}. Using original model.")
                return False