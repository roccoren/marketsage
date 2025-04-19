#!/usr/bin/env python3
"""
Model architectures for stock trading prediction.

This module defines various deep learning model architectures for stock trading prediction,
including LSTM/GRU and Transformer-based models.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)


class ModelArchitecture:
    """Base class for model architectures."""
    
    def __init__(self, name: str, framework: str):
        """
        Initialize ModelArchitecture.
        
        Args:
            name: Model name
            framework: Framework ('pytorch', 'tensorflow')
        """
        self.name = name
        self.framework = framework
    
    def create_model(self, input_shape: Tuple[int, int], num_classes: int = 3, **kwargs) -> Any:
        """
        Create model with the specified architecture.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            Model instance
        """
        if self.framework == 'pytorch':
            return self._create_pytorch_model(input_shape, num_classes, **kwargs)
        elif self.framework == 'tensorflow':
            return self._create_tensorflow_model(input_shape, num_classes, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _create_pytorch_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create PyTorch model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            PyTorch model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _create_tensorflow_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create TensorFlow model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            TensorFlow model
        """
        raise NotImplementedError("Subclasses must implement this method")


class LSTMArchitecture(ModelArchitecture):
    """LSTM-based architecture for stock trading prediction."""
    
    def __init__(self, framework: str = 'tensorflow'):
        """
        Initialize LSTMArchitecture.
        
        Args:
            framework: Framework ('pytorch', 'tensorflow')
        """
        super().__init__("LSTM", framework)
    
    def _create_pytorch_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create PyTorch LSTM model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            PyTorch model
        """
        try:
            import torch
            import torch.nn as nn
            
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            bidirectional = kwargs.get('bidirectional', False)
            
            class LSTMModel(nn.Module):
                def __init__(self):
                    super(LSTMModel, self).__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_shape[1],
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0,
                        bidirectional=bidirectional
                    )
                    
                    # Calculate output size of LSTM layer
                    lstm_output_size = hidden_size * (2 if bidirectional else 1)
                    
                    self.attention = nn.Sequential(
                        nn.Linear(lstm_output_size, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
                    
                    self.fc = nn.Sequential(
                        nn.Linear(lstm_output_size, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, num_classes)
                    )
                
                def forward(self, x):
                    # x shape: (batch_size, sequence_length, num_features)
                    lstm_out, _ = self.lstm(x)
                    
                    # Apply attention
                    attention_weights = self.attention(lstm_out)
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    
                    # Apply attention weights to LSTM output
                    context_vector = torch.sum(attention_weights * lstm_out, dim=1)
                    
                    # Pass through fully connected layers
                    output = self.fc(context_vector)
                    
                    return output
            
            return LSTMModel()
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return None
    
    def _create_tensorflow_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create TensorFlow LSTM model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            TensorFlow model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            bidirectional = kwargs.get('bidirectional', False)
            
            # Input layer
            inputs = layers.Input(shape=input_shape)
            
            # LSTM layers
            x = inputs
            for i in range(num_layers - 1):
                if bidirectional:
                    x = layers.Bidirectional(
                        layers.LSTM(hidden_size, return_sequences=True)
                    )(x)
                else:
                    x = layers.LSTM(hidden_size, return_sequences=True)(x)
                
                x = layers.Dropout(dropout)(x)
            
            # Last LSTM layer
            if bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(hidden_size, return_sequences=True)
                )(x)
            else:
                x = layers.LSTM(hidden_size, return_sequences=True)(x)
            
            # Attention mechanism
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention_weights = layers.Activation('softmax')(attention)
            attention_weights = layers.RepeatVector(hidden_size * (2 if bidirectional else 1))(attention_weights)
            attention_weights = layers.Permute([2, 1])(attention_weights)
            
            # Apply attention
            context_vector = layers.Multiply()([x, attention_weights])
            context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
            
            # Output layers
            x = layers.Dense(64, activation='relu')(context_vector)
            x = layers.Dropout(dropout)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=outputs)
            
            return model
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return None


class GRUArchitecture(ModelArchitecture):
    """GRU-based architecture for stock trading prediction."""
    
    def __init__(self, framework: str = 'tensorflow'):
        """
        Initialize GRUArchitecture.
        
        Args:
            framework: Framework ('pytorch', 'tensorflow')
        """
        super().__init__("GRU", framework)
    
    def _create_pytorch_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create PyTorch GRU model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            PyTorch model
        """
        try:
            import torch
            import torch.nn as nn
            
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            bidirectional = kwargs.get('bidirectional', False)
            
            class GRUModel(nn.Module):
                def __init__(self):
                    super(GRUModel, self).__init__()
                    self.gru = nn.GRU(
                        input_size=input_shape[1],
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                        dropout=dropout if num_layers > 1 else 0,
                        bidirectional=bidirectional
                    )
                    
                    # Calculate output size of GRU layer
                    gru_output_size = hidden_size * (2 if bidirectional else 1)
                    
                    self.attention = nn.Sequential(
                        nn.Linear(gru_output_size, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
                    
                    self.fc = nn.Sequential(
                        nn.Linear(gru_output_size, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, num_classes)
                    )
                
                def forward(self, x):
                    # x shape: (batch_size, sequence_length, num_features)
                    gru_out, _ = self.gru(x)
                    
                    # Apply attention
                    attention_weights = self.attention(gru_out)
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    
                    # Apply attention weights to GRU output
                    context_vector = torch.sum(attention_weights * gru_out, dim=1)
                    
                    # Pass through fully connected layers
                    output = self.fc(context_vector)
                    
                    return output
            
            return GRUModel()
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return None
    
    def _create_tensorflow_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create TensorFlow GRU model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            TensorFlow model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            bidirectional = kwargs.get('bidirectional', False)
            
            # Input layer
            inputs = layers.Input(shape=input_shape)
            
            # GRU layers
            x = inputs
            for i in range(num_layers - 1):
                if bidirectional:
                    x = layers.Bidirectional(
                        layers.GRU(hidden_size, return_sequences=True)
                    )(x)
                else:
                    x = layers.GRU(hidden_size, return_sequences=True)(x)
                
                x = layers.Dropout(dropout)(x)
            
            # Last GRU layer
            if bidirectional:
                x = layers.Bidirectional(
                    layers.GRU(hidden_size, return_sequences=True)
                )(x)
            else:
                x = layers.GRU(hidden_size, return_sequences=True)(x)
            
            # Attention mechanism
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention_weights = layers.Activation('softmax')(attention)
            attention_weights = layers.RepeatVector(hidden_size * (2 if bidirectional else 1))(attention_weights)
            attention_weights = layers.Permute([2, 1])(attention_weights)
            
            # Apply attention
            context_vector = layers.Multiply()([x, attention_weights])
            context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
            
            # Output layers
            x = layers.Dense(64, activation='relu')(context_vector)
            x = layers.Dropout(dropout)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=outputs)
            
            return model
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return None


class TransformerArchitecture(ModelArchitecture):
    """Transformer-based architecture for stock trading prediction."""
    
    def __init__(self, framework: str = 'tensorflow'):
        """
        Initialize TransformerArchitecture.
        
        Args:
            framework: Framework ('pytorch', 'tensorflow')
        """
        super().__init__("Transformer", framework)
    
    def _create_pytorch_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create PyTorch Transformer model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            PyTorch model
        """
        try:
            import torch
            import torch.nn as nn
            import math
            
            d_model = kwargs.get('d_model', 128)
            nhead = kwargs.get('nhead', 8)
            num_layers = kwargs.get('num_layers', 2)
            dim_feedforward = kwargs.get('dim_feedforward', 512)
            dropout = kwargs.get('dropout', 0.1)
            
            class PositionalEncoding(nn.Module):
                def __init__(self, d_model, max_len=5000):
                    super(PositionalEncoding, self).__init__()
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    pe = pe.unsqueeze(0)
                    self.register_buffer('pe', pe)
                
                def forward(self, x):
                    return x + self.pe[:, :x.size(1), :]
            
            class TransformerModel(nn.Module):
                def __init__(self):
                    super(TransformerModel, self).__init__()
                    self.embedding = nn.Linear(input_shape[1], d_model)
                    self.pos_encoder = PositionalEncoding(d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.fc = nn.Sequential(
                        nn.Linear(d_model, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, num_classes)
                    )
                
                def forward(self, x):
                    # x shape: (batch_size, sequence_length, num_features)
                    x = self.embedding(x)
                    x = self.pos_encoder(x)
                    
                    # Create mask for padding
                    mask = None  # No mask for now, all sequences are same length
                    
                    # Pass through transformer encoder
                    x = self.transformer_encoder(x, mask)
                    
                    # Use the output for the last token
                    x = x[:, -1, :]
                    
                    # Pass through fully connected layers
                    output = self.fc(x)
                    
                    return output
            
            return TransformerModel()
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return None
    
    def _create_tensorflow_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create TensorFlow Transformer model.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            TensorFlow model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            d_model = kwargs.get('d_model', 128)
            num_heads = kwargs.get('num_heads', 8)
            num_layers = kwargs.get('num_layers', 2)
            dff = kwargs.get('dff', 512)  # feed-forward dimension
            dropout = kwargs.get('dropout', 0.1)
            
            def positional_encoding(position, d_model):
                angle_rads = tf.range(position, dtype=tf.float32)[:, tf.newaxis] / \
                    tf.pow(10000, tf.range(0, d_model, 2, dtype=tf.float32) / d_model)
                
                # Apply sin to even indices
                sines = tf.math.sin(angle_rads)
                
                # Apply cos to odd indices
                cosines = tf.math.cos(angle_rads)
                
                pos_encoding = tf.concat([sines, cosines], axis=-1)
                pos_encoding = pos_encoding[tf.newaxis, ...]
                
                return tf.cast(pos_encoding, tf.float32)
            
            # Input layer
            inputs = layers.Input(shape=input_shape)
            
            # Embedding
            x = layers.Dense(d_model)(inputs)
            
            # Add positional encoding
            seq_len = input_shape[0]
            pos_encoding = positional_encoding(seq_len, d_model)
            x = x + pos_encoding[:, :seq_len, :]
            
            # Dropout
            x = layers.Dropout(dropout)(x)
            
            # Transformer blocks
            for i in range(num_layers):
                # Multi-head attention
                attn_output = layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=d_model // num_heads
                )(x, x, x)
                attn_output = layers.Dropout(dropout)(attn_output)
                
                # Add & Norm
                x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
                
                # Feed-forward network
                ffn_output = layers.Dense(dff, activation='relu')(x1)
                ffn_output = layers.Dense(d_model)(ffn_output)
                ffn_output = layers.Dropout(dropout)(ffn_output)
                
                # Add & Norm
                x = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn_output)
            
            # Use the output for the last token
            x = x[:, -1, :]
            
            # Output layers
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(dropout)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=outputs)
            
            return model
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return None


class DualOutputArchitecture(ModelArchitecture):
    """Architecture with dual outputs for classification and regression."""
    
    def __init__(self, base_architecture: str = 'LSTM', framework: str = 'tensorflow'):
        """
        Initialize DualOutputArchitecture.
        
        Args:
            base_architecture: Base architecture ('LSTM', 'GRU', 'Transformer')
            framework: Framework ('pytorch', 'tensorflow')
        """
        super().__init__(f"DualOutput{base_architecture}", framework)
        self.base_architecture = base_architecture
    
    def _create_pytorch_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create PyTorch model with dual outputs.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            PyTorch model
        """
        try:
            import torch
            import torch.nn as nn
            
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            
            class DualOutputModel(nn.Module):
                def __init__(self):
                    super(DualOutputModel, self).__init__()
                    
                    # Base architecture
                    if self.base_architecture == 'LSTM':
                        self.base = nn.LSTM(
                            input_size=input_shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0
                        )
                    elif self.base_architecture == 'GRU':
                        self.base = nn.GRU(
                            input_size=input_shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0
                        )
                    else:  # Transformer
                        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=hidden_size,
                            nhead=8,
                            dim_feedforward=512,
                            dropout=dropout,
                            batch_first=True
                        )
                        self.base = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                        self.embedding = nn.Linear(input_shape[1], hidden_size)
                    
                    # Common layers
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_size, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
                    
                    # Classification output
                    self.classification = nn.Sequential(
                        nn.Linear(hidden_size, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, num_classes)
                    )
                    
                    # Regression output
                    self.regression = nn.Sequential(
                        nn.Linear(hidden_size, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, 1)
                    )
                
                def forward(self, x):
                    # x shape: (batch_size, sequence_length, num_features)
                    
                    # Process with base architecture
                    if self.base_architecture == 'Transformer':
                        x = self.embedding(x)
                        base_out = self.base(x)
                    else:
                        base_out, _ = self.base(x)
                    
                    # Apply attention
                    attention_weights = self.attention(base_out)
                    attention_weights = torch.softmax(attention_weights, dim=1)
                    
                    # Apply attention weights to base output
                    context_vector = torch.sum(attention_weights * base_out, dim=1)
                    
                    # Generate outputs
                    classification_output = self.classification(context_vector)
                    regression_output = self.regression(context_vector)
                    
                    return classification_output, regression_output
            
            return DualOutputModel()
            
        except ImportError:
            logger.error("PyTorch is not installed. Please install it with: pip install torch")
            return None
    
    def _create_tensorflow_model(self, input_shape: Tuple[int, int], num_classes: int, **kwargs) -> Any:
        """
        Create TensorFlow model with dual outputs.
        
        Args:
            input_shape: Input shape (sequence_length, num_features)
            num_classes: Number of classes for classification output
            **kwargs: Additional arguments for model creation
            
        Returns:
            TensorFlow model
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            
            hidden_size = kwargs.get('hidden_size', 128)
            num_layers = kwargs.get('num_layers', 2)
            dropout = kwargs.get('dropout', 0.2)
            
            # Input layer
            inputs = layers.Input(shape=input_shape)
            
            # Base architecture
            x = inputs
            if self.base_architecture == 'LSTM':
                for i in range(num_layers - 1):
                    x = layers.LSTM(hidden_size, return_sequences=True)(x)
                    x = layers.Dropout(dropout)(x)
                x = layers.LSTM(hidden_size, return_sequences=True)(x)
            elif self.base_architecture == 'GRU':
                for i in range(num_layers - 1):
                    x = layers.GRU(hidden_size, return_sequences=True)(x)
                    x = layers.Dropout(dropout)(x)
                x = layers.GRU(hidden_size, return_sequences=True)(x)
            else:  # Transformer
                # Embedding
                x = layers.Dense(hidden_size)(x)
                
                # Transformer blocks
                for i in range(num_layers):
                    # Multi-head attention
                    attn_output = layers.MultiHeadAttention(
                        num_heads=8, key_dim=hidden_size // 8
                    )(x, x, x)
                    attn_output = layers.Dropout(dropout)(attn_output)
                    
                    # Add & Norm
                    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
                    
                    # Feed-forward network
                    ffn_output = layers.Dense(512, activation='relu')(x1)
                    ffn_output = layers.Dense(hidden_size)(ffn_output)
                    ffn_output = layers.Dropout(dropout)(ffn_output)
                    
                    # Add & Norm
                    x = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn_output)
            
            # Attention mechanism
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention_weights = layers.Activation('softmax')(attention)
            attention_weights = layers.RepeatVector(hidden_size)(attention_weights)
            attention_weights = layers.Permute([2, 1])(attention_weights)
            
            # Apply attention
            context_vector = layers.Multiply()([x, attention_weights])
            context_vector = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
            
            # Classification output
            classification = layers.Dense(64, activation='relu')(context_vector)
            classification = layers.Dropout(dropout)(classification)
            classification_output = layers.Dense(num_classes, activation='softmax', name='classification')(classification)
            
            # Regression output
            regression = layers.Dense(64, activation='relu')(context_vector)
            regression = layers.Dropout(dropout)(regression)
            regression_output = layers.Dense(1, name='regression')(regression)
            
            # Create model
            model = models.Model(inputs=inputs, outputs=[classification_output, regression_output])
            
            return model
            
        except ImportError:
            logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
            return None


def get_model_architecture(architecture_name: str, framework: str = 'tensorflow') -> ModelArchitecture:
    """
    Get model architecture by name.
    
    Args:
        architecture_name: Architecture name ('LSTM', 'GRU', 'Transformer', 'DualOutputLSTM', etc.)
        framework: Framework ('pytorch', 'tensorflow')
        
    Returns:
        ModelArchitecture instance
    """
    if architecture_name == 'LSTM':
        return LSTMArchitecture(framework)
    elif architecture_name == 'GRU':
        return GRUArchitecture(framework)
    elif architecture_name == 'Transformer':
        return TransformerArchitecture(framework)
    elif architecture_name.startswith('DualOutput'):
        base_arch = architecture_name[10:]  # Remove 'DualOutput' prefix
        return DualOutputArchitecture(base_arch, framework)
    else:
        raise ValueError(f"Unsupported architecture: {architecture_name}")