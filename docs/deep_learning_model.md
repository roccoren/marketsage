# Deep Learning Stock Prediction Model

This document describes the deep learning model implementation for stock price prediction using LSTM and Transformer architectures with attention mechanisms.

## Overview

The deep learning model (`DeepLearningModel`) is designed to predict stock prices using either LSTM or Transformer architectures with attention mechanisms. The model can identify which features are most important for predictions through its attention mechanism, providing insights into the decision-making process.

## Features

- **Dual Architecture Support**: Choose between LSTM or Transformer models
- **Attention Mechanism**: Identifies important features for predictions
- **Feature Importance Visualization**: Generates heatmaps of attention weights
- **GPU/CPU Compatibility**: Automatically detects and uses available hardware
- **Memory Optimization**: Efficiently handles large datasets with batch processing
- **Mixed Precision Training**: Accelerates training on compatible GPUs
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Model Persistence**: Save and load trained models
- **Performance Tracking**: Monitors training and inference performance

## Usage

### Command Line Interface

The model can be used through the main command line interface:

```bash
python main.py --symbols-file stock_symbols.txt --prediction-model deep
```

### Example Script

An example script is provided to demonstrate the model's capabilities:

```bash
# Train and evaluate using LSTM architecture
python examples/deep_learning_example.py --symbol AAPL --model lstm

# Train and evaluate using Transformer architecture
python examples/deep_learning_example.py --symbol MSFT --model transformer

# Compare both architectures
python examples/deep_learning_example.py --symbol GOOGL --model compare
```

## Model Parameters

The model accepts the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_type` | Model architecture ('lstm' or 'transformer') | 'lstm' |
| `hidden_dim` | Hidden layer dimension | 64 |
| `num_layers` | Number of LSTM/Transformer layers | 2 |
| `dropout` | Dropout probability | 0.5 |
| `learning_rate` | Learning rate for optimizer | 0.001 |
| `batch_size` | Batch size for training | 32 |
| `epochs` | Maximum training epochs | 100 |
| `sequence_length` | Length of input sequences | 10 |
| `attention_heads` | Number of attention heads (for transformer) | 4 |
| `weight_decay` | L2 regularization strength | 0.0001 |
| `device` | Computation device ('auto', 'cpu', or 'cuda') | 'auto' |
| `early_stopping_patience` | Patience for early stopping | 10 |
| `optimizer` | Optimizer type ('adam' or 'sgd') | 'adam' |

## Architecture

### LSTM with Attention

The LSTM architecture consists of:
1. Feature embedding layer
2. LSTM layers
3. Self-attention mechanism
4. Fully connected output layers

### Transformer with Attention

The Transformer architecture consists of:
1. Feature embedding layer
2. Positional encoding
3. Transformer encoder layers
4. Self-attention mechanism
5. Fully connected output layers

## Feature Importance

The model extracts feature importance from the attention weights, showing which features have the most influence on predictions. This can be visualized as a heatmap using the `visualize_attention()` method.

## GPU/CPU Compatibility

The model automatically detects available hardware and optimizes performance accordingly:

- **GPU Mode**: Uses CUDA acceleration, mixed precision training, and TorchScript optimization
- **CPU Mode**: Uses efficient batch processing and optional quantization

## Integration

The model is fully integrated with the existing forecasting system and can be used interchangeably with other models like LGBModel and EnsembleModel.