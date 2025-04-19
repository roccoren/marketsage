#!/usr/bin/env python3
"""
Example script for stock trading prediction.

This script demonstrates how to use the trading prediction modules to build
a complete stock trading prediction system.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from modules.trading_prediction.data_acquisition import get_data_manager
from modules.trading_prediction.feature_engineering import EnhancedFeatureEngineer, FeatureNormalizer
from modules.trading_prediction.dataset import prepare_dataset, TimeSeriesDataset
from modules.trading_prediction.model_training import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate stock trading prediction."""
    
    # Step 1: Data Acquisition
    logger.info("Step 1: Data Acquisition")
    
    # Initialize data manager
    data_manager = get_data_manager()
    
    # Define parameters
    symbol = "AAPL"  # Example stock symbol
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 years ago
    end_date = datetime.now().strftime('%Y-%m-%d')  # Today
    
    # Fetch data
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    data = data_manager.fetch_data_multi_source(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    
    if data.empty:
        logger.error("Failed to fetch data. Exiting.")
        return
    
    logger.info(f"Fetched {len(data)} data points")
    
    # Step 2: Feature Engineering
    logger.info("Step 2: Feature Engineering")
    
    # Initialize feature engineer
    feature_engineer = EnhancedFeatureEngineer()
    
    # Create features
    features = feature_engineer.create_features(data)
    logger.info(f"Created {len(features.columns)} features")
    
    # Normalize features
    normalizer = FeatureNormalizer(method='minmax')
    normalized_features = normalizer.fit_transform(features)
    
    # Step 3: Dataset Creation
    logger.info("Step 3: Dataset Creation")
    
    # Prepare dataset
    sequence_length = 60  # Use 60 days of data to predict
    prediction_horizon = 5  # Predict 5 days ahead
    
    dataset = prepare_dataset(
        features=normalized_features,
        prices=data[['Close']],
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        target_type='classification',
        test_size=0.2,
        validation_size=0.1
    )
    
    logger.info(f"Training set: {dataset['X_train'].shape}")
    logger.info(f"Validation set: {dataset['X_val'].shape}")
    logger.info(f"Test set: {dataset['X_test'].shape}")
    
    # Step 4: Model Training
    logger.info("Step 4: Model Training")
    
    # Initialize model trainer
    trainer = ModelTrainer(framework='tensorflow')
    
    # Build model
    input_shape = (sequence_length, dataset['X_train'].shape[2])
    num_classes = 3  # Sell (0), Hold (1), Buy (2)
    
    trainer.build_model(
        architecture='LSTM',
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    
    # Compile model
    trainer.compile_model(
        learning_rate=0.001,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = trainer.train(
        X_train=dataset['X_train'],
        y_train=dataset['y_train'],
        X_val=dataset['X_val'],
        y_val=dataset['y_val'],
        epochs=50,
        batch_size=32,
        early_stopping=True,
        patience=10,
        verbose=1
    )
    
    # Step 5: Model Evaluation
    logger.info("Step 5: Model Evaluation")
    
    # Evaluate model
    metrics = trainer.evaluate(dataset['X_test'], dataset['y_test'])
    
    logger.info("Test metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Step 6: Generate Predictions
    logger.info("Step 6: Generate Predictions")
    
    # Predict on test data
    predictions = trainer.predict(dataset['X_test'])
    
    # Map predictions to labels
    label_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
    pred_labels = [label_map[p] for p in predictions]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Actual': [label_map[y] for y in dataset['y_test']],
        'Predicted': pred_labels
    })
    
    logger.info("Prediction results (first 10 samples):")
    for i in range(min(10, len(results))):
        logger.info(f"  Sample {i+1}: Actual: {results['Actual'].iloc[i]}, Predicted: {results['Predicted'].iloc[i]}")
    
    # Calculate prediction distribution
    pred_distribution = results['Predicted'].value_counts()
    logger.info("\nPrediction distribution:")
    for label, count in pred_distribution.items():
        percentage = count / len(results) * 100
        logger.info(f"  {label}: {count} samples ({percentage:.1f}%)")
    
    # Step 7: Save Model
    logger.info("Step 7: Save Model")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model - Add .keras extension for TensorFlow models
    model_path = f'models/{symbol}_lstm_model.keras'
    trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Step 8: Generate visualization
    logger.info("Step 8: Generate visualization")
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot training history
    history_plot_path = f'plots/{symbol}_training_history.png'
    trainer.plot_history(save_path=history_plot_path)
    logger.info(f"Training history plot saved to {history_plot_path}")
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(dataset['y_test'], predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[label_map[i] for i in range(num_classes)],
                yticklabels=[label_map[i] for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    cm_plot_path = f'plots/{symbol}_confusion_matrix.png'
    plt.savefig(cm_plot_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {cm_plot_path}")
    
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Training history plot saved to: {history_plot_path}")
    logger.info(f"Confusion matrix plot saved to: {cm_plot_path}")
    logger.info("\nExample completed successfully")


if __name__ == "__main__":
    main()