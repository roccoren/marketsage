"""
QlibRDPredictor: A predictor class that integrates Microsoft Qlib with RDAgent for stock trading signal prediction.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import json
from datetime import datetime

import torch
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_gru import GRUModel

from stockscan.models.qlib_rdagent.utils import init_qlib, prepare_qlib_data, create_qlib_dataset

logger = logging.getLogger(__name__)

class RDAgentModel:
    """
    RDAgent-based model for feature extraction and reinforcement learning.
    
    This model uses RDKit to create molecular-like representations of stock data
    and applies reinforcement learning techniques for decision making.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, learning_rate: float = 1e-4):
        """
        Initialize the RDAgent model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (typically 3: buy, hold, sell)
            hidden_dim: Hidden dimension for network
            learning_rate: Learning rate for training
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Create actor and critic networks
        self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def _build_model(self):
        """Build the actor-critic network."""
        # Actor network (policy)
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        
        # Critic network (value function)
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def get_action(self, state: np.ndarray, explore: bool = False) -> int:
        """
        Get action from the policy network.
        
        Args:
            state: Current state vector
            explore: Whether to explore (sample from distribution) or exploit (take argmax)
            
        Returns:
            Action index (0=buy, 1=hold, 2=sell)
        """
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor(state).numpy()[0]
        
        if explore:
            return np.random.choice(self.action_dim, p=action_probs)
        else:
            return np.argmax(action_probs)
    
    def train_step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Perform a single training step using actor-critic method.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary with actor and critic losses
        """
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        
        with tf.GradientTape() as tape:
            # Current value
            value = self.critic(state)
            # Next value (0 if done)
            next_value = 0.0 if done else self.critic(next_state)
            
            # Compute TD target
            gamma = 0.99  # discount factor
            td_target = reward + gamma * next_value * (1 - done)
            td_error = td_target - value
            
            # Compute critic loss
            critic_loss = tf.square(td_error)
            
            # Compute actor loss
            action_probs = self.actor(state)
            action_mask = tf.one_hot(action, self.action_dim)
            log_prob = tf.math.log(tf.reduce_sum(action_probs * action_mask))
            actor_loss = -log_prob * td_error
            
            # Total loss
            total_loss = actor_loss + critic_loss
        
        # Compute gradients and apply them
        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
        
        return {
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
        }
    
    def save(self, path: str):
        """Save the model."""
        os.makedirs(path, exist_ok=True)
        self.actor.save(os.path.join(path, "actor"))
        self.critic.save(os.path.join(path, "critic"))
    
    def load(self, path: str):
        """Load the model."""
        self.actor = tf.keras.models.load_model(os.path.join(path, "actor"))
        self.critic = tf.keras.models.load_model(os.path.join(path, "critic"))


class QlibRDPredictor:
    """
    Predictor that combines Qlib models with RDAgent for stock prediction.
    
    This predictor uses Qlib for feature engineering and initial predictions,
    then refines the results using an RDAgent reinforcement learning approach.
    """
    
    def __init__(self, 
                qlib_model_type: str = "LSTM", 
                use_rdagent: bool = True,
                qlib_model_kwargs: Optional[Dict[str, Any]] = None,
                rdagent_hidden_dim: int = 128,
                device: str = "cpu"):
        """
        Initialize the QlibRDPredictor.
        
        Args:
            qlib_model_type: Type of Qlib model to use (LSTM or GRU)
            use_rdagent: Whether to use RDAgent for refinement
            qlib_model_kwargs: Additional kwargs for Qlib model
            rdagent_hidden_dim: Hidden dimension for RDAgent networks
            device: Device to run models on (cpu or cuda)
        """
        self.qlib_model_type = qlib_model_type
        self.use_rdagent = use_rdagent
        self.device = device
        self.rdagent_hidden_dim = rdagent_hidden_dim
        
        # Set default model kwargs if None
        if qlib_model_kwargs is None:
            self.qlib_model_kwargs = {
                "d_feat": 20,  # dimension of features
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "n_epochs": 100,
                "lr": 1e-3,
                "early_stop": 20,
                "batch_size": 1024,
                "metric": "loss",
                "loss": "mse",
                "GPU": 0 if device == "cuda" else -1,
            }
        else:
            self.qlib_model_kwargs = qlib_model_kwargs
            
        # Will be initialized during fit
        self.qlib_model = None
        self.rdagent_model = None
        self._is_fitted = False
    
    def _init_qlib_model(self, d_feat: int) -> Model:
        """Initialize the Qlib model based on the model type."""
        # Update d_feat in model kwargs
        model_kwargs = self.qlib_model_kwargs.copy()
        model_kwargs["d_feat"] = d_feat
        
        if self.qlib_model_type.upper() == "LSTM":
            return LSTMModel(**model_kwargs)
        elif self.qlib_model_type.upper() == "GRU":
            return GRUModel(**model_kwargs)
        else:
            raise ValueError(f"Unsupported Qlib model type: {self.qlib_model_type}")
    
    def _init_rdagent_model(self, state_dim: int) -> RDAgentModel:
        """Initialize the RDAgent model."""
        # 3 actions: buy, hold, sell
        return RDAgentModel(state_dim=state_dim, action_dim=3, hidden_dim=self.rdagent_hidden_dim)
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for RDAgent model."""
        # Extract features and target from data
        if isinstance(data, pd.DataFrame):
            features = data.drop(columns=["label"]).values if "label" in data.columns else data.values
            labels = data["label"].values if "label" in data.columns else None
            return features, labels
        else:
            # Assume data is already processed
            return data, None
    
    def _convert_to_rdagent_state(self, features: np.ndarray) -> np.ndarray:
        """Convert Qlib features to RDAgent state representation."""
        # For now, we'll just use the features directly
        # In a more advanced implementation, we could use RDKit to create molecular-like
        # representations of the financial data
        return features
    
    def fit(self, 
           train_data: Union[pd.DataFrame, DatasetH], 
           val_data: Optional[Union[pd.DataFrame, DatasetH]] = None,
           epochs: int = 100) -> Dict[str, Any]:
        """
        Fit the model to the training data.
        
        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of epochs for RDAgent training
            
        Returns:
            Dictionary with training metrics
        """
        # Initialize Qlib model
        if isinstance(train_data, pd.DataFrame):
            d_feat = train_data.shape[1] - 1 if "label" in train_data.columns else train_data.shape[1]
        else:
            # Assuming DatasetH
            d_feat = train_data.handler.get_feature_dim()
        
        self.qlib_model = self._init_qlib_model(d_feat)
        
        # Train Qlib model
        logger.info("Training Qlib model...")
        self.qlib_model.fit(train_data, val_data)
        
        # Initialize and train RDAgent if enabled
        if self.use_rdagent:
            logger.info("Training RDAgent model...")
            
            # Convert data to numpy arrays
            features, labels = self._preprocess_data(train_data)
            state_dim = features.shape[1]
            
            # Initialize RDAgent model
            self.rdagent_model = self._init_rdagent_model(state_dim)
            
            # Get initial predictions from Qlib model
            qlib_preds = self.qlib_model.predict(train_data)
            
            # Create environment-like data for RDAgent training
            # Each step is a day, state is feature vector, action is buy/hold/sell
            rdagent_metrics = {"actor_loss": [], "critic_loss": [], "rewards": []}
            
            # Simple environment simulation for training
            for epoch in range(epochs):
                total_reward = 0
                
                # Iterate through data points sequentially
                for i in range(len(features) - 1):  # -1 to have next state
                    state = self._convert_to_rdagent_state(features[i])
                    next_state = self._convert_to_rdagent_state(features[i+1])
                    
                    # Get action
                    action = self.rdagent_model.get_action(state, explore=True)
                    
                    # Calculate reward based on action and next day's price change
                    # Assuming last column of features is related to price change
                    price_change = next_state[-1]  
                    
                    # Reward scheme:
                    # Buy (0) + price up = positive reward, Buy + price down = negative reward
                    # Sell (2) + price down = positive reward, Sell + price up = negative reward
                    # Hold (1) = small reward based on abs(price_change)
                    if action == 0:  # Buy
                        reward = price_change
                    elif action == 2:  # Sell
                        reward = -price_change
                    else:  # Hold
                        reward = 0.1 * abs(price_change)
                    
                    # Train model
                    losses = self.rdagent_model.train_step(state, action, reward, next_state, i == len(features) - 2)
                    
                    # Track metrics
                    rdagent_metrics["actor_loss"].append(losses["actor_loss"])
                    rdagent_metrics["critic_loss"].append(losses["critic_loss"])
                    total_reward += reward
                
                rdagent_metrics["rewards"].append(total_reward)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Total reward: {total_reward:.4f}, "
                               f"Actor loss: {np.mean(rdagent_metrics['actor_loss'][-100:]):.4f}, "
                               f"Critic loss: {np.mean(rdagent_metrics['critic_loss'][-100:]):.4f}")
        
        self._is_fitted = True
        return {
            "qlib_model": self.qlib_model_type,
            "rdagent": self.use_rdagent,
            "rdagent_metrics": rdagent_metrics if self.use_rdagent else None
        }
    
    def predict(self, data: Union[pd.DataFrame, DatasetH]) -> np.ndarray:
        """
        Generate predictions for the input data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Array of predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict")
        
        # Get predictions from Qlib model
        qlib_preds = self.qlib_model.predict(data)
        
        # Refine predictions with RDAgent if enabled
        if self.use_rdagent and self.rdagent_model is not None:
            features, _ = self._preprocess_data(data)
            
            # Convert predictions to RDAgent actions
            rdagent_actions = np.zeros_like(qlib_preds)
            
            for i in range(len(features)):
                state = self._convert_to_rdagent_state(features[i])
                # Get action (0=buy, 1=hold, 2=sell)
                action = self.rdagent_model.get_action(state, explore=False)
                
                # Convert action to signal
                if action == 0:  # Buy
                    rdagent_actions[i] = 1.0
                elif action == 2:  # Sell
                    rdagent_actions[i] = -1.0
                else:  # Hold
                    rdagent_actions[i] = 0.0
            
            # Blend Qlib predictions with RDAgent actions
            # Here we simply use RDAgent actions as they're more direct
            return rdagent_actions
        else:
            return qlib_preds
    
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Directory to save model to
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model configuration
        config = {
            "qlib_model_type": self.qlib_model_type,
            "use_rdagent": self.use_rdagent,
            "qlib_model_kwargs": self.qlib_model_kwargs,
            "rdagent_hidden_dim": self.rdagent_hidden_dim,
            "device": self.device
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        
        # Save Qlib model
        if hasattr(self.qlib_model, "save"):
            self.qlib_model.save(os.path.join(path, "qlib_model"))
        
        # Save RDAgent model
        if self.use_rdagent and self.rdagent_model is not None:
            self.rdagent_model.save(os.path.join(path, "rdagent_model"))
    
    @classmethod
    def load(cls, path: str) -> 'QlibRDPredictor':
        """
        Load model from disk.
        
        Args:
            path: Directory to load model from
            
        Returns:
            Loaded model
        """
        # Load configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(
            qlib_model_type=config["qlib_model_type"],
            use_rdagent=config["use_rdagent"],
            qlib_model_kwargs=config["qlib_model_kwargs"],
            rdagent_hidden_dim=config["rdagent_hidden_dim"],
            device=config["device"]
        )
        
        # Load Qlib model
        if os.path.exists(os.path.join(path, "qlib_model")):
            instance.qlib_model = cls._load_qlib_model(path)
        
        # Load RDAgent model
        if config["use_rdagent"] and os.path.exists(os.path.join(path, "rdagent_model")):
            instance.rdagent_model = RDAgentModel(
                state_dim=0,  # Will be overwritten when loaded
                action_dim=3,
                hidden_dim=config["rdagent_hidden_dim"]
            )
            instance.rdagent_model.load(os.path.join(path, "rdagent_model"))
        
        instance._is_fitted = True
        return instance
    
    @staticmethod
    def _load_qlib_model(path: str) -> Model:
        """Helper method to load Qlib model."""
        # This would need to be implemented based on how Qlib models are saved
        # For now, return a placeholder
        return None 