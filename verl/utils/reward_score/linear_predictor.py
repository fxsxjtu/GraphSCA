"""
Linear MI/Position Predictor for Step Reward Calculation

This module provides functionality to load and use trained linear predictors
for computing step rewards based on activation embeddings.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional
import os


class LinearMIPredictor(nn.Module):
    """线性MI预测器"""

    def __init__(self, input_dim):
        super(LinearMIPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class RNNMIPredictor(nn.Module):
    """RNN-based MI预测器，支持LSTM和GRU"""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, rnn_type='lstm', dropout=0.2):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: RNN隐藏层维度
            num_layers: RNN层数
            rnn_type: RNN类型，'lstm' 或 'gru'
            dropout: dropout比例
        """
        super(RNNMIPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # 创建RNN层
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=1,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}，请选择 'lstm' 或 'gru'")

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, input_dim) or (input_dim,)
        Returns:
            output: shape (batch_size,) or scalar
        """
        # Handle single sample case
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, input_dim)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.size(0)

        # 将输入reshape为序列形式: (batch_size, seq_len, feature_dim)
        # 这里将每个特征看作序列中的一个时间步
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)

        # 通过RNN
        if self.rnn_type == 'lstm':
            rnn_out, (hidden, cell) = self.rnn(x)
            # 使用最后一层的hidden state
            last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        else:  # GRU
            rnn_out, hidden = self.rnn(x)
            last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # 通过全连接层
        output = self.fc(last_hidden)  # (batch_size, 1)
        output = output.squeeze(-1)  # (batch_size,)

        if squeeze_output:
            output = output.squeeze(0)  # scalar

        return output


class ActivationRewardPredictor:
    """
    Wrapper class for loading and using activation-based reward predictors

    This class handles loading trained linear or RNN predictors and provides
    a unified interface for computing step rewards from activation embeddings.
    """

    def __init__(
        self,
        model_path: str,
        model_type: Literal["linear", "rnn"] = "linear",
        label_type: Literal["mi", "position"] = "mi",
        device: str = "cuda",
        normalize_output: bool = False,
        output_scale: float = 1.0,
    ):
        """
        Initialize the activation reward predictor

        Args:
            model_path: Path to the saved model checkpoint (.pth file)
            model_type: Type of model - "linear" or "rnn"
            label_type: What the model predicts - "mi" or "position"
            device: Device to run inference on
            normalize_output: Whether to normalize output to [0, 1] range
            output_scale: Scale factor to multiply the output
        """
        self.model_path = model_path
        self.model_type = model_type
        self.label_type = label_type
        self.device = device
        self.normalize_output = normalize_output
        self.output_scale = output_scale

        # Load the model
        self.model = self._load_model()
        self.model.eval()

        print(f"[ActivationRewardPredictor] Loaded {model_type} {label_type} predictor from {model_path}")
        print(f"[ActivationRewardPredictor] Device: {device}, Scale: {output_scale}, Normalize: {normalize_output}")

    def _load_model(self) -> nn.Module:
        """Load the trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Validate checkpoint
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'model_state_dict'")
        if 'input_dim' not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'input_dim'")

        input_dim = checkpoint['input_dim']

        # Create model based on type
        if self.model_type == "linear":
            model = LinearMIPredictor(input_dim)
        elif self.model_type == "rnn":
            # Load RNN hyperparameters from checkpoint
            model = RNNMIPredictor(
                input_dim=input_dim,
                hidden_dim=checkpoint.get('hidden_dim', 128),
                num_layers=checkpoint.get('num_layers', 2),
                rnn_type=checkpoint.get('rnn_type', 'lstm'),
                dropout=checkpoint.get('dropout', 0.2),
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        # Print metrics if available
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"[ActivationRewardPredictor] Model metrics: {metrics}")

        return model

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute reward predictions from embeddings

        Args:
            embeddings: Activation embeddings of shape:
                - (batch_size, hidden_dim) for batch prediction
                - (hidden_dim,) for single prediction

        Returns:
            predictions: Predicted rewards of shape:
                - (batch_size,) for batch input
                - scalar for single input
        """
        # Ensure embeddings are on the correct device
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)

        # Get predictions from model
        predictions = self.model(embeddings)

        # Apply normalization if requested
        if self.normalize_output:
            # Normalize to [0, 1] range
            # Assuming MI values are typically in a certain range
            # You may need to adjust these based on your training data
            if self.label_type == "mi":
                # Example: normalize MI to [0, 1] assuming max MI ~ 10
                predictions = torch.sigmoid(predictions / 5.0)
            elif self.label_type == "position":
                # Position is already in [0, 1] from training
                predictions = torch.clamp(predictions, 0.0, 1.0)

        # Apply output scale
        predictions = predictions * self.output_scale

        return predictions

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Callable interface for easier use

        Args:
            embeddings: Activation embeddings

        Returns:
            predictions: Predicted rewards
        """
        return self.predict(embeddings)


def create_activation_reward_fn(
    model_path: str,
    model_type: Literal["linear", "rnn"] = "linear",
    label_type: Literal["mi", "position"] = "mi",
    device: str = "cuda",
    normalize_output: bool = False,
    output_scale: float = 1.0,
):
    """
    Factory function to create an activation reward function

    This is a convenience function that returns a callable that can be directly
    used as the `activation_reward_fn` parameter in GraphRewardManager.

    Args:
        model_path: Path to model checkpoint
        model_type: "linear" or "rnn"
        label_type: "mi" or "position"
        device: Device for inference
        normalize_output: Whether to normalize output
        output_scale: Scale factor for output

    Returns:
        A callable that takes embeddings and returns rewards

    Example:
        >>> reward_fn = create_activation_reward_fn(
        ...     model_path="/path/to/linear_mi_predictor.pth",
        ...     model_type="linear",
        ...     label_type="mi",
        ...     output_scale=0.1
        ... )
        >>> # Use in GraphRewardManager
        >>> reward_manager = GraphRewardManager(
        ...     tokenizer=tokenizer,
        ...     activation_reward_fn=reward_fn,
        ...     ...
        ... )
    """
    predictor = ActivationRewardPredictor(
        model_path=model_path,
        model_type=model_type,
        label_type=label_type,
        device=device,
        normalize_output=normalize_output,
        output_scale=output_scale,
    )

    return predictor
