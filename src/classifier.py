"""
Classifier head for binary prediction.

- Input: (B, input_dim), typically input_dim = 2 * lstm_hidden (e.g., 256)
- Architecture: MLP with optional hidden layers and dropout
- Output:
  - probability in [0,1] if return_logits=False (default)
  - raw logits if return_logits=True (for use with BCEWithLogitsLoss)
"""
from typing import Optional, Sequence
import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Optional[Sequence[int]] = (128, 64),
        dropout: float = 0.1,
        activation: str = "relu",
        return_logits: bool = False,
    ) -> None:
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else nn.GELU

        layers = []
        prev = input_dim
        if hidden_dims:
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(act())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
        layers.append(nn.Linear(prev, 1))  # final logit
        self.mlp = nn.Sequential(*layers)
        self.return_logits = return_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            probs: (B,) if return_logits=False, else logits: (B,)
        """
        logits = self.mlp(x).squeeze(-1)  # (B,)
        if self.return_logits:
            return logits
        return torch.sigmoid(logits)