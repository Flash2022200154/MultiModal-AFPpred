"""
Sequence encoder with residual connection and BiLSTM.

Pipeline:
- Input fused vector: (B, fusion_dim)
- Parallel 1D conv (k=3,5) with ReLU, keep length
- MaxPool1d(kernel=2) to downsample sequence length by 2
- BiLSTM over the temporal dimension
- Temporal mean pooling -> (B, 2*lstm_hidden)
- Residual: Linear(fusion_dim -> 2*lstm_hidden) and add

Return:
- (B, 2*lstm_hidden)
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBiLSTMBlock(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 512,
        conv_filters: int = 64,
        lstm_hidden: int = 128,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Parallel conv on temporal axis (treat fused as length=fusion_dim, channel=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=conv_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=conv_filters, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)  # length // 2

        # BiLSTM over concatenated conv features
        self.bi_lstm = nn.LSTM(
            input_size=2 * conv_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=lstm_dropout if lstm_dropout > 0 and 1 > 1 else 0.0,  # only applied if num_layers > 1
            bidirectional=True,
        )

        # Residual mapping: fusion_dim -> 2*lstm_hidden
        self.res_fc = nn.Linear(fusion_dim, 2 * lstm_hidden)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, fusion_dim)
        Returns:
            out: (B, 2*lstm_hidden)
        """
        B, feat_dim = fused.shape
        # 1) Parallel convs on temporal axis
        x = fused.unsqueeze(1)  # (B, 1, feat_dim) as a 1D signal with length=feat_dim
        c3 = F.relu(self.conv3(x))  # (B, C, feat_dim)
        c5 = F.relu(self.conv5(x))  # (B, C, feat_dim)
        feat = torch.cat([c3, c5], dim=1)  # (B, 2C, feat_dim)

        # 2) Downsample temporal length
        feat = self.pool(feat)  # (B, 2C, feat_dim//2)
        # 3) Prepare for LSTM: (B, T, 2C)
        feat = feat.transpose(1, 2).contiguous()  # (B, T, 2C)

        # 4) BiLSTM
        lstm_out, _ = self.bi_lstm(feat)  # (B, T, 2*lstm_hidden)

        # 5) Temporal mean pooling
        lstm_rep = lstm_out.mean(dim=1)  # (B, 2*lstm_hidden)

        # 6) Residual connection
        res = self.res_fc(fused)  # (B, 2*lstm_hidden)
        out = lstm_rep + res
        return out