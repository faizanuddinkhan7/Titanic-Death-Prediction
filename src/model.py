"""
PyTorch Neural Network Architectures for Titanic Survival Prediction.
"""

from typing import List
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual linear block with BatchNorm, SiLU activation, and Dropout."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class TitanicMLP(nn.Module):
    """
    Modern Tabular Neural Network with residual connections,
    Batch Normalization, SiLU non-linearities, and Dropout.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.25,
        use_residual: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_residual = use_residual

        # Input projection layer
        first_dim = hidden_dims[0]
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, first_dim),
            nn.BatchNorm1d(first_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Residual or MLP intermediate layers
        if use_residual:
            self.res_block1 = ResBlock(first_dim, dropout=dropout)
            self.transition = nn.Sequential(
                nn.Linear(first_dim, hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.res_block2 = ResBlock(hidden_dims[1], dropout=dropout)
            last_dim = hidden_dims[1]
        else:
            layers = []
            prev_dim = first_dim
            for h_dim in hidden_dims[1:]:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = h_dim
            self.hidden_layers = nn.Sequential(*layers)
            last_dim = prev_dim

        # Final classification head (outputs 1 logit)
        self.head = nn.Linear(last_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        if self.use_residual:
            out = self.res_block1(out)
            out = self.transition(out)
            out = self.res_block2(out)
        else:
            out = self.hidden_layers(out)
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

