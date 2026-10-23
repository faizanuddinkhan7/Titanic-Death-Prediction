"""
PyTorch Dataset and DataLoader definitions for Titanic tabular data.
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TitanicDataset(Dataset):
    """
    Dataset wrapper for Titanic features and optional survival targets.
    """
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation DataLoaders.
    """
    train_dataset = TitanicDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TitanicDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        )

    return train_loader, val_loader

