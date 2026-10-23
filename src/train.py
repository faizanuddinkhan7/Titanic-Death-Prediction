"""
Training Pipeline for Titanic Survival Prediction using PyTorch.
Performs 5-Fold Stratified Cross-Validation, out-of-fold evaluation,
plots training history, and generates submission.csv.
"""

import os
import sys
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocess import TitanicPreprocessor
from src.dataset import TitanicDataset, create_dataloaders
from src.model import TitanicMLP

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_y)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * len(batch_y)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        all_probs.extend(probs.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(batch_y.cpu().numpy().flatten())

    acc = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    return total_loss / len(all_targets), acc, auc, np.array(all_probs), np.array(all_preds)

def train_cv(
    train_csv_path: str = "titanic/train.csv",
    test_csv_path: str = "titanic/test.csv",
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    models_dir: str = "models",
    plots_dir: str = "plots",
    seed: int = 42
):
    seed_everything(seed)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load raw data
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_predictions = np.zeros(len(train_df))
    oof_probabilities = np.zeros(len(train_df))
    test_predictions_matrix = np.zeros((n_splits, len(test_df)))

    fold_metrics = []
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print("\n" + "=" * 65)
    print(f"STARTING {n_splits}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 65)

    y_train_full = train_df['Survived'].values

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train_full)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        train_fold_df = train_df.iloc[train_idx].copy()
        val_fold_df = train_df.iloc[val_idx].copy()

        # Fit preprocessor strictly on train_fold (avoiding leakage)
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(train_fold_df, test_df)

        X_train_f, y_train_f = preprocessor.transform(train_fold_df)
        X_val_f, y_val_f = preprocessor.transform(val_fold_df)
        X_test_f, _ = preprocessor.transform(test_df)

        input_dim = X_train_f.shape[1]

        train_loader, val_loader = create_dataloaders(
            X_train_f, y_train_f, X_val_f, y_val_f, batch_size=batch_size
        )

        model = TitanicMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.25).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_state = None
        best_oof_probs = None
        best_oof_preds = None

        fold_train_losses = []
        fold_val_losses = []
        fold_val_accs = []

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, v_auc, v_probs, v_preds = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            fold_train_losses.append(t_loss)
            fold_val_losses.append(v_loss)
            fold_val_accs.append(v_acc)

            # Checkpoint condition based on val accuracy & loss
            if v_acc > best_val_acc or (v_acc == best_val_acc and v_loss < best_val_loss):
                best_val_acc = v_acc
                best_val_loss = v_loss
                best_state = model.state_dict().copy()
                best_oof_probs = v_probs
                best_oof_preds = v_preds

        # Load best model for fold
        model.load_state_dict(best_state)
        model_save_path = os.path.join(models_dir, f"titanic_mlp_fold{fold + 1}.pth")
        torch.save(best_state, model_save_path)

        oof_probabilities[val_idx] = best_oof_probs
        oof_predictions[val_idx] = best_oof_preds

        # Predict on test set
        model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(X_test_f, dtype=torch.float32).to(device)
            test_probs = torch.sigmoid(model(test_tensor)).cpu().numpy().flatten()
            test_predictions_matrix[fold] = test_probs

        val_auc_score = roc_auc_score(y_val_f, best_oof_probs)
        val_f1 = f1_score(y_val_f, best_oof_preds)
        print(f"Fold {fold + 1} Best Validation Accuracy: {best_val_acc * 100:.2f}% | AUC: {val_auc_score:.4f} | F1: {val_f1:.4f}")
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': best_val_acc,
            'auc': val_auc_score,
            'f1': val_f1,
            'loss': best_val_loss
        })

        history['train_loss'].append(fold_train_losses)
        history['val_loss'].append(fold_val_losses)
        history['val_acc'].append(fold_val_accs)

    # Aggressive cross-checking and overall metrics
    oof_acc = accuracy_score(y_train_full, oof_predictions)
    oof_auc = roc_auc_score(y_train_full, oof_probabilities)
    oof_prec = precision_score(y_train_full, oof_predictions)
    oof_rec = recall_score(y_train_full, oof_predictions)
    oof_f1 = f1_score(y_train_full, oof_predictions)

    mean_acc = np.mean([m['accuracy'] for m in fold_metrics])
    std_acc = np.std([m['accuracy'] for m in fold_metrics])

    print("\n" + "=" * 65)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 65)
    for m in fold_metrics:
        print(f"Fold {m['fold']}: Accuracy = {m['accuracy']*100:.2f}%, AUC = {m['auc']:.4f}, F1 = {m['f1']:.4f}")
    print("-" * 65)
    print(f"Mean Fold Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
    print(f"Overall Out-Of-Fold Accuracy: {oof_acc * 100:.2f}% ({int(oof_acc * len(train_df))}/{len(train_df)} correct)")
    print(f"Overall OOF ROC-AUC:         {oof_auc:.4f}")
    print(f"Overall OOF Precision:       {oof_prec:.4f}")
    print(f"Overall OOF Recall:          {oof_rec:.4f}")
    print(f"Overall OOF F1-Score:        {oof_f1:.4f}")
    print("=" * 65)

    # Generate ensemble test predictions
    mean_test_probs = np.mean(test_predictions_matrix, axis=0)
    final_test_preds = (mean_test_probs >= 0.5).astype(int)

    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_test_preds
    })
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nFinal test predictions saved to: {submission_path}")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for f in range(n_splits):
        plt.plot(history['train_loss'][f], alpha=0.3, label=f'Fold {f+1} Train' if f==0 else "")
        plt.plot(history['val_loss'][f], label=f'Fold {f+1} Val')
    plt.title('Loss Curves across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    for f in range(n_splits):
        plt.plot(history['val_acc'][f], label=f'Fold {f+1}')
    plt.title('Validation Accuracy across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    curve_plot_path = os.path.join(plots_dir, 'training_curves.png')
    plt.savefig(curve_plot_path, dpi=150)
    plt.close()
    print(f"Training curves plot saved to: {curve_plot_path}")

    return {
        'fold_metrics': fold_metrics,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'oof_acc': oof_acc,
        'oof_auc': oof_auc,
        'oof_f1': oof_f1,
        'submission_path': submission_path
    }

if __name__ == "__main__":
    train_cv()
