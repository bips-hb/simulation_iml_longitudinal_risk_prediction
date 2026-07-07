"""
model_zoo.py
============
Single source of truth for all model definitions and the shared training
pipeline used across the simulation and eICU experiments.

Imported by both utils.py:

    from model_zoo import (
        GRUModel, LSTMModel, LogitWrapper,
        AttentionGRUModel, AttentionLSTMModel,
        CNNModel, run_training_pipeline,
    )

Sections
--------
1. Model definitions
   GRUModel, LSTMModel, LogitWrapper,
   AttentionGRUModel, AttentionLSTMModel, CNNModel
2. Training pipeline
   run_training_pipeline
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


# ==============================================================================
# 1.  MODEL DEFINITIONS
# ==============================================================================

class GRUModel(nn.Module):
    """Gated Recurrent Unit classifier.

    Parameters
    ----------
    input_size   : number of features per timestep
    hidden_size  : GRU hidden dimension
    output_size  : number of output logits (1 for binary)
    dropout_rate : dropout probability applied before the FC layer
    static_size  : number of static features concatenated at the FC layer (0 = none)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, dropout_rate: float = 0.5,
                 static_size: int = 0):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hidden_size + static_size, output_size)
        self.static_size = static_size

    def forward_logits(self, x: torch.Tensor,
                        x_static: torch.Tensor = None) -> torch.Tensor:
        _, h_n = self.gru(x)
        h = self.dropout(h_n[-1])
        if self.static_size > 0 and x_static is not None:
            h = torch.cat([h, x_static], dim=1)
        return self.fc(h)

    def forward(self, x: torch.Tensor,
                 x_static: torch.Tensor = None) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x, x_static))


class LSTMModel(nn.Module):
    """Long Short-Term Memory classifier.

    Parameters
    ----------
    input_size   : number of features per timestep
    hidden_size  : LSTM hidden dimension
    output_size  : number of output logits (1 for binary)
    dropout_rate : dropout probability applied before the FC layer
    static_size  : number of static features concatenated at the FC layer (0 = none)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, dropout_rate: float = 0.5,
                 static_size: int = 0):
        super().__init__()
        self.lstm       = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hidden_size + static_size, output_size)
        self.static_size = static_size

    def forward_logits(self, x: torch.Tensor,
                        x_static: torch.Tensor = None) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        if self.static_size > 0 and x_static is not None:
            h = torch.cat([h, x_static], dim=1)
        return self.fc(h)

    def forward(self, x: torch.Tensor,
                 x_static: torch.Tensor = None) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x, x_static))


class LogitWrapper(nn.Module):
    """Wraps any model so its output is always shape (B, 1).

    Required by SHAP GradientExplainer which expects a fixed output shape.
    Optionally holds a fixed x_static tensor that is concatenated on every
    forward call — use this when the explainer only passes sequence input.
    """

    def __init__(self, model: nn.Module, x_static: torch.Tensor = None):
        super().__init__()
        self.model    = model
        self.x_static = x_static  # (1, n_static) or None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.x_static
        if s is not None:
            s = s.expand(x.shape[0], -1).to(x.device)
        out = self.model.forward_logits(x, s)
        return out.unsqueeze(1) if out.dim() == 1 else out


class AttentionGRUModel(nn.Module):
    """GRU with additive (Bahdanau-style) attention over visit hidden states.

    The attention weights are directly interpretable: weight[t] = how much
    the model focused on visit t when making its prediction.
    """
    def __init__(self, input_size, hidden_size=32, output_size=1,
                 dropout_rate=0.5, static_size=0):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attn       = nn.Linear(hidden_size, 1)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hidden_size + static_size, output_size)
        self.static_size = static_size

    def forward_with_attn(self, x, x_static=None):
        """Returns (logit, attn_weights [B, T])."""
        h_all, _ = self.gru(x)                          # [B, T, H]
        scores   = self.attn(h_all).squeeze(-1)          # [B, T]
        weights  = torch.softmax(scores, dim=1)          # [B, T]
        context  = (weights.unsqueeze(-1) * h_all).sum(dim=1)  # [B, H]
        context  = self.dropout(context)
        if self.static_size > 0 and x_static is not None:
            context = torch.cat([context, x_static], dim=1)
        logit    = self.fc(context)                      # [B, 1]
        return logit, weights

    def forward_logits(self, x, x_static=None):
        logit, _ = self.forward_with_attn(x, x_static)
        return logit

    def forward(self, x, x_static=None):
        return torch.sigmoid(self.forward_logits(x, x_static))

    def get_attention_weights(self, x, x_static=None):
        """Return attention weights [B, T] as numpy (no grad)."""
        with torch.no_grad():
            _, w = self.forward_with_attn(x, x_static)
        return w.cpu().numpy()


class AttentionLSTMModel(nn.Module):
    """LSTM with additive attention over visit hidden states."""
    def __init__(self, input_size, hidden_size=32, output_size=1,
                 dropout_rate=0.5, static_size=0):
        super().__init__()
        self.lstm       = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn       = nn.Linear(hidden_size, 1)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hidden_size + static_size, output_size)
        self.static_size = static_size

    def forward_with_attn(self, x, x_static=None):
        h_all, _ = self.lstm(x)
        scores   = self.attn(h_all).squeeze(-1)
        weights  = torch.softmax(scores, dim=1)
        context  = (weights.unsqueeze(-1) * h_all).sum(dim=1)
        context  = self.dropout(context)
        if self.static_size > 0 and x_static is not None:
            context = torch.cat([context, x_static], dim=1)
        logit    = self.fc(context)
        return logit, weights

    def forward_logits(self, x, x_static=None):
        logit, _ = self.forward_with_attn(x, x_static)
        return logit

    def forward(self, x, x_static=None):
        return torch.sigmoid(self.forward_logits(x, x_static))

    def get_attention_weights(self, x, x_static=None):
        with torch.no_grad():
            _, w = self.forward_with_attn(x, x_static)
        return w.cpu().numpy()


class CNNModel(nn.Module):
    """1D CNN classifier with parallel multi-scale convolutions.

    Parallel Conv1d with kernel sizes 3 (local) and 5 (wider context),
    followed by global average + max pooling and a two-layer FC head.
    Architecture mirrors GRUModel/LSTMModel: supports optional static
    features appended before the FC head.

    Parameters
    ----------
    input_size  : number of features per timestep (= F)
    output_size : number of output logits (1 for binary)
    static_size : number of static features concatenated at the FC layer (0 = none)
    num_filters : number of Conv1d output channels per branch
    dropout     : dropout probability before FC layers
    """

    def __init__(self, input_size: int, output_size: int = 1,
                 static_size: int = 0, num_filters: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.conv3      = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.conv5      = nn.Conv1d(input_size, num_filters, kernel_size=5, padding=2)
        self.bn3        = nn.BatchNorm1d(num_filters)
        self.bn5        = nn.BatchNorm1d(num_filters)
        self.drop       = nn.Dropout(dropout)
        self.static_size = static_size
        # avg + max for each of the two branches → 4 × num_filters
        fc_in = num_filters * 4 + static_size
        self.fc1 = nn.Linear(fc_in, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward_logits(self, x: torch.Tensor,
                        x_static: torch.Tensor = None) -> torch.Tensor:
        # x : (B, T, F)  →  (B, F, T) for Conv1d
        x  = x.permute(0, 2, 1)
        h3 = torch.relu(self.bn3(self.conv3(x)))   # (B, C, T)
        h5 = torch.relu(self.bn5(self.conv5(x)))   # (B, C, T)
        h  = torch.cat([
            h3.mean(dim=2), h3.max(dim=2).values,
            h5.mean(dim=2), h5.max(dim=2).values,
        ], dim=1)                                    # (B, 4C)
        h  = self.drop(h)
        if self.static_size > 0 and x_static is not None:
            h = torch.cat([h, x_static], dim=1)
        return self.fc2(torch.relu(self.fc1(h)))

    def forward(self, x: torch.Tensor,
                 x_static: torch.Tensor = None) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x, x_static))


# ==============================================================================
# 2.  TRAINING PIPELINE
# ==============================================================================

def run_training_pipeline(model, X_train, y_train, X_val, y_val,
                           save_path, epochs=40, batch_size=32, lr=0.001,
                           pos_weight=None,
                           X_static_train=None, X_static_val=None):
    """Train *model* and save the best checkpoint (by val AUC).

    Parameters
    ----------
    X_static_train : np.ndarray (N_train, n_static) or None
    X_static_val   : np.ndarray (N_val,   n_static) or None
    pos_weight     : torch.Tensor scalar weight for positive class (BCEWithLogitsLoss)

    Returns
    -------
    history : dict with keys 'train_loss', 'train_auc', 'val_auc'
    """
    device = next(model.parameters()).device

    if X_static_train is not None:
        train_dataset = TensorDataset(
            torch.Tensor(X_train),
            torch.Tensor(y_train).view(-1, 1),
            torch.Tensor(X_static_train),
        )
    else:
        train_dataset = TensorDataset(
            torch.Tensor(X_train),
            torch.Tensor(y_train).view(-1, 1),
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_tensor   = torch.Tensor(X_val).to(device)
    val_static   = torch.Tensor(X_static_val).to(device) if X_static_val is not None else None

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history      = {"train_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss                 = 0.0
        y_true_train, y_pred_train = [], []

        for batch in train_loader:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            batch_s = batch[2].to(device) if len(batch) == 3 else None

            optimizer.zero_grad()
            logits = model.forward_logits(batch_x, batch_s)
            loss   = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            epoch_loss   += loss.item()
            y_true_train.extend(batch_y.detach().cpu().numpy().ravel())
            y_pred_train.extend(probs.detach().cpu().numpy().ravel())

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(
                model.forward_logits(val_tensor, val_static)
            ).cpu().numpy().ravel()

        val_auc   = roc_auc_score(y_val, np.array(val_probs))
        train_auc = roc_auc_score(np.array(y_true_train), np.array(y_pred_train))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)

        history["train_loss"].append(epoch_loss / len(train_loader))
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d} | "
                  f"Loss {history['train_loss'][-1]:.4f} | "
                  f"Val AUC {val_auc:.4f}")

    return history
