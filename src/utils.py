"""
utils.py
--------
Shared utility functions for the Multimodal Phishing Detection project.

Includes:
  * Metrics computation (accuracy, precision, recall, F1).
  * Checkpoint save / load helpers.
  * Reproducibility seed setter.
  * EarlyStopping callback.
  * Logging helpers.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Reproducibility                                                             #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    """
    Fix all relevant random seeds so experiments are reproducible.

    Covers Python's `random`, NumPy, PyTorch (CPU + CUDA), and CUDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA operations (may slightly slow down training).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global random seed set to %d.", seed)


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    positive_label: int = 1,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true        : Ground-truth class indices.
        y_pred        : Predicted class indices.
        positive_label: Index treated as the "positive" class (phishing=1).

    Returns:
        Dict with keys: accuracy, precision, recall, f1.
        All values are floats in [0, 1].
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def format_metrics(metrics: Dict[str, float], verbose: bool = False) -> str:
    """
    Format a metrics dict as a human-readable string.

    Args:
        metrics : Output of `compute_metrics`, optionally with a 'loss' key.
        verbose : If True, print each metric on its own line.

    Returns:
        Formatted string.
    """
    parts = []
    if "loss" in metrics:
        parts.append(f"loss={metrics['loss']:.4f}")
    parts.append(f"acc={metrics['accuracy']:.4f}")
    parts.append(f"prec={metrics['precision']:.4f}")
    parts.append(f"rec={metrics['recall']:.4f}")
    parts.append(f"f1={metrics['f1']:.4f}")

    sep = "\n  " if verbose else "  "
    return sep.join(parts)


def full_classification_report(
    y_true: List[int],
    y_pred: List[int],
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Return sklearn's full classification report as a string.

    Args:
        y_true       : Ground-truth labels.
        y_pred       : Predicted labels.
        target_names : Optional list of class name strings.
    """
    target_names = target_names or ["legitimate", "phishing"]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


# --------------------------------------------------------------------------- #
#  Checkpoint management                                                       #
# --------------------------------------------------------------------------- #

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch: int = 0,
    best_metric: float = 0.0,
) -> None:
    """
    Persist model (and optionally optimiser + scheduler) state to disk.

    Args:
        path         : File path for the checkpoint (e.g. 'checkpoints/best.pt').
        model        : The PyTorch model.
        optimizer    : (Optional) current optimiser state.
        scheduler    : (Optional) current LR scheduler state.
        epoch        : Current epoch number stored for resume.
        best_metric  : Best validation metric stored alongside the checkpoint.
    """
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved → %s (epoch %d, metric=%.4f)", path, epoch, best_metric)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Tuple[int, float]:
    """
    Load model (and optionally optimiser + scheduler) state from disk.

    Args:
        path      : Path to a checkpoint file created by `save_checkpoint`.
        model     : The PyTorch model to load state into.
        optimizer : (Optional) optimiser to restore state for.
        scheduler : (Optional) LR scheduler to restore state for.

    Returns:
        (next_epoch, best_metric) tuple.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    epoch = state.get("epoch", 0)
    best_metric = state.get("best_metric", 0.0)

    logger.info(
        "Checkpoint loaded ← %s (epoch %d, best_metric=%.4f)",
        path, epoch, best_metric,
    )
    return epoch + 1, best_metric


# --------------------------------------------------------------------------- #
#  Early stopping                                                              #
# --------------------------------------------------------------------------- #

class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    Args:
        patience  : How many epochs to wait after the last improvement.
        min_delta : Minimum change to qualify as an improvement.
        mode      : 'max' to monitor a metric that should increase (e.g. F1),
                    'min' for loss.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max") -> None:
        assert mode in ("max", "min"), "mode must be 'max' or 'min'."
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None

    def __call__(self, value: float) -> bool:
        """
        Call after each epoch.

        Returns True if training should stop, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            logger.debug("EarlyStopping counter: %d/%d", self.counter, self.patience)

        return self.counter >= self.patience


# --------------------------------------------------------------------------- #
#  Device helpers                                                              #
# --------------------------------------------------------------------------- #

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(
            "GPU: %s | Memory: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        logger.info("No GPU detected – running on CPU.")
    return device


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Move every tensor in `batch` to `device`.

    Non-tensor values (e.g. sample_id strings) are left unchanged.
    """
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }