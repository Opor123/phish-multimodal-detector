"""
train.py
--------
Training and evaluation pipeline for MultimodalPhishingDetector.

Usage (from project root):
    python src/train.py

Key features:
  * Mixed-precision training (torch.cuda.amp) for faster GPU throughput.
  * Cosine annealing LR scheduler with linear warm-up.
  * Gradient clipping to stabilise BERT fine-tuning.
  * Best-model checkpoint saved automatically.
  * Full classification report printed after training.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Allow running as a script from any working directory.
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.config import config
from src.dataset import build_dataloaders
from src.model import MultimodalPhishingDetector
from src.utils import (
    compute_metrics,
    format_metrics,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    EarlyStopping,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Training step                                                               #
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Run one full pass over the training set.

    Returns a dict with keys: loss, accuracy, precision, recall, f1.
    """
    model.train()

    all_preds, all_labels = [], []
    running_loss = 0.0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        # Move tensors to the target device.
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass under automatic mixed precision.
        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask, images)  # (B, num_classes)
            loss = criterion(logits, labels)

        # Backward pass with gradient scaling.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate predictions for epoch-level metrics.
        preds = logits.argmax(dim=1).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        running_loss += loss.item()

        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            logger.info(
                "Epoch %d | batch %d/%d | loss: %.4f",
                epoch, batch_idx + 1, num_batches, avg_loss,
            )

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / num_batches
    return metrics


# --------------------------------------------------------------------------- #
#  Evaluation step                                                             #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run inference over the validation (or test) set.

    Returns a dict with keys: loss, accuracy, precision, recall, f1.
    """
    model.eval()

    all_preds, all_labels = [], []
    running_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        running_loss += loss.item()

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader)
    return metrics


# --------------------------------------------------------------------------- #
#  Main training loop                                                          #
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    """Full training + validation routine."""

    # ── Reproducibility ──────────────────────────────────────────────── #
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    logger.info("Training on device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────── #
    logger.info("Building DataLoaders from: %s", config.DATASET_DIR)
    train_loader, val_loader = build_dataloaders(
        root_dir=config.DATASET_DIR,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    logger.info(
        "Dataset – train batches: %d | val batches: %d",
        len(train_loader), len(val_loader),
    )

    # ── Model ────────────────────────────────────────────────────────── #
    model = MultimodalPhishingDetector(
        text_model_name=config.TEXT_MODEL_NAME,
        freeze_bert_layers=args.freeze_bert_layers,
        freeze_resnet=args.freeze_resnet,
    ).to(device)

    # Log parameter counts.
    param_counts = model.count_parameters()
    for sub_name, counts in param_counts.items():
        logger.info(
            "  %-20s trainable: %10d / %10d",
            sub_name, counts["trainable"], counts["total"],
        )

    # ── Loss, Optimiser, Scheduler ───────────────────────────────────── #
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Use different LRs for the transformer backbone vs the rest.
    bert_params = list(model.text_encoder.parameters())
    other_params = (
        list(model.image_encoder.parameters())
        + list(model.fusion.parameters())
        + list(model.classifier.parameters())
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": config.LEARNING_RATE},
            {"params": other_params, "lr": config.LEARNING_RATE * 10},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-7)
    scaler = GradScaler(enabled=device.type == "cuda")
    early_stopper = EarlyStopping(patience=args.patience, min_delta=0.001)

    # ── Optional checkpoint resume ───────────────────────────────────── #
    start_epoch = 1
    best_val_f1 = 0.0
    if args.resume:
        start_epoch, best_val_f1 = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info("Resumed from %s – start epoch %d", args.resume, start_epoch)

    # ── Checkpoint directory ─────────────────────────────────────────── #
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    # ── Training loop ────────────────────────────────────────────────── #
    logger.info("Starting training for %d epochs.", config.NUM_EPOCHS)

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        t0 = time.time()

        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d (%.1fs) | "
            "Train – %s | "
            "Val   – %s",
            epoch, config.NUM_EPOCHS, elapsed,
            format_metrics(train_metrics),
            format_metrics(val_metrics),
        )

        # --- Checkpoint best model ---
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_val_f1,
            )
            logger.info("  ✓ New best F1: %.4f – checkpoint saved.", best_val_f1)

        # --- Early stopping ---
        if early_stopper(val_metrics["f1"]):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    # ── Final evaluation on best checkpoint ─────────────────────────── #
    logger.info("\n=== Final evaluation on validation set (best checkpoint) ===")
    load_checkpoint(best_ckpt_path, model)
    final_metrics = evaluate(model, val_loader, criterion, device)
    logger.info(format_metrics(final_metrics, verbose=True))


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MultimodalPhishingDetector")
    parser.add_argument(
        "--freeze-bert-layers",
        type=int,
        default=0,
        help="Number of initial BERT layers to freeze (0 = all trainable).",
    )
    parser.add_argument(
        "--freeze-resnet",
        action="store_true",
        help="Freeze most ResNet layers (layer4 stays trainable).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)