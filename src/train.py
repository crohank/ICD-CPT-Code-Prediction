"""
Training entry points for the transformer models (B and C).

One module on purpose: both share the same optimizer/scheduler shape, AMP
guards, and metric logging. Model D has its own script-style loop in the
notebooks but could call into pieces of this if you unify later.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from pathlib import Path

from .config import SEED, USE_AMP
from .evaluate import tune_global_threshold


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Focal loss down-weights "easy" negatives so rare ICD codes still move the model.

def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Focal Loss for multi-label classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Unlike BCEWithLogitsLoss + pos_weight, focal loss down-weights easy
    examples without distorting probability calibration.

    Args:
        logits:  (batch, num_labels) raw logits
        targets: (batch, num_labels) binary labels
        alpha:   weighting factor for positive class (0.25 default)
        gamma:   focusing parameter (higher = more focus on hard examples)
        reduction: 'mean', 'sum', or 'none'
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    focal_weight = alpha_t * (1 - p_t) ** gamma
    loss = focal_weight * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# Generic train/validate loop with checkpointing hooks.

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: Path,
    *,
    lr: float = 2e-5,
    epochs: int = 3,
    grad_accum: int = 1,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    use_amp: bool = USE_AMP,
    pos_weight: torch.Tensor = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    checkpoint_every: int = 2000,
    is_chunked: bool = False,
    early_stopping_patience: int = 0,
    device: str = None,
):
    """
    Train a multi-label classification model.

    Args:
        model:            nn.Module with forward() returning logits
        train_loader:     DataLoader yielding dicts with input_ids, attention_mask,
                          labels, and optionally chunk_count
        val_loader:       DataLoader for validation
        save_dir:         directory to save checkpoints and history
        lr:               learning rate
        epochs:           number of training epochs
        grad_accum:       gradient accumulation steps
        warmup_ratio:     fraction of total steps for linear warmup
        weight_decay:     AdamW weight decay
        max_grad_norm:    gradient clipping norm
        use_amp:          whether to use fp16 mixed precision
        pos_weight:       optional per-label positive class weights for BCE
                          (ignored if use_focal_loss=True)
        use_focal_loss:   if True, use focal loss instead of BCE
        focal_gamma:      focal loss gamma (focusing parameter)
        focal_alpha:      focal loss alpha (positive class weight)
        checkpoint_every: save mid-epoch checkpoint every N steps
        is_chunked:       True for Model C (passes chunk_counts to forward)
        early_stopping_patience: stop if val_f1 doesn't improve for N epochs (0=disabled)
        device:           'cuda' or 'cpu' (auto-detected if None)

    Returns:
        history: list of dicts with epoch, train_loss, val_micro_f1, val_tuned_f1, best_threshold
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer & scheduler
    total_steps  = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss function (FIX #2: focal loss option replaces aggressive pos_weight)
    if use_focal_loss:
        print(f"  Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
        criterion = None  # will call sigmoid_focal_loss directly
    else:
        pw = pos_weight.to(device) if pos_weight is not None else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        if pw is not None:
            print(f"  Using BCEWithLogitsLoss (pos_weight range: "
                  f"[{pw.min():.1f}, {pw.max():.1f}])")

    # Mixed precision
    use_amp = use_amp and device == 'cuda'
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best_val_f1 = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device, enabled=use_amp):
                if is_chunked:
                    chunk_counts = batch['chunk_count'].to(device) if 'chunk_count' in batch else None
                    logits = model(ids, mask, chunk_counts=chunk_counts)
                else:
                    logits = model(ids, mask)

                if use_focal_loss:
                    loss = sigmoid_focal_loss(logits, labels,
                                              alpha=focal_alpha,
                                              gamma=focal_gamma) / grad_accum
                else:
                    loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()
            # FIX #6: track true loss (undo grad_accum scaling for logging)
            total_loss += loss.item() * grad_accum
            num_batches += 1

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Progress logging
            if (step + 1) % 500 == 0:
                avg = total_loss / num_batches
                gpu_info = ""
                if device == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1e9
                    gpu_info = f"  GPU mem: {mem:.1f}GB"
                print(f'  Step {step+1}/{len(train_loader)}  '
                      f'loss={avg:.4f}{gpu_info}')

            # Mid-epoch checkpoint
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                torch.save({
                    'epoch': epoch, 'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, save_dir / f'checkpoint_e{epoch}_s{step+1}.pt')
                print(f'  -> mid-epoch checkpoint saved')

        # End-of-epoch validation
        avg_loss = total_loss / num_batches

        # FIX #3: Evaluate with threshold tuning (not hardcoded 0.5)
        _, P_val, Y_val = evaluate_predictions(
            model, val_loader, threshold=0.5,
            device=device, use_amp=use_amp, is_chunked=is_chunked,
        )
        # Tune threshold on validation predictions
        best_t, tuned_f1 = tune_global_threshold(P_val, Y_val)
        # Also compute F1 at default 0.5 for comparison
        default_f1 = f1_score(Y_val, (P_val >= 0.5).astype(int),
                               average='micro', zero_division=0)

        history.append({
            'epoch': epoch,
            'train_loss': round(avg_loss, 4),
            'val_f1_at_0.5': round(default_f1, 4),
            'val_f1_tuned': round(tuned_f1, 4),
            'best_threshold': round(best_t, 3),
        })
        print(f'Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  '
              f'val_F1@0.5={default_f1:.4f}  '
              f'val_F1@{best_t:.3f}={tuned_f1:.4f}')

        # Save best checkpoint (using TUNED F1, not default threshold)
        if tuned_f1 > best_val_f1:
            best_val_f1 = tuned_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            np.save(save_dir / 'P_val_best.npy', P_val)
            np.save(save_dir / 'Y_val_best.npy', Y_val)
            print(f'  -> saved best checkpoint (tuned F1={tuned_f1:.4f} @ t={best_t:.3f})')
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f'  -> Early stopping: no improvement for {early_stopping_patience} epochs')
                break

    # Save training history
    pd.DataFrame(history).to_csv(save_dir / 'training_history.csv', index=False)
    print(f'Training complete. Best val tuned micro-F1: {best_val_f1:.4f}')
    return history


@torch.no_grad()
def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    threshold: float = 0.5,
    device: str = 'cuda',
    use_amp: bool = True,
    is_chunked: bool = False,
):
    """
    Run model on a DataLoader, return micro-F1 and raw probabilities.

    Returns:
        micro_f1, P (n_samples, n_labels), Y (n_samples, n_labels)
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)

        with torch.amp.autocast(device, enabled=use_amp and device == 'cuda'):
            if is_chunked:
                chunk_counts = batch['chunk_count'].to(device) if 'chunk_count' in batch else None
                logits = model(ids, mask, chunk_counts=chunk_counts)
            else:
                logits = model(ids, mask)

        probs = torch.sigmoid(logits).cpu().float().numpy()
        all_preds.append(probs)
        all_labels.append(batch['labels'].numpy())

    P = np.vstack(all_preds)
    Y = np.vstack(all_labels)
    micro_f1 = f1_score(Y, (P >= threshold).astype(int),
                         average='micro', zero_division=0)
    return micro_f1, P, Y


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: str = 'cuda',
    use_amp: bool = True,
    is_chunked: bool = False,
):
    """
    Collect raw logits (before sigmoid) for temperature scaling calibration.

    Returns:
        logits (n_samples, n_labels), labels (n_samples, n_labels)
    """
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)

        with torch.amp.autocast(device, enabled=use_amp and device == 'cuda'):
            if is_chunked:
                chunk_counts = batch['chunk_count'].to(device) if 'chunk_count' in batch else None
                logits = model(ids, mask, chunk_counts=chunk_counts)
            else:
                logits = model(ids, mask)

        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(batch['labels'].numpy())

    return np.vstack(all_logits), np.vstack(all_labels)
