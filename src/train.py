"""
Unified training loop for Model B and Model C.
Supports fp16 mixed precision, gradient accumulation, mid-epoch checkpointing.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from pathlib import Path

from .config import SEED, USE_AMP


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    checkpoint_every: int = 2000,
    is_chunked: bool = False,
    device: str = None,
):
    """
    Train a multi-label classification model with BCEWithLogitsLoss.

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
        checkpoint_every: save mid-epoch checkpoint every N steps
        is_chunked:       True for Model C (passes chunk_counts to forward)
        device:           'cuda' or 'cpu' (auto-detected if None)

    Returns:
        history: list of dicts with epoch, train_loss, val_micro_f1
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

    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    # Mixed precision
    use_amp = use_amp and device == 'cuda'
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
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
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Progress logging
            if (step + 1) % 500 == 0:
                avg = total_loss / (step + 1)
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
        avg_loss = total_loss / len(train_loader)
        val_f1, P_val, Y_val = evaluate_predictions(
            model, val_loader, device=device, use_amp=use_amp,
            is_chunked=is_chunked,
        )
        history.append({
            'epoch': epoch,
            'train_loss': round(avg_loss, 4),
            'val_micro_f1': round(val_f1, 4),
        })
        print(f'Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  '
              f'val_micro_F1={val_f1:.4f}')

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            np.save(save_dir / 'P_val_best.npy', P_val)
            np.save(save_dir / 'Y_val_best.npy', Y_val)
            print('  -> saved best checkpoint')

    # Save training history
    pd.DataFrame(history).to_csv(save_dir / 'training_history.csv', index=False)
    print(f'Training complete. Best val micro-F1: {best_val_f1:.4f}')
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
