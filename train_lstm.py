"""LSTM path: 5-fold CV on 48-month sequences with SMOTE per fold.

Mixed precision keeps the model comfortable in 4GB VRAM on the T1200.
Windows/Jupyter: num_workers=0 (see CFG).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from config import CFG, set_global_seeds
from data import load_sequences
from cv import build_folds
from aug_wrapper import smote_only
from evaluate import (fold_metrics, aggregate_folds, save_result,
                      plot_roc, plot_confusion)
from models import SequenceLSTM


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                 shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=CFG.num_workers, pin_memory=(CFG.device == "cuda"))


def _train_one_fold(X_tr, y_tr, X_va, y_va, input_size: int) -> tuple[nn.Module, np.ndarray]:
    device = torch.device(CFG.device)
    model = SequenceLSTM(
        input_size=input_size,
        hidden=CFG.lstm_hidden,
        num_layers=CFG.lstm_layers,
        dropout=CFG.lstm_dropout,
        bidirectional=CFG.lstm_bidirectional,
    ).to(device)

    # Class weights for the loss — supplements SMOTE, not a replacement
    class_counts = np.bincount(y_tr)
    weights = torch.tensor(len(y_tr) / (2 * class_counts + 1e-8),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=CFG.lstm_lr,
                              weight_decay=CFG.lstm_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=CFG.lstm_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(CFG.use_amp and CFG.device == "cuda"))

    train_loader = _make_loader(X_tr, y_tr, CFG.lstm_batch_size, shuffle=True)
    val_loader   = _make_loader(X_va, y_va, CFG.lstm_batch_size, shuffle=False)

    best_auc, best_state, patience_left = -1.0, None, CFG.lstm_patience
    for epoch in range(CFG.lstm_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(CFG.use_amp and CFG.device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
        scheduler.step()

        # Val
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(CFG.use_amp and CFG.device == "cuda")):
                    logits = model(xb)
                probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        probs = np.concatenate(probs)
        m = fold_metrics(y_va, probs)
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_state = deepcopy(model.state_dict())
            patience_left = CFG.lstm_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    # Final val probs from the best model
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            probs.append(torch.softmax(model(xb).float(), dim=1)[:, 1].cpu().numpy())
    return model, np.concatenate(probs)


def train_lstm(model_name: str = "lstm") -> dict:
    set_global_seeds()
    seq, y = load_sequences()

    # Drop constant/zero-variance features (they contribute nothing and can cause NaN)
    feat_std = seq.reshape(-1, seq.shape[-1]).std(axis=0)
    keep = feat_std > 1e-8
    if (~keep).any():
        dropped = np.where(~keep)[0].tolist()
        print(f"[lstm] dropping constant sequence features at indices: {dropped}")
        seq = seq[:, :, keep]

    folds = build_folds(y)
    input_size = seq.shape[-1]

    per_fold, oof_prob = [], np.zeros(len(y), dtype=np.float32)
    for k, (tr, va) in enumerate(folds):
        # Per-fold standardization: fit on train, apply to val (no leakage)
        train_flat = seq[tr].reshape(-1, input_size)
        mean = train_flat.mean(axis=0, keepdims=True).astype(np.float32)
        std = train_flat.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        seq_tr = ((seq[tr] - mean) / std).astype(np.float32)
        seq_va = ((seq[va] - mean) / std).astype(np.float32)

        # SMOTE on standardized training data
        X_tr, y_tr = smote_only(seq_tr, y[tr], seed=CFG.seed + k)
        X_va, y_va = seq_va, y[va]

        model, probs = _train_one_fold(X_tr, y_tr, X_va, y_va, input_size)
        oof_prob[va] = probs
        m = fold_metrics(y_va, probs)
        per_fold.append(m)
        print(f"[lstm] fold {k}: AUC={m['auc']:.4f} F1={m['f1']:.4f} "
              f"Recall={m['recall']:.4f}")

        if k == CFG.n_folds - 1:
            torch.save(model.state_dict(),
                       CFG.artifacts_dir / "weights" / f"{model_name}_lastfold.pt")

    summary = aggregate_folds(per_fold)
    save_result(model_name, summary, per_fold)
    plot_roc(y, oof_prob, model_name)
    plot_confusion(y, oof_prob, model_name)
    print(f"[lstm] CV AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    return summary


if __name__ == "__main__":
    train_lstm()
