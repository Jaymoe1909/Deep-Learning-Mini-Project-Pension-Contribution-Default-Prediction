"""Shared evaluation: metrics, ROC curves, confusion matrices, results.json."""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, roc_curve, confusion_matrix)
from config import CFG


def fold_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                 threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc":       float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }


def aggregate_folds(fold_scores: list[dict]) -> dict:
    keys = fold_scores[0].keys()
    return {
        **{f"{k}_mean": float(np.mean([s[k] for s in fold_scores])) for k in keys},
        **{f"{k}_std":  float(np.std([s[k] for s in fold_scores]))  for k in keys},
        "n_folds": len(fold_scores),
    }


def save_result(model_name: str, summary: dict, per_fold: list[dict]) -> None:
    """Append/update this model's entry in artifacts/results.json."""
    data = {}
    if CFG.results_json.exists():
        data = json.loads(CFG.results_json.read_text())
    data[model_name] = {"summary": summary, "per_fold": per_fold}
    CFG.results_json.write_text(json.dumps(data, indent=2))
    print(f"[eval] wrote {model_name} -> {CFG.results_json.name}")


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC — {model_name}"); plt.legend(); plt.tight_layout()
    out = CFG.artifacts_dir / "plots" / f"roc_{model_name}.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"[eval] saved {out.name}")


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str, threshold: float = 0.5) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No default", "Default"])
    ax.set_yticklabels(["No default", "Default"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion — {model_name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out = CFG.artifacts_dir / "plots" / f"cm_{model_name}.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"[eval] saved {out.name}")


def build_comparison_table() -> None:
    """Print a comparison table of every model in results.json."""
    if not CFG.results_json.exists():
        print("[eval] no results yet"); return
    data = json.loads(CFG.results_json.read_text())
    header = f"{'Model':<25} {'AUC':>14} {'F1':>14} {'Recall':>14} {'Precision':>14}"
    print(header); print("-" * len(header))
    for name, entry in data.items():
        s = entry["summary"]
        row = (f"{name:<25} "
               f"{s['auc_mean']:.3f} ± {s['auc_std']:.3f}   "
               f"{s['f1_mean']:.3f} ± {s['f1_std']:.3f}   "
               f"{s['recall_mean']:.3f} ± {s['recall_std']:.3f}   "
               f"{s['precision_mean']:.3f} ± {s['precision_std']:.3f}")
        print(row)
