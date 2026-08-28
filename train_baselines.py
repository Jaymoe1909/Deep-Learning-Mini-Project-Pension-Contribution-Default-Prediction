"""Baselines: logistic regression and random forest on the same folds."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import CFG, set_global_seeds
from data import load_tabular
from cv import build_folds
from evaluate import (fold_metrics, aggregate_folds, save_result,
                      plot_roc, plot_confusion)


def _run_baseline(name: str, make_estimator) -> dict:
    set_global_seeds()
    X, y, _ = load_tabular()
    folds = build_folds(y)

    per_fold, oof_prob = [], np.zeros(len(y), dtype=np.float32)
    for k, (tr, va) in enumerate(folds):
        est = make_estimator()
        est.fit(X[tr], y[tr])
        probs = est.predict_proba(X[va])[:, 1]
        oof_prob[va] = probs
        m = fold_metrics(y[va], probs)
        per_fold.append(m)
        print(f"[{name}] fold {k}: AUC={m['auc']:.4f} F1={m['f1']:.4f}")

    summary = aggregate_folds(per_fold)
    save_result(name, summary, per_fold)
    plot_roc(y, oof_prob, name)
    plot_confusion(y, oof_prob, name)
    print(f"[{name}] CV AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    return summary


def train_logreg():
    return _run_baseline("logreg", lambda: Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=CFG.seed)),
    ]))


def train_rf():
    return _run_baseline("rf", lambda: RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=CFG.seed,
    ))


if __name__ == "__main__":
    train_logreg()
    train_rf()
