"""TabNet path: 5-fold CV with SMOTE + Gaussian noise per fold.

Requires: pip install pytorch-tabnet imbalanced-learn
"""
from __future__ import annotations
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from config import CFG, set_global_seeds
from data import load_tabular
from cv import build_folds
from aug_wrapper import augment_tabnet_fold
from evaluate import (fold_metrics, aggregate_folds, save_result,
                      plot_roc, plot_confusion)


def _make_model() -> TabNetClassifier:
    return TabNetClassifier(
        n_d=CFG.tabnet_n_d,
        n_a=CFG.tabnet_n_a,
        n_steps=CFG.tabnet_n_steps,
        gamma=CFG.tabnet_gamma,
        lambda_sparse=CFG.tabnet_lambda_sparse,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=CFG.tabnet_lr),
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        device_name=CFG.device,
        seed=CFG.seed,
        verbose=0,
    )


def train_tabnet(model_name: str = "tabnet", augmented: bool = True) -> dict:
    set_global_seeds()
    X, y, feat_names = load_tabular()
    folds = build_folds(y)

    per_fold, oof_prob = [], np.zeros(len(y), dtype=np.float32)
    for k, (tr, va) in enumerate(folds):
        # Per-fold standardization: fit on train, apply to val (no leakage)
        mean = X[tr].mean(axis=0, keepdims=True).astype(np.float32)
        std = X[tr].std(axis=0, keepdims=True).astype(np.float32) + 1e-6
        X_tr = ((X[tr] - mean) / std).astype(np.float32)
        X_va = ((X[va] - mean) / std).astype(np.float32)
        y_tr, y_va = y[tr], y[va]

        if augmented:
            X_tr, y_tr = augment_tabnet_fold(X_tr, y_tr, seed=CFG.seed + k,
                                             feature_names=feat_names)

        clf = _make_model()
        clf.fit(
            X_train=X_tr, y_train=y_tr,
            eval_set=[(X_va, y_va)],
            eval_name=["val"], eval_metric=["auc"],
            max_epochs=CFG.tabnet_max_epochs,
            patience=CFG.tabnet_patience,
            batch_size=CFG.tabnet_batch_size,
            virtual_batch_size=CFG.tabnet_virtual_batch_size,
            num_workers=CFG.num_workers,
            drop_last=False,
        )

        prob = clf.predict_proba(X_va)[:, 1]
        oof_prob[va] = prob
        m = fold_metrics(y_va, prob)
        per_fold.append(m)
        print(f"[tabnet] fold {k}: AUC={m['auc']:.4f} F1={m['f1']:.4f} "
              f"Recall={m['recall']:.4f}")

        # Save last-fold weights for the SHAP step
        if k == CFG.n_folds - 1:
            clf.save_model(str(CFG.artifacts_dir / "weights" / f"{model_name}_lastfold"))

    summary = aggregate_folds(per_fold)
    save_result(model_name, summary, per_fold)
    plot_roc(y, oof_prob, model_name)
    plot_confusion(y, oof_prob, model_name)
    print(f"[tabnet] CV AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    return summary


if __name__ == "__main__":
    # Run both variants so M2 can quantify the augmentation benefit (Phase 3, M2 task)
    train_tabnet("tabnet_augmented", augmented=True)
    train_tabnet("tabnet_realonly",  augmented=False)
