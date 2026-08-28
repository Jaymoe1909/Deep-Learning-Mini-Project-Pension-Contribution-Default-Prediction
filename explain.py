"""SHAP beeswarm plot for TabNet.

Two paths:
  1. Native TabNet feature attributions via explain() — fast, uses the
     attention masks. Good as an in-model explanation.
  2. SHAP KernelExplainer on a sampled background — slower but produces
     the canonical beeswarm plot M1 wants for domain interpretation.

We generate both. Requires: pip install shap
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import shap
from pytorch_tabnet.tab_model import TabNetClassifier

from config import CFG, set_global_seeds
from data import load_tabular
from cv import build_folds


def run_shap(model_name: str = "tabnet_augmented", n_background: int = 100,
             n_explain: int = 300) -> None:
    set_global_seeds()
    X, y, feat_names = load_tabular()
    folds = build_folds(y)
    _, va = folds[-1]  # matches the fold whose weights we saved

    clf = TabNetClassifier()
    clf.load_model(str(CFG.artifacts_dir / "weights" / f"{model_name}_lastfold.zip"))

    # -- 1. Native TabNet attribution (fast) --
    attributions, _ = clf.explain(X[va])
    mean_abs = np.abs(attributions).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    plt.figure(figsize=(7, 5))
    plt.barh([feat_names[i] for i in order][::-1], mean_abs[order][::-1])
    plt.xlabel("Mean |attribution|")
    plt.title(f"{model_name} — native TabNet feature attributions")
    plt.tight_layout()
    out = CFG.artifacts_dir / "plots" / f"tabnet_native_attr_{model_name}.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"[shap] saved {out.name}")

    # -- 2. SHAP KernelExplainer beeswarm --
    rng = np.random.default_rng(CFG.seed)
    bg_idx = rng.choice(len(X), size=min(n_background, len(X)), replace=False)
    ex_idx = rng.choice(va, size=min(n_explain, len(va)), replace=False)

    def f(data):
        return clf.predict_proba(data.astype(np.float32))[:, 1]

    explainer = shap.KernelExplainer(f, X[bg_idx])
    shap_values = explainer.shap_values(X[ex_idx], nsamples=100)

    shap.summary_plot(shap_values, X[ex_idx], feature_names=feat_names,
                      show=False, max_display=15)
    fig = plt.gcf()
    fig.suptitle(f"SHAP beeswarm — {model_name}", y=1.02)
    out = CFG.artifacts_dir / "plots" / f"shap_beeswarm_{model_name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[shap] saved {out.name}")


if __name__ == "__main__":
    run_shap()
