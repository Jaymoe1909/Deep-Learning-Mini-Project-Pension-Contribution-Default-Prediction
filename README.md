# M3 — Model training & evaluation

Phase 3 of the pension-default project. Trains four models on identical
5-fold splits and produces a comparison table + plots.

## Layout

```
config.py            central paths, seeds, hyperparameters
data.py              loads data/features.csv + data/sequences.npy
                     (labels: DEFAULT_LABEL column, sequences row-aligned)
cv.py                shared stratified 5-fold indices, cached to disk
aug_wrapper.py       thin wrapper around M2's augmentation.py
evaluate.py          metrics, ROC, confusion, results.json, comparison table
models/lstm.py       nn.LSTM classifier for 48-month sequences
train_tabnet.py      TabNet CV (augmented + real-only for M2's ablation)
train_lstm.py        LSTM CV with SMOTE per fold, AMP, early stopping
train_baselines.py   LogReg + RandomForest
explain.py           SHAP beeswarm + native TabNet attributions
run_all.py           orchestrator
```

## Run

```bash
# Windows / conda env `deeplearning`
pip install -r requirements.txt
python run_all.py
```

Outputs land in `artifacts/`:
- `results.json` — every model's per-fold + summary metrics
- `plots/roc_*.png`, `plots/cm_*.png` — per-model ROC and confusion
- `plots/shap_beeswarm_tabnet_augmented.png` — SHAP beeswarm (M1 interprets)
- `weights/*_lastfold.*` — last-fold weights for SHAP

## Notes

- Every model uses the same folds via `cv.build_folds()` — same seed, same
  indices cached to `artifacts/fold_indices.npz`. Delete that file to
  regenerate.
- `augmentation.py` first tries to import M2's module (`augmentation_m2`);
  falls back to an inline SMOTE + Gaussian noise so M3 isn't blocked.
- LSTM uses `nn.LSTM`. If the assignment enforces from-scratch, swap
  `models/lstm.py` — the training loop stays the same.
- Windows/Jupyter: `NUM_WORKERS=0` in `config.py`. Do not raise it.
- Mixed precision is on by default for LSTM; disable via `CFG.use_amp = False`
  if you see NaN losses.
```
