"""Run the full M3 pipeline: baselines → LSTM → TabNet → SHAP → comparison.

Usage:
    python run_all.py                # everything
    python run_all.py --skip-shap    # skip SHAP (slowest step)
    python run_all.py --only tabnet  # single model
"""
from __future__ import annotations
import argparse
from evaluate import build_comparison_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["logreg", "rf", "lstm", "tabnet", "shap"],
                        default=None, help="Run one step only")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    steps = ["logreg", "rf", "lstm", "tabnet"]
    if not args.skip_shap:
        steps.append("shap")
    if args.only:
        steps = [args.only]

    if "logreg" in steps:
        from train_baselines import train_logreg; train_logreg()
    if "rf" in steps:
        from train_baselines import train_rf; train_rf()
    if "lstm" in steps:
        from train_lstm import train_lstm; train_lstm()
    if "tabnet" in steps:
        from train_tabnet import train_tabnet
        train_tabnet("tabnet_augmented", augmented=True)
        train_tabnet("tabnet_realonly",  augmented=False)
    if "shap" in steps:
        from explain import run_shap; run_shap("tabnet_augmented")

    print("\n=== Final comparison ===")
    build_comparison_table()


if __name__ == "__main__":
    main()
