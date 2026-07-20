
"""augmentation.py
Gaussian noise regularisation pipeline

Usage in 03_modelling.ipynb (inside each CV fold):
    from augmentation import augment_training_fold, CONTINUOUS_COLS
    X_aug, y_aug = augment_training_fold(X_train_scaled, y_train, CONTINUOUS_COLS)

Rules:
    - Call ONLY on training folds
    - NEVER augment validation or test folds
    - Fit StandardScaler on training fold only before calling this
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# Continuous features to apply noise to
CONTINUOUS_COLS = [
    'TOTAL_ARREARS', 'MAX_CONSECUTIVE_MISSED', 'AVG_CONTRIBUTION_RATIO',
    'PAYMENT_RATE', 'AMOUNT_CV', 'AMOUNT_TREND',
    'EMPLOYER_TENURE', 'AVG_PAYMENT_DELAY'
]


def add_gaussian_noise(X, continuous_cols, sigma=0.05, rng=None):
    """Add Gaussian noise to continuous features only."""
    rng = rng or np.random.default_rng()
    X_noisy = X.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=(len(X), len(continuous_cols)))
    X_noisy[continuous_cols] = X_noisy[continuous_cols].values + noise
    return X_noisy


def augment_training_fold(X_train, y_train,
                          continuous_cols=None,
                          sigma=0.05,
                          random_state=42):
    """SMOTE + Gaussian noise augmentation for one training fold."""
    continuous_cols = continuous_cols or CONTINUOUS_COLS
    rng = np.random.default_rng(random_state)

    # Step 1: SMOTE-- balance classes
    sm = SMOTE(random_state=random_state)
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    X_bal = pd.DataFrame(X_bal, columns=X_train.columns)
    y_bal = pd.Series(y_bal, name=y_train.name)

    # Step 2: Gaussian noise -- regularise continuous features
    X_aug = add_gaussian_noise(X_bal, continuous_cols, sigma=sigma, rng=rng)
    y_aug = y_bal.reset_index(drop=True)

    return X_aug.reset_index(drop=True), y_aug
