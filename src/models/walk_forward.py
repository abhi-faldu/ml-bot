"""
src/models/walk_forward.py

Walk-forward validation for time series ML models.

Why this matters
----------------
Random cross-validation on time series causes look-ahead bias —
test folds contain data BEFORE some training samples. This inflates
apparent performance. Walk-forward validation strictly ensures the
model only ever trains on the past and tests on the future.

Method
------
1. Start with an initial training window (e.g. first 60% of data)
2. Train model on that window
3. Evaluate on the next N candles (test fold)
4. Slide the window forward by N candles
5. Repeat until the end of the dataset
6. Aggregate all out-of-sample predictions for final metrics
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pipeline import load_raw, build_features


def walk_forward_validate(
    csv_name: str,
    lookback: int = 10,
    n_splits: int = 5,
    train_ratio: float = 0.6,
    n_estimators: int = 300,
    max_depth: int = 8,
) -> Dict:
    """
    Run walk-forward validation and return aggregated metrics.

    Parameters
    ----------
    csv_name    : raw data CSV filename
    lookback    : number of lag features
    n_splits    : number of test folds
    train_ratio : fraction used as minimum training window
    n_estimators, max_depth : RF hyperparameters

    Returns
    -------
    dict with per-fold and aggregate metrics
    """
    df_raw  = load_raw(csv_name)
    df      = build_features(df_raw, lookback)

    feature_cols = [c for c in df.columns if c.startswith("ret_lag_")]
    X = df[feature_cols].values
    y = df["target"].values
    dates = df.index

    n = len(df)
    initial_train = int(n * train_ratio)
    fold_size = (n - initial_train) // n_splits

    if fold_size < 20:
        raise ValueError(
            f"Fold size too small ({fold_size}). "
            "Download more data or reduce n_splits."
        )

    results: List[Dict] = []
    all_preds  = []
    all_actual = []

    print(f"\nWalk-Forward Validation — {n_splits} folds")
    print(f"Total samples  : {n}")
    print(f"Initial train  : {initial_train} ({train_ratio*100:.0f}%)")
    print(f"Fold size      : {fold_size} candles each")
    print("-" * 55)

    for fold in range(n_splits):
        train_end = initial_train + fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test,  y_test  = X[test_start:test_end], y[test_start:test_end]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        fold_result = {
            "fold":       fold + 1,
            "train_rows": train_end,
            "test_rows":  test_end - test_start,
            "train_from": str(dates[0].date()),
            "train_to":   str(dates[train_end - 1].date()),
            "test_from":  str(dates[test_start].date()),
            "test_to":    str(dates[test_end - 1].date()),
            "accuracy":   round(acc, 4),
        }
        results.append(fold_result)
        all_preds.extend(preds)
        all_actual.extend(y_test)

        print(
            f"Fold {fold+1}/{n_splits}  "
            f"train→{fold_result['train_to']}  "
            f"test {fold_result['test_from']}→{fold_result['test_to']}  "
            f"acc={acc:.3f}"
        )

    # aggregate out-of-sample metrics
    overall_acc = accuracy_score(all_actual, all_preds)
    accs = [r["accuracy"] for r in results]

    summary = {
        "folds":            results,
        "oos_accuracy":     round(overall_acc, 4),
        "mean_fold_acc":    round(float(np.mean(accs)), 4),
        "std_fold_acc":     round(float(np.std(accs)), 4),
        "min_fold_acc":     round(float(np.min(accs)), 4),
        "max_fold_acc":     round(float(np.max(accs)), 4),
    }

    print("\n" + "=" * 55)
    print(f"Out-of-sample accuracy : {overall_acc:.4f}")
    print(f"Mean fold accuracy     : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Range                  : [{np.min(accs):.4f}, {np.max(accs):.4f}]")
    print("\nFull OOS classification report:")
    print(classification_report(all_actual, all_preds))

    if overall_acc < 0.50:
        print("WARNING: OOS accuracy below 50%. Strategy expected to lose money after fees.")
    elif overall_acc < 0.52:
        print("NOTE: OOS accuracy marginally above 50%. Borderline after fees.")
    else:
        print("OOS accuracy above 52%. Promising — verify with backtest.")

    return summary


if __name__ == "__main__":
    walk_forward_validate("binance_BTCUSDT_1h.csv")
