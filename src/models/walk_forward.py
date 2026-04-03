"""
src/models/walk_forward.py

Walk-forward validation using Random Forest.

Why walk-forward matters
------------------------
Random cross-validation on time series leaks future data into training.
Walk-forward strictly ensures the model only ever trains on the past
and is tested on data it has never seen.

Method
------
1. Start with initial training window (first 60% of data)
2. Train RandomForest on that window
3. Evaluate on next N candles
4. Slide window forward by N candles
5. Repeat until end of dataset
6. Aggregate all out-of-sample predictions for final metrics
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pipeline import load_raw, build_features
from src.data.make_features import get_feature_cols


def walk_forward_validate(
    csv_name: str,
    lookback: int = 10,
    n_splits: int = 5,
    train_ratio: float = 0.6,
) -> Dict:
    """
    Run walk-forward validation and return aggregated metrics.

    Parameters
    ----------
    csv_name    : raw data CSV filename
    lookback    : number of lag features
    n_splits    : number of test folds
    train_ratio : fraction used as minimum training window
    """
    df_raw = load_raw(csv_name)
    df     = build_features(df_raw, lookback)

    feature_cols = get_feature_cols(df)
    X     = df[feature_cols].values
    y     = df["target"].values
    dates = df.index

    n             = len(df)
    initial_train = int(n * train_ratio)
    fold_size     = (n - initial_train) // n_splits

    if fold_size < 50:
        raise ValueError(
            f"Fold size too small ({fold_size} rows). "
            "Download more data with: python src/data/download_crypto.py"
        )

    results: List[Dict] = []
    all_preds  = []
    all_actual = []

    print(f"\nWalk-Forward Validation — {n_splits} folds — RandomForest")
    print(f"Total samples  : {n}")
    print(f"Features       : {len(feature_cols)}")
    print(f"Initial train  : {initial_train} ({train_ratio*100:.0f}%)")
    print(f"Fold size      : {fold_size} candles each")
    print("-" * 60)

    for fold in range(n_splits):
        train_end  = initial_train + fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        X_train, y_train = X[:train_end], y[:train_end]
        X_test,  y_test  = X[test_start:test_end], y[test_start:test_end]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)

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
            f"train -> {fold_result['train_to']}  "
            f"test {fold_result['test_from']} -> {fold_result['test_to']}  "
            f"acc={acc:.4f}"
        )

    overall_acc = accuracy_score(all_actual, all_preds)
    accs = [r["accuracy"] for r in results]

    summary = {
        "folds":         results,
        "oos_accuracy":  round(overall_acc, 4),
        "mean_fold_acc": round(float(np.mean(accs)), 4),
        "std_fold_acc":  round(float(np.std(accs)), 4),
        "min_fold_acc":  round(float(np.min(accs)), 4),
        "max_fold_acc":  round(float(np.max(accs)), 4),
    }

    print("\n" + "=" * 60)
    print(f"Out-of-sample accuracy : {overall_acc:.4f}")
    print(f"Mean fold accuracy     : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Range                  : [{np.min(accs):.4f}, {np.max(accs):.4f}]")
    print("\nFull OOS classification report:")
    print(classification_report(all_actual, all_preds))

    if overall_acc < 0.50:
        print("WARNING: OOS accuracy below 50%. Strategy expected to lose money after fees.")
        print("ACTION : Re-run download_crypto.py to fetch 2 years of data, then retry.")
    elif overall_acc < 0.52:
        print("NOTE: OOS accuracy marginally above 50%. Borderline after fees.")
    else:
        print("OOS accuracy above 52%. Promising — verify with backtest.")

    return summary


if __name__ == "__main__":
    walk_forward_validate("binance_BTCUSDT_1h.csv")
