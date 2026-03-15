"""
src/models/train_model.py

Trains a LightGBM classifier on the full feature set.

Why LightGBM over Random Forest
---------------------------------
- Handles tabular financial data better at medium dataset sizes
- Built-in L1/L2 regularisation reduces overfitting
- Faster to train, easier to tune
- predict_proba() gives well-calibrated confidence scores
  which the live loop can use to filter low-confidence signals
"""
from pathlib import Path
import joblib
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import MODEL_NAME
from src.data.pipeline import build_and_split
from src.data.make_features import get_feature_cols

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train(csv_name: str = "binance_BTCUSDT_1h.csv") -> RandomForestClassifier:
    train_df, test_df = build_and_split(csv_name)

    feature_cols = get_feature_cols(train_df)
    print(f"Training LightGBM with {len(feature_cols)} features: {feature_cols}")

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test,  y_test  = test_df[feature_cols],  test_df["target"]

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
    print(f"\nTest accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    out = MODEL_DIR / MODEL_NAME
    joblib.dump(model, out)
    print(f"model saved -> {out}")
    return model


if __name__ == "__main__":
    train()
