"""
src/models/train_model.py

Trains a Random Forest classifier on the full feature set.
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
    print(f"Training RandomForest with {len(feature_cols)} features: {feature_cols}")

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

    # save with the dashboard alias so dashboard/utils/load_model.py can find it
    dashboard_out = MODEL_DIR / "lgbm_model.pkl"
    joblib.dump(model, dashboard_out)
    print(f"model saved -> {dashboard_out}")

    return model


if __name__ == "__main__":
    train()
