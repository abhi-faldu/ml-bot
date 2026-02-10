from pathlib import Path
import joblib
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import MODEL_NAME
from src.data.pipeline import build_and_split   # uses leakage-safe pipeline

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train(csv_name: str = "binance_BTCUSDT_1h.csv") -> RandomForestClassifier:
    # pipeline handles temporal split — no shuffle, no leakage
    train_df, test_df = build_and_split(csv_name)

    feature_cols = [c for c in train_df.columns if c.startswith("ret_lag_")]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test,  y_test  = test_df[feature_cols],  test_df["target"]

    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    out = MODEL_DIR / MODEL_NAME
    joblib.dump(model, out)
    print(f"model saved → {out}")
    return model


if __name__ == "__main__":
    train()

