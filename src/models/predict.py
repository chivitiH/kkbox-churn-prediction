import pandas as pd
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed")
MODEL_PATH = Path("model.pkl")


def load_model():
    print("📦 Loading model...")
    return joblib.load(MODEL_PATH)


def load_data():
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH / "dataset.csv")
    return df


def prepare_data(df):
    X = df.drop(columns=["msno", "is_churn"], errors="ignore")
    X = pd.get_dummies(X)
    X = X.fillna(0)
    return X


def main():
    df = load_data()
    model = load_model()

    X = prepare_data(df)

    print("🔮 Predicting...")

    preds = model.predict_proba(X)[:, 1]

    output = pd.DataFrame({
        "msno": df["msno"],
        "churn_probability": preds
    })

    output.to_csv("predictions.csv", index=False)

    print("✅ Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()
