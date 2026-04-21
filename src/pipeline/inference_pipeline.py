import pandas as pd
import joblib

def main():
    print("📥 Loading data...")
    df = pd.read_csv("data/processed/dataset.csv")

    print("📦 Loading model...")
    model = joblib.load("artifacts/model.pkl")

    print("📦 Loading feature schema...")
    train_columns = pd.read_json("artifacts/features.json", typ="series").tolist()

    print("⚙️ Preparing features...")
    X = df.drop(columns=["msno", "is_churn"], errors="ignore")
    X = pd.get_dummies(X)
    X = X.fillna(0)

    # 🔥 alignement train / inference
    X = X.reindex(columns=train_columns, fill_value=0)

    print("🔮 Predicting...")
    preds = model.predict_proba(X)[:, 1]

    df_out = pd.DataFrame({
        "msno": df["msno"],
        "prediction": preds
    })

    df_out.to_csv("artifacts/predictions.csv", index=False)

    print("✅ Inference done")


if __name__ == "__main__":
    main()
