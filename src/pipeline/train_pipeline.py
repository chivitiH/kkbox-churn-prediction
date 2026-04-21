import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from src.models.evaluate import evaluate

DATA_PATH = Path("data/processed")


def load_data():
    print("📥 Loading dataset...")
    return pd.read_csv(DATA_PATH / "dataset.csv")


def prepare_data(df):
    y = df["is_churn"]

    X = df.drop(columns=[
        "msno",
        "is_churn",
        "is_cancel",
        "cancel_rate",
        "last_activity_date",
        "last_transaction_date",
        "days_since_last_activity",
        "days_since_last_transaction"
    ], errors="ignore")

    X = pd.get_dummies(X)
    X = X.fillna(0)

    return X, y


def main():
    df = load_data()
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Training model...")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced"  # 🔥 améliore churn
    )

    model.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred)

    print("💾 Saving artifacts...")

    Path("artifacts").mkdir(exist_ok=True)

    joblib.dump(model, "artifacts/model.pkl")
    X_train.columns.to_series().to_json("artifacts/features.json")
    pd.Series(metrics).to_json("artifacts/metrics.json")

    # 🔥 FEATURE IMPORTANCE
    print("📊 Saving feature importance...")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    feat_imp.sort_values(ascending=False).to_csv("artifacts/feature_importance.csv")

    print("✅ Pipeline complete")


if __name__ == "__main__":
    main()
