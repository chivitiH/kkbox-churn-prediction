import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

DATA_PATH = Path("data/processed")


def load_data():
    df = pd.read_csv(DATA_PATH / "dataset.csv")
    return df


def prepare_data(df):
    y = df["is_churn"]

    # 🔥 ON GARDE QUE LES FEATURES "SAFE"
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

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05
    )

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    print(f"🔥 AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
