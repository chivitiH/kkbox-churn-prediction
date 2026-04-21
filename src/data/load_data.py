import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")


def load_sample():
    print("📥 Loading small sample for debug...")

    train = pd.read_csv(DATA_PATH / "train.csv", nrows=100_000)
    members = pd.read_csv(DATA_PATH / "members_v3.csv", nrows=100_000)
    transactions = pd.read_csv(DATA_PATH / "transactions.csv", nrows=100_000)
    logs = pd.read_csv(DATA_PATH / "user_logs.csv", nrows=100_000)

    return train, members, transactions, logs


def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def main():
    train, members, transactions, logs = load_sample()

    print("⚡ Optimizing memory...")

    train = reduce_memory(train)
    members = reduce_memory(members)
    transactions = reduce_memory(transactions)
    logs = reduce_memory(logs)

    print("\n📊 Shapes:")
    print("Train:", train.shape)
    print("Members:", members.shape)
    print("Transactions:", transactions.shape)
    print("Logs:", logs.shape)


if __name__ == "__main__":
    main()