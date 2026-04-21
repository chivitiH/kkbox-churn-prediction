import polars as pl
from pathlib import Path

DATA_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/processed")

CUTOFF_DATE = pl.date(2017, 2, 28)


def load_data(sample_size=20000):
    print("📥 Loading data (lazy mode)...")

    train = pl.read_csv(DATA_PATH / "train.csv")

    sampled_users = (
        train.sample(n=sample_size, seed=42)
        .select("msno")
        .to_series()
        .to_list()
    )

    train = train.filter(pl.col("msno").is_in(sampled_users))

    print(f"👤 Users sampled: {len(sampled_users)}")

    members = pl.read_csv(DATA_PATH / "members_v3.csv")
    members = members.filter(pl.col("msno").is_in(sampled_users))

    transactions = (
        pl.scan_csv(DATA_PATH / "transactions.csv")
        .filter(pl.col("msno").is_in(sampled_users))
    )

    logs = (
        pl.scan_csv(DATA_PATH / "user_logs.csv")
        .filter(pl.col("msno").is_in(sampled_users))
    )

    return train, members, transactions, logs


def aggregate_logs(logs):
    print("🎧 Aggregating logs...")

    logs = logs.with_columns(
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d")
    )

    logs = logs.filter(pl.col("date") <= CUTOFF_DATE)

    agg = (
        logs
        .group_by("msno")
        .agg([
            pl.col("date").max().alias("last_activity_date"),
            pl.col("total_secs").sum().alias("total_secs"),
            pl.len().alias("total_sessions"),
        ])
    )

    agg = agg.with_columns([
        pl.min_horizontal(
            (CUTOFF_DATE - pl.col("last_activity_date")).dt.total_days(),
            pl.lit(30)
        ).alias("days_since_last_activity"),

        (pl.col("total_secs") / (pl.col("total_sessions") + 1)).alias("avg_daily_secs"),
    ])

    return agg.collect()


def aggregate_transactions(transactions):
    print("💳 Aggregating transactions...")

    transactions = transactions.with_columns(
        pl.col("transaction_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d")
    )

    transactions = transactions.filter(
        pl.col("transaction_date") <= CUTOFF_DATE
    )

    agg = (
        transactions
        .group_by("msno")
        .agg([
            pl.col("transaction_date").max().alias("last_transaction_date"),
            pl.len().alias("num_transactions"),
            pl.col("actual_amount_paid").sum().alias("total_paid"),
        ])
    )

    agg = agg.with_columns([
        pl.min_horizontal(
            (CUTOFF_DATE - pl.col("last_transaction_date")).dt.total_days(),
            pl.lit(30)
        ).alias("days_since_last_transaction"),
    ])

    return agg.collect()


def process_members(members):
    print("👤 Processing members...")

    members = members.with_columns(
        pl.col("registration_init_time").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d")
    )

    members = members.with_columns(
        (CUTOFF_DATE - pl.col("registration_init_time")).dt.total_days().alias("account_age_days")
    )

    return members


def main():
    train, members, transactions, logs = load_data(sample_size=20000)

    logs_agg = aggregate_logs(logs)
    trans_agg = aggregate_transactions(transactions)
    members = process_members(members)

    print("🔗 Merging datasets...")

    df = (
        train
        .join(members, on="msno", how="left")
        .join(logs_agg, on="msno", how="left")
        .join(trans_agg, on="msno", how="left")
    )

    print("💾 Saving dataset...")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUTPUT_PATH / "dataset.csv")

    print("✅ Dataset ready:", df.shape)


if __name__ == "__main__":
    main()
