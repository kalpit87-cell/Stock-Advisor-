"""Tiny storage helper: Parquet when pyarrow is installed, otherwise gzip CSV.
Pass the path WITHOUT extension, e.g. 'data/daily'."""
import os
import pandas as pd


def save_df(df: pd.DataFrame, base: str) -> str:
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        path = base + ".parquet"
        df.to_parquet(path, index=False)
    except ImportError:
        path = base + ".csv.gz"
        df.to_csv(path, index=False, compression="gzip")
    return path


def load_df(base: str) -> pd.DataFrame:
    base = base.replace(".parquet", "").replace(".csv.gz", "")
    if os.path.exists(base + ".parquet"):
        return pd.read_parquet(base + ".parquet")
    if os.path.exists(base + ".csv.gz"):
        return pd.read_csv(base + ".csv.gz", parse_dates=["date"])
    raise FileNotFoundError(f"No data found at {base}.parquet or {base}.csv.gz - run build_db.py first")
