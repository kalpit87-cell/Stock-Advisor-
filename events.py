"""Results-calendar support: download NSE board-meeting dates, and build a blackout mask.

  python events.py fetch --start 2026-10-01 --end 2026-12-15 --out data/events.csv
Or write data/events.csv by hand: columns  symbol,date  (one row per results date).

fetch() is NOT tested against live NSE (no internet where it was built); the API path or field names
may differ. It prints the keys it sees if it cannot find the expected ones.
"""
import argparse
from datetime import datetime
import numpy as np
import pandas as pd


def load_events(path: str) -> pd.DataFrame:
    try:
        ev = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["symbol", "date"])
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce", format="mixed")
    return ev.dropna(subset=["date"])[["symbol", "date"]].drop_duplicates()


def results_within(events: pd.DataFrame, symbol: str, day, days: int):
    """Earliest results date in [day, day+days] calendar days, else None."""
    if events is None or events.empty:
        return None
    day = pd.Timestamp(day)
    e = events[(events["symbol"] == symbol) & (events["date"] >= day) & (events["date"] <= day + pd.Timedelta(days=days))]
    return None if e.empty else e["date"].min()


def load_dividends(path: str) -> pd.DataFrame:
    try:
        d = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["symbol", "ex_date", "dividend"])
    d["ex_date"] = pd.to_datetime(d["ex_date"], errors="coerce")
    return d.dropna(subset=["ex_date"])[["symbol", "ex_date", "dividend"]]


def dividend_within(div: pd.DataFrame, symbol: str, day, days: int = 5):
    """(ex_date, amount) of the next dividend with ex-date in (day, day+days], else None."""
    if div is None or div.empty:
        return None
    day = pd.Timestamp(day)
    x = div[(div["symbol"] == symbol) & (div["ex_date"] > day) & (div["ex_date"] <= day + pd.Timedelta(days=days))]
    return None if x.empty else (x["ex_date"].min(), float(x["dividend"].sum()))


def blackout_mask(index: pd.DatetimeIndex, columns, events: pd.DataFrame, days: int) -> np.ndarray:
    """True where a NEW entry signal on that date should be ignored because results are due within `days`."""
    mask = np.zeros((len(index), len(columns)), dtype=bool)
    col = {c: k for k, c in enumerate(columns)}
    for sym, d in zip(events["symbol"], events["date"]):
        if sym not in col:
            continue
        lo = index.searchsorted(d - pd.Timedelta(days=days))
        hi = index.searchsorted(d, side="right")
        mask[lo:hi, col[sym]] = True
    return mask


def fetch(start: str, end: str, out: str):
    import requests
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
         "Accept": "application/json,*/*", "Referer": "https://www.nseindia.com/"}
    s = requests.Session(); s.headers.update(h)
    s.get("https://www.nseindia.com", timeout=30)
    f = datetime.strptime(start, "%Y-%m-%d").strftime("%d-%m-%Y")
    t = datetime.strptime(end, "%Y-%m-%d").strftime("%d-%m-%Y")
    r = s.get(f"https://www.nseindia.com/api/event-calendar?index=equities&from_date={f}&to_date={t}", timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise SystemExit("empty response")
    keys = {k.lower(): k for k in rows[0]}
    if "symbol" not in keys or "date" not in keys:
        raise SystemExit(f"unexpected fields: {list(rows[0])} - edit events.py")
    pk = keys.get("purpose") or keys.get("bm_desc")
    out_rows = []
    for x in rows:
        text = " ".join(str(x.get(k, "")) for k in (keys.get("purpose"), keys.get("bm_desc")) if k).lower()
        if "result" in text:
            out_rows.append({"symbol": x[keys["symbol"]], "date": x[keys["date"]]})
    pd.DataFrame(out_rows).drop_duplicates().to_csv(out, index=False)
    print(f"saved {len(out_rows)} results dates to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch"])
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="data/events.csv")
    a = ap.parse_args()
    fetch(a.start, a.end, a.out)
