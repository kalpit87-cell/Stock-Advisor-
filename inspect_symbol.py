"""Look at one symbol around a date: adjusted price, adjustment factor, raw (unadjusted) price, and the corporate actions on file.

  python inspect_symbol.py CANBK --date 2024-04-08          # rows around the date + actions + applied adjustments
  python inspect_symbol.py --dates                          # which dates the suspect jumps cluster on
Reading it: 'raw_close' = close / factor is the price NSE actually printed. If raw_close moves smoothly but the adjusted close
jumps, an adjustment is on the wrong date or has the wrong ratio. If raw_close itself jumps, the data (or an unlisted event) is the cause.
"""
import argparse, os
import pandas as pd
from store import load_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?")
    ap.add_argument("--date")
    ap.add_argument("--dates", action="store_true", help="show the dates on which suspect jumps cluster")
    ap.add_argument("--window", type=int, default=5, help="trading days either side")
    ap.add_argument("--data", default="data/daily")
    a = ap.parse_args()
    base = os.path.dirname(a.data) or "."
    if a.dates:
        j = pd.read_csv(os.path.join(base, "suspect_jumps.csv"))
        print(j["date"].value_counts().head(15).to_string())
        return
    if not a.symbol:
        raise SystemExit("give a SYMBOL, or use --dates")
    d = load_df(a.data)
    d = d[d["symbol"] == a.symbol.upper()].sort_values("date").reset_index(drop=True)
    if d.empty:
        raise SystemExit(f"{a.symbol} not found in {a.data}")
    d["raw_close"] = d["close"] / d["factor"]
    d["adj_ret_%"] = (d["close"].pct_change() * 100).round(1)
    d["raw_ret_%"] = (d["raw_close"].pct_change() * 100).round(1)
    if a.date:
        i = int(d["date"].searchsorted(pd.Timestamp(a.date)))
        d = d.iloc[max(0, i - a.window): i + a.window + 1]
    print(f"\n{a.symbol.upper()}: prices")
    print(d[["date", "close", "factor", "raw_close", "adj_ret_%", "raw_ret_%"]].assign(date=d["date"].dt.date).round(4).to_string(index=False))
    for name, col in (("corporate_actions_all.csv", "ex_date"), ("applied_adjustments.csv", "ex_date")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            x = pd.read_csv(p)
            x = x[x["symbol"] == a.symbol.upper()]
            if a.date and len(x):
                x = x[(pd.to_datetime(x[col], errors="coerce") - pd.Timestamp(a.date)).abs() <= pd.Timedelta(days=200)]
            print(f"\n{name} (within 200 days of the date):")
            print(x.to_string(index=False) if len(x) else "  none")


if __name__ == "__main__":
    main()
