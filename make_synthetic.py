"""Create SYNTHETIC test data (not real prices) to test the pipeline offline.
   python make_synthetic.py --mode random|trending --out data/daily
'random'   : no exploitable edge (strategies should NOT show a convincing profit)
'trending' : persistent stock-specific drift (momentum/breakouts should work)"""
import argparse, os, zipfile
import numpy as np
import pandas as pd
from store import save_df
from build_db import adjust_prices


def make(mode="random", seed=7, n=150, start="2016-01-01", end="2026-10-16"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    mkt = np.zeros(T)
    state = 1
    for t in range(T):
        if rng.random() < (0.004 if state == 1 else 0.01):
            state = 1 - state
        mu, sg = (0.0006, 0.008) if state == 1 else (-0.0008, 0.015)
        mkt[t] = mu + sg * rng.standard_normal()
    beta = rng.uniform(0.6, 1.4, n)
    idio = rng.uniform(0.012, 0.028, n)
    rets = beta[None, :] * mkt[:, None] + idio[None, :] * rng.standard_normal((T, n))
    if mode == "trending":
        drift = np.zeros((T, n))
        for t in range(1, T):
            drift[t] = 0.99 * drift[t - 1] + 0.0002 * rng.standard_normal(n)
        rets = rets + drift
    rets = np.clip(rets, -0.2, 0.2)
    close = rng.uniform(50, 3000, n)[None, :] * np.cumprod(1 + rets, axis=0)
    prev = np.vstack([close[:1], close[:-1]])
    op = prev * (1 + 0.003 * rng.standard_normal((T, n)))
    hi = np.maximum(op, close) * (1 + np.abs(0.006 * rng.standard_normal((T, n))))
    lo = np.minimum(op, close) * (1 - np.abs(0.006 * rng.standard_normal((T, n))))
    base = np.exp(rng.normal(13, 1.0, n))
    vol = base[None, :] * np.exp(0.4 * rng.standard_normal((T, n))) * (1 + 2.0 * (np.abs(rets) > 2.5 * idio[None, :]))
    # staggered listings
    first = np.where(rng.random(n) < 0.4, rng.integers(0, 1500, n), 0)
    first[0], first[1] = 0, 0
    last = np.full(n, T)
    last[1] = int(np.searchsorted(dates, pd.Timestamp("2023-03-01")))   # DELISTCO
    ts = int(np.searchsorted(dates, pd.Timestamp("2021-06-01")))        # SPLITCO 2:1 split
    prevc = prev.copy()
    close_u, op_u, hi_u, lo_u, vol_u = close.copy(), op.copy(), hi.copy(), lo.copy(), vol.copy()
    for arr in (close_u, op_u, hi_u, lo_u):
        arr[ts:, 0] = arr[ts:, 0] / 2.0
    vol_u[ts:, 0] *= 2.0
    prevc[:, 0] = np.vstack([close_u[:1, 0], close_u[:-1, 0:1]])[:, 0]
    # NOTE: like the real PR files, prev_close is NOT adjusted on the ex-date; adjustment comes from the action list
    z = (rets - rets.mean(axis=0)) / rets.std(axis=0)
    deliv = np.clip(50 + 6 * z * (1 if mode == 'trending' else 0) + 12 * rng.standard_normal((T, n)), 5, 95)
    names = ["SPLITCO", "DELISTCO"] + [f"STK{k:03d}" for k in range(2, n)]
    frames = []
    for j in range(n):
        idx = np.arange(first[j], last[j])
        frames.append(pd.DataFrame({"date": dates[idx], "symbol": names[j], "open": op_u[idx, j],
                                    "high": hi_u[idx, j], "low": lo_u[idx, j], "close": close_u[idx, j],
                                    "prev_close": prevc[idx, j], "volume": vol_u[idx, j],
                                    "value": vol_u[idx, j] * close_u[idx, j]}))
    df = pd.concat(frames, ignore_index=True)
    dfs = []
    for j in range(n):
        idx = np.arange(first[j], last[j])
        dfs.append(pd.DataFrame({'date': dates[idx], 'symbol': names[j], 'deliv_qty': 0.0, 'deliv_per': deliv[idx, j]}))
    df.attrs['delivery'] = pd.concat(dfs, ignore_index=True)
    nifty = pd.DataFrame({'date': dates, 'name': 'Nifty 50', 'open': 10000 * np.cumprod(1 + mkt),
                          'high': 10000 * np.cumprod(1 + mkt), 'low': 10000 * np.cumprod(1 + mkt), 'close': 10000 * np.cumprod(1 + mkt)})
    events = pd.DataFrame({'symbol': ['SPLITCO'], 'ex_date': [dates[ts]], 'ratio': [0.5]})
    return df, nifty, events


def write_raw_samples(df, out_dir):
    """Write a few UDiFF-style raw files so the parser can be tested."""
    os.makedirs(out_dir, exist_ok=True)
    for d in sorted(df["date"].unique())[-5:]:
        x = df[df["date"] == d]
        raw = pd.DataFrame({"TradDt": x["date"].dt.strftime("%Y-%m-%d"), "BizDt": x["date"].dt.strftime("%Y-%m-%d"),
                            "Sgmt": "CM", "TckrSymb": x["symbol"], "SctySrs": "EQ", "OpnPric": x["open"],
                            "HghPric": x["high"], "LwPric": x["low"], "ClsPric": x["close"],
                            "PrvsClsgPric": x["prev_close"], "TtlTradgVol": x["volume"], "TtlTrfVal": x["value"]})
        name = f"cm_{pd.Timestamp(d).strftime('%Y%m%d')}"
        raw.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
        with zipfile.ZipFile(os.path.join(out_dir, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(out_dir, name + ".csv"), arcname=name + ".csv")
        os.remove(os.path.join(out_dir, name + ".csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/daily")
    ap.add_argument("--raw-samples", default=None)
    a = ap.parse_args()
    df, nifty, events = make(a.mode, a.seed)
    adj, applied = adjust_prices(df, events)
    print("adjustments applied:\n", applied.to_string(index=False))
    save_df(nifty, os.path.join(os.path.dirname(a.out) or '.', 'index_daily'))
    save_df(df.attrs['delivery'], os.path.join(os.path.dirname(a.out) or '.', 'delivery'))
    print("saved", save_df(adj, a.out))
    if a.raw_samples:
        write_raw_samples(df, a.raw_samples)
