"""Shared panel loading, features and signals.
Used by BOTH the backtester and the daily report so live signals match what was tested."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from config import Config
import os
from store import load_df


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    value: pd.DataFrame


@dataclass
class Features:
    in_universe: pd.DataFrame = None
    regime_on: pd.Series = None
    market: pd.Series = None
    score: pd.DataFrame = None
    atr: pd.DataFrame = None
    core_eligible: pd.DataFrame = None
    swing_signal: pd.DataFrame = None
    regime_source_used: str = ""
    breakout_signal: pd.DataFrame = None
    pullback_signal: pd.DataFrame = None


def load_panel(path: str, exclude_funds: bool = True) -> Panel:
    df = load_df(path)
    sp = os.path.join(os.path.dirname(path) or ".", "securities.csv")
    if exclude_funds and os.path.exists(sp):                # ETFs/funds trade in the same series as stocks: keep them out
        sec = pd.read_csv(sp)
        funds = set(sec.loc[sec["is_fund"].astype(str).str.lower() == "true", "symbol"])
        df = df[~df["symbol"].isin(funds)]
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "symbol"], keep="last")
    cols = {}
    for c in ["open", "high", "low", "close", "volume", "value"]:
        cols[c] = df.pivot(index="date", columns="symbol", values=c).sort_index()
    return Panel(**cols)


def load_index_close(data_path: str, name: str = "Nifty 50"):
    """Close series of an index from index_daily.* beside the daily file (built by build_db.py), or None."""
    base = os.path.join(os.path.dirname(data_path) or ".", "index_daily")
    try:
        ix = load_df(base)
    except FileNotFoundError:
        return None
    ix = ix[ix["name"] == name]
    if ix.empty:
        return None
    return ix.drop_duplicates("date").set_index(pd.to_datetime(ix.drop_duplicates("date")["date"]))["close"].sort_index()


def load_delivery(data_path: str):
    """Delivery % panel (date x symbol) from delivery.* beside the daily file, or None."""
    base = os.path.join(os.path.dirname(data_path) or ".", "delivery")
    try:
        d = load_df(base)
    except FileNotFoundError:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.pivot(index="date", columns="symbol", values="deliv_per").sort_index()


def wilder_atr(h, l, c, n):
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def wilder_rsi(c, n=14):
    d = c.diff()
    au = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = (-d).clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)


def compute_features(P: Panel, cfg: Config, index_close=None, delivery=None) -> Features:
    c, h, l, v, val = P.close, P.high, P.low, P.volume, P.value

    # point-in-time liquidity universe (no look-ahead; delisted names remain in history)
    liq = val.rolling(cfg.liq_window, min_periods=int(cfg.liq_window * 0.8)).mean()
    liq = liq.where(c.notna().cumsum() >= cfg.min_history)      # rank only stocks with enough history
    rank = liq.rank(axis=1, ascending=False, method="first")
    in_u = (rank <= cfg.universe_size) & (c >= cfg.min_price)

    # equal-weight market proxy of yesterday's universe -> regime filter
    ret = c.pct_change(fill_method=None).clip(-0.25, 0.25)
    mret = ret.where(in_u.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    market = (1.0 + mret).cumprod() * 100.0
    regime = market > market.rolling(cfg.regime_sma).mean()
    use_nifty = cfg.regime_source in ("auto", "nifty") and index_close is not None
    used = "proxy"
    if use_nifty:
        ic = index_close.reindex(c.index).ffill()
        if ic.notna().mean() >= 0.9:                      # need near-full coverage, else fall back
            market = ic / ic.dropna().iloc[0] * 100.0
            regime = market > market.rolling(cfg.regime_sma).mean()
            used = "nifty"
        elif cfg.regime_source == "nifty":
            raise ValueError("Nifty index data covers less than 90% of the price dates")
    reg_df = pd.DataFrame(np.repeat(regime.values[:, None], c.shape[1], axis=1),
                          index=c.index, columns=c.columns)

    atr = wilder_atr(h, l, c, cfg.atr_n)
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    sma_core = c.rolling(cfg.core_sma).mean()
    score = c.shift(cfg.core_roc_skip) / c.shift(cfg.core_roc_skip + cfg.core_roc_long) - 1.0
    hh_prev = h.rolling(cfg.swing_breakout).max().shift(1)
    vol_ratio = v / v.rolling(20).mean().shift(1)

    breakout = (in_u & reg_df & (c > hh_prev) & (sma50 > sma200) & (c > sma50)
                & (vol_ratio > cfg.swing_vol_mult) & atr.notna())
    breakout_raw = breakout.copy()       # pullbacks exclude any raw breakout, whether or not delivery passes

    if cfg.delivery_min > 0:
        if delivery is None:
            raise ValueError("delivery_min > 0 but no delivery data found (build_db.py needs sec_bhavdata_full_*.csv files)")
        deliv_ok = (delivery.reindex(index=c.index, columns=c.columns) >= cfg.delivery_min)
        breakout = breakout & deliv_ok          # missing delivery data fails the filter (conservative)

    # pullback in an established uptrend: dip to the 20-EMA zone in last 5 days, then a reversal bar
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    rsi = wilder_rsi(c)
    trend = (sma50 > sma200) & (c > sma200)
    dip = l.rolling(5).min() <= ema20 + 0.25 * atr
    trigger = (c > ema20) & (c > h.shift(1)) & (rsi >= 45) & (rsi <= 65)
    pullback = (in_u & reg_df & trend & dip & trigger & (score > 0) & atr.notna() & ~breakout_raw)
    if cfg.delivery_min > 0:
        pullback = pullback & deliv_ok

    swing = pd.DataFrame(False, index=c.index, columns=c.columns)
    if cfg.use_breakout:
        swing = swing | breakout
    if cfg.use_pullback:
        swing = swing | pullback
    core_el = in_u & reg_df & (c > sma_core) & score.notna() & (score > 0)

    return Features(in_universe=in_u, regime_on=regime, market=market, score=score,
                    atr=atr, core_eligible=core_el, swing_signal=swing,
                    breakout_signal=breakout, pullback_signal=pullback, regime_source_used=used)
