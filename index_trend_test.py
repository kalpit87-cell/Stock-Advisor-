"""One fixed rule, tested once: hold the Nifty 50 while its close is above its 200-day average, otherwise sit in a liquid fund.

  python index_trend_test.py --data data\\daily

PRE-REGISTERED: the rule (200-day average, daily check, switch at the next open) is fixed before looking at results. Do not try other
averages and keep the best one: that would be tuning, and on this much data it finds fake winners. --sma exists only to run the
same rule for a sensitivity check (e.g. 150 and 250) AFTER the main result, to see whether the answer is stable.

Modelled: switches at the next day's open (signal known at the previous close), costs on every switch (taxes + 0.10% slippage per side),
the liquid-fund yield on cash days, and an assumed dividend yield on days invested and on buy-and-hold (the index file is price only).
Not modelled: capital-gains tax, tracking error of an actual ETF/index fund, and the delay between signal and your order.
"""
import argparse
import numpy as np
import pandas as pd
import os
from store import load_df
from config import Config
from metrics import equity_stats, yearly_returns


def load_index_ohlc(data_path: str, name: str = "Nifty 50") -> pd.DataFrame:
    ix = load_df(os.path.join(os.path.dirname(data_path) or ".", "index_daily"))
    ix = ix[ix["name"] == name].drop_duplicates("date").sort_values("date")
    ix = ix[(ix["close"] > 0) & (ix["open"] > 0)]
    return ix.set_index(pd.to_datetime(ix["date"]))[["open", "close"]]


def run_trend(ix: pd.DataFrame, sma_days: int = 200, cash_yield: float = 0.06, div_yield: float = 0.012,
              capital: float = 150_000.0, costs=None):
    cfg = Config()
    c = costs or cfg.costs
    buy_cost = c.buy_fee_rate() + c.slippage
    sell_cost = c.sell_fee_rate() + c.slippage
    o, cl = ix["open"], ix["close"]
    sig = (cl > cl.rolling(sma_days).mean()).astype(float)        # known at the close of day t
    pos_new = sig.shift(1).fillna(0.0)                            # acted on at the open of day t
    pos_old = pos_new.shift(1).fillna(0.0)                        # what was held through the previous close
    overnight = o / cl.shift(1) - 1
    intraday = cl / o - 1
    cash_d, div_d = (1 + cash_yield) ** (1 / 252) - 1, div_yield / 252
    r = (pos_old * (overnight + div_d / 2) + pos_new * (intraday + div_d / 2) + (1 - pos_new) * cash_d).fillna(0.0)
    delta = pos_new - pos_old
    r = r - np.where(delta > 0, delta * buy_cost, -delta * sell_cost) - (delta.abs() > 0) * c.dp_charge_sell / capital * (delta < 0)
    eq = capital * (1 + r).cumprod()
    bh = capital * (1 + (cl.pct_change().fillna(0.0) + div_d)).cumprod()
    cash = capital * (1 + cash_d) ** np.arange(len(cl))
    cash = pd.Series(cash, index=cl.index)
    start = cl.index[sma_days]                                    # the first day a signal exists
    return eq[start:], bh[start:], cash[start:], int((delta.abs() > 0).sum()), float(pos_new[start:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/daily")
    ap.add_argument("--sma", type=int, default=200)
    ap.add_argument("--cash-yield", type=float, default=0.06)
    ap.add_argument("--div-yield", type=float, default=0.012)
    ap.add_argument("--split", default=None, help="show two sub-periods either side of this date")
    a = ap.parse_args()
    ix = load_index_ohlc(a.data)
    eq, bh, cash, switches, invested = run_trend(ix, a.sma, a.cash_yield, a.div_yield)
    rebase = lambda s: s / s.iloc[0] * 100
    print(f"Nifty 50, {eq.index[0].date()} -> {eq.index[-1].date()} | rule: invested when close > {a.sma}-day average, else liquid fund "
          f"({a.cash_yield:.1%} assumed) | dividends {a.div_yield:.1%} assumed")
    rows = {"Trend rule": equity_stats(eq), "Buy & hold Nifty": equity_stats(bh), "Liquid fund only": equity_stats(cash)}
    t = pd.DataFrame(rows).T[["total_return", "cagr", "max_drawdown", "ann_vol"]]
    print((t * 100).round(1).to_string())
    print(f"switches: {switches} (about {switches / equity_stats(eq)['years']:.1f} a year) | share of days invested: {invested:.0%}")
    print("\nCalendar-year returns (%):")
    print((yearly_returns({"trend": eq, "buy_hold": bh, "liquid": cash}) * 100).round(1).to_string())
    if a.split:
        sp = pd.Timestamp(a.split)
        for name, m in (("before " + a.split, eq.index < sp), ("from " + a.split, eq.index >= sp)):
            if m.sum() > 60:
                s_, b_ = equity_stats(eq[m]), equity_stats(bh[m])
                print(f"{name}: trend CAGR {s_['cagr']:.1%}, max DD {s_['max_drawdown']:.1%} | buy&hold CAGR {b_['cagr']:.1%}, max DD {b_['max_drawdown']:.1%}")


if __name__ == "__main__":
    main()
