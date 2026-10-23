"""Run the two-sleeve backtest on your local data.

  python run_backtest.py --data data/daily --start 2018-01-01 --split 2023-01-01

--split marks the start of an out-of-sample period: look at it ONLY after freezing the rules.
"""
import argparse, os
import numpy as np
import pandas as pd
from config import Config
import dataclasses
from signals import load_panel, compute_features, load_index_close, load_delivery
from engine import run_core, run_swing
from metrics import equity_stats, yearly_returns, trade_stats, trade_stats_by_setup, position_table
from events import load_events, blackout_mask

WARMUP = 260  # trading days needed by the slowest indicator (200-day SMA etc.)


PCT_KEYS = {"total_return", "cagr", "max_drawdown", "ann_vol", "win_rate"}


def fmt(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = f"{v:.1%}" if k in PCT_KEYS else f"{v:,.2f}"
        else:
            out[k] = v
    return out


def show(title, d):
    print(f"\n--- {title}")
    for k, v in fmt(d).items():
        print(f"  {k:<24}{v}")


def spearman_boot(x, y, n=2000, seed=0, groups=None):
    """Rank correlation with a 95% bootstrap interval. With `groups` (e.g. the signal month) whole groups are resampled, which
    keeps signals that share the same market conditions together and gives an honest (wider) interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)

    def rho(i):
        return pd.Series(x[i]).rank().corr(pd.Series(y[i]).rank())
    base = rho(np.arange(len(x)))
    if groups is None:
        def sample():
            return rng.integers(0, len(x), len(x))
    else:
        g = np.asarray(groups).astype(str)
        labels = pd.unique(g)
        by = {l: np.where(g == l)[0] for l in labels}

        def sample():
            return np.concatenate([by[labels[k]] for k in rng.integers(0, len(labels), len(labels))])
    boots = [rho(sample()) for _ in range(n)]
    return base, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def study_mode(a, cfg, P, ixc, dlv):
    """Does delivery % on the signal day predict the outcome of the SAME signals? Every raw signal is simulated as an independent
    trade (no slot limit, so no path dependence) and delivery is related to the R-multiple. One test, no threshold search."""
    if dlv is None:
        raise SystemExit("No delivery data: put sec_bhavdata_full_*.csv files in data/raw and re-run build_db.py")
    dates = P.close.index
    i0 = max(WARMUP, int(dates.searchsorted(pd.Timestamp(a.start))) if a.start else WARMUP)
    i1 = int(dates.searchsorted(pd.Timestamp(a.end), side="right")) - 1 if a.end else len(dates) - 1
    cfg.use_breakout = a.setups in ("both", "breakout")
    cfg.use_pullback = a.setups in ("both", "pullback")
    cfg.delivery_min = 0.0
    F = compute_features(P, cfg, ixc, dlv)
    big = dataclasses.replace(cfg, swing_max_pos=10 ** 6, swing_max_pos_pct=10.0, total_capital=1e12, swing_risk_pct=1e-6)
    _, tr = run_swing(P, F, big, 1e12, i0, i1)
    if tr.empty:
        raise SystemExit("No signals in the test window")
    g = position_table(tr, "swing")
    g = g[g.init_risk > 0].copy()
    g["R"] = g["pnl"] / g["init_risk"]
    sig_i = dates.get_indexer(pd.to_datetime(g["entry"])) - 1
    g["deliv"] = [dlv.at[dates[k], s] if k >= 0 and s in dlv.columns and dates[k] in dlv.index else np.nan for k, s in zip(sig_i, g["symbol"])]
    g = g.dropna(subset=["deliv"])
    print(f"Signals studied: {len(g)} (delivery known) | mean R {g.R.mean():+.3f} | win rate {(g.R > 0).mean():.1%}")
    bins = pd.cut(g["deliv"], [0, 40, 50, 60, 101], labels=["<40", "40-50", "50-60", ">=60"], right=False)
    t = g.groupby(bins, observed=True).agg(signals=("R", "size"), mean_R=("R", "mean"), win_rate=("R", lambda s: (s > 0).mean()),
                                         PF=("pnl", lambda s: s[s > 0].sum() / max(-s[s <= 0].sum(), 1e-9)))
    print(t.round(3).to_string())
    for name, sub in [("all", g)] + [(s, g[g.setup == s]) for s in sorted(g.setup.unique())]:
        if len(sub) >= 50:
            rho, lo, hi = spearman_boot(sub["deliv"], sub["R"], n=1000, groups=pd.to_datetime(sub["entry"]).dt.to_period("M"))
            verdict = "no evidence of an effect (CI includes 0)" if lo <= 0 <= hi else ("delivery % HELPS" if rho > 0 else "delivery % HURTS (higher delivery, worse outcome)")
            print(f"Rank correlation delivery vs R [{name}, n={len(sub)}]: {rho:+.3f}  95% CI (months resampled) [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    yrs = pd.to_datetime(g["entry"]).dt.year
    print("By calendar year (a real effect should keep the same sign):")
    for y_, sub in g.groupby(yrs):
        if len(sub) >= 50:
            print(f"  {y_}: n={len(sub):4d}  rank correlation {sub['deliv'].rank().corr(sub['R'].rank()):+.3f}  mean R {sub['R'].mean():+.3f}")
    print("Note: one test on one data set. Even a 'significant' result should be confirmed on a later out-of-sample period.")


def grid_mode(a, cfg, P, ixc, dlv):
    """Swing sleeve only, at several delivery thresholds. Beware: every extra variant you try raises the chance of a fluke.
    Path-dependent (slot limits): run --delivery-study first. A useless filter can look good here."""
    if dlv is None:
        raise SystemExit("No delivery data: put sec_bhavdata_full_*.csv files in data/raw and re-run build_db.py")
    dates = P.close.index
    i0 = max(WARMUP, int(dates.searchsorted(pd.Timestamp(a.start))) if a.start else WARMUP)
    i1 = int(dates.searchsorted(pd.Timestamp(a.end), side="right")) - 1 if a.end else len(dates) - 1
    cov = dlv.reindex(dates).notna().sum(axis=1)
    print(f"Delivery coverage: {int((cov > 0).sum())} of {len(dates)} trading dates have delivery data")
    cfg.use_breakout = a.setups in ("both", "breakout")
    cfg.use_pullback = a.setups in ("both", "pullback")
    cfg.swing_priority = a.priority
    rows, yearly = [], {}
    for thr in [float(x) for x in a.delivery_grid.split(",")]:
        c2 = dataclasses.replace(cfg, delivery_min=thr)
        F = compute_features(P, c2, ixc, dlv)
        eq, tr = run_swing(P, F, c2, c2.total_capital * c2.alloc_swing, i0, i1)
        row = {"min_deliv_%": thr}
        st = trade_stats(tr, "swing") if len(tr) else {"positions": 0}
        row.update({"positions": st.get("positions", 0), "win_rate": st.get("win_rate", np.nan), "profit_factor": st.get("profit_factor", np.nan),
                    "expectancy_R": st.get("expectancy_R", np.nan), "swing_CAGR": equity_stats(eq).get("cagr", np.nan),
                    "max_DD": equity_stats(eq).get("max_drawdown", np.nan)})
        if a.split and len(tr):
            oos = trade_stats(tr[pd.to_datetime(tr["exit_date"]) >= pd.Timestamp(a.split)], "swing")
            row.update({"OOS_positions": oos.get("positions", 0), "OOS_PF": oos.get("profit_factor", np.nan)})
        if len(tr):
            g = position_table(tr, "swing")
            g["yr"] = pd.to_datetime(g["exit"]).dt.year
            yearly[thr] = g.groupby("yr")["pnl"].agg(gw=lambda s: s[s > 0].sum(), gl=lambda s: -s[s <= 0].sum(), n="size")
        rows.append(row)
    base_thr = float(a.delivery_grid.split(",")[0])
    for row in rows:
        thr = row["min_deliv_%"]
        if thr != base_thr and thr in yearly and base_thr in yearly:
            y0, y1 = yearly[base_thr], yearly[thr]
            ok = [(y, (y1.loc[y, "gw"] / max(y1.loc[y, "gl"], 1e-9)) > (y0.loc[y, "gw"] / max(y0.loc[y, "gl"], 1e-9)))
                  for y in y0.index.intersection(y1.index) if y0.loc[y, "n"] >= 10 and y1.loc[y, "n"] >= 10]
            row["years_PF_better"] = f"{sum(v for _, v in ok)}/{len(ok)}"
    t = pd.DataFrame(rows)
    for c in ("win_rate", "swing_CAGR", "max_DD"):
        t[c] = (t[c] * 100).round(1)
    print(t.round(2).to_string(index=False))
    print("Reading guide: a useless filter can still raise the profit factor by 0.2-0.3 by chance (see README). Trust a threshold only if the "
          "gain is large, rises steadily with the threshold, helps in most years (years_PF_better), helps out-of-sample, and keeps 100+ trades.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/daily")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--split", default=None, help="out-of-sample start date")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--setups", default="both", choices=["both", "breakout", "pullback"])
    ap.add_argument("--priority", default="score", choices=["score", "breakout", "pullback"])
    ap.add_argument("--delivery-min", type=float, default=0.0, help="require delivery %% >= this on swing signal day (0 = off)")
    ap.add_argument("--delivery-grid", default=None, help="e.g. 0,40,50,60: compare the swing sleeve at several thresholds and exit")
    ap.add_argument("--delivery-study", action="store_true", help="statistical test of whether delivery %% predicts signal outcomes (run this FIRST)")
    ap.add_argument("--events", default=None, help="results-calendar CSV (symbol,date) to apply the entry blackout")
    a = ap.parse_args()

    cfg = Config()
    if a.capital:
        cfg.total_capital = a.capital
    cfg.swing_priority = a.priority
    cfg.use_breakout = a.setups in ("both", "breakout")
    cfg.use_pullback = a.setups in ("both", "pullback")
    P = load_panel(a.data)
    ixc = load_index_close(a.data)
    dlv = load_delivery(a.data)
    cfg.delivery_min = a.delivery_min
    if a.delivery_study:
        study_mode(a, cfg, P, ixc, dlv)
        return
    if a.delivery_grid:
        grid_mode(a, cfg, P, ixc, dlv)
        return
    F = compute_features(P, cfg, ixc, dlv)
    if cfg.delivery_min > 0:
        print(f"Delivery filter ON: swing signals need delivery % >= {cfg.delivery_min:g}")
    print('Regime/benchmark source:', 'Nifty 50 index' if (ixc is not None and cfg.regime_source != 'proxy') else 'equal-weight universe proxy')
    blk = None
    if a.events:
        ev = load_events(a.events)
        blk = blackout_mask(P.close.index, P.close.columns, ev, cfg.results_blackout_days)
        print(f"Results blackout applied: {len(ev)} events, {cfg.results_blackout_days} calendar days before each")
    else:
        print("NOTE: no --events file, so the results blackout is NOT applied in this backtest "
              "(live reports will apply it; expect live gap risk to be lower).")
    dates = P.close.index
    i0 = max(WARMUP, int(dates.searchsorted(pd.Timestamp(a.start))) if a.start else WARMUP)
    i1 = int(dates.searchsorted(pd.Timestamp(a.end), side="right")) - 1 if a.end else len(dates) - 1
    print(f"Data {dates[0].date()} -> {dates[-1].date()} | testing {dates[i0].date()} -> {dates[i1].date()} "
          f"| {P.close.shape[1]} symbols | capital {cfg.total_capital:,.0f}")

    core_eq, core_tr = run_core(P, F, cfg, cfg.total_capital * cfg.alloc_core, i0, i1)
    swing_eq, swing_tr = run_swing(P, F, cfg, cfg.total_capital * cfg.alloc_swing, i0, i1, blackout=blk)
    n = len(core_eq)
    cash_start = cfg.total_capital * cfg.alloc_cash
    cash_eq = pd.Series(cash_start * (1 + cfg.cash_yield) ** (np.arange(n) / 252), index=core_eq.index)
    comb = core_eq + swing_eq + cash_eq
    bench = F.market.iloc[i0:i1 + 1]
    bench = bench / bench.iloc[0] * cfg.total_capital
    liquid = pd.Series(cfg.total_capital * (1 + cfg.cash_yield) ** (np.arange(n) / 252), index=core_eq.index)

    trades = pd.concat([t for t in (core_tr, swing_tr) if not t.empty], ignore_index=True) if (len(core_tr) or len(swing_tr)) else pd.DataFrame()
    s_comb, s_bench = equity_stats(comb), equity_stats(bench)
    show("COMBINED (core + swing + cash buffer)", s_comb)
    show("Benchmark: " + ("Nifty 50 index, price return (no dividends)" if F.regime_source_used == "nifty" else "equal-weight universe buy&hold (proxy for index)"), s_bench)
    show("Core sleeve", equity_stats(core_eq))
    show("Swing sleeve", equity_stats(swing_eq))
    if not trades.empty:
        for sl in ("core", "swing"):
            show(f"Trade statistics: {sl}", trade_stats(trades, sl))
        for setup, st in trade_stats_by_setup(trades).items():
            show(f"Swing by setup: {setup}", st)
    print("\n--- Calendar-year returns")
    yr = yearly_returns({"combined": comb, "core": core_eq, "swing": swing_eq, "benchmark": bench})
    print((yr * 100).round(1).to_string())

    if a.split:
        sp = pd.Timestamp(a.split)
        for name, seg in (("IN-SAMPLE", comb[comb.index < sp]), ("OUT-OF-SAMPLE", comb[comb.index >= sp])):
            if len(seg) > 20:
                show(f"{name} combined", equity_stats(seg))
        if not trades.empty:
            oos = trades[pd.to_datetime(trades["exit_date"]) >= sp]
            if len(oos):
                show("OUT-OF-SAMPLE swing trades", trade_stats(oos, "swing"))

    sw = trade_stats(trades, "swing") if not trades.empty else {"positions": 0}
    print("\n=== EVIDENCE CHECKLIST (all must pass before real money) ===")
    checks = [
        ("Swing: at least 100 positions", sw.get("positions", 0) >= 100),
        ("Swing: profit factor >= 1.3 after costs", sw.get("profit_factor", 0) >= 1.3),
        ("Swing: expectancy >= +0.10R per position", sw.get("expectancy_R", -1) >= 0.10),
        ("Combined CAGR beats benchmark", s_comb.get("cagr", -1) > s_bench.get("cagr", 0)),
        ("Combined CAGR beats liquid-fund yield", s_comb.get("cagr", -1) > cfg.cash_yield),
        ("Combined max drawdown better than -25%", s_comb.get("max_drawdown", -1) > -0.25),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("Reminder: passing in-sample proves little; require out-of-sample and 2-3 months of paper trading.")

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame({"combined": comb, "core": core_eq, "swing": swing_eq, "cash": cash_eq,
                  "benchmark": bench, "liquid_fund": liquid}).to_csv(os.path.join(a.out, "equity.csv"))
    if not trades.empty:
        trades.to_csv(os.path.join(a.out, "trades.csv"), index=False)
    print(f"\nSaved {a.out}/equity.csv and {a.out}/trades.csv")


if __name__ == "__main__":
    main()
