"""Daily report: regime, core-momentum ranking, new swing signals (breakout + pullback) with stop/target/
quantity, results-calendar flags, open-position status, optional headline scan.

  python daily_report.py --data data/daily --capital 150000 --positions positions.csv --events data/events.csv --news

positions.csv columns (you maintain it):
  sleeve,symbol,entry_date,entry_price,qty,stop,target1,t1_done
Every signal is appended to paper_signals.csv so live results can be audited later.
"""
import argparse, os
import numpy as np
import pandas as pd
from config import Config
from signals import load_panel, compute_features, load_index_close, load_delivery
from events import load_events, results_within, load_dividends, dividend_within
from news_check import news_section


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/daily")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--positions", default="positions.csv")
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--delivery-min", type=float, default=0.0, help="require delivery %% >= this for swing signals (needs delivery data)")
    ap.add_argument("--dividends", default="data/dividends.csv")
    ap.add_argument("--news", action="store_true", help="scan headlines for symbols in this report (needs internet)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--asof", default=None, help="run the report as of a past date (for review/testing)")
    a = ap.parse_args()
    cfg = Config()
    if a.capital:
        cfg.total_capital = a.capital
    P = load_panel(a.data)
    if a.asof:
        cut = pd.Timestamp(a.asof)
        for k in ("open", "high", "low", "close", "volume", "value"):
            setattr(P, k, getattr(P, k).loc[:cut])
    dlv = load_delivery(a.data)
    cfg.delivery_min = a.delivery_min
    notes = []
    if cfg.delivery_min > 0:
        last_cov = 0 if dlv is None or P.close.index[-1] not in dlv.index else int(dlv.loc[P.close.index[-1]].notna().sum())
        if last_cov < 500:
            notes.append(f"> WARNING: delivery data missing for {P.close.index[-1].date()} (only {last_cov} symbols): delivery filter NOT applied today. Download sec_bhavdata_full for this date and rebuild.")
            cfg.delivery_min = 0.0
    F = compute_features(P, cfg, load_index_close(a.data), dlv)
    ev = load_events(a.events)
    divs = load_dividends(a.dividends)
    dates = P.close.index
    i = len(dates) - 1
    day = dates[i]
    syms = list(P.close.columns)
    watch = []  # symbols for the headline scan
    lines = [f"# Daily report - data as of {day.date()}", ""]
    lines += notes + ([""] if notes else [])
    if ev.empty:
        lines += [f"> WARNING: no results calendar loaded from {a.events}. Check results dates manually before every entry.", ""]

    def res_flag(sym):
        d = results_within(ev, sym, day, cfg.results_blackout_days)
        return d

    # ---- 1 regime
    reg = bool(F.regime_on.iloc[i])
    lines += ["## 1. Market regime",
              f"Market trend filter (Nifty 50 if available, else universe proxy) is {'ABOVE' if reg else 'BELOW'} its {cfg.regime_sma}-day average -> "
              f"**{'new longs allowed' if reg else 'NO new long entries; core sleeve moves to cash; swing sleeve idle'}**.", ""]

    # ---- 2 core ranking
    sc = np.where(F.core_eligible.iloc[i].values, F.score.iloc[i].values, np.nan)
    order = [j for j in np.argsort(-np.nan_to_num(sc, nan=-9e9)) if np.isfinite(sc[j])]
    lines += [f"## 2. Core momentum ranking (top {cfg.core_hold_rank}; buy top {cfg.core_k}; hold while rank <= {cfg.core_hold_rank})"]
    if not reg or not order:
        lines += ["No eligible candidates (regime off or nothing passes filters).", ""]
    else:
        lines += ["| Rank | Symbol | Close | 6m momentum | ATR % | Buy size | Results |", "|---|---|---|---|---|---|---|"]
        per = cfg.total_capital * cfg.alloc_core / cfg.core_k
        for r, j in enumerate(order[:cfg.core_hold_rank], 1):
            c = P.close.iloc[i, j]
            ok = c * cfg.core_min_shares <= per
            size = f"{int(per // c)} sh (~{per:,.0f})" if ok else "too expensive for slot size: skip"
            rf = res_flag(syms[j])
            if ok and r <= cfg.core_k and rf is None:
                watch.append(syms[j])
            lines.append(f"| {r} | {syms[j]} | {c:,.2f} | {sc[j]:.1%} | {F.atr.iloc[i, j] / c:.1%} | {size} | "
                         f"{'**' + str(rf.date()) + ' - wait**' if rf is not None else '-'} |")
        lines += ["Buy the highest-ranked affordable names (top 10 only) until 5 slots are filled, skipping names with results "
                  "due. Core rebalances weekly (last trading day of the week); act at next open.", ""]

    # ---- 3 swing signals
    pos_df = pd.read_csv(a.positions) if os.path.exists(a.positions) else None
    n_open = int((pos_df["sleeve"] == "swing").sum()) if pos_df is not None and len(pos_df) else 0
    slots = max(cfg.swing_max_pos - n_open, 0)
    risk_amt = cfg.swing_risk_pct * cfg.total_capital

    def key(j):
        s = F.score.iloc[i, j]
        s = s if np.isfinite(s) else -9e9
        if cfg.swing_priority in ("breakout", "pullback"):
            return (1 if bool(F.breakout_signal.iloc[i, j]) == (cfg.swing_priority == "breakout") else 0, s)
        return (0, s)

    sig = sorted(np.where(F.swing_signal.iloc[i].values)[0], key=key, reverse=True)
    lines += [f"## 3. New swing signals (risk Rs {risk_amt:,.0f} = {cfg.swing_risk_pct:.2%} of capital per trade; "
              f"{n_open} open, {slots} free slot(s) of {cfg.swing_max_pos})"]
    new_rows, blocked, taken = [], [], 0
    if not sig:
        lines += ["No signals today.", ""]
    else:
        lines += ["| Action | Setup | Symbol | Signal close | Deliv % | Entry rule | Stop | T1 (+2R, sell 50%) | Qty | Exposure | Max loss | Note |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for j in sig:
            c, atr = P.close.iloc[i, j], F.atr.iloc[i, j]
            R = cfg.swing_stop_atr * atr
            setup = "breakout" if F.breakout_signal.iloc[i, j] else "pullback"
            rf = res_flag(syms[j])
            if rf is not None:
                blocked.append(f"{syms[j]} ({setup}): results on {rf.date()}")
                continue
            stop, t1 = c - R, c + cfg.swing_t1_r * R
            qty = min(int(risk_amt / R), int(cfg.swing_max_pos_pct * cfg.total_capital / c))
            take = taken < slots
            act = "**TAKE**" if take else "backup"
            dv = None if dlv is None or day not in dlv.index or syms[j] not in dlv.columns else dlv.loc[day, syms[j]]
            xd = dividend_within(divs, syms[j], day, 5)
            note = f"ex-dividend {xd[0].date()} (Rs {xd[1]:g}): price drops by about that; broker stops are NOT adjusted" if xd else "-"
            lines.append(f"| {act} | {setup} | {syms[j]} | {c:,.2f} | {'n/a' if dv is None or pd.isna(dv) else f'{dv:.0f}'} | "
                         f"buy at next open, SKIP if open > {c + cfg.swing_max_gap_atr * atr:,.2f} | "
                         f"{stop:,.2f} | {t1:,.2f} | {qty} | {qty * c:,.0f} | {qty * R:,.0f} | {note} |")
            if take:
                taken += 1
                watch.append(syms[j])
                new_rows.append(dict(date=day.date(), sleeve="swing", setup=setup, symbol=syms[j], signal_close=round(c, 2),
                                     stop=round(stop, 2), t1=round(t1, 2), qty=qty))
        if blocked:
            lines += ["", "Blocked by the results calendar (do not enter): " + "; ".join(blocked)]
        lines += ["", "TAKE = fills your free slots in priority order; backup = use only if a TAKE is skipped (gap-up filter) or fails to fill. "
                  "Stop/T1 are re-based on your actual fill: stop = fill - 1.5 ATR, T1 = fill + 2R. After T1: sell half, stop to entry, "
                  "then trail at highest high - 3 ATR. Time stop: 30 trading days.", ""]
    if new_rows:
        path = "paper_signals.csv"
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        if len(old):
            old = old[old["date"].astype(str) != str(day.date())]
        pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True).to_csv(path, index=False)

    # ---- 4 open positions
    lines += ["## 4. Open positions"]
    if pos_df is not None:
        pos = pos_df
        lines += ["| Sleeve | Symbol | Close | P&L % | R-multiple | Action |", "|---|---|---|---|---|---|"]
        for _, r in pos.iterrows():
            s = r["symbol"]
            if s not in P.close.columns:
                lines.append(f"| {r['sleeve']} | {s} | n/a | n/a | n/a | symbol not in data: check manually |")
                continue
            watch.append(s)
            c, hi, lo = P.close[s].iloc[i], P.high[s].iloc[i], P.low[s].iloc[i]
            ent = float(r["entry_price"])
            acts, rmult = [], ""
            since = P.close.index >= pd.Timestamp(r["entry_date"])
            atr_now = F.atr[s].iloc[i]
            if r["sleeve"] == "swing" and pd.notna(r.get("stop")):
                stop, t1 = float(r["stop"]), float(r["target1"])
                done = int(r.get("t1_done", 0)) == 1
                R0 = (t1 - ent) / cfg.swing_t1_r
                rmult = f"{(c - ent) / R0:+.2f}R" if R0 > 0 else ""
                if lo <= stop:
                    acts.append(f"STOP HIT today (low {lo:,.2f} <= {stop:,.2f}): exit")
                if (not done) and hi >= t1:
                    acts.append("T1 reached: sell 50%, move stop to entry")
                if done:
                    trail = max(stop, P.high[s][since].max() - cfg.swing_trail_atr * atr_now)
                    if trail > stop:
                        acts.append(f"raise stop to {trail:,.2f}")
                if since.sum() - 1 >= cfg.swing_time_stop:
                    acts.append("time stop due: exit at close")
            elif r["sleeve"] == "core":
                hc = P.close[s][since].max()
                if not reg:
                    acts.append("regime OFF: exit at next open")
                if c < hc - cfg.core_trail_atr * atr_now:
                    acts.append("trailing stop breached: exit at next open")
                rank = order.index(syms.index(s)) + 1 if syms.index(s) in order else None
                if rank is None or rank > cfg.core_hold_rank:
                    acts.append("outside hold-rank: exit at next weekly rebalance")
            xd = dividend_within(divs, s, day, 5)
            if xd:
                acts.append(f"ex-dividend {xd[0].date()} (Rs {xd[1]:g}): expect a price drop of about that; check your stop")
            rf = res_flag(s)
            if rf is not None:
                acts.append(f"**RESULTS on {rf.date()}: gap risk beyond your stop - decide to exit/reduce BEFORE results**")
            lines.append(f"| {r['sleeve']} | {s} | {c:,.2f} | {c / ent - 1:+.1%} | {rmult} | {'; '.join(acts) or 'hold'} |")
        lines.append("")
    else:
        lines += [f"No {a.positions} found - create it to track open trades.", ""]

    # ---- 5 news
    if a.news:
        lines += news_section(sorted(set(watch)))
    lines += ["## Reminders", "- Prices are end-of-day; verify against your broker before ordering. Update the results calendar and read headlines yourself.",
              "- Decision support, not advice. Log every override you make and why."]
    text = "\n".join(lines)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{day.date()}.md")
    open(out, "w").write(text)
    print(text)
    print(f"\n[saved {out}]")


if __name__ == "__main__":
    main()
