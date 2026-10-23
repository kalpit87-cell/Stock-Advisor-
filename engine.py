"""Event-driven daily backtest engine for two long-only sleeves.

Timing convention (no look-ahead): signals are computed on the CLOSE of day t and executed
at the OPEN of day t+1. Stops are checked against intraday low/open of later days.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import Config
from signals import Panel, Features


class Book:
    """Cash + positions + trade log for one sleeve. All costs are applied here."""

    def __init__(self, capital: float, cfg: Config, sleeve: str, symbols):
        self.cash, self.cfg, self.sleeve, self.syms = capital, cfg, sleeve, list(symbols)
        self.pos, self.trades, self._tid = {}, [], 0

    def buy(self, j, date, open_px, qty, **extra):
        c = self.cfg.costs
        fill = open_px * (1 + c.slippage)
        fee = qty * fill * c.buy_fee_rate()
        self.cash -= qty * fill + fee
        self._tid += 1
        self.pos[j] = dict(tid=self._tid, qty=qty, qty0=qty, entry=fill, entry_date=date,
                           cost=qty * fill + fee, fee_buy=fee, slip_buy=qty * open_px * c.slippage, **extra)

    def sell(self, j, date, px, qty, reason):
        p = self.pos[j]
        qty = min(qty, p["qty"])
        c = self.cfg.costs
        fill = px * (1 - c.slippage)
        value = qty * fill
        fee = value * c.sell_fee_rate() + c.dp_charge_sell
        self.cash += value - fee
        basis = p["cost"] * qty / p["qty"]
        pnl = value - fee - basis
        frac = qty / p["qty"]
        costs = fee + qty * px * c.slippage + p["fee_buy"] * frac + p["slip_buy"] * frac     # everything paid to trade this lot
        p["fee_buy"] -= p["fee_buy"] * frac
        p["slip_buy"] -= p["slip_buy"] * frac
        p["cost"] -= basis
        p["qty"] -= qty
        self.trades.append(dict(sleeve=self.sleeve, trade_id=p["tid"], symbol=self.syms[j],
                                entry_date=p["entry_date"], exit_date=date, qty=qty,
                                entry_px=p["entry"], exit_px=fill, pnl=pnl, reason=reason,
                                init_risk=p.get("init_risk", np.nan), setup=p.get("setup", ""), costs=costs))
        if p["qty"] <= 0:
            del self.pos[j]

    def equity(self, last_prices) -> float:
        return self.cash + sum(p["qty"] * last_prices[j] for j, p in self.pos.items())


def _prep(P: Panel):
    return dict(O=P.open.values, H=P.high.values, L=P.low.values, C=P.close.values,
                Cff=P.close.ffill().values,
                stale=P.close.isna().rolling(10).sum().eq(10).values)


def run_core(P: Panel, F: Features, cfg: Config, capital: float, i0: int, i1: int):
    """Weekly-rebalanced momentum rotation with regime filter and ATR trailing stop."""
    d = _prep(P)
    O, C, Cff, stale = d["O"], d["C"], d["Cff"], d["stale"]
    ATR, SC, EL, REG = F.atr.values, F.score.values, F.core_eligible.values, F.regime_on.values
    dates, syms = P.close.index, P.close.columns
    wk = pd.Series(dates).dt.to_period("W")
    is_reb = (wk != wk.shift(-1)).values
    book = Book(capital, cfg, "core", syms)
    pend_exit, pend_entry = {}, []
    eq = np.zeros(i1 - i0 + 1)
    slip, bfee = cfg.costs.slippage, cfg.costs.buy_fee_rate()

    for i in range(i0, i1 + 1):
        dt = dates[i]
        # --- execute orders created at yesterday's close, at today's open
        for j in list(pend_exit):
            if j not in book.pos:
                del pend_exit[j]
            elif not np.isnan(O[i, j]):
                book.sell(j, dt, O[i, j], book.pos[j]["qty"], pend_exit.pop(j))
        for j, tv in pend_entry:
            if j in book.pos or np.isnan(O[i, j]):
                continue
            fill = O[i, j] * (1 + slip)
            qty = int(min(tv, book.cash) / (fill * (1 + bfee)))
            if qty >= 1:
                book.buy(j, dt, O[i, j], qty, hc=fill)
        pend_entry = []

        # --- end of day: update trailing high, flag exits
        for j, p in list(book.pos.items()):
            if np.isnan(C[i, j]):
                if stale[i, j]:
                    book.sell(j, dt, Cff[i, j], p["qty"], "stale_or_delisted")
                continue
            p["hc"] = max(p["hc"], C[i, j])
            if j in pend_exit:
                continue
            if not REG[i]:
                pend_exit[j] = "regime_off"
            elif not np.isnan(ATR[i, j]) and C[i, j] < p["hc"] - cfg.core_trail_atr * ATR[i, j]:
                pend_exit[j] = "trail_stop"
        eq[i - i0] = book.equity(Cff[i])

        # --- weekly rebalance decisions (executed next open)
        if is_reb[i] and i < i1 and REG[i]:
            sc = np.where(EL[i], SC[i], -np.inf)
            order = np.argsort(-sc)
            ranked = [j for j in order if np.isfinite(sc[j])]
            keep = set(ranked[:cfg.core_hold_rank])
            for j in list(book.pos):
                if j not in keep and j not in pend_exit:
                    pend_exit[j] = "rank_drop"
            slots = cfg.core_k - len([j for j in book.pos if j not in pend_exit])
            tv = eq[i - i0] / cfg.core_k
            buyable = [j for j in ranked[:cfg.core_hold_rank] if C[i, j] * cfg.core_min_shares <= tv]
            for j in buyable:
                if slots <= 0:
                    break
                if j not in book.pos:
                    pend_entry.append((j, tv))
                    slots -= 1

    for j in list(book.pos):
        book.sell(j, dates[i1], Cff[i1, j], book.pos[j]["qty"], "end_of_test")
    eq[-1] = book.cash
    return pd.Series(eq, index=dates[i0:i1 + 1]), pd.DataFrame(book.trades)


def run_swing(P: Panel, F: Features, cfg: Config, capital: float, i0: int, i1: int, blackout=None):
    """Breakout swing trades: ATR stop, 50% off at +2R, then breakeven + chandelier trail."""
    d = _prep(P)
    O, H, L, C, Cff, stale = d["O"], d["H"], d["L"], d["C"], d["Cff"], d["stale"]
    ATR, SC, SIG = F.atr.values, F.score.values, F.swing_signal.values
    BRK = F.breakout_signal.values if F.breakout_signal is not None else None
    dates, syms = P.close.index, P.close.columns
    book = Book(capital, cfg, "swing", syms)
    pending = []
    eq = np.zeros(i1 - i0 + 1)
    last_eq = capital
    slip, bfee = cfg.costs.slippage, cfg.costs.buy_fee_rate()

    for i in range(i0, i1 + 1):
        dt = dates[i]
        scale = last_eq / capital
        # --- entries at today's open
        for j, sig_close, a, setup in pending:
            if j in book.pos or np.isnan(O[i, j]) or len(book.pos) >= cfg.swing_max_pos:
                continue
            o = O[i, j]
            if o > sig_close + cfg.swing_max_gap_atr * a:
                continue  # gapped up too far: do not chase
            fill = o * (1 + slip)
            R = cfg.swing_stop_atr * a
            qty = int(cfg.swing_risk_pct * cfg.total_capital * scale / R)
            qty = min(qty, int(cfg.swing_max_pos_pct * cfg.total_capital * scale / fill),
                      int(book.cash / (fill * (1 + bfee))))
            if qty < 1:
                continue
            book.buy(j, dt, o, qty, stop=fill - R, R=R, t1=fill + cfg.swing_t1_r * R,
                     t1_done=False, hh=fill, atr=a, init_risk=qty * R, entry_i=i, setup=setup)
        pending = []

        # --- manage open positions with today's bar (stop checked before target: conservative)
        for j in list(book.pos):
            p = book.pos[j]
            o, h, l, c = O[i, j], H[i, j], L[i, j], C[i, j]
            if np.isnan(c):
                if stale[i, j]:
                    book.sell(j, dt, Cff[i, j], p["qty"], "stale_or_delisted")
                continue
            today = p["entry_i"] == i
            if (not today) and o <= p["stop"]:
                book.sell(j, dt, o, p["qty"], "stop_gap")
                continue
            if l <= p["stop"]:
                book.sell(j, dt, p["stop"], p["qty"], "stop")
                continue
            if (not p["t1_done"]) and h >= p["t1"]:
                px = p["t1"] if today else max(p["t1"], o)
                q1 = p["qty"] // 2
                if q1 >= 1:
                    book.sell(j, dt, px, q1, "T1")
                p["t1_done"] = True
                p["stop"] = max(p["stop"], p["entry"])
            p["hh"] = max(p["hh"], h)
            if p["t1_done"]:
                p["stop"] = max(p["stop"], p["hh"] - cfg.swing_trail_atr * p["atr"])
            if i - p["entry_i"] >= cfg.swing_time_stop:
                book.sell(j, dt, c, p["qty"], "time_stop")
        last_eq = book.equity(Cff[i])
        eq[i - i0] = last_eq

        # --- new signals at today's close, executed next open
        if i < i1:
            slots = cfg.swing_max_pos - len(book.pos)
            if slots > 0:
                cand = [j for j in np.where(SIG[i])[0]
                        if j not in book.pos and not (blackout is not None and blackout[i, j])]
                def _key(j):
                    s = SC[i, j] if np.isfinite(SC[i, j]) else -1e9
                    if BRK is not None and cfg.swing_priority in ("breakout", "pullback"):
                        first = bool(BRK[i, j]) == (cfg.swing_priority == "breakout")
                        return (1 if first else 0, s)
                    return (0, s)
                cand.sort(key=_key, reverse=True)
                pending = [(j, C[i, j], ATR[i, j],
                            "swing" if BRK is None else ("breakout" if BRK[i, j] else "pullback"))
                           for j in cand[:slots]]

    for j in list(book.pos):
        book.sell(j, dates[i1], Cff[i1, j], book.pos[j]["qty"], "end_of_test")
    eq[-1] = book.cash
    return pd.Series(eq, index=dates[i0:i1 + 1]), pd.DataFrame(book.trades)
