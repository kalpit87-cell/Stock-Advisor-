"""Performance statistics."""
import numpy as np
import pandas as pd


def equity_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    ratio = eq.iloc[-1] / eq.iloc[0]
    cagr = ratio ** (1 / years) - 1 if years > 0 and ratio > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    r = eq.pct_change().dropna()
    vol = r.std() * np.sqrt(252)
    return {"total_return": ratio - 1, "cagr": cagr, "max_drawdown": dd,
            "ann_vol": vol, "calmar": (cagr / abs(dd)) if dd < 0 else np.nan, "years": years}


def yearly_returns(curves: dict) -> pd.DataFrame:
    out = {}
    for name, eq in curves.items():
        eq = eq.dropna()
        ye = eq.groupby(eq.index.year).last()
        prev = pd.concat([pd.Series([eq.iloc[0]], index=[ye.index[0] - 1]), ye.iloc[:-1]])
        out[name] = (ye / prev.values - 1)
    return pd.DataFrame(out)


def position_table(trades: pd.DataFrame, sleeve: str) -> pd.DataFrame:
    t = trades[trades["sleeve"] == sleeve]
    if t.empty:
        return pd.DataFrame()
    g = t.groupby("trade_id").agg(pnl=("pnl", "sum"), init_risk=("init_risk", "first"),
                                  entry=("entry_date", "first"), exit=("exit_date", "last"),
                                  symbol=("symbol", "first"), setup=("setup", "first"), costs=("costs", "sum"))
    g["hold_days"] = (pd.to_datetime(g["exit"]) - pd.to_datetime(g["entry"])).dt.days
    return g


def trade_stats(trades: pd.DataFrame, sleeve: str) -> dict:
    g = position_table(trades, sleeve)
    if g.empty:
        return {"positions": 0}
    wins, losses = g[g.pnl > 0], g[g.pnl <= 0]
    gw, gl = wins.pnl.sum(), -losses.pnl.sum()
    out = {"positions": len(g), "win_rate": len(wins) / len(g),
           "avg_win": wins.pnl.mean() if len(wins) else 0.0,
           "avg_loss": losses.pnl.mean() if len(losses) else 0.0,
           "profit_factor": (gw / gl) if gl > 0 else np.inf,
           "net_pnl": g.pnl.sum(), "costs_paid": g.costs.sum(), "gross_pnl_before_costs": (g.pnl + g.costs).sum(),
           "avg_hold_days": g.hold_days.mean()}
    ir = g[g.init_risk > 0]
    if len(ir):
        out["expectancy_R"] = (ir.pnl / ir.init_risk).mean()
    return out


def trade_stats_by_setup(trades: pd.DataFrame, sleeve: str = "swing") -> dict:
    t = trades[trades["sleeve"] == sleeve]
    out = {}
    if t.empty or "setup" not in t:
        return out
    for s in sorted(t["setup"].unique()):
        out[s] = trade_stats(t[t["setup"] == s], sleeve)
    return out
