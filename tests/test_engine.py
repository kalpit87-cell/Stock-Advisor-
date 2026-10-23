"""Hand-computed checks of the swing engine (zero costs so numbers are exact).
Run:  python tests/test_engine.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from config import Config, Costs
from signals import Panel, Features
from engine import run_swing


def make(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    o, h, l, c = [pd.DataFrame({"AAA": [r[k] for r in rows]}, index=idx) for k in range(4)]
    P = Panel(open=o, high=h, low=l, close=c, volume=c * 0 + 1e6, value=c * 1e6)
    sig = pd.DataFrame({"AAA": [True] + [False] * (len(rows) - 1)}, index=idx)
    F = Features(swing_signal=sig, atr=c * 0 + 2.0, score=c * 0 + 1.0)
    cfg = Config(total_capital=100_000, swing_risk_pct=0.01, swing_max_pos_pct=0.5,
                 costs=Costs(0, 0, 0, 0, 0))
    return P, F, cfg


def test_target_then_trail():
    #        open  high  low   close
    rows = [(100, 100, 100, 100),      # day0 signal day (close 100, ATR 2)
            (100, 101, 99, 100.5),     # day1 entry @100, stop 97, R 3, T1 106
            (101, 107, 100, 106),      # day2 T1 hit -> sell half @106, stop -> 101
            (105, 106, 100.5, 101),    # day3 stopped @101
            (101, 101, 101, 101), (101, 101, 101, 101)]
    P, F, cfg = make(rows)
    eq, tr = run_swing(P, F, cfg, 100_000, 0, len(rows) - 1)
    assert list(tr.reason) == ["T1", "stop"], tr
    assert list(tr.qty) == [166, 167]
    assert abs(tr.pnl.sum() - (166 * 6 + 167 * 1)) < 1e-6, tr.pnl.sum()
    assert abs(eq.iloc[-1] - (100_000 + 1163)) < 1e-6, eq.iloc[-1]
    assert abs(tr.init_risk.iloc[0] - 999) < 1e-9


def test_gap_through_stop():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100.5),
            (95, 96, 94, 95),          # gaps below stop 97 -> exit at the OPEN (95)
            (95, 95, 95, 95), (95, 95, 95, 95)]
    P, F, cfg = make(rows)
    eq, tr = run_swing(P, F, cfg, 100_000, 0, len(rows) - 1)
    assert list(tr.reason) == ["stop_gap"], tr
    assert abs(tr.pnl.sum() - 333 * (95 - 100)) < 1e-6
    assert abs(eq.iloc[-1] - (100_000 - 1665)) < 1e-6


def test_no_chase_when_gap_up():
    rows = [(100, 100, 100, 100), (103, 104, 102, 103), (103, 103, 103, 103)]  # open 103 > 100+0.5*2
    P, F, cfg = make(rows)
    eq, tr = run_swing(P, F, cfg, 100_000, 0, len(rows) - 1)
    assert tr.empty and abs(eq.iloc[-1] - 100_000) < 1e-9


def test_blackout_blocks_entry():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100.5), (101, 101, 101, 101)]
    P, F, cfg = make(rows)
    blk = np.zeros((3, 1), dtype=bool); blk[0, 0] = True       # results due near signal day 0
    eq, tr = run_swing(P, F, cfg, 100_000, 0, 2, blackout=blk)
    assert tr.empty and abs(eq.iloc[-1] - 100_000) < 1e-9


def test_events_mask_and_lookup():
    from events import blackout_mask, results_within
    idx = pd.bdate_range("2026-10-01", periods=15)
    ev = pd.DataFrame({"symbol": ["AAA"], "date": [pd.Timestamp("2026-10-12")]})
    m = blackout_mask(idx, ["AAA", "BBB"], ev, 4)
    blocked = list(idx[m[:, 0]].strftime("%m-%d"))
    assert blocked == ["10-08", "10-09", "10-12"], blocked      # 8th..12th (calendar window of 4 days)
    assert not m[:, 1].any()
    assert results_within(ev, "AAA", "2026-10-09", 4) == pd.Timestamp("2026-10-12")
    assert results_within(ev, "AAA", "2026-10-13", 4) is None
    assert results_within(ev, "BBB", "2026-10-09", 4) is None


def test_news_parse_and_flag():
    from news_check import parse_rss, flag, news_section
    xml = ("<rss><channel><item><title>SEBI probe into ABC over disclosures</title><pubDate>x</pubDate><link>l</link></item>"
           "<item><title>ABC launches new product</title><pubDate>y</pubDate><link>l2</link></item></channel></rss>")
    hs = flag(parse_rss(xml))
    assert hs[0]["risk"] == ["probe", "sebi"] and hs[1]["risk"] == []
    out = "\n".join(news_section(["ABC"], fetcher=lambda s: parse_rss(xml)))
    assert "1 with risk words" in out
    bad = "\n".join(news_section(["ABC"], fetcher=lambda s: (_ for _ in ()).throw(OSError("offline"))))
    assert "news fetch failed" in bad


def test_parse_ratio_and_bm():
    from build_db import parse_ratio, parse_board_meetings
    assert parse_ratio("FV SPLT FRM RS 5 TO RE 1") == 0.2 and parse_ratio("FVSPLT FRM RS 10 TO RS 2") == 0.2
    assert parse_ratio("BONUS 1:1") == 0.5 and abs(parse_ratio("BONUS 3:2") - 0.4) < 1e-12
    assert parse_ratio("INTDIV - RS 12 PER SH") is None and parse_ratio("MERGER") is None
    txt = ("COMPANY NAME    SYMBOL     : BM DATE    : BM PURPOSE\n"
           "KPIT Technologies Limited KPITTECH : 28-OCT-2026 : Financial Results To consider and approve\nthe results\n"
           "Bajaj Auto Limited BAJAJ-AUTO : 29-OCT-2026 : Dividend\n")
    bm = parse_board_meetings(txt)
    assert list(bm.symbol) == ["KPITTECH", "BAJAJ-AUTO"] and "the results" in bm.purpose.iloc[0]
    assert bm.date.iloc[0] == pd.Timestamp("2026-10-28")


def test_adjust_prices_split_and_future_event():
    from build_db import adjust_prices
    d = pd.bdate_range("2026-01-01", periods=6)
    px = [100, 100, 100, 50, 50, 50]
    df = pd.DataFrame({"date": d, "symbol": "AAA", "open": px, "high": px, "low": px, "close": px,
                       "prev_close": px, "volume": 10.0, "value": 1.0})
    ev = pd.DataFrame({"symbol": ["AAA", "AAA"], "ex_date": [d[3], pd.Timestamp("2027-01-01")], "ratio": [0.5, 0.1]})
    out, applied = adjust_prices(df, ev)
    assert list(out.close) == [50, 50, 50, 50, 50, 50], list(out.close)   # future event (2027) ignored
    assert list(out.volume)[:3] == [20.0] * 3 and len(applied) == 1


def test_dividend_parse_and_adjust():
    from build_db import parse_dividend, adjust_prices
    assert parse_dividend("INTDIV - RS 2.5 PER SH") == 2.5 and parse_dividend("DIV - RE 0.30 PER SH") == 0.3
    assert parse_dividend("BONUS 1:1") is None
    d = pd.bdate_range("2026-01-01", periods=4)
    df = pd.DataFrame({"date": d, "symbol": "AAA", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                       "prev_close": 100.0, "volume": 1.0, "value": 1.0})
    dv = pd.DataFrame({"symbol": ["AAA"], "ex_date": [d[2]], "dividend": [5.0]})
    plain, _ = adjust_prices(df, None)
    adj, applied = adjust_prices(df, None, dv)
    assert list(plain.close) == [100.0] * 4                       # default: dividends NOT adjusted
    assert [round(x, 4) for x in adj.close] == [95.0, 95.0, 100.0, 100.0] and applied.kind.iloc[0] == "dividend"


def test_delivery_filter():
    from signals import compute_features
    from config import Config
    import numpy as np
    idx = pd.bdate_range("2023-01-02", periods=320)
    rng = np.random.default_rng(1)
    base = 100 * np.cumprod(1 + 0.004 + 0.01 * rng.standard_normal((320, 2)), axis=0)
    c = pd.DataFrame(base, index=idx, columns=["AAA", "BBB"])
    P = Panel(open=c.shift(1).fillna(c), high=c * 1.01, low=c * 0.99, close=c, volume=c * 0 + 1e6 * (1 + (np.arange(320) % 7 == 0)[:, None] * 2.0), value=c * 1e8)
    cfg = Config(universe_size=10, min_price=1.0)
    F0 = compute_features(P, cfg)
    dl = pd.DataFrame(30.0, index=idx, columns=["AAA", "BBB"])
    dl["AAA"] = 80.0                                              # AAA always high delivery, BBB always low
    cfg.delivery_min = 60.0
    F1 = compute_features(P, cfg, None, dl)
    assert F1.swing_signal["BBB"].sum() == 0 and F0.swing_signal.values.sum() > 0
    assert (F1.swing_signal["AAA"] <= F0.swing_signal["AAA"]).all()
    try:
        compute_features(P, cfg, None, None); assert False
    except ValueError:
        pass


def test_spearman_boot():
    import numpy as np
    from run_backtest import spearman_boot
    rng = np.random.default_rng(0)
    x = rng.uniform(20, 80, 400)
    rho, lo, hi = spearman_boot(x, 0.05 * x + rng.standard_normal(400), n=300)
    assert lo > 0 and rho > 0.1                          # real relationship is detected
    rho, lo, hi = spearman_boot(x, rng.standard_normal(400), n=300)
    assert lo <= 0 <= hi                                 # independent data: interval includes zero


def test_ex_date_parsing_no_day_month_swap():
    from build_db import parse_ex_dates
    s = pd.Series(["2026-10-06", "2026-10-16", "06/10/2026", "28/10/2024", "06-OCT-2026", "", "garbage"])
    got = parse_ex_dates(s)
    exp = ["2026-10-06", "2026-10-16", "2026-10-06", "2024-10-28", "2026-10-06"]
    assert [d.strftime("%Y-%m-%d") for d in got[:5]] == exp, got.tolist()
    assert got[5:].isna().all()


def test_bc_tolerant_and_jump_diagnosis():
    from build_db import read_bc_tolerant, suspect_jumps, diagnose_jumps, adjust_prices
    h = "SERIES,SYMBOL,SECURITY,RECORD_DT,BC_STRT_DT,BC_END_DT,EX_DT,ND_STRT_DT,ND_END_DT,PURPOSE\n"
    df = read_bc_tolerant((h + "EQ,AAA,Alpha Ltd.,2026-10-23,,,2026-10-23,,,FVSPLT FRM RS 10 TO RS 2\n"
                           "EQ,BBB,Beta Industries, Ltd.,2026-10-23,,,2026-10-22,,,BONUS 1:1\n"
                           "EQ,CCC,Gamma,2026-10-23,,,2026-10-24,,,INTDIV - RS 2 PER SH, SPLDIV - RS 1 PER SH\n"
                           "EQ,DDD,broken\n").encode())
    assert list(df.SYMBOL) == ["AAA", "BBB", "CCC"] and df.SECURITY[1] == "Beta Industries,Ltd." and df.EX_DT[1] == "2026-10-22"
    assert df.PURPOSE[2].startswith("INTDIV") and "SPLDIV" in df.PURPOSE[2]
    d = pd.bdate_range("2026-03-02", periods=12)
    def frame(sym, closes):
        return pd.DataFrame({"date": d, "symbol": sym, "open": closes, "high": closes, "low": closes, "close": closes,
                             "prev_close": closes, "volume": 1.0, "value": 1.0})
    px = pd.concat([frame("MISSED", [100.0] * 6 + [50.0] * 6),           # unlisted 2:1 split / bonus
                    frame("DEMERG", [100.0] * 6 + [55.0] * 6),            # -45%, a demerger is listed nearby
                    frame("CRASH", [100.0] * 6 + [43.0] * 6),             # -57%, no action, not a typical ratio
                    frame("OKAY", [100.0] * 12)])
    acts = pd.DataFrame({"symbol": ["DEMERG"], "ex_date": [d[6]], "purpose": ["DEMERGER"], "ratio": [None], "dividend": [None]})
    j = diagnose_jumps(suspect_jumps(px), acts).set_index("symbol")
    assert set(j.index) == {"MISSED", "DEMERG", "CRASH"}
    assert j.loc["MISSED", "cause"].startswith("probable split") and j.loc["MISSED", "suggested_ratio"] == 0.5
    assert j.loc["DEMERG", "cause"].startswith("corporate action not auto-adjusted")
    assert j.loc["CRASH", "cause"].startswith("no corporate action found")
    assert j["apply"].all() and abs(j.loc["DEMERG", "suggested_ratio"] - 0.55) < 1e-9        # falls of 40%+ get a continuity ratio
    acts2 = pd.DataFrame({"symbol": ["MISSED", "DEMERG"], "ex_date": [d[6], d[6]], "purpose": ["FV SPLT FRM RS # TO", "FVSPLT FRM RS 10 TO RS 5"],
                          "ratio": [None, 0.5], "dividend": [None, None]})
    j2 = diagnose_jumps(suspect_jumps(px), acts2).set_index("symbol")
    assert j2.loc["MISSED", "cause"].startswith("split/bonus wording unparsed") and j2.loc["MISSED", "apply"]
    assert j2.loc["DEMERG", "cause"].startswith("split/bonus listed nearby") and not j2.loc["DEMERG", "apply"]   # needs a human look
    fixed, _ = adjust_prices(px, pd.DataFrame({"symbol": ["MISSED"], "ex_date": [d[6]], "ratio": [0.5]}))
    assert "MISSED" not in set(suspect_jumps(fixed).symbol)               # the suggestion removes the jump


def test_rescheduled_bonus_uses_the_date_the_price_supports():
    from build_db import adjust_prices, suspect_jumps
    d = pd.bdate_range("2026-03-02", periods=14)
    closes = [100.0] * 9 + [9.2] * 5                     # real drop (bonus 10:1) on d[9]
    df = pd.DataFrame({"date": d, "symbol": "VST", "open": closes, "high": closes, "low": closes, "close": closes,
                       "prev_close": closes, "volume": 1.0, "value": 1.0})
    ev = pd.DataFrame({"symbol": ["VST", "VST"], "ex_date": [d[4], d[9]], "ratio": [1 / 11, 1 / 11]})   # first schedule d[4] was moved to d[9]
    out, applied = adjust_prices(df, ev)
    ok = applied[applied.kind == "split/bonus"]
    assert list(ok.ex_date) == [d[9]], applied                                   # the supported date is kept
    assert list(applied[applied.kind.str.startswith("IGNORED")].ex_date) == [d[4]]
    assert len(suspect_jumps(out)) == 0 and out.close.max() / out.close.min() < 1.1   # series is smooth
    # two genuinely different events months apart are both kept (and each is visible in the price)
    d2 = pd.bdate_range("2026-01-05", periods=120)
    c2 = [100.0] * 20 + [50.0] * 80 + [25.0] * 20
    df2 = pd.DataFrame({"date": d2, "symbol": "VST", "open": c2, "high": c2, "low": c2, "close": c2,
                        "prev_close": c2, "volume": 1.0, "value": 1.0})
    ev2 = pd.DataFrame({"symbol": ["VST", "VST"], "ex_date": [d2[20], d2[100]], "ratio": [0.5, 0.5]})
    _, ap2 = adjust_prices(df2, ev2)
    assert (ap2.kind == "split/bonus").sum() == 2


def test_universe_requires_history():
    from signals import compute_features
    from config import Config
    import numpy as np
    n = 600
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(3)
    px = 100 * np.cumprod(1 + 0.001 + 0.01 * rng.standard_normal((n, 3)), axis=0)
    c = pd.DataFrame(px, index=idx, columns=["OLD1", "OLD2", "IPO"])
    c.loc[idx[:250], "IPO"] = np.nan                                  # listed on day 251
    val = c * 1e8
    val["IPO"] = val["IPO"] * 1000                                    # by far the most traded
    P = Panel(open=c, high=c * 1.01, low=c * 0.99, close=c, volume=c * 0 + 1e6, value=val)
    cfg = Config(universe_size=1, min_price=1.0, min_history=200)
    F = compute_features(P, cfg)
    assert F.in_universe["IPO"].iloc[:449].sum() == 0                 # fewer than 200 rows of history: excluded despite top liquidity
    assert F.in_universe["IPO"].iloc[-1] and not F.in_universe["OLD1"].iloc[-1]
    assert F.in_universe["OLD1"].iloc[300] or F.in_universe["OLD2"].iloc[300]   # an old stock fills the single slot meanwhile


def test_costs_are_tracked_exactly():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100.5), (95, 96, 94, 95), (95, 95, 95, 95), (95, 95, 95, 95)]
    P, F, cfg = make(rows)
    cfg = Config(total_capital=100_000, swing_risk_pct=0.01, swing_max_pos_pct=0.5)       # default costs this time
    eq, tr = run_swing(P, F, cfg, 100_000, 0, len(rows) - 1)
    c = cfg.costs
    qty = int(0.01 * 100_000 / 3.0)
    exp = (qty * 100 * c.slippage + qty * 100 * (1 + c.slippage) * c.buy_fee_rate()
           + qty * 95 * c.slippage + qty * 95 * (1 - c.slippage) * c.sell_fee_rate() + c.dp_charge_sell)
    assert abs(tr.costs.sum() - exp) < 1e-6, (tr.costs.sum(), exp)
    from metrics import trade_stats
    st = trade_stats(tr, "swing")
    assert abs(st["gross_pnl_before_costs"] - st["net_pnl"] - st["costs_paid"]) < 1e-6 and st["costs_paid"] > 100


def test_spearman_cluster_bootstrap_is_wider():
    import numpy as np
    from run_backtest import spearman_boot
    rng = np.random.default_rng(1)
    months = np.repeat(np.arange(30), 20)                     # 30 months x 20 signals
    shared = np.repeat(rng.standard_normal(30), 20)           # signals in a month share the same market effect
    x = shared + 0.5 * rng.standard_normal(600)
    y = shared + 0.5 * rng.standard_normal(600)
    _, lo1, hi1 = spearman_boot(x, y, n=400)
    _, lo2, hi2 = spearman_boot(x, y, n=400, groups=months)
    assert (hi2 - lo2) > (hi1 - lo1)


def test_penny_stock_ticks_are_not_suspect_jumps():
    from build_db import suspect_jumps, adjust_prices
    d = pd.bdate_range("2016-06-13", periods=6)
    def frame(sym, closes):
        return pd.DataFrame({"date": d, "symbol": sym, "open": closes, "high": closes, "low": closes, "close": closes,
                             "prev_close": closes, "volume": 1.0, "value": 1.0})
    px = pd.concat([frame("PENNY", [0.05, 0.10, 0.05, 0.05, 0.10, 0.10]),          # one-tick moves on a 5-paise stock
                    frame("REAL", [100.0, 100.0, 50.0, 50.0, 50.0, 50.0])], ignore_index=True)   # a genuine -50% overnight
    out, _ = adjust_prices(px, None)
    assert list(suspect_jumps(out).symbol) == ["REAL"]


def test_old_file_quirks_stray_byte_and_missing_actions():
    import tempfile, zipfile, os
    from build_db import parse_pr_zip, diagnose_jumps
    import numpy as np
    pd_head = ("MKT,SERIES,SYMBOL,SECURITY,PREV_CL_PR,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,NET_TRDVAL,NET_TRDQTY,"
               "IND_SEC,CORP_IND,TRADES,HI_52_WK,LO_52_WK\n")
    row = b"N,EQ,AAA,ALPHA \xbfINDUSTRIES LTD,100,101,103,99,102,1000000,10000,N, ,500,120,80\n"       # 0xbf is not valid UTF-8
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "PR060418.zip")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("Pd060418.csv", pd_head.encode() + row)          # no corporate-action or meeting file at all
        prices, index, actions, meetings = parse_pr_zip(p)
    assert list(prices.symbol) == ["AAA"] and prices.close.iloc[0] == 102
    assert actions.empty and str(actions["ex_date"].dtype).startswith("datetime64")
    # an empty actions frame mixed with a real one must not break the jump diagnosis
    real = pd.DataFrame({"symbol": ["AAA"], "ex_date": pd.to_datetime(["2016-06-14"]), "purpose": ["FV SPLT FRM RS 10 TO RS 2"],
                         "ratio": [0.2], "dividend": [np.nan]})
    mixed = pd.concat([real, pd.DataFrame(columns=real.columns)], ignore_index=True)       # object-dtype ex_date, as seen on real data
    jumps = pd.DataFrame({"symbol": ["AAA"], "date": pd.to_datetime(["2016-06-14"]), "ret": [-0.8], "gap_days": [1]})
    assert len(diagnose_jumps(jumps, mixed)) == 1


def test_bad_weekday_removed_not_the_rebound_day():
    from build_db import drop_abnormal_days, adjust_prices, suspect_jumps, read_bc_tolerant
    import numpy as np
    rng = np.random.default_rng(5)
    d = pd.bdate_range("2018-01-01", periods=40)
    n = 300
    base = 100 * np.cumprod(1 + 0.004 * rng.standard_normal((40, n)), axis=0)
    base[30:] *= 0.92                                            # a genuine market-wide fall of 8%: must NOT be dropped
    bad = base.copy()
    bad[20, :150] *= 0.1                                         # day 20: half the stocks print at a tenth of their price
    rows = [(d[i], f"S{j}", bad[i, j]) for i in range(40) for j in range(n)]
    px = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    for c in ("open", "high", "low"):
        px[c] = px["close"]
    px["prev_close"], px["volume"], px["value"] = px["close"], 1.0, 1.0
    assert len(suspect_jumps(adjust_prices(px, None)[0])) > 100                      # both the fall and the rebound are flagged
    kept, rep = drop_abnormal_days(px)
    dropped = list(rep[rep.action == "dropped"].date)
    assert dropped == [d[20]], dropped                                             # the bad day, not the rebound day, and not the real fall
    assert len(suspect_jumps(adjust_prices(kept, None)[0])) == 0
    assert read_bc_tolerant(b"").empty                                                # empty corporate-action file


def test_cluster_of_jumps_is_not_auto_adjusted():
    from build_db import diagnose_jumps
    jumps = pd.DataFrame({"symbol": [f"S{i}" for i in range(10)], "date": pd.Timestamp("2018-01-03"), "ret": -0.9, "gap_days": 1})
    acts = pd.DataFrame(columns=["symbol", "ex_date", "purpose", "ratio", "dividend"])
    acts["ex_date"] = pd.to_datetime(acts["ex_date"])
    res = diagnose_jumps(jumps, acts)
    assert (~res["apply"]).all() and res["cause"].str.startswith("cluster").all()


def test_sparse_dates_are_never_judged_abnormal():
    from build_db import drop_abnormal_days
    import numpy as np
    rng = np.random.default_rng(2)
    d = pd.to_datetime(["2024-10-15", "2025-10-15", "2026-10-16"])           # a year apart: big moves are normal
    rows = [(dt, f"S{j}", 100 * np.exp(rng.normal(0, 0.5))) for dt in d for j in range(200)]
    px = pd.DataFrame(rows, columns=["date", "symbol", "close"])
    kept, rep = drop_abnormal_days(px)
    assert len(kept) == len(px) and (rep.empty or (rep.action == "kept").all())


def _one_stock(closes, start="2022-08-01", sym="KRIT"):
    d = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame({"date": d, "symbol": sym, "open": closes, "high": closes, "low": closes, "close": closes,
                       "prev_close": closes, "volume": 1.0, "value": 1.0})
    return d, df


def test_ex_date_snaps_to_the_day_the_price_fell():
    from build_db import adjust_prices, suspect_jumps
    d, df = _one_stock([100.0] * 20 + [20.0] * 10)                                    # real 1:5 split effective on d[20]
    early = pd.DataFrame({"symbol": ["KRIT"], "ex_date": [d[19]], "ratio": [0.2]})   # listed one day early
    out, applied = adjust_prices(df, early)
    assert len(suspect_jumps(out)) == 0 and list(applied.ex_date) == [d[20]] and "moved" in applied.kind.iloc[0]
    exact = pd.DataFrame({"symbol": ["KRIT"], "ex_date": [d[20]], "ratio": [0.2]})
    assert adjust_prices(df, exact)[1].kind.iloc[0] == "split/bonus"                 # correct dates are left alone


def test_same_day_actions_are_combined_and_bogus_listing_ignored():
    from build_db import adjust_prices, suspect_jumps
    # LINC-like: a 10->5 split and a 1:1 bonus on the same day (x0.5 each): the price falls 75%
    d, df = _one_stock([100.0] * 20 + [25.0] * 10, sym="LINC")
    both = pd.DataFrame({"symbol": ["LINC", "LINC"], "ex_date": [d[20], d[20]], "ratio": [0.5, 0.5]})
    out, ap = adjust_prices(df, both)
    assert (ap.kind == "split/bonus").sum() == 2 and len(suspect_jumps(out)) == 0
    # CUB-like: a 1:1 bonus (x0.5) and a 1:10 bonus (x0.909) listed, but the price only falls ~5%
    d, df = _one_stock([100.0] * 20 + [95.0] * 10, sym="CUB")
    cub = pd.DataFrame({"symbol": ["CUB", "CUB"], "ex_date": [d[20], d[20]], "ratio": [0.5, 10 / 11]})
    out, ap = adjust_prices(df, cub)
    assert list(ap[ap.kind == "split/bonus"].ratio.round(3)) == [0.909] and ap.kind.str.startswith("IGNORED").sum() == 1
    assert len(suspect_jumps(out)) == 0
    # VAISHALI-like: split x0.2 and bonus x0.5 on one day, price falls to a tenth: both kept
    d, df = _one_stock([100.0] * 20 + [10.5] * 10, sym="VAI")
    vai = pd.DataFrame({"symbol": ["VAI", "VAI"], "ex_date": [d[20], d[20]], "ratio": [0.2, 0.5]})
    out, ap = adjust_prices(df, vai)
    assert (ap.kind == "split/bonus").sum() == 2 and len(suspect_jumps(out)) == 0


def test_dates_are_never_moved_far_on_illiquid_stocks():
    from build_db import adjust_prices
    # UEL-like: rows are months apart, a 50% fall appears in the data 3 months after the listed ex-date
    d = pd.to_datetime(["2024-09-02", "2024-09-10", "2024-09-18", "2024-09-25", "2024-12-20", "2025-03-10", "2025-03-11", "2025-03-12"])
    closes = [100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    df = pd.DataFrame({"date": d, "symbol": "UEL", "open": closes, "high": closes, "low": closes, "close": closes,
                       "prev_close": closes, "volume": 1.0, "value": 1.0})
    ev = pd.DataFrame({"symbol": ["UEL"], "ex_date": [pd.Timestamp("2024-09-20")], "ratio": [0.5]})
    _, ap = adjust_prices(df, ev)
    assert pd.Timestamp("2025-03-10") not in set(ap.ex_date)                       # not dragged months away
    assert ap.kind.str.startswith("IGNORED").sum() == 1                            # and not applied where the price never shows it


def test_manual_duplicate_of_listed_event_is_not_applied_twice():
    from build_db import merge_manual
    listed = pd.DataFrame({"symbol": ["AAA"], "ex_date": pd.to_datetime(["2020-01-10"]), "ratio": [0.5]})
    man = pd.DataFrame({"symbol": ["AAA", "BBB"], "ex_date": ["2020-01-10", "2020-02-01"], "ratio": [0.5, 0.2]})
    assert list(merge_manual(listed, man).symbol) == ["AAA", "BBB"]


def test_index_trend_rule_has_no_lookahead_and_dodges_a_crash():
    import numpy as np
    from index_trend_test import run_trend
    n = 700
    d = pd.bdate_range("2016-01-01", periods=n)
    path = np.concatenate([np.linspace(100, 200, 350), np.linspace(200, 80, 150), np.linspace(80, 150, 200)])   # up, crash, recovery
    cl = pd.Series(path, index=d)
    ix = pd.DataFrame({"open": cl.shift(1).fillna(cl.iloc[0]), "close": cl})
    eq, bh, cash, switches, invested = run_trend(ix, sma_days=100, cash_yield=0.0, div_yield=0.0)
    dd = lambda s: (s / s.cummax() - 1).min()
    assert dd(eq) > dd(bh) + 0.15 and switches >= 2                    # it steps aside in the crash
    # no look-ahead: a signal that only turns on at the LAST bar cannot have changed anything before that bar
    cl2 = cl.copy(); cl2.iloc[-1] *= 3
    ix2 = pd.DataFrame({"open": cl2.shift(1).fillna(cl2.iloc[0]), "close": cl2})
    eq2 = run_trend(ix2, sma_days=100, cash_yield=0.0, div_yield=0.0)[0]
    assert np.allclose(eq.iloc[:-1].values, eq2.iloc[:-1].values)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
