"""Build the research database from NSE daily files.

Supported inputs in --raw (mixed is fine):
  * PRddmmyy.zip  (NSE "PR" daily package: prices incl. index rows, corporate actions, board meetings)  <- recommended
  * cm_YYYYMMDD.zip (UDiFF bhavcopy) or legacy bhavcopy zips/csv (prices only)

Outputs (next to --out):  daily.*  index_daily.*  delivery.*  dividends.csv  events.csv  applied_adjustments.csv
                          corporate_actions_review.csv  suspect_jumps.csv

IMPORTANT (verified on real PR files): PREV_CL_PR is NOT adjusted on the ex-date of a split/bonus, so prices are
back-adjusted from the corporate-action file (bc) instead. Rights issues, mergers, demergers and capital reductions
are NOT auto-adjusted: they are listed in corporate_actions_review.csv for you to check. Dividends are listed in dividends.csv and adjusted only with --adjust-dividends.
Add fixes in data/manual_adjustments.csv  (symbol,ex_date,ratio)  ratio = new price basis / old (2:1 split -> 0.5).
"""
import argparse, csv, glob, io, itertools, os, re, zipfile
import numpy as np
import pandas as pd
from store import save_df

KEEP = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "value"]
PRICE_NUM = ["open", "high", "low", "close", "prev_close", "volume", "value"]


# ------------------------------------------------------------------ helpers
def file_date(name: str):
    m = re.search(r"(\d{2})(\d{2})(\d{4}|\d{2})\.\w+$", os.path.basename(name))
    if not m:
        return None
    dd, mm, yy = m.groups()
    yy = int(yy)
    yy = yy + 2000 if yy < 100 else yy
    try:
        return pd.Timestamp(year=yy, month=int(mm), day=int(dd))
    except ValueError:
        return None


SPLIT_RE = re.compile(r"(SPL\w*|CONSOL\w*)\s*(?:FRM|FROM)\s*R[SE]\.?\s*([\d.]+)\s*(?:/-)?\s*(?:PER\s*SH\w*)?\s*TO\s*R[SE]\.?\s*([\d.]+)", re.I)
DIV_RE = re.compile(r"DIV\w*\s*-?\s*R[SE]\.?\s*([\d.]+)", re.I)


def parse_dividend(purpose: str):
    """Cash dividend per share (sum if several are listed), else None."""
    if not isinstance(purpose, str):
        return None
    amts = [float(x) for x in DIV_RE.findall(purpose) if x.replace(".", "", 1).isdigit()]
    return sum(amts) if amts else None


BONUS_RE = re.compile(r"BONUS\s*(\d+)\s*:\s*(\d+)", re.I)


def parse_ratio(purpose: str):
    """Price-basis ratio (new/old) for splits, consolidations and bonuses; None if not auto-adjustable."""
    if not isinstance(purpose, str):
        return None
    m = SPLIT_RE.search(purpose)
    if m:
        old, new = float(m.group(2)), float(m.group(3))
        return new / old if old > 0 and new > 0 else None
    m = BONUS_RE.search(purpose)
    if m:                                   # A new shares for every B held
        a, b = float(m.group(1)), float(m.group(2))
        return b / (a + b) if (a + b) > 0 and b > 0 else None
    return None


BM_RE = re.compile(r"^(?P<company>.+?)\s+(?P<symbol>[A-Z0-9&\-]+)\s*:\s*(?P<date>\d{2}-[A-Za-z]{3}-\d{4})\s*:\s*(?P<purpose>.*)$")


def parse_board_meetings(text: str) -> pd.DataFrame:
    rows, cur = [], None
    for line in text.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        m = BM_RE.match(line)
        if m:
            cur = {"symbol": m["symbol"], "date": m["date"], "purpose": m["purpose"]}
            rows.append(cur)
        elif cur is not None:
            cur["purpose"] += " " + line      # purpose text continues on the next line
    df = pd.DataFrame(rows, columns=["symbol", "date", "purpose"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    return df.dropna(subset=["date"])


def drop_abnormal_days(prices: pd.DataFrame, max_moved_pct: float = 2.0, min_moved: int = 15,
                       drop_all_weekends: bool = False, max_drops: int = 60):
    """Remove days whose prices do not fit the days around them (weekday or weekend): a day on which more than `max_moved_pct` percent of
    stocks 'moved' more than 25% against the previous trading day (and at least `min_moved` stocks did). Real examples: Sat 6 Apr 2024
    (9.8% of stocks, CANBK 109.85 between 615.00 and 612.45). A bad day appears as a fall followed by a rebound, so each round picks
    the candidate whose removal leaves the following day looking normal (the middle of the 'V'), not the rebound day.
    Genuine market-wide moves are not caught: even in March 2020 only a tiny share of stocks moved >25% in a day.
    Returns (kept rows, report of every weekend date plus every dropped weekday)."""
    close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    dates, C = close.index, close.values.astype(float)
    T = len(dates)

    def moved(cur, prev):
        ok = np.isfinite(cur) & np.isfinite(prev) & (prev > 0)
        n = int(ok.sum())
        m = int((np.abs(cur[ok] / prev[ok] - 1) > 0.25).sum())
        return m, 100.0 * m / max(n, 1)

    def gap_ok(i, j):                                  # only compare days that are close together (files can be missing)
        return (dates[i] - dates[j]).days <= 5

    mv, pct = np.zeros(T, dtype=int), np.zeros(T)
    for i in range(1, T):
        if gap_ok(i, i - 1):
            mv[i], pct[i] = moved(C[i], C[i - 1])
    pct0 = pct.copy()
    active, dropped = list(range(T)), []
    for _ in range(max_drops):
        flagged = sorted(((pct[active[k]], k) for k in range(1, len(active))
                          if pct[active[k]] > max_moved_pct and mv[active[k]] >= min_moved), reverse=True)
        if not flagged:
            break
        best = None
        for p, k in flagged[:2]:
            after = (moved(C[active[k + 1]], C[active[k - 1]])[1]
                     if k + 1 < len(active) and gap_ok(active[k + 1], active[k - 1]) else 0.0)
            score = (round(after, 6), -p)
            if best is None or score < best[0]:
                best = (score, k)
        k = best[1]
        dropped.append(active[k])
        if k + 1 < len(active):
            nxt = active[k + 1]
            mv[nxt], pct[nxt] = moved(C[nxt], C[active[k - 1]]) if gap_ok(nxt, active[k - 1]) else (0, 0.0)
        del active[k]
    drop_dates = {dates[i] for i in dropped}
    if drop_all_weekends:
        drop_dates |= {d for d in dates if d.dayofweek >= 5}
    counts = prices.groupby("date").size()
    rows = [{"date": d, "weekday": d.day_name()[:3], "symbols": int(counts.get(d, 0)), "pct_moved_over_25": round(pct0[i], 1),
             "action": "dropped" if d in drop_dates else "kept"}
            for i, d in enumerate(dates) if d.dayofweek >= 5 or d in drop_dates]
    rep = pd.DataFrame(rows, columns=["date", "weekday", "symbols", "pct_moved_over_25", "action"])
    return prices[~prices["date"].isin(drop_dates)], rep


def parse_ex_dates(s: pd.Series) -> pd.Series:
    """NSE writes dates as YYYY-MM-DD (newer files), DD/MM/YYYY (older files) or DD-MON-YYYY. Parsed per format on purpose:
    'dayfirst' would silently swap day and month in ISO dates such as 2026-10-06."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt_ in ("%d/%m/%Y", "%d-%b-%Y", "%d-%m-%Y"):
        m = out.isna()
        if m.any():
            out[m] = pd.to_datetime(s[m], format=fmt_, errors="coerce")
    return out


PARSE_WARNINGS = []
NAMES = {}
FUND_RE = re.compile(r"\bETFS?\b|\bLIQUID\b|\bFUND\b|\bGILT\b|AMC\s*-\s|\bNETF", re.I)   # NSE abbreviates fund names, e.g. "ZERODHAAMC - LIQUIDCAS"
FUND_SHAPE = re.compile(r"^[A-Z0-9&]+\s*-\s*[A-Z0-9&]+$")      # "<AMC> - <FUNDSYMBOL>", e.g. "BFAM - NIFTYBETF", "TATAAML-TATAGOLD"

FUND_SHAPE2 = re.compile(r"^[A-Z0-9]+AMC\s+[A-Z0-9]+$")          # "HDFCAMC HDFCPSUBK" (two words, first ends in AMC)


def is_fund_name(name: str) -> bool:
    s = str(name).strip().upper()
    return bool(FUND_RE.search(s) or FUND_SHAPE.match(s) or FUND_SHAPE2.match(s))
DATE_OR_BLANK = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-[A-Za-z]{3}-\d{4})?\s*$")


def read_bc_tolerant(raw: bytes) -> pd.DataFrame:
    """Corporate-action file with stray commas inside company names or purposes (a plain CSV read fails on those).
    Rows with extra fields are repaired by checking which merge leaves the six date columns looking like dates."""
    reader = csv.reader(io.StringIO(raw.decode("utf-8", "replace")), skipinitialspace=True)
    header = next(reader, None)
    if header is None:                                   # an empty file (seen in PR100122.zip)
        return pd.DataFrame(columns=["SERIES", "SYMBOL", "SECURITY", "RECORD_DT", "BC_STRT_DT", "BC_END_DT", "EX_DT", "ND_STRT_DT", "ND_END_DT", "PURPOSE"])
    header = [h.strip() for h in header]
    n, rows, bad = len(header), [], 0
    for r in reader:
        if not r:
            continue
        if len(r) == n:
            rows.append(r)
        elif len(r) > n:
            extra = len(r) - n
            in_purpose = r[:n - 1] + [",".join(r[n - 1:])]                    # comma inside the last field
            in_name = r[:2] + [",".join(r[2:3 + extra])] + r[3 + extra:]       # comma inside SECURITY
            if all(DATE_OR_BLANK.match(x) for x in in_purpose[3:9]):
                rows.append(in_purpose)
            elif all(DATE_OR_BLANK.match(x) for x in in_name[3:9]):
                rows.append(in_name)
            else:
                bad += 1
        else:
            bad += 1
    if bad:
        PARSE_WARNINGS.append(f"{bad} unreadable corporate-action row(s) ignored")
    return pd.DataFrame(rows, columns=header)


def _member(z: zipfile.ZipFile, prefix: str):
    pat = re.compile(rf"^{prefix}\d{{6,8}}\.(csv|txt)$", re.I)
    for n in z.namelist():
        if pat.match(os.path.basename(n)):
            return n
    return None


# ------------------------------------------------------------------ PR package
def parse_pr_zip(path: str):
    d = file_date(path)
    if d is None:
        raise ValueError("cannot read date from file name")
    with zipfile.ZipFile(path) as z:
        n = _member(z, "pd")
        if n is None:
            raise ValueError("no pd file inside")
        raw = pd.read_csv(z.open(n), skipinitialspace=True, dtype={"SERIES": str, "SYMBOL": str, "SECURITY": str},
                          encoding="utf-8", encoding_errors="replace")      # a few old files contain stray non-UTF-8 bytes in names
        raw.columns = [c.strip() for c in raw.columns]
        need = ["SERIES", "SYMBOL", "SECURITY", "PREV_CL_PR", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "NET_TRDVAL", "NET_TRDQTY"]
        miss = [c for c in need if c not in raw.columns]
        if miss:
            raise ValueError(f"pd file missing columns {miss}; found {list(raw.columns)}")
        sym = raw["SYMBOL"].fillna("").str.strip()
        ser = raw["SERIES"].fillna("").str.strip()
        m = {"PREV_CL_PR": "prev_close", "OPEN_PRICE": "open", "HIGH_PRICE": "high", "LOW_PRICE": "low",
             "CLOSE_PRICE": "close", "NET_TRDVAL": "value", "NET_TRDQTY": "volume"}
        eq = raw[(ser == "EQ") & (sym != "")].rename(columns=m).copy()
        eq["symbol"] = sym[eq.index]
        NAMES.update(zip(eq["symbol"], raw.loc[eq.index, "SECURITY"].fillna("").astype(str).str.strip()))
        eq["date"] = d
        for c in PRICE_NUM:
            eq[c] = pd.to_numeric(eq[c], errors="coerce")
        prices = eq[KEEP]
        idx = raw[sym == ""].rename(columns=m).copy()
        idx["name"] = idx["SECURITY"].fillna("").str.strip()
        idx["date"] = d
        for c in ["open", "high", "low", "close"]:
            idx[c] = pd.to_numeric(idx[c], errors="coerce")
        index = idx[["date", "name", "open", "high", "low", "close"]]

        actions = pd.DataFrame({"symbol": pd.Series(dtype="object"), "ex_date": pd.Series(dtype="datetime64[ns]"),
                                "purpose": pd.Series(dtype="object"), "ratio": pd.Series(dtype="float64"),
                                "dividend": pd.Series(dtype="float64")})
        n = _member(z, "bc")
        if n:
            try:
                bc = read_bc_tolerant(z.read(n))
                bc.columns = [c.strip() for c in bc.columns]
                bc = bc[bc["SERIES"].fillna("").str.strip() == "EQ"]
                actions = pd.DataFrame({"symbol": bc["SYMBOL"].str.strip(),
                                        "ex_date": parse_ex_dates(bc["EX_DT"]),
                                        "purpose": bc["PURPOSE"].fillna("").str.strip()})
                actions["ratio"] = actions["purpose"].map(parse_ratio)
                actions["dividend"] = actions["purpose"].map(parse_dividend)
            except Exception as e:                       # never lose a day of prices because of a side file
                PARSE_WARNINGS.append(f"{os.path.basename(path)}: corporate actions ignored ({type(e).__name__}: {e})")
        meetings = pd.DataFrame(columns=["symbol", "date", "purpose"])
        n = _member(z, "bm")
        if n:
            try:
                meetings = parse_board_meetings(z.read(n).decode("utf-8", "replace"))
            except Exception as e:
                PARSE_WARNINGS.append(f"{os.path.basename(path)}: board meetings ignored ({type(e).__name__}: {e})")
    return prices, index, actions, meetings


# ------------------------------------------------------------------ UDiFF / legacy bhavcopy (prices only)
UDIFF = {"TradDt": "date", "TckrSymb": "symbol", "SctySrs": "series", "OpnPric": "open", "HghPric": "high",
         "LwPric": "low", "ClsPric": "close", "PrvsClsgPric": "prev_close", "TtlTradgVol": "volume", "TtlTrfVal": "value"}
LEGACY = {"TIMESTAMP": "date", "SYMBOL": "symbol", "SERIES": "series", "OPEN": "open", "HIGH": "high", "LOW": "low",
          "CLOSE": "close", "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume", "TOTTRDVAL": "value"}


def parse_bhavcopy(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    m = UDIFF if "TckrSymb" in raw.columns else LEGACY if "SYMBOL" in raw.columns else None
    if m is None:
        raise ValueError(f"{path}: unknown layout, columns: {list(raw.columns)}")
    miss = [k for k in m if k not in raw.columns]
    if miss:
        raise ValueError(f"{path}: missing columns {miss}")
    df = raw[list(m)].rename(columns=m)
    df = df[df["series"].astype(str).str.strip() == "EQ"].copy()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    for c in PRICE_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[KEEP]


# ------------------------------------------------------------------ delivery (sec_bhavdata_full_DDMMYYYY.csv)
def parse_delivery(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, skipinitialspace=True, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    need = ["SYMBOL", "SERIES", "DATE1", "DELIV_QTY", "DELIV_PER"]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise ValueError(f"missing columns {miss}; found {list(raw.columns)}")
    df = raw[raw["SERIES"].str.strip() == "EQ"]
    return pd.DataFrame({"date": pd.to_datetime(df["DATE1"].str.strip(), format="%d-%b-%Y", errors="coerce"),
                         "symbol": df["SYMBOL"].str.strip(),
                         "deliv_qty": pd.to_numeric(df["DELIV_QTY"].str.strip(), errors="coerce"),
                         "deliv_per": pd.to_numeric(df["DELIV_PER"].str.strip(), errors="coerce")}).dropna(subset=["date"])


# ------------------------------------------------------------------ adjustment
def resolve_reschedules(ev: pd.DataFrame, groups, dates, raw_close, window_days: int = 45):
    """The same split/bonus (same symbol and ratio) listed with several DIFFERENT ex-dates within `window_days` is a rescheduled action:
    only one date is real. Keep the date whose raw price move best matches the ratio (ties: the latest date). Several entries on the
    SAME date are separate actions (e.g. a split and a bonus with the same ratio) and are all kept."""
    keep, dropped = [], []
    for (sym, ratio), g in ev.sort_values("ex_date").groupby(["symbol", "ratio"]):
        rows = list(g.itertuples(index=False))
        clusters, cur = [], [rows[0]]
        for r in rows[1:]:
            if (r.ex_date - cur[-1].ex_date).days <= window_days:
                cur.append(r)
            else:
                clusters.append(cur)
                cur = [r]
        clusters.append(cur)
        pos = groups.get(sym)

        def score(d):
            if pos is None:
                return 9.0
            k = int(np.searchsorted(dates[pos], np.datetime64(d), side="left"))
            if k <= 0 or k >= len(pos):
                return 9.0
            m = raw_close[pos[k]] / raw_close[pos[k - 1]]
            return abs(float(np.log(m / ratio))) if m > 0 else 9.0
        for cl in clusters:
            uniq = sorted({r.ex_date for r in cl})
            if len(uniq) == 1:
                keep.extend(cl)
                continue
            best = min(uniq, key=lambda d: (round(score(d), 6), -d.value))
            keep.extend(r for r in cl if r.ex_date == best)
            dropped.extend(r for r in cl if r.ex_date != best)
    cols = ["symbol", "ex_date", "ratio"]
    return (pd.DataFrame(keep, columns=cols) if keep else pd.DataFrame(columns=cols),
            pd.DataFrame(dropped, columns=cols) if dropped else pd.DataFrame(columns=cols))


def refine_events(ev: pd.DataFrame, groups, dates, raw_close, max_days: int = 7, max_rows: int = 5):
    """Check large listed adjustments (ratio below 0.8 or above 1.25) against the raw price and correct the listing, per (symbol, ex-date):
      * the actions of one day are judged TOGETHER (a 1:1 bonus and a 10->5 split on the same day make x0.25);
      * if the listed day shows the move (within a factor of 1.65) keep the best-fitting combination and ignore listed actions the price
        does not show (e.g. a bogus 1:1 bonus listed next to a real 1:10 bonus);
      * if the listed day does not, but a day within `max_days` calendar days (and `max_rows` rows) clearly does, move the actions there;
      * if no day in that window shows any large ratio, ignore the large ones (a 20%+ adjustment must be visible in the price).
    Small ratios (0.8 to 1.25) cannot be verified from prices and are always kept.
    Returns (events, moved[(symbol, listed, new)], ignored[(symbol, ex_date, ratio)])."""
    kept, moved, ignored = [], [], []
    for (sym, ex), g in ev.groupby(["symbol", "ex_date"], sort=False):
        rows = list(g.itertuples(index=False))
        large = [r for r in rows if not (0.8 < r.ratio < 1.25)]
        small = [r for r in rows if 0.8 < r.ratio < 1.25]
        pos = groups.get(sym)
        k = int(np.searchsorted(dates[pos], np.datetime64(ex), side="left")) if pos is not None else -1
        if not large or pos is None or len(pos) < 3 or len(large) > 5 or k < 1 or k >= len(pos):
            kept.extend((r.symbol, r.ex_date, r.ratio) for r in rows)
            continue
        small_prod = float(np.prod([r.ratio for r in small])) if small else 1.0
        subsets = [c for n in range(len(large), 0, -1) for c in itertools.combinations(large, n)]   # larger first: ties keep more actions
        prods = [float(np.prod([r.ratio for r in c])) * small_prod for c in subsets]

        def best_at(j):
            m = raw_close[pos[j]] / raw_close[pos[j - 1]]
            if not m > 0:
                return 9.0, 0
            return min((abs(float(np.log(m / p))), i) for i, p in enumerate(prods))
        s_listed, i_listed = best_at(k)
        chosen, new_ex = None, ex
        if s_listed <= 0.5:
            chosen = subsets[i_listed]
        else:
            ex_ts = pd.Timestamp(ex)
            cand = [j for j in range(max(1, k - max_rows), min(len(pos), k + max_rows + 1))
                    if j != k and abs((pd.Timestamp(dates[pos[j]]) - ex_ts).days) <= max_days]
            if cand:
                (s_nb, i_nb), j_nb = min((best_at(j), j) for j in cand)
                if s_nb <= 0.35 and s_nb + 0.25 < s_listed:
                    chosen, new_ex = subsets[i_nb], pd.Timestamp(dates[pos[j_nb]])
                    moved.append((sym, ex, new_ex))
        for r in (list(chosen) if chosen else []) + small:
            kept.append((r.symbol, new_ex, r.ratio))
        for r in large:
            if chosen is None or r not in chosen:
                ignored.append((r.symbol, r.ex_date, r.ratio))
    return pd.DataFrame(kept, columns=["symbol", "ex_date", "ratio"]), moved, ignored


def adjust_prices(df: pd.DataFrame, events: pd.DataFrame, dividends: pd.DataFrame = None):
    """Back-adjust O/H/L/C (and volume) for events (symbol, ex_date, ratio). Only events on or before the last data date apply.
    If `dividends` (symbol, ex_date, dividend) is given, prices are also adjusted for cash dividends (total-return style)."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    raw_close = df["close"].values.copy()
    factor = np.ones(len(df))
    applied = []
    last = df["date"].max()
    ev = events.dropna(subset=["ratio"]) if events is not None and len(events) else pd.DataFrame(columns=["symbol", "ex_date", "ratio"])
    ev = ev[ev["ex_date"].notna() & (ev["ex_date"] <= last)]
    groups = df.groupby("symbol", sort=False).indices
    dates = df["date"].values
    ev, dropped = resolve_reschedules(ev[["symbol", "ex_date", "ratio"]], groups, dates, raw_close)
    ev, snapped, not_seen = refine_events(ev, groups, dates, raw_close)
    moved_from = {(s, new): old for s, old, new in snapped}
    for s_, ex_, r_ in not_seen:
        applied.append({"symbol": s_, "ex_date": ex_, "ratio": r_, "rows_adjusted": 0, "kind": "IGNORED: ratio not visible in the price"})
    for r in dropped.itertuples(index=False):
        applied.append({"symbol": r.symbol, "ex_date": r.ex_date, "ratio": r.ratio, "rows_adjusted": 0, "kind": "IGNORED: rescheduled duplicate"})
    for sym, ex, ratio in zip(ev["symbol"], ev["ex_date"], ev["ratio"]):
        pos = groups.get(sym)
        if pos is None:
            continue
        k = int(np.searchsorted(dates[pos], np.datetime64(ex), side="left"))
        if k > 0:
            factor[pos[:k]] *= ratio
        kind = "split/bonus" if (sym, ex) not in moved_from else f"split/bonus (listed {moved_from[(sym, ex)].date()}, moved to price-supported date)"
        applied.append({"symbol": sym, "ex_date": ex, "ratio": ratio, "rows_adjusted": k, "kind": kind})
    if dividends is not None and len(dividends):
        dv = dividends.dropna(subset=["dividend", "ex_date"])
        dv = dv[dv["ex_date"] <= last].drop_duplicates(["symbol", "ex_date", "dividend"])
        for sym, ex, amt in zip(dv["symbol"], dv["ex_date"], dv["dividend"]):
            pos = groups.get(sym)
            if pos is None:
                continue
            k = int(np.searchsorted(dates[pos], np.datetime64(ex), side="left"))
            if k > 0 and raw_close[pos[k - 1]] > 0 and 0 < amt / raw_close[pos[k - 1]] < 0.5:
                r = 1.0 - amt / raw_close[pos[k - 1]]
                factor[pos[:k]] *= r
                applied.append({"symbol": sym, "ex_date": ex, "ratio": r, "rows_adjusted": k, "kind": "dividend"})
    df["factor"] = factor
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] * df["factor"]
    df["volume"] = df["volume"] / df["factor"]
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "value", "factor"]], pd.DataFrame(applied)


def merge_manual(listed: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    """Listed actions plus manual/continuity fixes; a manual row identical to a listed one (same symbol, date, ratio) is not applied twice."""
    have = set(zip(listed["symbol"], pd.to_datetime(listed["ex_date"]), listed["ratio"].astype(float).round(6)))
    m = man.copy()
    m["ex_date"] = pd.to_datetime(m["ex_date"])
    keep = [(s, d, round(float(r), 6)) not in have for s, d, r in zip(m["symbol"], m["ex_date"], m["ratio"])]
    return pd.concat([listed[["symbol", "ex_date", "ratio"]], m.loc[keep, ["symbol", "ex_date", "ratio"]]], ignore_index=True)


def suspect_jumps(df: pd.DataFrame, thresh: float = 0.40, max_gap_days: int = 5, min_raw_price: float = 5.0) -> pd.DataFrame:
    """Large overnight moves on adjacent trading days that no adjustment explains. Stocks priced below `min_raw_price` rupees
    BEFORE the move are ignored: a one-tick move on a Rs 0.05 stock is a 100% 'jump' (e.g. Rs 0.05 -> Rs 0.10) and is not a corporate action."""
    d = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    prev_c = d.groupby("symbol")["close"].shift(1)
    gap = (d["date"] - d.groupby("symbol")["date"].shift(1)).dt.days
    ret = d["close"] / prev_c - 1
    raw = d["close"] / d["factor"] if "factor" in d.columns else d["close"]
    prev_raw = raw.groupby(d["symbol"]).shift(1)
    out = d.loc[(ret.abs() > thresh) & (gap <= max_gap_days) & (prev_raw >= min_raw_price), ["symbol", "date"]].copy()
    out["ret"] = ret[out.index]
    out["gap_days"] = gap[out.index]
    return out.reset_index(drop=True)


NICE_RATIOS = [0.05, 0.1, 0.2, 0.25, 1 / 3, 0.4, 0.5, 0.6, 2 / 3, 0.75, 0.8, 1.25, 1.5, 2.0, 2.5, 4.0, 5.0, 10.0]
NON_PRICE_ACTIONS = r"DIV|INTEREST|REDEMPTION|PAYMENT|AGM|EGM|MATURITY|ANNUAL GEN"


LISTED = "split/bonus listed nearby but jump remains: check ex-date/ratio"
WORDING = "split/bonus wording unparsed: ratio taken from the price move"
OTHER_ACTION = "corporate action not auto-adjusted (merger/demerger/rights/etc.)"
MISSING = "probable split/bonus missing from the file"
NOTHING = "no corporate action found: genuine move, relisting or missing data"


def diagnose_jumps(jumps: pd.DataFrame, actions: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Explain each suspect jump and say whether a continuity adjustment is safe ('apply').
    Exchange circuit limits (normally 5-20% a day) make an overnight fall of 40%+ in a traded stock almost certainly a corporate
    action (split, bonus, demerger, merger...) or bad data, so such falls are adjusted for continuity at the implied ratio.
    Jumps UP of 40%+ are adjusted only when they match a typical split/consolidation ratio. Jumps next to a listed split/bonus
    that has a parsed ratio are NOT auto-adjusted (the ex-date or ratio needs a human look)."""
    if jumps.empty:
        return jumps.assign(cause=[], nearby_action=[], suggested_ratio=[], apply=[])
    act = actions.copy()
    act["ex_date"] = pd.to_datetime(act["ex_date"], errors="coerce")
    act = act.dropna(subset=["ex_date"])
    act = act[~act["purpose"].str.contains(NON_PRICE_ACTIONS, case=False, na=False)]
    by_sym = {s: g for s, g in act.groupby("symbol")}
    cause, near_txt, sugg, apply = [], [], [], []
    for r in jumps.itertuples():
        implied = 1.0 + r.ret
        nice = min(NICE_RATIOS, key=lambda x: abs(x - implied) / x)
        nice_ok = abs(nice - implied) <= max(0.03 * nice, 0.015)
        g = by_sym.get(r.symbol)
        near = g[(g["ex_date"] - r.date).abs() <= pd.Timedelta(days=window)] if g is not None else None
        if near is not None and len(near):
            p = near.iloc[0]
            near_txt.append(p["purpose"])
            if pd.notna(p["ratio"]):
                c = LISTED
            elif re.search(r"SPLT|SPLIT|SUB-?DIV|BONUS|CONSOL", p["purpose"], re.I):
                c = WORDING
            else:
                c = OTHER_ACTION
        else:
            near_txt.append("")
            c = MISSING if nice_ok else NOTHING
        cause.append(c)
        sugg.append(round(nice if nice_ok else implied, 4))
        apply.append(c != LISTED and (r.ret <= -0.40 or (r.ret >= 0.40 and nice_ok)))
    res = jumps.assign(cause=cause, nearby_action=near_txt, suggested_ratio=sugg, apply=apply)
    n_same_day = res.groupby("date")["symbol"].transform("size")
    cl = n_same_day >= 8                      # many unrelated stocks jumping on one date is a data problem, never a set of corporate actions
    res.loc[cl, "cause"] = "cluster: many jumps on this date, probably bad data for the day (not adjusted)"
    res.loc[cl, "apply"] = False
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/daily")
    ap.add_argument("--apply-suggested-fixes", "--apply-probable-splits", dest="apply_fixes", action="store_true",
                    help="continuity-adjust the suspect jumps marked apply=True (see suspect_jumps.csv): appended to manual_adjustments.csv, then rebuild")
    ap.add_argument("--keep-weekend-sessions", "--keep-abnormal-days", dest="keep_weekend_sessions", action="store_true", help="never drop abnormal days")
    ap.add_argument("--drop-all-weekend-sessions", action="store_true", help="drop every weekend session (default: only abnormal ones)")
    ap.add_argument("--adjust-dividends", action="store_true", help="also adjust prices for cash dividends (default: off, split/bonus only)")
    a = ap.parse_args()
    files = sorted(set(glob.glob(os.path.join(a.raw, "*.zip")) + glob.glob(os.path.join(a.raw, "*.csv"))))
    P, I, A, M, D = [], [], [], [], []
    bad = 0
    for k, f in enumerate(files, 1):
        base = os.path.basename(f).lower()
        try:
            if base.startswith("sec_bhavdata_full"):
                D.append(parse_delivery(f))
            elif base.startswith("pr") and base.endswith(".zip"):
                p, i, ac, m = parse_pr_zip(f)
                P.append(p); I.append(i); A.append(ac); M.append(m)
            else:
                P.append(parse_bhavcopy(f))
        except Exception as e:
            bad += 1
            print("SKIPPED", f, "->", e)
        if k % 250 == 0:
            print(f"  parsed {k}/{len(files)}")
    if not P:
        raise SystemExit("No price files parsed")
    prices = pd.concat(P, ignore_index=True).dropna(subset=["date", "close"]).drop_duplicates(["date", "symbol"], keep="last")
    base_dir = os.path.dirname(a.out) or "."
    os.makedirs(base_dir, exist_ok=True)
    dropped_dates = set()
    if not a.keep_weekend_sessions:
        prices, wk_report = drop_abnormal_days(prices, drop_all_weekends=a.drop_all_weekend_sessions)
        if len(wk_report):
            dropped_dates = set(wk_report.loc[wk_report["action"] == "dropped", "date"])
            wk_report.to_csv(os.path.join(base_dir, "weekend_sessions.csv"), index=False)
            print("\nWeekend sessions and abnormal days (a day is dropped when >2% of its stocks moved >25% vs the previous day and its removal fixes the next day):")
            print(wk_report.assign(date=wk_report["date"].dt.date).to_string(index=False))
    A = [x for x in A if len(x)]
    actions = (pd.concat(A, ignore_index=True) if A else pd.DataFrame(columns=["symbol", "ex_date", "purpose", "ratio", "dividend"]))
    actions["ex_date"] = pd.to_datetime(actions["ex_date"], errors="coerce")
    for c_ in ("ratio", "dividend"):
        actions[c_] = pd.to_numeric(actions[c_], errors="coerce")
    actions = actions.drop_duplicates(["symbol", "ex_date", "purpose"])
    ev = actions[["symbol", "ex_date", "ratio"]].copy()
    divs = actions.dropna(subset=["dividend", "ex_date"])[["symbol", "ex_date", "dividend", "purpose"]].sort_values(["ex_date", "symbol"])
    mp = os.path.join(base_dir, "manual_adjustments.csv")
    if os.path.exists(mp):
        man = pd.read_csv(mp)
        man["ex_date"] = pd.to_datetime(man["ex_date"])
        ev = merge_manual(ev, man)
    out, applied = adjust_prices(prices, ev, divs if a.adjust_dividends else None)
    actions.to_csv(os.path.join(base_dir, "corporate_actions_all.csv"), index=False)
    names = pd.DataFrame({"symbol": list(NAMES), "name": list(NAMES.values())})
    names["is_fund"] = names["name"].map(is_fund_name)
    names.to_csv(os.path.join(base_dir, "securities.csv"), index=False)
    funds = set(names.loc[names["is_fund"], "symbol"])
    print(f"\nETFs/funds recognised by name: {len(funds)} symbols (excluded from the stock universe; list in securities.csv). Examples: "
          + ", ".join(sorted(funds)[:8]))
    recent = out[out["date"] >= out["date"].max() - pd.Timedelta(days=60)]
    top = recent.groupby("symbol")["value"].mean().sort_values(ascending=False)
    top = top[~top.index.isin(funds)].head(60)
    nm = dict(zip(names["symbol"], names["name"]))
    print("\nTop 60 most-traded symbols NOT treated as funds (check that no ETF/fund slipped through; the stock universe is built from these):")
    print(", ".join(f"{s} [{nm.get(s, '')[:22]}]" for s in top.index))
    jumps = diagnose_jumps(suspect_jumps(out[~out["symbol"].isin(funds)]), actions)
    if a.apply_fixes:
        pick = jumps[jumps["apply"]]
        if len(pick):
            add = pd.DataFrame({"symbol": pick["symbol"], "ex_date": pick["date"].dt.strftime("%Y-%m-%d"), "ratio": pick["suggested_ratio"]})
            old_m = pd.read_csv(mp) if os.path.exists(mp) else pd.DataFrame(columns=["symbol", "ex_date", "ratio"])
            allm = pd.concat([old_m, add], ignore_index=True).drop_duplicates(["symbol", "ex_date"], keep="first")
            allm.to_csv(mp, index=False)
            print(f"Added {len(add)} continuity adjustments to {mp} (review that file); rebuilding with them applied")
            man = allm.copy()
            man["ex_date"] = pd.to_datetime(man["ex_date"])
            ev = merge_manual(actions[["symbol", "ex_date", "ratio"]], man)
            out, applied = adjust_prices(prices, ev, divs if a.adjust_dividends else None)
            jumps = diagnose_jumps(suspect_jumps(out[~out["symbol"].isin(funds)]), actions)
    divs.to_csv(os.path.join(base_dir, "dividends.csv"), index=False)
    if D:
        dl = pd.concat(D, ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
        dl = dl[~dl["date"].isin(dropped_dates)]
        save_df(dl, os.path.join(base_dir, "delivery"))
    path = save_df(out, a.out)
    if I:
        ix = pd.concat(I, ignore_index=True).drop_duplicates(["date", "name"], keep="last")
        ix = ix[~ix["date"].isin(dropped_dates)]
        save_df(ix, os.path.join(base_dir, "index_daily"))
    if M:
        mt = pd.concat(M, ignore_index=True).drop_duplicates(["symbol", "date"])
        mt[mt["purpose"].str.contains("result", case=False, na=False)][["symbol", "date"]].sort_values(["date", "symbol"]).to_csv(os.path.join(base_dir, "events.csv"), index=False)
    applied.to_csv(os.path.join(base_dir, "applied_adjustments.csv"), index=False)
    fx = actions[actions["ratio"].isna() & actions["symbol"].isin(out["symbol"].unique())
                 & actions["purpose"].str.contains(r"MERGER|DEMERGER|CAPITAL RED|RIGHT|SCHEME|AMALG|BUYBACK|CONSOL|SPLT|SPLIT|SUB-?DIV|BONUS", case=False, na=False)]
    fx.to_csv(os.path.join(base_dir, "corporate_actions_review.csv"), index=False)
    if len(fx):
        pat = fx["purpose"].str.replace(r"[0-9.]+", "#", regex=True).str.slice(0, 60).value_counts().head(12)
        print("\nPrice-affecting actions the parser could not turn into a ratio (wording patterns, # = number):")
        print(pat.to_string())
    jumps.to_csv(os.path.join(base_dir, "suspect_jumps.csv"), index=False)
    print(f"{len(out):,} rows, {out.symbol.nunique()} symbols, {out.date.min().date()} -> {out.date.max().date()} saved to {path}")
    print(f"files skipped: {bad} | adjustments applied: {len(applied)} | needing manual review: {len(fx)} | suspect jumps: {len(jumps)}")
    for w in PARSE_WARNINGS[:10]:
        print("WARNING:", w)
    if len(PARSE_WARNINGS) > 10:
        print(f"WARNING: {len(PARSE_WARNINGS) - 10} more warnings")
    if len(jumps):
        print("\nSuspect jumps by likely cause (see data/suspect_jumps.csv):")
        print(jumps["cause"].value_counts().to_string())
        print("Largest moves:")
        print(jumps.reindex(jumps["ret"].abs().sort_values(ascending=False).index).head(12)[["symbol", "date", "ret", "cause", "suggested_ratio"]]
              .assign(date=lambda x: x["date"].dt.date, ret=lambda x: x["ret"].round(3)).to_string(index=False))
        print()
    print(f"index rows: {sum(len(x) for x in I):,} | results dates: {len(mt) if M else 0} | dividends listed: {len(divs)} | delivery rows: {sum(len(x) for x in D):,}")


if __name__ == "__main__":
    main()
