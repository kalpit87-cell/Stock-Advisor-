# nse_trader - offline NSE research, backtest and daily-signal toolkit

Decision-support only. Long-only delivery trading in liquid NSE equities. Not investment advice.
Nothing here guarantees profit; synthetic-data tests prove the plumbing works, not that the rules earn money on NSE.

## Setup
    pip install -r requirements.txt
Run all commands from this folder (modules import each other by name).

## Workflow
| Step | Command | Output |
|---|---|---|
| 0. Test the links first (changes nothing) | `python download.py --check` | OK / problem per source |
| 1. Download full history (resumable, ~1 hour) | `python download.py --start 2016-01-01 --out data/raw` (PR packages + delivery files; add `--sources pr,sec,cm` for UDiFF too) | `data/raw/PRddmmyy.zip`, `sec_bhavdata_full_ddmmyyyy.csv` |
| 1b. Daily update + rebuild | `python download.py --update --out data/raw --then-build` | new files, refreshed database |
| 2. Build adjusted table | `python build_db.py --raw data/raw --out data/daily` | `data/daily.*` (prices), `index_daily.*` (Nifty etc.), `events.csv` (results dates), `applied_adjustments.csv`, `corporate_actions_review.csv`, `suspect_jumps.csv` |
| 3. Backtest | `python run_backtest.py --data data/daily --start 2018-01-01 --split 2023-01-01` | stats, checklist, `results/equity.csv`, `results/trades.csv` |
| 4. Daily report (after close, after steps 1-2) | `python daily_report.py --data data/daily --capital 150000 --positions positions.csv` | `reports/YYYY-MM-DD.md`, `paper_signals.csv` |
| Results calendar (weekly) | `python events.py fetch --start 2026-10-17 --end 2026-12-31 --out data/events.csv` (or write `symbol,date` rows by hand) | `data/events.csv` |
| Report with news scan | add `--news --events data/events.csv` to the daily report command | headline section |
| Delivery % files (optional) | add `--delivery` to the download command, or put `sec_bhavdata_full_DDMMYYYY.csv` files in `data/raw` | picked up by `build_db.py` -> `data/delivery.*` |
| Delivery test, step 1 (run FIRST) | `python run_backtest.py --data data/daily --delivery-study` | rank correlation of delivery % vs trade outcome with a 95% interval |
| Delivery test, step 2 (only if step 1 shows an effect) | `python run_backtest.py --data data/daily --delivery-grid 0,40,50,60 --split 2023-01-01` | swing sleeve at several thresholds |
| Use the filter | add `--delivery-min 55` to `run_backtest.py` and `daily_report.py` | signals need delivery % >= 55 on the signal day |
| Self-test | `python tests/test_engine.py` | hand-computed trade checks |
| Pipeline test without internet | `python make_synthetic.py --mode random --out data/syn_random` then `run_backtest.py --data data/syn_random` | synthetic data only |

## Downloader notes (`download.py`)
Retries with back-off, re-warms the NSE session on blocks, validates every file (a blocked request often returns a web page with status 200), writes atomically, remembers holidays in `data/raw/_manifest.json`, lists failures in `data/raw/failed.txt` (re-run the same command to retry), and tries weekends too (special sessions). It stops early with a message if a link returns nothing for the last 10 weekdays. Tested against a local mock server only (retries, blocked pages, resume, update mode); the NSE links themselves are untested. If `--check` fails, copy the link pattern from nseindia.com and pass `--url-pr`, `--url-sec` or `--url-cm`. Use `--delay 1.5` if NSE starts blocking you.

## Rules implemented (edit `config.py`)
Timing: signals on the close of day t, orders at the open of day t+1. Stops use intraday low and gap-open.
Universe: top 100 by 60-day average traded value, price >= Rs 50 (point-in-time, delisted names stay in history).
Regime: Nifty 50 above its 100-day average (from the PR index rows; falls back to an equal-weight universe proxy if index data is missing), else no new longs (core to cash).

**Core momentum sleeve (40% of capital)** - weekly rebalance (last trading day of week)
- Score: close[t-5] / close[t-131] - 1 (6-month momentum skipping the last week); eligible if score > 0 and close > 100-day SMA.
- Hold up to 5 equal-weight positions; buy highest-ranked affordable names from the top 10 (min 2 shares per slot).
- Exit: rank falls below 10, close < highest close - 3 ATR(14), or regime off.

**Swing sleeve (35% of capital)** - daily signals
- Entry signal: close > prior 20-day high, volume > 1.5x prior 20-day average, close > 50-SMA > 200-SMA, regime on.
- Max 3 open; ranked by 6-month momentum. Skip if next open > signal close + 0.5 ATR (no chasing).
- Stop = fill - 1.5 ATR; R = fill - stop; risk = 0.5% of total capital per trade; notional cap 25% of capital.
- At +2R sell 50%, move stop to entry, then trail at highest high - 3 ATR. Time stop: 30 trading days. Stop is checked before target (conservative).

**Pullback setup (inside the swing sleeve, same stop/target/sizing rules)**
- Uptrend: 50-SMA > 200-SMA and close > 200-SMA, regime on, 6-month momentum > 0.
- Dip: lowest low of last 5 days <= 20-EMA + 0.25 ATR. Trigger: close > 20-EMA, close > prior day's high, RSI(14) between 45 and 65. Not counted if it is also a breakout.
- Shared 3-slot cap: `swing_priority` in config ('score', 'breakout', 'pullback') decides who gets slots when both fire. Backtest each alone with `--setups breakout|pullback` and together with `--setups both`.

**Results-calendar blackout**: no new entry if results fall within 4 calendar days (`results_blackout_days`). The daily report flags open positions with results due (gap risk). The backtest applies it only if you pass `--events` with historical results dates; otherwise it prints a warning.

**Headline scan (`--news`)**: pulls recent Google News RSS headlines for the symbols in the report and highlights risk words (SEBI, probe, fraud, downgrade, pledge, resign...). It does not score sentiment: you read the headlines. Search by NSE symbol can match the wrong company.

**Delivery % filter (optional, OFF by default)**: swing signals need delivery % >= `delivery_min` on the signal day. It is a hypothesis, not a rule. In testing on synthetic data a useless filter still looked helpful: on one random dataset the threshold grid showed a steadily rising profit factor, gains in 8 of 10 years and a better out-of-sample result, yet the delivery data was unrelated to returns by construction. Across six random datasets a useless 60% filter changed the profit factor by +0.07 on average (std 0.16, up to +0.31). The signal-level `--delivery-study` (every signal simulated independently, one rank-correlation test, no threshold search) correctly found nothing on that data. Use the study first; use the grid only to size a threshold if the study shows a real effect.

**Dividends**: `dividends.csv` lists every cash dividend (symbol, ex-date, Rs per share) taken from the corporate-action files. Prices are NOT dividend-adjusted by default (signals use actual prices); `build_db.py --adjust-dividends` produces total-return-style prices. The daily report flags TAKE signals and open positions with an ex-dividend date in the next 5 days, because the price drops by about the dividend on that day and broker stop orders are not adjusted.

**Cash buffer (25%)**: assumed liquid-fund yield (config).
Costs modelled: STT, stamp duty, exchange/SEBI/GST (approx), 0.10% slippage per side, Rs 16 DP charge per sell. VERIFY current rates.

## What was verified on real NSE files, and what was not
Verified on three real PR packages (15 Oct 2024, 15 Oct 2025, 16 Oct 2026): price and index parsing (EQ series, Nifty 50 rows), corporate-action ratio parsing for every split/bonus wording present, board-meeting (results) parsing, back-adjustment (e.g. DR Reddy's 5:1 split), and that NSE's PREV_CL_PR is NOT adjusted on ex-dates (so adjustment comes from the corporate-action file).

Not verified / known limitations:
1. `download.py` has never run against live NSE (the PR URL is my best recollection: compare with the link on NSE's site and edit URL_PR if needed).
2. Only three non-consecutive days were available, so `suspect_jumps.csv` (unadjusted overnight moves) and multi-year adjustment completeness are untested on real history. Run `build_db.py` on the full download and review `suspect_jumps.csv`, `corporate_actions_review.csv` and `applied_adjustments.csv`.
3. NOT auto-adjusted: rights issues, mergers, demergers, capital reductions (listed for review). Splits and bonuses are adjusted; dividends only with `--adjust-dividends`. Fix misses in `data/manual_adjustments.csv` (symbol,ex_date,ratio).
4. Only series EQ is used. A stock moved to BE (trade-to-trade) vanishes from the data; if held it is force-sold at the last price after 10 missing days.
5. Parquet storage is untested here (gzip CSV tested); install pyarrow and run once.
6. Results calendar comes from the board-meeting files inside the daily packages; meetings are announced only days ahead, so historical coverage depends on having every daily file. `events.py fetch` (NSE API) is untested.
7. Rules are tested on synthetic data only (engine checked by hand-computed unit tests). They have NOT been shown to work on NSE data: that is the next step.
8. The delivery file link in `download.py` (`URL_SEC`) is untested; the delivery parser was verified on one real file (15 Oct 2024: 1,910 EQ rows, values match NSE's PR file). Whether delivery % actually helps on NSE data is unknown.
9. Synthetic datasets include positive market drift, so even the 'random' set shows slightly positive average R: only compare variants against each other.
10. Options sleeve is NOT implemented. FII/DII flows, crude, rupee and macro news are manual.

## Before real money
Pass the checklist printed by `run_backtest.py` on out-of-sample data, then paper trade 2-3 months using `paper_signals.csv`, compare live against backtest, start with 0.5% risk per trade.
Tax/regulatory: keep contract notes and a trade log; confirm tax treatment (capital gains vs business income) and Section 44AB for the relevant AY; do not share signals with others without SEBI registration.
