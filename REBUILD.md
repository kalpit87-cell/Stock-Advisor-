# Rebuild specification

## Confirmed requirements

- Reuse the existing Python work as the base.
- Personal use; total capital approximately INR 300,000–400,000.
- Equity calls for long, medium and short horizons, plus derivatives.
- Interpret “all three” as futures, option buying and defined-risk spreads; confirm this interpretation.
- Calls need entry, target, stop, reason, validity and position sizing.
- Download reliable data and incorporate relevant political, macro and company news.
- Desired win rate above 80%, treated as a research objective, never a guaranteed result.

## Open decisions

End-of-day versus live intraday signals; broker; monthly data/news budget; exact holding periods; acceptable portfolio drawdown and per-trade risk. No paid subscriptions or broker access have been configured.

## Baseline review

Imported source from the supplied nse_trader1.zip; historical datasets and personal output files excluded. Existing architecture: downloader, corporate-action builder, equity feature generation, equity backtest, metrics, daily report, index trend test and headline scan. The earlier user-provided summary reports no dependable edge for previous stock strategies; those full historical results have not been reproduced here. Existing news code uses Google News RSS and keyword matching, not a validated event-impact model. A pandas compatibility FutureWarning occurs in an existing test fixture.

## Implementation sequence

1. Reproduce historical equity results and audit data handling, cost assumptions, delistings, dividends, events and as-of-date handling. Keep original results reproducible.
2. Add one explicit account configuration and portfolio-wide exposure/risk limits. Display research/paper/live state in every report. Do not allow missing risk inputs to produce a trade-ready call.
3. Add separate, versioned strategy definitions for investment, medium-term and short-term research. Avoid tuning the unsuccessful legacy rules to force an 80% result.
4. Add futures and option contract ingestion: historical lot sizes, expiry, settlement, premiums, volume, OI and bid/ask where available. Check current broker margin; do not infer affordability from notional or stop distance alone. Long-option sizing must consider full premium loss; spread sizing must consider maximum loss, execution and expiry/settlement exposure. Futures stop loss is not a guaranteed loss cap.
5. Add news ingestion with publisher, original URL, publication/first-seen time, deduplication, company/sector mapping and separate evidence versus inferred impact. Cover official filings, political/policy events, macro announcements and licensed news as available. Record unavailable or stale coverage explicitly.
6. Add dashboard/report and immutable signal/outcome logs. Keep order execution disabled in the initial research product.

## Validation contract

- Register entry, exit, target, stop, time limit and primary metrics before each experiment.
- Define target-before-stop hit rate separately from net profitable-position rate. Aggregate partial exits per position and disclose time exits, open positions and excluded trades.
- Use only information observable when the signal was generated; no current fundamentals or current news in historical decisions.
- Treat already examined 2016–2026 data as development history. Use chronological walk-forward checks and an untouched future paper-trading period; do not relabel an already examined interval as new holdout evidence.
- Include transaction costs, realistic slippage, taxes/fees at applicable dates, contract rolls and liquidity assumptions. Investor income tax, if omitted, must be disclosed separately.
- Report net expectancy, profit factor, drawdown, turnover, benchmark comparison, sample size and uncertainty by category and regime, plus portfolio results with shared capital constraints.
- Historical news impact remains unvalidated unless timestamped historical content is available. Current news collection cannot retrospectively establish performance.
- Suppress unsupported, stale, unaffordable or unvalidated calls. Never display an invented probability or an aggregate 80% label across unrelated strategies.

## Scope boundaries

This baseline has no implemented derivatives engine, live feed, fundamental investment engine, political impact model or proven 80% strategy. Automated regression checks establish code behavior, not market profitability. Data licenses and current provider endpoints require separate verification.
