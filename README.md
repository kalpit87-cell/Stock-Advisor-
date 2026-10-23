# Stock Advisor

Personal Indian-market research toolkit rebuilt from the existing nse_trader project.

## Current status

The legacy Python equity research code is preserved at the repository root. This is a research baseline, not a validated call service. Futures, option buying, defined-risk spreads, fundamental investment models, live intraday data and a political-news impact model are not implemented yet.

Confirmed scope: personal use; capital INR 300,000–400,000; reuse the existing work; all three derivatives categories (interpretation pending confirmation). Signal frequency, broker, data budget and precise holding periods remain open.

Read [REBUILD.md](REBUILD.md) for the implementation plan and validation boundaries, and [docs/LEGACY_README.md](docs/LEGACY_README.md) for existing commands and limitations.

## Run the existing research tools

Install Python and dependencies with `python -m pip install -r requirements.txt`. Keep your historical data on your own computer under `data/`; it is excluded from Git.

Use your actual capital explicitly; for example:

```text
python run_backtest.py --data data/daily --capital 350000 --setups breakout --split 2023-01-01
python daily_report.py --data data/daily --capital 350000 --positions positions.csv
python tests/test_engine.py
python tests/test_download.py
```

INR 350,000 is an illustrative midpoint, not a confirmed account balance. Legacy defaults and strategy allocations are preserved for reproducibility; do not treat them as new portfolio recommendations. All historical periods already examined are development data, not an untouched validation set.

No 80% win rate has been demonstrated. Results must report all closed positions, costs, time exits and drawdowns. A partial target exit is not a separate winning trade. News retrieval failure must not be interpreted as no risk.

Do not commit broker credentials, personal positions, trade logs, historical datasets or restricted news content. No order placement is enabled.
