"""Central configuration. Edit values here only."""
from dataclasses import dataclass, field


@dataclass
class Costs:
    # NSE delivery (equity cash) charges. VERIFY current rates before relying on results.
    stt: float = 0.001              # 0.1% on buy and sell (delivery)
    stamp_buy: float = 0.00015      # 0.015% on buy
    exch_sebi_gst: float = 0.0000325  # approx exchange + SEBI fee + GST on them, both sides
    slippage: float = 0.001         # 0.10% per side (price impact / spread)
    dp_charge_sell: float = 16.0    # approx depository charge per sell per scrip per day

    def buy_fee_rate(self) -> float:
        return self.stt + self.stamp_buy + self.exch_sebi_gst

    def sell_fee_rate(self) -> float:
        return self.stt + self.exch_sebi_gst


@dataclass
class Config:
    total_capital: float = 150_000.0
    alloc_core: float = 0.40
    alloc_swing: float = 0.35
    alloc_cash: float = 0.25
    cash_yield: float = 0.06        # assumed liquid-fund yield p.a. (verify)

    # universe (point-in-time, by traded value) and data hygiene
    universe_size: int = 100
    liq_window: int = 60
    min_price: float = 50.0
    min_history: int = 200          # trading days of data before a stock can enter the universe (new IPOs have no usable history)
    atr_n: int = 14

    # market regime: equal-weight proxy of the universe above its SMA
    regime_sma: int = 100
    regime_source: str = "auto"     # 'auto' = Nifty 50 if index data is available, else equal-weight proxy; or 'nifty' / 'proxy'

    # core momentum sleeve (weekly rebalance, long only)
    core_k: int = 5
    core_hold_rank: int = 10
    core_roc_long: int = 126
    core_roc_skip: int = 5
    core_sma: int = 100
    core_trail_atr: float = 3.0
    core_min_shares: int = 2        # skip stocks too expensive to hold >= 2 shares per slot

    # swing breakout sleeve
    use_breakout: bool = True
    use_pullback: bool = True
    swing_priority: str = "score"   # slot priority when both setups fire: 'score' | 'breakout' | 'pullback'
    results_blackout_days: int = 4   # calendar days: no new entry if results fall within this window
    delivery_min: float = 0.0       # swing signals need delivery % >= this on the signal day (0 = off)
    swing_risk_pct: float = 0.005   # of TOTAL capital per trade (0.5%)
    swing_max_pos: int = 3
    swing_breakout: int = 20
    swing_vol_mult: float = 1.5
    swing_stop_atr: float = 1.5
    swing_t1_r: float = 2.0
    swing_trail_atr: float = 3.0
    swing_time_stop: int = 30
    swing_max_gap_atr: float = 0.5  # skip entry if open > signal close + 0.5 ATR
    swing_max_pos_pct: float = 0.25  # max notional per trade as % of total capital

    costs: Costs = field(default_factory=Costs)
