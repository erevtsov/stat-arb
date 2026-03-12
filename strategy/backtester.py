"""
Portfolio backtester for the statistical arbitrage strategy.

Runs a day-by-day portfolio simulation:
  - SOD: recalibrate cointegrated pairs over the trailing 84 calendar days.
  - Intraday: bar-by-bar signal loop, enter/exit positions based on signal transitions.
  - EOD: force-close all open positions at the close of the last bar.
  - PnL: compute gross and net P&L per trade; accumulate portfolio value.

Execution prices:
  - Intraday entry/exit: (high + low) / 2 of the *next* bar after signal.
  - EOD forced exit: close of the last bar.

Position sizing:
  - Equal (default): notional_per_pair = portfolio_value / max_pairs
  - P-value weighted (use_pvalue_weights=True): notional proportional to -log(p_value),
    normalized over top max_pairs pairs by p_value; weights sum to 1.
  - shares_a = notional / entry_price_a
  - shares_b = hedge_ratio * notional / entry_price_b  (dollar-neutral: each leg has notional β×N)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from analysis.cointegration import find_cointegrated_pairs
from analysis.preprocessing import load_processed
from analysis.signals import compute_pvalue_weights, generate_pair_signals_for_day
from utils.config import BARS_PER_DAY, CONFIG, Config, MINUTES_PER_BAR, get_all_tickers

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PositionState:
    """State of an open spread position."""

    direction: int  # +1 (long spread) or -1 (short spread)
    entry_time: dt.datetime
    entry_price_a: float
    entry_price_b: float
    shares_a: float  # notional / entry_price_a
    shares_b: float  # hedge_ratio * shares_a
    hedge_ratio: float
    ticker_a: str
    ticker_b: str
    p_value: float


# ---------------------------------------------------------------------------
# PnL helpers
# ---------------------------------------------------------------------------


def _compute_pnl(
    pos: PositionState,
    exit_price_a: float,
    exit_price_b: float,
    cost_bps: float,
) -> tuple[float, float, float]:
    """
    Compute gross PnL, transaction cost, and net PnL for closing a position.

    Returns:
        (gross_pnl, transaction_cost, net_pnl)
    """
    gross_pnl = pos.direction * (
        (exit_price_a - pos.entry_price_a) * pos.shares_a
        - (exit_price_b - pos.entry_price_b) * pos.shares_b
    )

    leg_cost_fraction = cost_bps / 10_000.0
    cost_a_in = abs(pos.shares_a) * pos.entry_price_a * leg_cost_fraction
    cost_b_in = abs(pos.shares_b) * pos.entry_price_b * leg_cost_fraction
    cost_a_out = abs(pos.shares_a) * exit_price_a * leg_cost_fraction
    cost_b_out = abs(pos.shares_b) * exit_price_b * leg_cost_fraction
    transaction_cost = cost_a_in + cost_b_in + cost_a_out + cost_b_out

    return gross_pnl, transaction_cost, gross_pnl - transaction_cost


# ---------------------------------------------------------------------------
# NYSE trading calendar
# ---------------------------------------------------------------------------


def _get_nyse_trading_days(start_date: str, end_date: str) -> list[dt.date]:
    """Return sorted list of NYSE trading days between start and end (inclusive)."""
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        return [d.date() for d in schedule.index]
    except ImportError as exc:
        raise ImportError(
            "pandas_market_calendars is required. Install with: pip install pandas-market-calendars"
        ) from exc


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def _load_day_prices(
    tickers: list[str],
    start_date: dt.date,
    end_date: dt.date,
    timeframe: str,
) -> dict[str, pl.DataFrame]:
    """
    Load intraday OHLCV for each ticker over [start_date, end_date].

    Includes the z-score lookback window so that rolling statistics are
    fully populated on the trading day being simulated.
    """
    prices: dict[str, pl.DataFrame] = {}
    for ticker in tickers:
        try:
            df = load_processed(
                ticker, timeframe,
                start_date=str(start_date),
                end_date=str(end_date),
            )
            if df is not None and len(df) > 0:
                prices[ticker] = df
        except FileNotFoundError:
            pass
    return prices


def _build_bar_lookup(
    prices: dict[str, pl.DataFrame],
) -> dict[str, dict[Any, dict]]:
    """
    Build a fast lookup: ticker → {timestamp → {open, high, low, close}}.
    """
    lookup: dict[str, dict[Any, dict]] = {}
    for ticker, df in prices.items():
        rows = df.select(["timestamp", "open", "high", "low", "close"]).iter_rows(
            named=True
        )
        lookup[ticker] = {r["timestamp"]: r for r in rows}
    return lookup


def _get_exec_price(
    exec_bar_lookup: dict[str, dict[Any, dict]],
    ticker: str,
    next_bar_ts: Any,
    execution_lag_minutes: int,
    execution_price_field: str,
) -> float | None:
    """
    Look up the execution price from a 1-min bar lookup.

    Args:
        exec_bar_lookup:       1-min bar lookup built by _build_bar_lookup().
        ticker:                Ticker symbol.
        next_bar_ts:           Timestamp of the next signal-timeframe bar
                               (i.e., the bar after the signal fires).
        execution_lag_minutes: Minutes after next_bar_ts to look up.
                               0 = first 1-min bar of the next bar period.
        execution_price_field: OHLC field or "mid" ((high + low) / 2).

    Returns:
        Price as float, or None if the 1-min bar is not found (caller should
        fall back to the signal-timeframe midpoint).
    """
    exec_ts = next_bar_ts + dt.timedelta(minutes=execution_lag_minutes)
    bar = exec_bar_lookup.get(ticker, {}).get(exec_ts)
    if bar is None:
        return None
    if execution_price_field == "mid":
        return (bar["high"] + bar["low"]) / 2.0
    return float(bar[execution_price_field])


# ---------------------------------------------------------------------------
# Main backtest entry point
# ---------------------------------------------------------------------------


def run_backtest(config: Config | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run the full intraday pairs-trading backtest.

    All parameters are read from a ``Config`` instance. Pass a custom ``Config``
    to override any setting; omit it (or pass ``None``) to use the module-level
    ``CONFIG`` singleton.

    Key config sections:
        config.portfolio   — start_date, end_date, timeframe, capital, max_pairs,
                             transaction_cost_bps, min_bars_remaining,
                             execution_lag_minutes, execution_price_field
        config.cointegration — rolling_window_days, p_value_threshold,
                               min_half_life, max_half_life, require_split_window
        config.signal      — z_entry, z_exit, z_stop, max_holding_minutes,
                             zscore_window_days, fixed_exit_norm

    Returns:
        (trades_df, daily_pnl_df) — see module docstring for schemas.
    """
    cfg = config if config is not None else CONFIG

    # Portfolio / execution
    start_date            = cfg.portfolio.start_date
    end_date              = cfg.portfolio.end_date
    timeframe             = cfg.portfolio.timeframe
    min_bars_remaining    = cfg.portfolio.min_bars_remaining
    execution_lag_minutes = cfg.portfolio.execution_lag_minutes
    execution_price_field = cfg.portfolio.execution_price_field
    max_pairs             = cfg.portfolio.max_pairs
    capital               = cfg.portfolio.capital
    cost_bps              = cfg.portfolio.transaction_cost_bps
    use_pvalue_weights    = cfg.portfolio.use_pvalue_weights

    # Cointegration
    rolling_window_days  = cfg.cointegration.rolling_window_days
    p_value_threshold    = cfg.cointegration.p_value_threshold
    min_half_life        = cfg.cointegration.min_half_life
    max_half_life        = cfg.cointegration.max_half_life
    require_split_window = cfg.cointegration.require_split_window

    # Signal
    z_entry             = cfg.signal.z_entry
    z_exit              = cfg.signal.z_exit
    z_stop              = cfg.signal.z_stop
    max_holding_minutes = cfg.signal.max_holding_minutes
    zscore_window_days  = cfg.signal.zscore_window_days
    fixed_exit_norm     = cfg.signal.fixed_exit_norm

    zscore_window        = zscore_window_days * BARS_PER_DAY[timeframe]
    # Number of lookback trading days needed to warm up the z-score window
    zscore_lookback_days = zscore_window_days + 1
    max_holding_bars_val = max_holding_minutes // MINUTES_PER_BAR[timeframe]

    trading_days = _get_nyse_trading_days(start_date, end_date)

    # Pre-load all universe prices for the full backtest span (including formation
    # lookback) into memory.  Each day then slices from RAM instead of re-reading
    # Parquet files, eliminating 100 × n_days disk reads.
    cache_start = trading_days[0] - dt.timedelta(days=rolling_window_days + zscore_lookback_days + 5)
    _price_cache: dict[str, pl.DataFrame] = _load_day_prices(
        get_all_tickers(), cache_start, trading_days[-1], timeframe
    )

    # Chunk-based 1-min cache: loads 30 trading days at a time instead of one day
    # at a time.  Bounds memory to O(chunk × tickers) regardless of history length,
    # while eliminating ~95% of per-day parquet I/O (amortises 100 file opens over
    # 30 days instead of 1).  At 300 tickers + 10 years the chunk stays ~165 MB.
    _1MIN_CHUNK_DAYS = 30
    _1min_chunk: dict[str, pl.DataFrame] = {}
    _1min_chunk_through: dt.date = dt.date.min   # sentinel → triggers load on day 0

    portfolio_value = capital
    open_positions: dict[tuple[str, str], PositionState] = {}
    all_trades: list[dict] = []
    daily_pnl_rows: list[dict] = []

    for day_idx, day in enumerate(trading_days):
        formation_start = day - dt.timedelta(days=rolling_window_days)
        formation_end = day - dt.timedelta(days=1)

        # SOD: find cointegrated pairs for the formation window (cache avoids I/O)
        try:
            pairs_df = find_cointegrated_pairs(
                start_date=str(formation_start),
                end_date=str(formation_end),
                timeframe=timeframe,
                price_cache=_price_cache,
                p_value_threshold=p_value_threshold,
                min_half_life=min_half_life,
                max_half_life=max_half_life,
                require_split_window=require_split_window,
            )
        except Exception:
            pairs_df = None

        if pairs_df is None or len(pairs_df) == 0:
            daily_pnl_rows.append(
                {
                    "date": str(day),
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                    "n_trades": 0,
                    "n_open_eod": 0,
                    "portfolio_value": portfolio_value,
                }
            )
            continue

        pairs_df = compute_pvalue_weights(pairs_df)

        notional_per_pair = portfolio_value / max_pairs

        if use_pvalue_weights:
            import math
            _top = pairs_df.sort("p_value").head(max_pairs)
            _raw = [-math.log(p) for p in _top["p_value"].to_list()]
            _total = sum(_raw)
            _pair_notionals: dict[tuple[str, str], float] = {
                (row["ticker_a"], row["ticker_b"]): portfolio_value * (w / _total)
                for row, w in zip(_top.iter_rows(named=True), _raw)
            }
        else:
            _pair_notionals = {}

        # Determine all tickers needed today
        all_tickers_today: list[str] = list(
            set(pairs_df["ticker_a"].to_list() + pairs_df["ticker_b"].to_list())
        )

        # Slice intraday prices from the in-memory cache (no Parquet I/O).
        lookback_start = trading_days[max(0, day_idx - zscore_lookback_days)]
        day_prices = {
            t: _price_cache[t].filter(
                (pl.col("timestamp").dt.date() >= lookback_start)
                & (pl.col("timestamp").dt.date() <= day)
            )
            for t in all_tickers_today
            if t in _price_cache
        }

        if not day_prices:
            daily_pnl_rows.append(
                {
                    "date": str(day),
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                    "n_trades": 0,
                    "n_open_eod": 0,
                    "portfolio_value": portfolio_value,
                }
            )
            continue

        # Generate signals for the full lookback + today.
        # max_holding_bars enforces an intraday time stop inside the signal
        # generator; all remaining positions are also force-closed at EOD below.
        try:
            signals_day = generate_pair_signals_for_day(
                pairs_df=pairs_df,
                date=day,
                intraday_prices=day_prices,
                zscore_window=zscore_window,
                z_entry=z_entry,
                z_exit=z_exit,
                z_stop=z_stop,
                max_holding_bars=max_holding_bars_val,
                fixed_exit_norm=fixed_exit_norm,
            )
        except Exception:
            daily_pnl_rows.append(
                {
                    "date": str(day),
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                    "n_trades": 0,
                    "n_open_eod": 0,
                    "portfolio_value": portfolio_value,
                }
            )
            continue

        # Filter signals to the current trading day only.
        # signals_day contains the full lookback window (needed for z-score
        # warmup), but we only trade on bars belonging to today.
        signals_day = signals_day.filter(
            pl.col("timestamp").dt.date() == day
        )

        if len(signals_day) == 0:
            daily_pnl_rows.append(
                {
                    "date": str(day),
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                    "n_trades": 0,
                    "n_open_eod": 0,
                    "portfolio_value": portfolio_value,
                }
            )
            continue

        bar_lookup = _build_bar_lookup(day_prices)

        # Load 1-min execution prices when requested.
        # Only the current trading day is needed — all signal-based exits
        # use next_bar_ts which is always within the current day.
        if execution_lag_minutes is not None:
            # Reload chunk when the current day has advanced past it
            if day > _1min_chunk_through:
                chunk_end_idx = min(day_idx + _1MIN_CHUNK_DAYS - 1, len(trading_days) - 1)
                _1min_chunk_through = trading_days[chunk_end_idx]
                _1min_chunk = _load_day_prices(
                    get_all_tickers(), day, _1min_chunk_through, "1min"
                )
            # Per-day slice from chunk: no parquet I/O, just a DataFrame filter
            exec_prices_1min = {
                t: df.filter(pl.col("timestamp").dt.date() == day)
                for t, df in _1min_chunk.items()
                if t in set(all_tickers_today)
            }
            exec_bar_lookup: dict[str, dict[Any, dict]] = _build_bar_lookup(exec_prices_1min)
        else:
            exec_bar_lookup = {}


        # Get sorted unique bars for this day
        bars: list = sorted(signals_day["timestamp"].unique().to_list())
        n_bars_day = len(bars)

        # Build quick-access lookups for signal and z-score
        sig_map: dict[tuple, int] = {}
        zscore_map: dict[tuple, float | None] = {}
        for row in signals_day.iter_rows(named=True):
            key = (row["ticker_a"], row["ticker_b"], row["timestamp"])
            sig_map[key] = row["signal_binary"]
            zscore_map[key] = row["zscore"]

        # Ordered pairs list (by p_value ascending = best signal first)
        pairs_ordered = list(pairs_df.sort("p_value").iter_rows(named=True))

        day_gross_pnl = 0.0
        day_net_pnl = 0.0
        day_n_trades = 0

        for bar_idx, bar_ts in enumerate(bars):
            is_last_bar = bar_idx == n_bars_day - 1
            next_bar_ts = bars[bar_idx + 1] if not is_last_bar else None

            # --- Exit checks (existing positions) ---
            pairs_to_close: list[tuple[str, str]] = []

            for pair_key, pos in open_positions.items():
                ticker_a, ticker_b = pair_key
                sig = sig_map.get((ticker_a, ticker_b, bar_ts), 0)

                # Determine if we should exit
                # Intraday-only: force-close all positions at EOD every day.
                force_close = is_last_bar
                should_exit = force_close

                if not should_exit and not is_last_bar:
                    # Check if signal went to 0 (mean reversion or stop loss).
                    # On bar_idx=0, use pos.direction as the "previous" signal
                    # since we know a position is already open.
                    # Skip this check on the last bar of non-final days: the
                    # correct execution would be at next bar (next trading day's
                    # open), so carry overnight instead of falling back to the
                    # current bar's midpoint.
                    prev_bar_ts = bars[bar_idx - 1] if bar_idx > 0 else None
                    prev_sig = (
                        sig_map.get((ticker_a, ticker_b, prev_bar_ts), 0)
                        if prev_bar_ts
                        else pos.direction
                    )
                    if sig == 0 and prev_sig != 0:
                        # Signal cleared: z_exit or z_stop fired.
                        should_exit = True
                    elif sig != 0 and prev_sig != 0 and sig != pos.direction:
                        # Direction flip: z crossed z_exit then z_entry in the
                        # opposite direction within one bar (rare but possible
                        # overnight).  Exit on the next bar — our position is now
                        # wrong-way relative to the new signal.
                        should_exit = True

                if not should_exit:
                    continue

                # Determine exit price and reason
                if force_close:
                    exit_reason = "eod"
                    bars_a = bar_lookup.get(ticker_a, {})
                    bars_b = bar_lookup.get(ticker_b, {})
                    bar_a = bars_a.get(bar_ts)
                    bar_b = bars_b.get(bar_ts)
                    if bar_a is None or bar_b is None:
                        pairs_to_close.append(pair_key)
                        continue
                    exit_price_a = bar_a["close"]
                    exit_price_b = bar_b["close"]
                else:
                    # Infer granular exit reason from z-score at the exit bar.
                    # The signal generator counts bars_held starting at the bar
                    # AFTER the entry signal, but pos.entry_time is next_bar_ts
                    # (one bar later).  So when the time stop fires after
                    # max_holding_bars_val bars, elapsed from pos.entry_time is
                    # (max_holding_bars_val - 1) bar-intervals, not
                    # max_holding_minutes.  Use (max_holding_bars_val - 1) bars
                    # as the threshold to correctly detect time-stop exits.
                    elapsed_minutes = (
                        bar_ts - pos.entry_time
                    ).total_seconds() / 60.0
                    minutes_per_bar = MINUTES_PER_BAR[timeframe]
                    max_hold_threshold = (max_holding_bars_val - 1) * minutes_per_bar
                    z_at_exit = zscore_map.get((ticker_a, ticker_b, bar_ts))
                    if (
                        z_stop is not None
                        and z_at_exit is not None
                        and abs(z_at_exit) > z_stop
                    ):
                        exit_reason = "z_stop"
                    elif elapsed_minutes >= max_hold_threshold:
                        exit_reason = "max_hold"
                    else:
                        exit_reason = "z_exit"

                    bars_a = bar_lookup.get(ticker_a, {})
                    bars_b = bar_lookup.get(ticker_b, {})
                    nbar_a = bars_a.get(next_bar_ts)
                    nbar_b = bars_b.get(next_bar_ts)
                    if nbar_a is None or nbar_b is None:
                        # Fall back to current bar midpoint
                        bar_a = bars_a.get(bar_ts)
                        bar_b = bars_b.get(bar_ts)
                        if bar_a is None or bar_b is None:
                            pairs_to_close.append(pair_key)
                            continue
                        exit_price_a = (bar_a["high"] + bar_a["low"]) / 2.0
                        exit_price_b = (bar_b["high"] + bar_b["low"]) / 2.0
                    else:
                        # Use 1-min execution price when requested; fall back
                        # to signal-timeframe midpoint if bar is missing.
                        p_a = (
                            _get_exec_price(
                                exec_bar_lookup, ticker_a, next_bar_ts,
                                execution_lag_minutes, execution_price_field,
                            )
                            if execution_lag_minutes is not None
                            else None
                        )
                        p_b = (
                            _get_exec_price(
                                exec_bar_lookup, ticker_b, next_bar_ts,
                                execution_lag_minutes, execution_price_field,
                            )
                            if execution_lag_minutes is not None
                            else None
                        )
                        exit_price_a = p_a if p_a is not None else (nbar_a["high"] + nbar_a["low"]) / 2.0
                        exit_price_b = p_b if p_b is not None else (nbar_b["high"] + nbar_b["low"]) / 2.0

                gross, cost, net = _compute_pnl(
                    pos, exit_price_a, exit_price_b, cost_bps
                )

                exit_ts = bar_ts if is_last_bar else (next_bar_ts or bar_ts)

                all_trades.append(
                    {
                        "entry_time": pos.entry_time,
                        "exit_time": exit_ts,
                        "ticker_a": pos.ticker_a,
                        "ticker_b": pos.ticker_b,
                        "direction": pos.direction,
                        "entry_price_a": pos.entry_price_a,
                        "entry_price_b": pos.entry_price_b,
                        "exit_price_a": exit_price_a,
                        "exit_price_b": exit_price_b,
                        "shares_a": pos.shares_a,
                        "shares_b": pos.shares_b,
                        "gross_pnl": gross,
                        "transaction_cost": cost,
                        "net_pnl": net,
                        "exit_reason": exit_reason,
                    }
                )

                day_gross_pnl += gross
                day_net_pnl += net
                day_n_trades += 1
                pairs_to_close.append(pair_key)

            for pk in pairs_to_close:
                open_positions.pop(pk, None)

            if is_last_bar:
                break

            # How many bars remain after the entry execution bar (next_bar_ts)?
            # bars_after_entry = n_bars_day - bar_idx - 2
            # (subtract current bar and the entry bar itself)
            bars_after_entry = n_bars_day - bar_idx - 2

            # --- Entry checks (new positions) ---
            for pair_row in pairs_ordered:
                if len(open_positions) >= max_pairs:
                    break

                ticker_a = pair_row["ticker_a"]
                ticker_b = pair_row["ticker_b"]
                pair_key = (ticker_a, ticker_b)

                if pair_key in open_positions:
                    continue

                sig = sig_map.get((ticker_a, ticker_b, bar_ts), 0)
                if sig == 0:
                    continue

                # Require sufficient bars remaining so mean reversion has time
                # to occur before the EOD forced close.
                if bars_after_entry < min_bars_remaining:
                    continue

                # Only enter on a genuine z_entry crossing.
                # On bar_idx=0 the "previous" signal comes from the lookback
                # simulation; gate entries using the actual z-score to avoid
                # acting on stale continuation signals carried over from the
                # lookback window.
                prev_bar_ts = bars[bar_idx - 1] if bar_idx > 0 else None
                prev_sig = (
                    sig_map.get((ticker_a, ticker_b, prev_bar_ts), 0)
                    if prev_bar_ts
                    else 0
                )
                if prev_sig != 0:
                    continue

                if bar_idx == 0:
                    # Verify the z-score actually crossed the entry threshold
                    # on this bar (not a stale carry-over from prior day).
                    z_now = zscore_map.get((ticker_a, ticker_b, bar_ts))
                    if z_now is None or abs(z_now) < z_entry:
                        continue

                # Execute at next bar — use 1-min price when requested,
                # fall back to signal-timeframe midpoint if bar is missing.
                bars_a = bar_lookup.get(ticker_a, {})
                bars_b = bar_lookup.get(ticker_b, {})
                nbar_a = bars_a.get(next_bar_ts)
                nbar_b = bars_b.get(next_bar_ts)
                if nbar_a is None or nbar_b is None:
                    continue

                if execution_lag_minutes is not None:
                    p_a = _get_exec_price(
                        exec_bar_lookup, ticker_a, next_bar_ts,
                        execution_lag_minutes, execution_price_field,
                    )
                    p_b = _get_exec_price(
                        exec_bar_lookup, ticker_b, next_bar_ts,
                        execution_lag_minutes, execution_price_field,
                    )
                    entry_price_a = p_a if p_a is not None else (nbar_a["high"] + nbar_a["low"]) / 2.0
                    entry_price_b = p_b if p_b is not None else (nbar_b["high"] + nbar_b["low"]) / 2.0
                else:
                    entry_price_a = (nbar_a["high"] + nbar_a["low"]) / 2.0
                    entry_price_b = (nbar_b["high"] + nbar_b["low"]) / 2.0

                if entry_price_a <= 0 or entry_price_b <= 0:
                    continue

                _notional = _pair_notionals.get((ticker_a, ticker_b), notional_per_pair)
                shares_a = _notional / entry_price_a
                shares_b = pair_row["hedge_ratio"] * _notional / entry_price_b

                open_positions[pair_key] = PositionState(
                    direction=sig,
                    entry_time=next_bar_ts,
                    entry_price_a=entry_price_a,
                    entry_price_b=entry_price_b,
                    shares_a=shares_a,
                    shares_b=shares_b,
                    hedge_ratio=pair_row["hedge_ratio"],
                    ticker_a=ticker_a,
                    ticker_b=ticker_b,
                    p_value=pair_row["p_value"],
                )

        portfolio_value += day_net_pnl

        daily_pnl_rows.append(
            {
                "date": str(day),
                "gross_pnl": day_gross_pnl,
                "net_pnl": day_net_pnl,
                "n_trades": day_n_trades,
                "n_open_eod": 0,  # always 0: all positions force-closed at EOD
                "portfolio_value": portfolio_value,
            }
        )

    # Build output DataFrames
    if not all_trades:
        trades_df = pl.DataFrame(
            schema={
                "entry_time": pl.Datetime,
                "exit_time": pl.Datetime,
                "ticker_a": pl.Utf8,
                "ticker_b": pl.Utf8,
                "direction": pl.Int32,
                "entry_price_a": pl.Float64,
                "entry_price_b": pl.Float64,
                "exit_price_a": pl.Float64,
                "exit_price_b": pl.Float64,
                "shares_a": pl.Float64,
                "shares_b": pl.Float64,
                "gross_pnl": pl.Float64,
                "transaction_cost": pl.Float64,
                "net_pnl": pl.Float64,
                "exit_reason": pl.Utf8,
            }
        )
    else:
        trades_df = pl.DataFrame(all_trades).with_columns(
            pl.col("direction").cast(pl.Int32)
        )

    daily_pnl_df = pl.DataFrame(daily_pnl_rows).with_columns(
        pl.col("n_trades").cast(pl.Int32),
        pl.col("n_open_eod").cast(pl.Int32),
    )

    return trades_df, daily_pnl_df


# ---------------------------------------------------------------------------
# Save results helpers
# ---------------------------------------------------------------------------


def save_results(
    trades_df: pl.DataFrame,
    daily_pnl_df: pl.DataFrame,
    results_dir: str | None = None,
    label: str = "15min",
) -> None:
    """
    Save backtest results to Parquet files under results/.

    Args:
        trades_df:     Trade log DataFrame.
        daily_pnl_df:  Daily PnL DataFrame.
        results_dir:   Root results directory. Defaults to CONFIG.paths.results_dir.
        label:         Filename suffix (e.g., "15min").
    """
    root = Path(results_dir or CONFIG.paths.results_dir)
    trades_path = root / "trades" / f"trades_{label}.parquet"
    pnl_path = root / "portfolio" / f"daily_pnl_{label}.parquet"

    trades_path.parent.mkdir(parents=True, exist_ok=True)
    pnl_path.parent.mkdir(parents=True, exist_ok=True)

    trades_df.write_parquet(trades_path)
    daily_pnl_df.write_parquet(pnl_path)

    print(f"Trades saved to:     {trades_path}")
    print(f"Daily PnL saved to:  {pnl_path}")
