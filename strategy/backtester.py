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
  - notional_per_pair = portfolio_value / max_pairs
  - shares_a = notional / entry_price_a
  - shares_b = hedge_ratio * shares_a  (for dollar-neutral pair)
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
from utils.config import CONFIG, MINUTES_PER_BAR, max_holding_bars, zscore_window_bars

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


# ---------------------------------------------------------------------------
# Main backtest entry point
# ---------------------------------------------------------------------------


def run_backtest(
    start_date: str = "2022-07-01",
    end_date: str = "2024-12-31",
    timeframe: str = "15min",
    rolling_window_days: int | None = None,
    z_entry: float | None = None,
    z_exit: float | None = None,
    z_stop: float | None = None,
    max_pairs: int | None = None,
    capital: float | None = None,
    cost_bps: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Run the full intraday pairs-trading backtest.

    Args:
        start_date:           First trading day (YYYY-MM-DD).
        end_date:             Last trading day (YYYY-MM-DD).
        timeframe:            Bar frequency (e.g., "15min").
        rolling_window_days:  Calendar days for the formation window.
                              Defaults to CONFIG.cointegration.rolling_window_days.
        z_entry:              Entry z-score threshold. Defaults to CONFIG.signal.z_entry.
        z_exit:               Exit z-score threshold. Defaults to CONFIG.signal.z_exit.
        z_stop:               Stop-loss threshold. Defaults to CONFIG.signal.z_stop.
        max_pairs:            Max simultaneous open positions.
                              Defaults to CONFIG.portfolio.max_pairs.
        capital:              Starting portfolio value.
                              Defaults to CONFIG.portfolio.capital.
        cost_bps:             One-way cost per leg in bps.
                              Defaults to CONFIG.portfolio.transaction_cost_bps.

    Returns:
        (trades_df, daily_pnl_df) — see module docstring for schemas.
    """
    # Resolve defaults
    rolling_window_days = (
        rolling_window_days or CONFIG.cointegration.rolling_window_days
    )
    z_entry = z_entry if z_entry is not None else CONFIG.signal.z_entry
    z_exit = z_exit if z_exit is not None else CONFIG.signal.z_exit
    z_stop = z_stop if z_stop is not None else CONFIG.signal.z_stop
    max_pairs = max_pairs if max_pairs is not None else CONFIG.portfolio.max_pairs
    capital = capital if capital is not None else CONFIG.portfolio.capital
    cost_bps = (
        cost_bps if cost_bps is not None else CONFIG.portfolio.transaction_cost_bps
    )

    zscore_window = zscore_window_bars(timeframe)
    max_hold_bars = max_holding_bars(timeframe)
    minutes_per_bar = MINUTES_PER_BAR[timeframe]
    # Number of lookback trading days needed to warm up the z-score window
    zscore_lookback_days = CONFIG.signal.zscore_window_days + 1

    trading_days = _get_nyse_trading_days(start_date, end_date)

    portfolio_value = capital
    open_positions: dict[tuple[str, str], PositionState] = {}
    all_trades: list[dict] = []
    daily_pnl_rows: list[dict] = []

    for day_idx, day in enumerate(trading_days):
        formation_start = day - dt.timedelta(days=rolling_window_days)
        formation_end = day - dt.timedelta(days=1)

        # SOD: find cointegrated pairs for the formation window
        try:
            pairs_df = find_cointegrated_pairs(
                start_date=str(formation_start),
                end_date=str(formation_end),
                timeframe=timeframe,
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

        # Determine all tickers needed today
        all_tickers_today: list[str] = list(
            set(pairs_df["ticker_a"].to_list() + pairs_df["ticker_b"].to_list())
        )

        # Load intraday prices: include zscore lookback window so that the
        # rolling z-score is fully populated when signals are generated.
        # load_processed end_date is exclusive, so pass day + 1 to include today.
        lookback_start = trading_days[max(0, day_idx - zscore_lookback_days)]
        day_end = day + dt.timedelta(days=1)
        day_prices = _load_day_prices(all_tickers_today, lookback_start, day_end, timeframe)

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

        # Generate signals for the full day
        try:
            signals_day = generate_pair_signals_for_day(
                pairs_df=pairs_df,
                date=day,
                intraday_prices=day_prices,
                zscore_window=zscore_window,
                z_entry=z_entry,
                z_exit=z_exit,
                z_stop=z_stop,
                max_holding_bars=max_hold_bars,
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

        # Get sorted unique bars for this day
        bars: list = sorted(signals_day["timestamp"].unique().to_list())
        n_bars_day = len(bars)

        # Build quick-access: (ticker_a, ticker_b, bar_ts) → signal_binary
        sig_map: dict[tuple, int] = {}
        for row in signals_day.iter_rows(named=True):
            key = (row["ticker_a"], row["ticker_b"], row["timestamp"])
            sig_map[key] = row["signal_binary"]

        # Ordered pairs list (by p_value ascending = best signal first)
        pairs_list = pairs_df.sort("p_value").iter_rows(named=True)
        pairs_ordered = list(pairs_list)

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
                should_exit = is_last_bar

                if not is_last_bar:
                    # Check if signal went to 0 (mean reversion / stoploss / time stop)
                    prev_bar_ts = bars[bar_idx - 1] if bar_idx > 0 else None
                    prev_sig = (
                        sig_map.get((ticker_a, ticker_b, prev_bar_ts), 0)
                        if prev_bar_ts
                        else pos.direction
                    )
                    if sig == 0 and prev_sig != 0:
                        should_exit = True

                if not should_exit:
                    continue

                # Determine exit price
                if is_last_bar:
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
                    exit_reason = "signal"
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
                        exit_price_a = (nbar_a["high"] + nbar_a["low"]) / 2.0
                        exit_price_b = (nbar_b["high"] + nbar_b["low"]) / 2.0

                gross, cost, net = _compute_pnl(
                    pos, exit_price_a, exit_price_b, cost_bps
                )

                # Determine exit time
                if is_last_bar:
                    exit_ts = bar_ts
                else:
                    exit_ts = next_bar_ts if next_bar_ts else bar_ts

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

                # Only enter on signal transition (0 → ±1)
                prev_bar_ts = bars[bar_idx - 1] if bar_idx > 0 else None
                prev_sig = (
                    sig_map.get((ticker_a, ticker_b, prev_bar_ts), 0)
                    if prev_bar_ts
                    else 0
                )
                if prev_sig != 0:
                    continue

                # Execute at midpoint of next bar
                bars_a = bar_lookup.get(ticker_a, {})
                bars_b = bar_lookup.get(ticker_b, {})
                nbar_a = bars_a.get(next_bar_ts)
                nbar_b = bars_b.get(next_bar_ts)
                if nbar_a is None or nbar_b is None:
                    continue

                entry_price_a = (nbar_a["high"] + nbar_a["low"]) / 2.0
                entry_price_b = (nbar_b["high"] + nbar_b["low"]) / 2.0

                if entry_price_a <= 0 or entry_price_b <= 0:
                    continue

                shares_a = notional_per_pair / entry_price_a
                shares_b = pair_row["hedge_ratio"] * shares_a

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
