"""
Vectorized backtesting engine.

Simulates mean-reversion pairs trades using pre-computed signals.
Tracks positions, calculates P&L, and applies transaction costs.
"""

from pathlib import Path

import polars as pl
from tqdm import tqdm

from analysis.preprocessing import load_processed
from analysis.spread import build_spread_frame
from strategy.signals import apply_signals_with_holding, generate_signals
from utils.config import CONFIG


def backtest_pair(
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float,
    timeframe: str = "1min",
    z_entry: float | None = None,
    z_exit: float | None = None,
    z_stop: float | None = None,
    max_holding_minutes: int | None = None,
    zscore_window: int | None = None,
    transaction_cost_bps: float | None = None,
    processed_dir: str | None = None,
) -> pl.DataFrame:
    """
    Backtest a single pair.

    Steps:
        1. Load processed data for both tickers.
        2. Calculate spread and z-score.
        3. Generate and apply signals.
        4. Extract trades from position changes.
        5. Calculate P&L with transaction costs.

    Args:
        ticker_a:             First ticker.
        ticker_b:             Second ticker.
        hedge_ratio:          Cointegration hedge ratio.
        timeframe:            Data timeframe to use.
        z_entry:              Z-score entry threshold.
        z_exit:               Z-score exit threshold.
        z_stop:               Z-score stop loss threshold.
        max_holding_minutes:  Max position holding time.
        zscore_window:        Rolling z-score window.
        transaction_cost_bps: Cost per leg in basis points.
        processed_dir:        Processed data directory.

    Returns:
        Trade log as Polars DataFrame with columns:
        [pair, entry_time, exit_time, direction, entry_spread, exit_spread,
         holding_minutes, pnl_gross, pnl_net, transaction_cost, reason_exit]
    """
    processed_dir = processed_dir or CONFIG["processed_dir"]
    transaction_cost_bps = (
        transaction_cost_bps
        if transaction_cost_bps is not None
        else CONFIG["transaction_cost_bps"]
    )

    # Load data
    df_a = load_processed(ticker_a, timeframe, processed_dir)
    df_b = load_processed(ticker_b, timeframe, processed_dir)

    # Build spread with z-score
    spread_df = build_spread_frame(df_a, df_b, hedge_ratio, zscore_window)

    # Drop rows with null z-scores (initial warm-up period)
    spread_df = spread_df.drop_nulls(subset=["z_score"])

    if len(spread_df) == 0:
        return _empty_trade_log()

    # Generate signals
    signals = generate_signals(
        spread_df, z_entry, z_exit, z_stop, max_holding_minutes
    )

    # Apply signal logic with position tracking
    signals_with_pos = apply_signals_with_holding(signals, max_holding_minutes)

    # Merge spread data back for price info
    full = spread_df.join(
        signals_with_pos.select(
            ["timestamp", "position", "entry_bar", "exit_bar", "exit_reason"]
        ),
        on="timestamp",
        how="inner",
    )

    # Extract trades
    trades = _extract_trades(
        full, ticker_a, ticker_b, transaction_cost_bps
    )

    return trades


def _extract_trades(
    df: pl.DataFrame,
    ticker_a: str,
    ticker_b: str,
    transaction_cost_bps: float,
) -> pl.DataFrame:
    """
    Extract individual trades from a DataFrame with position tracking.

    Scans entry_bar / exit_bar columns to pair up entries and exits,
    then calculates P&L for each trade.
    """
    pair_label = f"{ticker_a}/{ticker_b}"

    timestamps = df["timestamp"].to_list()
    spreads = df["spread"].to_list()
    positions = df["position"].to_list()
    entry_bars = df["entry_bar"].to_list()
    exit_bars = df["exit_bar"].to_list()
    exit_reasons = df["exit_reason"].to_list()
    prices_a = df["price_a"].to_list()
    prices_b = df["price_b"].to_list()
    hedge_ratios = df.get_column("hedge_ratio").to_list() if "hedge_ratio" in df.columns else None

    trades: list[dict] = []
    pending_entry = None

    for i in range(len(timestamps)):
        if entry_bars[i]:
            pending_entry = {
                "entry_time": timestamps[i],
                "entry_spread": spreads[i],
                "direction": "long" if positions[i] == 1 else "short",
                "entry_price_a": prices_a[i],
                "entry_price_b": prices_b[i],
                "entry_index": i,
            }

        if exit_bars[i] and pending_entry is not None:
            entry_time = pending_entry["entry_time"]
            exit_time = timestamps[i]

            # Calculate holding period
            if hasattr(exit_time, "timestamp") and hasattr(
                entry_time, "timestamp"
            ):
                holding_minutes = (
                    exit_time - entry_time
                ).total_seconds() / 60
            else:
                holding_minutes = 0.0

            entry_spread = pending_entry["entry_spread"]
            exit_spread = spreads[i]

            # P&L depends on direction
            if pending_entry["direction"] == "long":
                pnl_gross = exit_spread - entry_spread
            else:
                pnl_gross = entry_spread - exit_spread

            # Transaction cost calculation - FIXED
            # We need the hedge ratio used to construct the spread
            # Since spread = price_a - hedge_ratio * price_b
            # A trade involves: 1 share of A, hedge_ratio shares of B
            # Get hedge ratio from the row (will be added when we pass it through)
            # For now, reconstruct it from the spread
            entry_price_a = pending_entry["entry_price_a"]
            entry_price_b = pending_entry["entry_price_b"]
            exit_price_a = prices_a[i]
            exit_price_b = prices_b[i]

            # Reconstruct hedge ratio from spread: spread = price_a - hr * price_b
            # We'll get this from the parent function via the dataframe
            # For backward compatibility, estimate if not available
            if hedge_ratios is not None:
                hedge_ratio = hedge_ratios[pending_entry["entry_index"]]
            else:
                # Fallback: estimate from the spread values
                # This is imperfect but maintains backward compatibility
                hedge_ratio = (entry_price_a - entry_spread) / entry_price_b if entry_price_b != 0 else 1.0

            # Calculate actual notional traded
            # Entry: buy/sell 1 share of A + sell/buy hedge_ratio shares of B
            entry_notional = abs(entry_price_a) + abs(hedge_ratio * entry_price_b)
            exit_notional = abs(exit_price_a) + abs(hedge_ratio * exit_price_b)

            # Transaction cost = (entry + exit notional) * bps / 10000
            # Note: bps is per leg, and we have 2 legs (A and B) at entry and exit
            # So total is 4 legs, but we count each leg's notional separately
            cost = (entry_notional + exit_notional) * (transaction_cost_bps / 10000)

            trades.append(
                {
                    "pair": pair_label,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "direction": pending_entry["direction"],
                    "entry_spread": entry_spread,
                    "exit_spread": exit_spread,
                    "holding_minutes": holding_minutes,
                    "pnl_gross": pnl_gross,
                    "pnl_net": pnl_gross - cost,
                    "transaction_cost": cost,
                    "reason_exit": exit_reasons[i],
                }
            )
            pending_entry = None

    if not trades:
        return _empty_trade_log()

    return pl.DataFrame(trades)


def _empty_trade_log() -> pl.DataFrame:
    """Return an empty trade log with the correct schema."""
    return pl.DataFrame(
        schema={
            "pair": pl.Utf8,
            "entry_time": pl.Datetime,
            "exit_time": pl.Datetime,
            "direction": pl.Utf8,
            "entry_spread": pl.Float64,
            "exit_spread": pl.Float64,
            "holding_minutes": pl.Float64,
            "pnl_gross": pl.Float64,
            "pnl_net": pl.Float64,
            "transaction_cost": pl.Float64,
            "reason_exit": pl.Utf8,
        }
    )


def backtest_all_pairs(
    pairs_df: pl.DataFrame,
    timeframe: str = "1min",
    max_pairs: int | None = None,
    **kwargs,
) -> pl.DataFrame:
    """
    Backtest multiple pairs and aggregate all trades.

    Args:
        pairs_df:  Cointegration results DataFrame with columns
                   [ticker_a, ticker_b, hedge_ratio, ...].
        timeframe: Data timeframe.
        max_pairs: Max number of pairs to backtest. Defaults to CONFIG['max_pairs'].
        **kwargs:  Additional parameters passed to backtest_pair().

    Returns:
        Combined trade log for all pairs.
    """
    max_pairs = max_pairs or CONFIG["max_pairs"]
    pairs_to_test = pairs_df.head(max_pairs)

    all_trades: list[pl.DataFrame] = []

    for row in tqdm(
        pairs_to_test.iter_rows(named=True),
        total=len(pairs_to_test),
        desc="Backtesting pairs",
    ):
        trades = backtest_pair(
            ticker_a=row["ticker_a"],
            ticker_b=row["ticker_b"],
            hedge_ratio=row["hedge_ratio"],
            timeframe=timeframe,
            **kwargs,
        )
        if len(trades) > 0:
            all_trades.append(trades)

    if not all_trades:
        return _empty_trade_log()

    combined = pl.concat(all_trades).sort("entry_time")
    return combined


def save_trades(
    trades: pl.DataFrame,
    results_dir: str | None = None,
) -> Path:
    """
    Save trade log to Parquet.

    Args:
        trades:      Trade log DataFrame.
        results_dir: Output directory.

    Returns:
        Path to saved file.
    """
    results_dir = results_dir or CONFIG["results_dir"]
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trades.parquet"
    trades.write_parquet(out_path)
    print(f"Saved {len(trades)} trades to {out_path}")
    return out_path
