"""
Signal generation module.

Generates entry and exit signals based on z-score thresholds
for a mean-reversion pairs trading strategy.
"""

import polars as pl

from utils.config import CONFIG


def generate_signals(
    spread_df: pl.DataFrame,
    z_entry: float | None = None,
    z_exit: float | None = None,
    z_stop: float | None = None,
    max_holding_minutes: int | None = None,
) -> pl.DataFrame:
    """
    Generate trading signals from a spread DataFrame with z-scores.

    Signal logic:
        - Entry long  (buy spread): z_score < -z_entry
        - Entry short (sell spread): z_score > +z_entry
        - Exit: z_score crosses 0 (changes sign toward zero)
        - Stop loss: |z_score| > z_stop
        - Max hold: position held longer than max_holding_minutes

    Args:
        spread_df:           DataFrame with columns [timestamp, z_score, ...].
        z_entry:             Z-score threshold for entry (absolute value).
        z_exit:              Z-score threshold for exit (crossing toward).
        z_stop:              Z-score threshold for stop loss (absolute value).
        max_holding_minutes: Maximum minutes to hold a position.

    Returns:
        Polars DataFrame with columns:
        [timestamp, z_score, entry_long, entry_short, exit, stop_loss]
    """
    z_entry = z_entry if z_entry is not None else CONFIG["z_score_entry"]
    z_exit = z_exit if z_exit is not None else CONFIG["z_score_exit"]
    z_stop = z_stop if z_stop is not None else CONFIG["z_score_stop"]
    max_holding_minutes = (
        max_holding_minutes
        if max_holding_minutes is not None
        else CONFIG["max_holding_minutes"]
    )

    result = spread_df.select(["timestamp", "z_score"]).with_columns(
        # Entry signals
        (pl.col("z_score") < -z_entry).alias("entry_long"),
        (pl.col("z_score") > z_entry).alias("entry_short"),
        # Exit: z-score crosses zero (absolute value below exit threshold)
        (pl.col("z_score").abs() <= z_exit).alias("exit_mean_reversion"),
        # Stop loss
        (pl.col("z_score").abs() > z_stop).alias("stop_loss"),
    )

    # Combine exit signals
    result = result.with_columns(
        (pl.col("exit_mean_reversion") | pl.col("stop_loss")).alias("exit")
    )

    return result


def apply_signals_with_holding(
    signals_df: pl.DataFrame,
    max_holding_minutes: int | None = None,
) -> pl.DataFrame:
    """
    Apply signals sequentially, enforcing max holding period and
    preventing overlapping positions. Uses a stateful scan.

    Adds columns:
        - position: 1 (long spread), -1 (short spread), 0 (flat)
        - entry_bar: boolean, True on the bar where position is opened
        - exit_bar: boolean, True on the bar where position is closed

    Args:
        signals_df:          DataFrame from generate_signals().
        max_holding_minutes: Max holding period.

    Returns:
        DataFrame with position tracking columns added.
    """
    max_holding_minutes = (
        max_holding_minutes
        if max_holding_minutes is not None
        else CONFIG["max_holding_minutes"]
    )

    timestamps = signals_df["timestamp"].to_list()
    z_scores = signals_df["z_score"].to_list()
    entry_long = signals_df["entry_long"].to_list()
    entry_short = signals_df["entry_short"].to_list()
    exits = signals_df["exit"].to_list()
    stop_losses = signals_df["stop_loss"].to_list()

    n = len(timestamps)
    positions = [0] * n
    entry_bars = [False] * n
    exit_bars = [False] * n
    exit_reasons = [""] * n

    current_pos = 0
    entry_time = None

    for i in range(n):
        ts = timestamps[i]

        if current_pos != 0:
            # Check exit conditions
            should_exit = False
            reason = ""

            if exits[i]:
                should_exit = True
                reason = "stop_loss" if stop_losses[i] else "mean_reversion"

            # Check max holding period
            if entry_time is not None:
                if hasattr(ts, "timestamp") and hasattr(entry_time, "timestamp"):
                    held_minutes = (ts - entry_time).total_seconds() / 60
                else:
                    held_minutes = 0
                if held_minutes >= max_holding_minutes:
                    should_exit = True
                    reason = "max_hold"

            if should_exit:
                exit_bars[i] = True
                exit_reasons[i] = reason
                current_pos = 0
                entry_time = None
            else:
                positions[i] = current_pos
                continue

        # Check for new entry (only if flat)
        if current_pos == 0:
            if entry_long[i]:
                current_pos = 1
                entry_time = ts
                entry_bars[i] = True
                positions[i] = current_pos
            elif entry_short[i]:
                current_pos = -1
                entry_time = ts
                entry_bars[i] = True
                positions[i] = current_pos

    return signals_df.with_columns(
        pl.Series("position", positions),
        pl.Series("entry_bar", entry_bars),
        pl.Series("exit_bar", exit_bars),
        pl.Series("exit_reason", exit_reasons),
    )
