"""
Multi-pair portfolio management module.

Handles position sizing, capital allocation, and portfolio-level
return calculations across multiple concurrent pairs.
"""

import polars as pl

from utils.config import CONFIG


def allocate_capital(
    pairs_df: pl.DataFrame,
    capital: float | None = None,
    max_pairs: int | None = None,
) -> pl.DataFrame:
    """
    Allocate capital equally across selected pairs.

    Args:
        pairs_df:  Cointegration results DataFrame (sorted by quality).
        capital:   Total portfolio capital.
        max_pairs: Maximum concurrent pairs.

    Returns:
        DataFrame with added 'allocation' and 'position_size' columns.
    """
    capital = capital or CONFIG["capital"]
    max_pairs = max_pairs or CONFIG["max_pairs"]

    selected = pairs_df.head(max_pairs)
    n_pairs = len(selected)

    if n_pairs == 0:
        return selected.with_columns(
            pl.lit(0.0).alias("allocation"),
            pl.lit(0.0).alias("position_size"),
        )

    per_pair = capital / n_pairs

    selected = selected.with_columns(
        pl.lit(per_pair).alias("allocation"),
        pl.lit(per_pair).alias("position_size"),
    )

    return selected


def build_equity_curve(
    trades: pl.DataFrame,
    capital: float | None = None,
) -> pl.DataFrame:
    """
    Build a portfolio-level equity curve from a trade log.

    Each trade's P&L is added to the cumulative equity at the trade's
    exit time.

    Args:
        trades:  Trade log DataFrame with columns [exit_time, pnl_net].
        capital: Starting capital.

    Returns:
        Polars DataFrame with columns [timestamp, equity, drawdown].
    """
    capital = capital or CONFIG["capital"]

    if len(trades) == 0:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "equity": pl.Float64,
                "drawdown": pl.Float64,
            }
        )

    # Sort trades by exit time and accumulate P&L
    sorted_trades = trades.sort("exit_time")

    equity_df = sorted_trades.select(
        pl.col("exit_time").alias("timestamp"),
        pl.col("pnl_net"),
    )

    equity_df = equity_df.with_columns(
        (pl.lit(capital) + pl.col("pnl_net").cum_sum()).alias("equity")
    )

    # Calculate drawdown
    equity_df = equity_df.with_columns(
        pl.col("equity").cum_max().alias("running_max")
    )
    equity_df = equity_df.with_columns(
        (
            (pl.col("equity") - pl.col("running_max")) / pl.col("running_max")
        ).alias("drawdown")
    )

    return equity_df.select(["timestamp", "equity", "drawdown"])


def build_daily_returns(
    equity_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregate equity curve to daily returns.

    Args:
        equity_df: Equity curve DataFrame with [timestamp, equity].

    Returns:
        DataFrame with columns [date, equity, daily_return].
    """
    if len(equity_df) == 0:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "equity": pl.Float64,
                "daily_return": pl.Float64,
            }
        )

    daily = (
        equity_df.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("equity").last())
        .sort("date")
    )

    daily = daily.with_columns(
        (pl.col("equity") / pl.col("equity").shift(1) - 1).alias("daily_return")
    )

    return daily


def monthly_returns_table(
    equity_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a monthly returns table for heatmap visualization.

    Args:
        equity_df: Equity curve DataFrame.

    Returns:
        DataFrame with columns [year, month, monthly_return_pct].
    """
    if len(equity_df) == 0:
        return pl.DataFrame(
            schema={
                "year": pl.Int32,
                "month": pl.Int32,
                "monthly_return_pct": pl.Float64,
            }
        )

    monthly = (
        equity_df.with_columns(
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month"),
        )
        .group_by(["year", "month"])
        .agg(
            pl.col("equity").first().alias("equity_start"),
            pl.col("equity").last().alias("equity_end"),
        )
        .sort(["year", "month"])
    )

    monthly = monthly.with_columns(
        (
            (pl.col("equity_end") - pl.col("equity_start"))
            / pl.col("equity_start")
            * 100
        ).alias("monthly_return_pct")
    )

    return monthly.select(["year", "month", "monthly_return_pct"])


def pair_performance_summary(
    trades: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calculate per-pair performance summary.

    Args:
        trades: Trade log DataFrame.

    Returns:
        DataFrame with per-pair statistics:
        [pair, num_trades, win_rate, avg_pnl, total_pnl, avg_holding_min].
    """
    if len(trades) == 0:
        return pl.DataFrame(
            schema={
                "pair": pl.Utf8,
                "num_trades": pl.UInt32,
                "win_rate": pl.Float64,
                "avg_pnl": pl.Float64,
                "total_pnl": pl.Float64,
                "avg_holding_min": pl.Float64,
            }
        )

    summary = (
        trades.group_by("pair")
        .agg(
            pl.len().alias("num_trades"),
            (pl.col("pnl_net") > 0).mean().alias("win_rate"),
            pl.col("pnl_net").mean().alias("avg_pnl"),
            pl.col("pnl_net").sum().alias("total_pnl"),
            pl.col("holding_minutes").mean().alias("avg_holding_min"),
        )
        .sort("total_pnl", descending=True)
    )

    return summary
