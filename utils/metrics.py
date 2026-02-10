"""
Performance metric calculations.

All functions assume pre-loaded trade data as Polars DataFrames or Series.
"""

import math

import polars as pl


def sharpe_ratio(
    returns: pl.Series,
    annualization_factor: float = 252.0,
) -> float:
    """
    Calculate annualized Sharpe ratio (assuming risk-free rate = 0).

    Args:
        returns:               Series of period returns.
        annualization_factor:  Periods per year (252 for daily, 252*390 for 1-min).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    if std_ret is None or std_ret == 0:
        return 0.0
    return float(mean_ret / std_ret * math.sqrt(annualization_factor))


def max_drawdown(equity_curve: pl.Series) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve: Cumulative equity series.

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.15 = 15% drawdown).
    """
    if len(equity_curve) < 2:
        return 0.0

    df = pl.DataFrame({"equity": equity_curve})
    df = df.with_columns(
        pl.col("equity").cum_max().alias("running_max")
    )
    df = df.with_columns(
        ((pl.col("equity") - pl.col("running_max")) / pl.col("running_max")).alias(
            "drawdown"
        )
    )

    min_dd = df["drawdown"].min()
    return float(min_dd) if min_dd is not None else 0.0


def win_rate(trades: pl.DataFrame, pnl_col: str = "pnl_net") -> float:
    """
    Calculate win rate (fraction of profitable trades).

    Args:
        trades:  Trade log DataFrame.
        pnl_col: Column name for P&L values.

    Returns:
        Win rate as fraction [0, 1].
    """
    if len(trades) == 0:
        return 0.0
    wins = trades.filter(pl.col(pnl_col) > 0)
    return len(wins) / len(trades)


def profit_factor(trades: pl.DataFrame, pnl_col: str = "pnl_net") -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        trades:  Trade log DataFrame.
        pnl_col: Column name for P&L values.

    Returns:
        Profit factor (> 1 is profitable). Returns inf if no losses.
    """
    if len(trades) == 0:
        return 0.0

    gross_profit = trades.filter(pl.col(pnl_col) > 0)[pnl_col].sum()
    gross_loss = trades.filter(pl.col(pnl_col) < 0)[pnl_col].sum()

    if gross_profit is None:
        gross_profit = 0.0
    if gross_loss is None or gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(abs(gross_profit / gross_loss))


def avg_trade_pnl(trades: pl.DataFrame, pnl_col: str = "pnl_net") -> float:
    """
    Calculate average P&L per trade.

    Args:
        trades:  Trade log DataFrame.
        pnl_col: Column name for P&L values.

    Returns:
        Average P&L per trade.
    """
    if len(trades) == 0:
        return 0.0
    mean_val = trades[pnl_col].mean()
    return float(mean_val) if mean_val is not None else 0.0


def avg_holding_period(trades: pl.DataFrame) -> float:
    """
    Calculate average holding period in minutes.

    Args:
        trades: Trade log DataFrame with 'holding_minutes' column.

    Returns:
        Average holding period in minutes.
    """
    if len(trades) == 0 or "holding_minutes" not in trades.columns:
        return 0.0
    mean_val = trades["holding_minutes"].mean()
    return float(mean_val) if mean_val is not None else 0.0


def total_return(trades: pl.DataFrame, pnl_col: str = "pnl_net") -> float:
    """
    Calculate total return from all trades.

    Args:
        trades:  Trade log DataFrame.
        pnl_col: Column name for P&L values.

    Returns:
        Sum of all trade P&L.
    """
    if len(trades) == 0:
        return 0.0
    total = trades[pnl_col].sum()
    return float(total) if total is not None else 0.0


def calculate_all_metrics(
    trades: pl.DataFrame,
    equity_curve: pl.Series | None = None,
    capital: float | None = None,
) -> dict:
    """
    Calculate all performance metrics.

    Args:
        trades:       Trade log DataFrame.
        equity_curve: Optional cumulative equity series.
        capital:      Starting capital for return calculations.

    Returns:
        Dict with all computed metrics.
    """
    from utils.config import CONFIG as cfg

    capital = capital or cfg["capital"]

    metrics = {
        "total_trades": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "avg_pnl_per_trade": avg_trade_pnl(trades),
        "total_pnl": total_return(trades),
        "avg_holding_minutes": avg_holding_period(trades),
        "total_return_pct": total_return(trades) / capital * 100,
    }

    if equity_curve is not None and len(equity_curve) > 1:
        daily_returns = equity_curve.diff() / equity_curve.shift(1)
        daily_returns = daily_returns.drop_nulls()
        metrics["sharpe_ratio"] = sharpe_ratio(daily_returns)
        metrics["max_drawdown"] = max_drawdown(equity_curve)
    else:
        metrics["sharpe_ratio"] = 0.0
        metrics["max_drawdown"] = 0.0

    # Breakdown by exit reason
    if "reason_exit" in trades.columns and len(trades) > 0:
        reason_counts = (
            trades.group_by("reason_exit")
            .len()
            .sort("len", descending=True)
        )
        metrics["exit_reason_breakdown"] = {
            row["reason_exit"]: row["len"]
            for row in reason_counts.iter_rows(named=True)
        }

    return metrics
