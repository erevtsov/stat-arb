"""
Performance metrics module.

Calculates risk-adjusted return metrics and equity curves from trade results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


@dataclass
class Trade:
    """A single completed pair trade."""

    ticker_a: str
    ticker_b: str
    direction: int           # +1 = long spread, -1 = short spread
    entry_time: datetime
    exit_time: datetime
    entry_zscore: float
    exit_zscore: float
    pnl_gross: float         # before transaction costs
    pnl_net: float           # after transaction costs
    exit_reason: str          # "mean_reversion", "stop_loss", "time_stop", "end_of_data"
    holding_bars: int
    notional: float           # dollar value of one leg at entry


@dataclass
class BacktestResult:
    """Aggregated results from a backtest run."""

    trades: list[Trade] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def calculate_metrics(
    trades: list[Trade],
    capital: float | None = None,
    annualization_factor: float = 252.0,
) -> dict:
    """
    Calculate performance metrics from a list of completed trades.

    Args:
        trades:                List of Trade objects.
        capital:               Starting capital (for return calculation).
                               Defaults to CONFIG value.
        annualization_factor:  Trading days per year for annualization.

    Returns:
        Dict with performance metrics.
    """
    from utils.config import CONFIG
    capital = capital if capital is not None else CONFIG["capital"]

    if not trades:
        return {
            "num_trades": 0,
            "total_pnl_gross": 0.0,
            "total_pnl_net": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl_net": 0.0,
            "avg_holding_bars": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "tail_ratio": 0.0,
            "pct_mean_reversion": 0.0,
            "pct_stop_loss": 0.0,
            "pct_time_stop": 0.0,
        }

    pnls = np.array([t.pnl_net for t in trades])
    pnls_gross = np.array([t.pnl_gross for t in trades])
    num_trades = len(trades)

    # Basic P&L
    total_pnl_gross = float(pnls_gross.sum())
    total_pnl_net = float(pnls.sum())
    total_return_pct = total_pnl_net / capital * 100.0

    # Win rate and profit factor
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    win_rate = len(winners) / num_trades if num_trades > 0 else 0.0
    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualized, from per-trade returns)
    if num_trades > 1 and pnls.std() > 0:
        # Estimate trades per year from the data
        first_trade = min(t.entry_time for t in trades)
        last_trade = max(t.exit_time for t in trades)
        span_days = max((last_trade - first_trade).total_seconds() / 86400, 1.0)
        trades_per_year = num_trades / span_days * annualization_factor
        sharpe_ratio = float(
            (pnls.mean() / pnls.std()) * np.sqrt(trades_per_year)
        )
    else:
        sharpe_ratio = 0.0

    # Sortino ratio (uses downside deviation)
    downside_returns = pnls[pnls < 0]
    if len(downside_returns) > 1 and pnls.mean() != 0:
        downside_std = float(np.sqrt(np.mean(downside_returns**2)))
        if downside_std > 0:
            first_trade = min(t.entry_time for t in trades)
            last_trade = max(t.exit_time for t in trades)
            span_days = max((last_trade - first_trade).total_seconds() / 86400, 1.0)
            trades_per_year = num_trades / span_days * annualization_factor
            sortino_ratio = float(
                (pnls.mean() / downside_std) * np.sqrt(trades_per_year)
            )
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0

    # Maximum drawdown from cumulative P&L
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_drawdown = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    max_drawdown_pct = max_drawdown / capital * 100.0

    # VaR and CVaR (95%)
    if num_trades >= 20:
        var_95 = float(-np.percentile(pnls, 5))
        cvar_95 = float(-pnls[pnls <= -var_95].mean()) if np.any(pnls <= -var_95) else var_95
    else:
        var_95 = float(-pnls.min()) if num_trades > 0 else 0.0
        cvar_95 = var_95

    # Tail ratio (95th percentile gain / 5th percentile loss)
    if num_trades >= 20:
        upper = float(np.percentile(pnls, 95))
        lower = float(abs(np.percentile(pnls, 5)))
        tail_ratio = upper / lower if lower > 0 else float("inf")
    else:
        tail_ratio = 0.0

    # Exit reason breakdown
    exit_reasons = [t.exit_reason for t in trades]
    pct_mean_reversion = exit_reasons.count("mean_reversion") / num_trades
    pct_stop_loss = exit_reasons.count("stop_loss") / num_trades
    pct_time_stop = exit_reasons.count("time_stop") / num_trades

    # Average holding time
    avg_holding_bars = float(np.mean([t.holding_bars for t in trades]))

    return {
        "num_trades": num_trades,
        "total_pnl_gross": total_pnl_gross,
        "total_pnl_net": total_pnl_net,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_net": float(pnls.mean()),
        "avg_holding_bars": avg_holding_bars,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "tail_ratio": tail_ratio,
        "pct_mean_reversion": pct_mean_reversion,
        "pct_stop_loss": pct_stop_loss,
        "pct_time_stop": pct_time_stop,
    }


def build_equity_curve(
    trades: list[Trade],
    capital: float | None = None,
) -> pl.DataFrame:
    """
    Build time-indexed equity curve from trades.

    Args:
        trades:  List of Trade objects (sorted by exit_time).
        capital: Starting capital.

    Returns:
        Polars DataFrame with [timestamp, equity, drawdown_pct].
    """
    from utils.config import CONFIG
    capital = capital if capital is not None else CONFIG["capital"]

    if not trades:
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime,
            "equity": pl.Float64,
            "drawdown_pct": pl.Float64,
        })

    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    timestamps = []
    equities = []
    equity = capital

    for t in sorted_trades:
        equity += t.pnl_net
        timestamps.append(t.exit_time)
        equities.append(equity)

    equity_arr = np.array(equities)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown_pct = ((running_max - equity_arr) / running_max * 100.0).tolist()

    return pl.DataFrame({
        "timestamp": timestamps,
        "equity": equities,
        "drawdown_pct": drawdown_pct,
    })


def bootstrap_sharpe(
    trades: list[Trade],
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict:
    """
    Bootstrap confidence intervals for Sharpe ratio.

    Resamples trades with replacement to estimate the sampling
    distribution of the Sharpe ratio.

    Args:
        trades:        List of Trade objects.
        n_iterations:  Number of bootstrap samples.
        confidence:    Confidence level (e.g., 0.95 for 95% CI).
        seed:          Random seed for reproducibility.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, std_error.
    """
    if len(trades) < 10:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_error": 0.0,
        }

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl_net for t in trades])
    n = len(pnls)

    sharpes = []
    for _ in range(n_iterations):
        sample = rng.choice(pnls, size=n, replace=True)
        if sample.std() > 0:
            # Simple per-trade Sharpe (no annualization in bootstrap)
            sharpes.append(float(sample.mean() / sample.std()))
        else:
            sharpes.append(0.0)

    sharpes = np.array(sharpes)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(sharpes, alpha / 2 * 100))
    ci_upper = float(np.percentile(sharpes, (1 - alpha / 2) * 100))

    return {
        "point_estimate": float(pnls.mean() / pnls.std()) if pnls.std() > 0 else 0.0,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": float(sharpes.std()),
    }
