"""
Backtesting engine for cointegrated pairs trading.

Provides PairBacktester (single pair) and PortfolioBacktester (multiple pairs)
with configurable parameters, transaction costs, and risk controls.
"""

from __future__ import annotations

import polars as pl

from analysis.preprocessing import load_processed
from strategy.metrics import BacktestResult, Trade, calculate_metrics
from strategy.signals import compute_spread, compute_zscore
from utils.config import CONFIG


class PairBacktester:
    """
    Backtest a single cointegrated pair.

    Simulates trades using z-score signals with configurable entry/exit
    thresholds, stop losses, time stops, and transaction costs.
    """

    def __init__(
        self,
        z_entry: float | None = None,
        z_exit: float | None = None,
        z_stop: float | None = None,
        max_holding_bars: int | None = None,
        transaction_cost_bps: float | None = None,
        zscore_window: int | None = None,
        notional_per_pair: float | None = None,
    ):
        """
        Args:
            z_entry:              Entry threshold. Defaults to CONFIG value.
            z_exit:               Exit threshold. Defaults to CONFIG value.
            z_stop:               Stop-loss threshold. Defaults to CONFIG value.
                                  Use float('inf') to disable.
            max_holding_bars:     Max bars to hold. None = no time stop.
            transaction_cost_bps: Cost per leg in basis points. Defaults to CONFIG.
            zscore_window:        Rolling window for z-score. Defaults to CONFIG.
            notional_per_pair:    Dollar notional per leg at entry.
        """
        self.z_entry = z_entry if z_entry is not None else CONFIG["z_score_entry"]
        self.z_exit = z_exit if z_exit is not None else CONFIG["z_score_exit"]
        self.z_stop = z_stop if z_stop is not None else CONFIG["z_score_stop"]
        self.max_holding_bars = max_holding_bars
        self.transaction_cost_bps = (
            transaction_cost_bps
            if transaction_cost_bps is not None
            else CONFIG["transaction_cost_bps"]
        )
        self.zscore_window = zscore_window or CONFIG["zscore_window"]
        self.notional_per_pair = notional_per_pair or (
            CONFIG["capital"] / CONFIG["max_pairs"]
        )

    def backtest_pair(
        self,
        ticker_a: str,
        ticker_b: str,
        hedge_ratio: float,
        timeframe: str = "daily",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Trade]:
        """
        Run backtest for a single pair over a period.

        Args:
            ticker_a:    First ticker.
            ticker_b:    Second ticker.
            hedge_ratio: Cointegration hedge ratio.
            timeframe:   Data timeframe.
            start_date:  Trading period start.
            end_date:    Trading period end.

        Returns:
            List of completed Trade objects.
        """
        df_a = load_processed(ticker_a, timeframe, start_date=start_date, end_date=end_date)
        df_b = load_processed(ticker_b, timeframe, start_date=start_date, end_date=end_date)

        # Inner join on timestamp
        merged = df_a.select(["timestamp", "close"]).rename({"close": "close_a"}).join(
            df_b.select(["timestamp", "close"]).rename({"close": "close_b"}),
            on="timestamp",
            how="inner",
        )

        if len(merged) < self.zscore_window + 1:
            return []

        prices_a = merged["close_a"]
        prices_b = merged["close_b"]
        timestamps = merged["timestamp"].to_list()

        spread = compute_spread(prices_a, prices_b, hedge_ratio)
        zscore = compute_zscore(spread, window=self.zscore_window)

        # Walk through bars and simulate trades
        zvals = zscore.to_list()
        pa = prices_a.to_list()
        pb = prices_b.to_list()
        n = len(zvals)

        trades: list[Trade] = []
        position = 0  # +1 long spread, -1 short spread, 0 flat
        entry_idx = 0
        entry_z = 0.0
        bars_held = 0

        for i in range(n):
            z = zvals[i]
            if z is None:
                # Close any open position at end of valid data
                if position != 0:
                    trades.append(self._close_trade(
                        ticker_a, ticker_b, position, entry_idx, i,
                        entry_z, 0.0, pa, pb, timestamps, bars_held,
                        "end_of_data",
                    ))
                    position = 0
                    bars_held = 0
                continue

            if position == 0:
                # Check entry
                if z < -self.z_entry:
                    position = 1
                    entry_idx = i
                    entry_z = z
                    bars_held = 0
                elif z > self.z_entry:
                    position = -1
                    entry_idx = i
                    entry_z = z
                    bars_held = 0
            else:
                bars_held += 1
                exit_reason = None

                # Mean reversion exit
                if position == 1 and z >= -self.z_exit:
                    exit_reason = "mean_reversion"
                elif position == -1 and z <= self.z_exit:
                    exit_reason = "mean_reversion"

                # Stop loss exit (checked after mean reversion so stop takes priority
                # if both fire on same bar)
                if abs(z) > self.z_stop:
                    exit_reason = "stop_loss"

                # Time stop
                if (
                    self.max_holding_bars is not None
                    and bars_held >= self.max_holding_bars
                ):
                    exit_reason = "time_stop"

                if exit_reason:
                    trades.append(self._close_trade(
                        ticker_a, ticker_b, position, entry_idx, i,
                        entry_z, z, pa, pb, timestamps, bars_held,
                        exit_reason,
                    ))
                    position = 0
                    bars_held = 0

        # Close any remaining position at end of data
        if position != 0:
            trades.append(self._close_trade(
                ticker_a, ticker_b, position, entry_idx, n - 1,
                entry_z, zvals[-1] or 0.0, pa, pb, timestamps, bars_held,
                "end_of_data",
            ))

        return trades

    def _close_trade(
        self,
        ticker_a: str,
        ticker_b: str,
        direction: int,
        entry_idx: int,
        exit_idx: int,
        entry_z: float,
        exit_z: float,
        prices_a: list,
        prices_b: list,
        timestamps: list,
        bars_held: int,
        exit_reason: str,
    ) -> Trade:
        """Construct a Trade object from entry/exit indices."""
        pa_entry, pa_exit = prices_a[entry_idx], prices_a[exit_idx]
        pb_entry, pb_exit = prices_b[entry_idx], prices_b[exit_idx]

        # P&L calculation:
        # Long spread (+1): buy A, sell B → profit when spread widens then reverts
        #   P&L_A = notional * (pa_exit - pa_entry) / pa_entry
        #   P&L_B = notional * (pb_entry - pb_exit) / pb_entry  (short B)
        # Short spread (-1): sell A, buy B → mirror
        notional = self.notional_per_pair
        if pa_entry > 0 and pb_entry > 0:
            ret_a = (pa_exit - pa_entry) / pa_entry
            ret_b = (pb_exit - pb_entry) / pb_entry
            if direction == 1:
                pnl_gross = notional * (ret_a - ret_b)
            else:
                pnl_gross = notional * (-ret_a + ret_b)
        else:
            pnl_gross = 0.0

        # Transaction costs: 4 legs total (buy/sell each stock, entry + exit)
        cost = 4 * (self.transaction_cost_bps / 10_000) * notional
        pnl_net = pnl_gross - cost

        return Trade(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            direction=direction,
            entry_time=timestamps[entry_idx],
            exit_time=timestamps[exit_idx],
            entry_zscore=entry_z,
            exit_zscore=exit_z,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            exit_reason=exit_reason,
            holding_bars=bars_held,
            notional=notional,
        )


class PortfolioBacktester:
    """
    Backtest a portfolio of cointegrated pairs.

    Runs PairBacktester on each pair and aggregates results.
    """

    def __init__(
        self,
        capital: float | None = None,
        max_pairs: int | None = None,
        **pair_kwargs,
    ):
        """
        Args:
            capital:      Total portfolio capital. Defaults to CONFIG value.
            max_pairs:    Maximum concurrent pairs. Defaults to CONFIG value.
            **pair_kwargs: Passed to PairBacktester constructor.
        """
        self.capital = capital if capital is not None else CONFIG["capital"]
        self.max_pairs = max_pairs if max_pairs is not None else CONFIG["max_pairs"]
        self.pair_kwargs = pair_kwargs

        # Set notional per pair based on capital and max_pairs
        self.pair_kwargs["notional_per_pair"] = self.capital / self.max_pairs

    def backtest(
        self,
        pairs_df: pl.DataFrame,
        timeframe: str = "daily",
        start_date: str | None = None,
        end_date: str | None = None,
        max_pairs_to_trade: int | None = None,
    ) -> BacktestResult:
        """
        Backtest a portfolio of pairs.

        Takes the top N pairs from pairs_df (sorted by p-value ascending,
        as returned by find_cointegrated_pairs) and runs individual
        pair backtests.

        Args:
            pairs_df:           DataFrame from find_cointegrated_pairs().
            timeframe:          Data timeframe.
            start_date:         Trading period start.
            end_date:           Trading period end.
            max_pairs_to_trade: Override max pairs to trade from this set.
                                Defaults to self.max_pairs.

        Returns:
            BacktestResult with all trades and aggregated metrics.
        """
        n_pairs = max_pairs_to_trade or self.max_pairs
        if len(pairs_df) == 0:
            return BacktestResult(params=self._params_dict())

        # Take top N pairs by p-value (already sorted)
        top_pairs = pairs_df.head(n_pairs)

        pair_bt = PairBacktester(**self.pair_kwargs)
        all_trades: list[Trade] = []

        for row in top_pairs.iter_rows(named=True):
            pair_trades = pair_bt.backtest_pair(
                ticker_a=row["ticker_a"],
                ticker_b=row["ticker_b"],
                hedge_ratio=row["hedge_ratio"],
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            all_trades.extend(pair_trades)

        metrics = calculate_metrics(all_trades, capital=self.capital)

        return BacktestResult(
            trades=all_trades,
            params=self._params_dict(),
            metrics=metrics,
        )

    def _params_dict(self) -> dict:
        """Return dict of all parameters for reproducibility."""
        return {
            "capital": self.capital,
            "max_pairs": self.max_pairs,
            **self.pair_kwargs,
        }
