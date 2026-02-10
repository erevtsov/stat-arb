"""
Spread and z-score calculation module.

Calculates the spread between two cointegrated stocks and
computes rolling z-scores for mean-reversion signal generation.
"""

import polars as pl

from utils.config import CONFIG


def calculate_spread(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    hedge_ratio: float,
) -> pl.DataFrame:
    """
    Calculate spread between two price series.

    spread = price_A - hedge_ratio * price_B

    Args:
        df_a:        DataFrame for stock A with columns [timestamp, ..., close].
        df_b:        DataFrame for stock B with columns [timestamp, ..., close].
        hedge_ratio: Beta coefficient from cointegration regression.

    Returns:
        Polars DataFrame with columns:
        [timestamp, price_a, price_b, spread]
    """
    # Rename close columns before joining
    a = df_a.select(
        pl.col("timestamp"),
        pl.col("close").alias("price_a"),
    )
    b = df_b.select(
        pl.col("timestamp"),
        pl.col("close").alias("price_b"),
    )

    merged = a.join(b, on="timestamp", how="inner").sort("timestamp")

    merged = merged.with_columns(
        (pl.col("price_a") - hedge_ratio * pl.col("price_b")).alias("spread")
    )

    return merged


def calculate_zscore(
    spread_df: pl.DataFrame,
    window: int | None = None,
) -> pl.DataFrame:
    """
    Calculate rolling z-score of the spread.

    z_score = (spread - rolling_mean) / rolling_std

    Args:
        spread_df: DataFrame with at least [timestamp, spread] columns.
        window:    Rolling window size in bars. Defaults to CONFIG['zscore_window'].

    Returns:
        Polars DataFrame with additional columns:
        [rolling_mean, rolling_std, z_score]
    """
    window = window or CONFIG["zscore_window"]

    result = spread_df.with_columns(
        pl.col("spread").rolling_mean(window_size=window).alias("rolling_mean"),
        pl.col("spread").rolling_std(window_size=window).alias("rolling_std"),
    )

    result = result.with_columns(
        pl.when(pl.col("rolling_std") > 0)
        .then(
            (pl.col("spread") - pl.col("rolling_mean")) / pl.col("rolling_std")
        )
        .otherwise(0.0)
        .alias("z_score")
    )

    return result


def build_spread_frame(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    hedge_ratio: float,
    window: int | None = None,
) -> pl.DataFrame:
    """
    Convenience function: calculate spread and z-score in one step.

    Args:
        df_a:        DataFrame for stock A.
        df_b:        DataFrame for stock B.
        hedge_ratio: Hedge ratio (beta).
        window:      Rolling z-score window.

    Returns:
        Polars DataFrame with columns:
        [timestamp, price_a, price_b, spread, rolling_mean, rolling_std, z_score]
    """
    spread_df = calculate_spread(df_a, df_b, hedge_ratio)
    return calculate_zscore(spread_df, window)
