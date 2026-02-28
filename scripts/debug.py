"""
Phase 1 systematic debugging: verify Sharpe 3.5 is real.

Investigates:
1. Last-bar fallback bug (z_exit/z_stop at 15:45 use current-bar midpoint)
2. Why same-day z_exit has avg_net=$300 vs multi-day $1.81
3. Cost sensitivity at 5bps and 10bps
4. Whether the PnL distribution is consistent with the economic hypothesis
"""

import sys
sys.path.insert(0, "/Users/erevtsov/dev/stat-arb")

import polars as pl

from strategy.backtester import run_backtest

# ─────────────────────────────────────────────────────────────────────────────
# Run baseline backtest
# ─────────────────────────────────────────────────────────────────────────────
print("Running full-period backtest at 1.5 bps...")
trades, daily_pnl = run_backtest(
    start_date="2022-07-01",
    end_date="2024-12-31",
    cost_bps=1.5,
)

final_value = daily_pnl["portfolio_value"].tail(1)[0]
ret = (final_value - 100_000) / 100_000 * 100
print(f"Total trades: {len(trades)}")
print(f"Return: {ret:.2f}%  Final portfolio: ${final_value:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Annotate trades
# ─────────────────────────────────────────────────────────────────────────────
t = trades.with_columns([
    pl.col("exit_time").dt.hour().alias("exit_hour"),
    pl.col("exit_time").dt.minute().alias("exit_min"),
    (pl.col("exit_time").dt.date() == pl.col("entry_time").dt.date()).alias("same_day"),
]).with_columns(
    ((pl.col("exit_hour") == 15) & (pl.col("exit_min") == 45)).alias("exit_at_eod_bar"),
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. EXIT-REASON BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. EXIT-REASON BREAKDOWN")
print("=" * 70)
print(
    t.group_by("exit_reason")
     .agg([
         pl.len().alias("n"),
         pl.col("net_pnl").mean().alias("avg_net"),
         pl.col("net_pnl").sum().alias("total_net"),
         (pl.col("net_pnl") > 0).mean().alias("win_rate"),
         pl.col("same_day").mean().alias("pct_same_day"),
     ])
     .sort("exit_reason")
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SAME-DAY VS MULTI-DAY Z_EXIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. SAME-DAY vs MULTI-DAY z_exit BREAKDOWN")
print("=" * 70)
z_exit_t = t.filter(pl.col("exit_reason") == "z_exit")
print(
    z_exit_t.group_by("same_day")
     .agg([
         pl.len().alias("n"),
         pl.col("net_pnl").mean().alias("avg_net"),
         pl.col("gross_pnl").mean().alias("avg_gross"),
         pl.col("net_pnl").sum().alias("total_net"),
         (pl.col("net_pnl") > 0).mean().alias("win_rate"),
         pl.col("exit_at_eod_bar").mean().alias("pct_at_15_45"),
     ])
     .sort("same_day")
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. LAST-BAR FALLBACK (15:45) vs EARLIER EXIT TIMES — same-day z_exit only
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. LAST-BAR FALLBACK: 15:45 vs earlier (same-day z_exit)")
print("=" * 70)
print(
    z_exit_t.filter(pl.col("same_day"))
     .group_by("exit_at_eod_bar")
     .agg([
         pl.len().alias("n"),
         pl.col("net_pnl").mean().alias("avg_net"),
         pl.col("gross_pnl").mean().alias("avg_gross"),
         (pl.col("net_pnl") > 0).mean().alias("win_rate"),
     ])
     .sort("exit_at_eod_bar")
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. HOLDING-PERIOD BUCKETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. HOLDING-PERIOD BUCKETS")
print("=" * 70)
t2 = t.with_columns(
    (pl.col("exit_time") - pl.col("entry_time"))
    .dt.total_minutes()
    .cast(pl.Float64)
    .alias("hold_minutes")
).with_columns(
    pl.when(pl.col("hold_minutes") <= 60).then(pl.lit("0-1h"))
     .when(pl.col("hold_minutes") <= 240).then(pl.lit("1-4h"))
     .when(pl.col("hold_minutes") <= 480).then(pl.lit("4-8h"))
     .when(pl.col("hold_minutes") <= 1440).then(pl.lit("8-24h"))
     .otherwise(pl.lit(">24h"))
     .alias("hold_bucket")
)
print(
    t2.group_by("hold_bucket")
     .agg([
         pl.len().alias("n"),
         pl.col("net_pnl").mean().alias("avg_net"),
         pl.col("net_pnl").sum().alias("total_net"),
         (pl.col("net_pnl") > 0).mean().alias("win_rate"),
     ])
     .sort("hold_bucket")
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. GROSS PNL SANITY: same-day z_exit
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. GROSS PNL SANITY CHECK — same-day z_exit")
print("=" * 70)
sd_ze = t2.filter((pl.col("exit_reason") == "z_exit") & pl.col("same_day"))
print(f"n={len(sd_ze)}")
print(
    sd_ze.select([
        pl.col("shares_a").mean().alias("avg_shares_a"),
        pl.col("gross_pnl").mean().alias("avg_gross"),
        pl.col("transaction_cost").mean().alias("avg_cost"),
        pl.col("net_pnl").mean().alias("avg_net"),
        ((pl.col("exit_price_a") - pl.col("entry_price_a")).abs()).mean().alias("avg_abs_da"),
        ((pl.col("exit_price_b") - pl.col("entry_price_b")).abs()).mean().alias("avg_abs_db"),
        pl.col("entry_price_a").mean().alias("avg_entry_pa"),
        pl.col("hold_minutes").mean().alias("avg_hold_min"),
    ])
)

# Check if gross PnL is consistent with price movement × shares
print("\nImplied spread move per share (gross / shares_a):")
implied = sd_ze.with_columns(
    (pl.col("gross_pnl") / pl.col("shares_a")).alias("implied_spread_move_per_share")
)
print(implied.select(
    pl.col("implied_spread_move_per_share").mean().alias("mean"),
    pl.col("implied_spread_move_per_share").std().alias("std"),
    pl.col("implied_spread_move_per_share").quantile(0.25).alias("q25"),
    pl.col("implied_spread_move_per_share").quantile(0.75).alias("q75"),
))

# ─────────────────────────────────────────────────────────────────────────────
# 6. COST SENSITIVITY (analytical from baseline run)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. COST SENSITIVITY (rescaled from 1.5 bps baseline)")
print("=" * 70)

# Cost per trade scales linearly with bps.
# net_pnl(bps_new) = gross_pnl - cost_baseline × (bps_new / 1.5)
# This is an approximation (position sizing depends on portfolio_value
# which changes differently at each cost level).
total_gross = trades["gross_pnl"].sum()
total_cost_baseline = trades["transaction_cost"].sum()

for bps in [1.5, 5.0, 10.0]:
    scale = bps / 1.5
    approx_total_net = total_gross - total_cost_baseline * scale
    approx_return = approx_total_net / 100_000 * 100  # rough: ignores compounding
    approx_sharpe = approx_return / (100 * (252 ** 0.5))  # very rough
    print(
        f"  {bps:.1f} bps: ~total_net=${approx_total_net:,.0f}  "
        f"~return≈{approx_return:+.1f}%  "
        f"total_cost≈${total_cost_baseline * scale:,.0f}"
    )

print(f"\n  Baseline gross: ${total_gross:,.0f}  baseline_cost@1.5bps: ${total_cost_baseline:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ANNUAL BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. ANNUAL BREAKDOWN")
print("=" * 70)
pnl_with_year = daily_pnl.with_columns(
    pl.col("date").str.slice(0, 4).alias("year")
)
for yr in sorted(pnl_with_year["year"].unique().to_list()):
    yr_df = pnl_with_year.filter(pl.col("year") == yr)
    start_val = yr_df["portfolio_value"].head(1)[0] - yr_df["net_pnl"].head(1)[0]
    end_val = yr_df["portfolio_value"].tail(1)[0]
    ret = (end_val - start_val) / start_val * 100
    n_tr = yr_df["n_trades"].sum()
    print(f"  {yr}: return={ret:+.1f}%  n_trades={n_tr}  end_value=${end_val:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY-BAR CHECK: are we ever entering on bar_idx > 0 with non-zero prev?
#    (potential stale-signal entry bug)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. ENTRY TIME DISTRIBUTION")
print("=" * 70)
t_entry = t.with_columns([
    pl.col("entry_time").dt.hour().alias("entry_hour"),
    pl.col("entry_time").dt.minute().alias("entry_min"),
])
print(
    t_entry.with_columns(
        (pl.col("entry_hour") * 100 + pl.col("entry_min")).alias("entry_hhmm")
    )
    .group_by("entry_hhmm")
    .agg(pl.len().alias("n"))
    .sort("entry_hhmm")
    .head(20)
)

print("\nEntries at first bar of day (9:30 or 9:45):")
early_entries = t_entry.filter(
    (pl.col("entry_hour") == 9) & (pl.col("entry_min").is_in([30, 45]))
)
print(f"  n={len(early_entries)}, pct={len(early_entries)/len(t)*100:.1f}%")
print(
    early_entries.group_by("exit_reason")
     .agg([
         pl.len().alias("n"),
         pl.col("net_pnl").mean().alias("avg_net"),
         (pl.col("net_pnl") > 0).mean().alias("win_rate"),
     ])
     .sort("exit_reason")
)

print("\nDone.")
