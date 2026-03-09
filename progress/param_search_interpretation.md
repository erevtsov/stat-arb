# Parameter Search Metric Interpretation

## Metrics Used

The parameter search evaluates each combo using three metrics:

| Metric | What It Measures |
|--------|-----------------|
| `ic_gross` | Spearman rank correlation between signal and forward returns (directional skill) |
| `hit_rate` | Fraction of bars where signal and return agree in sign |
| `mean_net_return` | Average per-bar net return while in position (cost deducted each bar) |

---

## `ic_gross` — Primary Ranking Criterion

IC is the standard industry metric for signal quality. It answers: **"Does the signal predict direction?"**

- Scale: dimensionless [-1, +1]; IC = 0 means no skill
- Not affected by spread size or holding length
- IC > 0.05 is meaningful; IC > 0.10 is considered good
- **Key finding:** IC > 0 consistently at `z_entry ≥ 2.5` — this is the main actionable result from the grid search

Use IC as the primary criterion for ranking and filtering parameter combos.

---

## `hit_rate` — Sanity / Filter Check

Hit rate is the fraction of signal bars where the return was in the predicted direction.

- Should be > 50% for a mean-reversion strategy
- Less sensitive than IC: two combos can have identical hit rate but very different IC if the winning trades are larger
- Use as a sanity check, not as a primary ranking criterion

---

## `mean_net_return` — Per-Bar Net Return

`mean_net_return` averages the net return across every bar while a position is open. Cost (`cost_bps`) is deducted proportionally on each bar.

**Known bias:** Long-holding parameter sets (`max_holding_minutes`, `formation_window_days`) are penalized because:
1. **Averaging dilution** — a 40-bar trade contributes 40 observations to the mean; a 4-bar trade contributes 4. If both earn the same total P&L, the 40-bar trade looks worse per-bar.
2. **Cost double-counting** — cost is deducted per bar rather than once per round-trip, so longer holds pay more cost in aggregate.

**Practical consequence:** Do not use `mean_net_return` to rank `max_holding_minutes` or `formation_window_days`. It reliably ranks `z_entry` and `z_stop` (which don't change hold length) and is useful as a sign filter (`mean_net_return > 0` = strategy is cost-viable on a per-bar basis).
