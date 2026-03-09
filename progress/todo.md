# TODO

## Active

- [ ] test the optimal half life bounds 

## Backlog

## Done

- [ ] Analyze max simultaneous pairs: check how often the `max_pairs=20` cap is actually hit, whether it's binding, and whether raising/lowering it changes P&L and risk.
- [ ] Analyze decay of exit z-score as position life grows. Meaning, the longer it takes to revert to mean the more we assume the spread will stay.
- [ ] how can the gross PnL be negative on a pair trade if the trade reason is z_exit? if we enter a trade and the spread moves in the expected direction, shouldn't the gross PnL always be positive? only should turn negative net of t costs.

