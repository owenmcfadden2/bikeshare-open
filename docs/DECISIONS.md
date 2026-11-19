# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Test set touched once, after all feature work.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) is within 0.07 R^2 of ridge and beats it on MAE; that caveat goes in the report.
- 2026-11-18: Added `cnt_lag_14` (cnt 14 calendar days before d). Availability: d-14 is strictly before d, so it is known at prediction time. Raised LAG_WARMUP 7 -> 14 (train rows 348 -> 340).
- 2026-11-18: Decoupled `cnt_lag_7` and `cnt_roll_7` from LAG_WARMUP (now a literal 7). Raising LAG_WARMUP had silently turned both into 14-day features; the hook caught `cnt_roll_7` via test_rolling_mean_excludes_today, but no test covered `cnt_lag_7`.
- 2026-11-18: Kept `cnt_lag_14` despite a null result: valid R^2 ridge 0.6475 -> 0.6430, hgb 0.2185 -> 0.2135, both inside the 0.01 noise floor. Kept so the Exercise 1 feature stays in the pipeline; in production I would drop it. The delta also mixes the feature with 8 fewer training rows.
