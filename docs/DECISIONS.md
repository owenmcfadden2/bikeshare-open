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
- 2026-11-19: Tightened /seed-fault twice after watching it fail. v2: paste every line of the hook stderr (v1 kept only the summary). v3: edit the existing line in place, no fault-marking comment (v2 added an override line labelled "# FAULT", which would have handed the fixing agent the answer).
- 2026-11-19: Added test_weather_encoder_rejects_unknown_labels. Removing the encoder's raise left the starter suite 16/16 green: every weather label in the data is known, so no data-driven test can see it.
- 2026-11-19: Added a sha256 check on data/raw/bike_daily.csv (RAW_SHA256 in config.py). A Bash cp into data/raw/ passed the deny rules, the hook, and all 17 tests.
- 2026-11-19: Classified every raw column in config.py (RAW_FEATURE_COLS, RAW_SOURCE_COLS, RAW_SUPERSEDED_COLS, INDEX_COL) and added test_every_raw_column_is_classified. Kept feature_columns() as a denylist and added detection at the raw level rather than switching to an allowlist, so the starter pipeline is unchanged; the failure I chose is "a new column makes make test fail" over "a new column is silently dropped". Without it, a new numeric column entered the feature set silently (demonstrated with new_col). season, mnth and weekday are treated as superseded by dteday-derived features.
- 2026-11-19: Exercise 4 target transform: fit ridge and hgb on cnt / cnt_roll_7, multiply back (src/train.py --target-ratio; default path unchanged). One transformation tried, chosen because it removes the diagnosed cause: valid ratio range [0.16, 1.69] lies inside train [0.158, 1.878]. Validation: hgb 0.2135 -> 0.7409 R^2 (MAE 1295.2 -> 691.8); ridge 0.643 -> 0.3906 (its raw-count inputs still extrapolate); new baseline_roll7 0.5945 (MAE 823.9). Verdict: defensible as "hgb on a ratio target", reported alongside ridge getting worse. Decided on validation only; test run once after this commit.
