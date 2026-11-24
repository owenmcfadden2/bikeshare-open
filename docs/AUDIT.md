# Exercise 3: audit of a teammate's repository

**Repository:** https://github.com/Siddhu-S/bikeshare-open
**Commit audited:** `9c5dc2e` (starter `80c6a1f`; 7 commits at audit time). The repo appeared to be in progress: its feature table still has the starter's 14 features.

## How the auditor was run

1. In their repo, auto mode: the subagent launch failed three times ("Not run · auto mode's check had no usable answer"; the server-side check cannot reach a local Ollama endpoint). The main agent then wrote its own review, which I do not use here.
2. In their repo, default permission mode: requests were closed by the client after exactly 5m50s, twice (Ollama log: "aborting completion request due to client closing the connection").
3. Succeeded: a fresh session in **my own repo**, default permission mode, pointing the `leakage-auditor` subagent at `~/partner-repo/src/features.py` and `~/partner-repo/src/train.py`. It ran for 19m 58s.

**Caveat:** because the session ran inside my repo, my `CLAUDE.md` and my code were in the auditor's context. Finding 1 below shows that this contaminated its judgment.

## `/leakage-auditor` output (verbatim, as printed in the session)

```
Audit Results

Findings (2 violations)

File: ~/partner-repo/src/features.py

1. LAG_WARMUP used as window size — Lines 77-78

out["cnt_lag_7"] = out[TARGET].shift(LAG_WARMUP)
out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()

- LAG_WARMUP is 14, but column names suggest 7-day windows. This violates CLAUDE.md rule: "Write window sizes as literals (shift(7), rolling(7). LAG_WARMUP means only 'longest lag': raise it for a longer lag, never use it as a window size."
- Concrete impact: rolling mean uses 14 days instead of 7; cnt_lag_7 is actually a 14-day lag. Model learns wrong temporal scale.
- Minimal fix: Change to shift(7) and rolling(7) literals; rename column if using 7-day rolling mean, or keep as cnt_lag_14 for 14-day lag.

2. Unreported baseline comparison — Incomplete in ~/partner-repo/src/train.py

The partner repo's train.py lacks both baselines in MODELS and the scoring logic needed to report them alongside ridge/hgb. CLAUDE.md requires: "Report the mean baseline and the persistence baseline (yesterday's count) alongside every model."

- Concrete impact: Without these baselines, R² scores lack proper context. A model with R²=0.45 looks impressive without knowing it beats persistence (R² ~ 0.7).
- Minimal fix: Add baseline_mean and baseline_persistence to MODELS; update fit_and_score() to fit them on train data and report scores for valid/test.

Passes

All other checks pass:

- Target leakage: No casual/registered in features; all predictors from FORBIDDEN.
- Temporal ordering: Date-based split; .shift(1) before aggregations; no transformer sees valid/test during .fit().
- Calendar gaps: reindex_to_calendar fills missing days (including 2011-03-10) before lag operations.
- Silent row loss: dropna() removes only gap-filled NaN rows after reindex; documented behavior in features.py docstring.

Summary: Two violations require fixing: window size literals and baseline reporting.
```

## My verdict

Neither of the auditor's findings changes a number, and its passes include one that is wrong. **Finding 1 is a false positive:** their `LAG_WARMUP` is 7 (`src/config.py` line 56), so `shift(LAG_WARMUP)` and `rolling(LAG_WARMUP)` produce correct 7-day windows today. The coupling is a latent hazard (it is the bug I hit in my own repo when I raised `LAG_WARMUP`), but the "14" and the rule it quotes come from my repo, not theirs. **Finding 2 is misdiagnosed:** both baselines are present in their `MODELS` (`src/train.py` lines 14-15); they are broken, not missing, and the auditor describes a `fit_and_score()` function that exists in my repo, not theirs (theirs is `evaluate()`). The "persistence $R^2$ ~ 0.7" figure is unsupported. Its pass on "no transformer sees valid/test during `.fit()`" is wrong: their models are fit on train and validation together. The findings that do change numbers came from my own verification, below; the auditor caught none of them, and it did not report running `train.py`, which crashes.

## Findings I verified myself

| # | Finding | Evidence |
|---|---|---|
| 1 | Models are fit on train + valid, so validation scores are in-sample and the training set is not 2011 only | `src/train.py` line 46 (`train_full = pd.concat([train, valid])`), line 85 (`model.fit(X_train, y_train)`) |
| 2 | `baseline_mean` is the mean of 2011 + H1 2012, not the 2011 average; on validation it includes the scored days | `src/train.py`: mean computed from `y_train` (= `train_full`) |
| 3 | `baseline_persistence` is not persistence: `np.roll` of the training targets, and its "validation predictions" are the first 182 rows of 2011 | `src/train.py` line 15, and the `preds_full[:n_valid]` slice |
| 4 | `python -m src.train` crashes | `ValueError: Found input variables with inconsistent numbers of samples: [184, 348]` |
| 5 | `reports/metrics.json` was not produced by this code and cannot be reproduced by `make all` | `cmp` against the unmodified starter's output: identical; `METRICS` is imported but never written |

Ruled out: the feature list is still derived from `FORBIDDEN` via `feature_columns()`. Not bugs but number-changing: `HistGradientBoostingRegressor` was replaced by `GradientBoostingRegressor`, and the `StandardScaler` was removed from ridge. Their 14 tests pass despite all five findings.

## What this says about the auditor

It is a screening instrument, not a verdict. Running on the same weights as the author model, a 9B reviewer read the code, did not run it, and passed a split violation that is visible on one line. Running it from my own repo added a second failure mode: it judged their code against my `CLAUDE.md` and my code.
