# Bikeshare demand forecasting: verified results

## Results

Temporal split: train = 2011, valid = H1 2012 (182 days), test = H2 2012 (184 days). "Before" is the unmodified starter (14 features, 348 training rows). "After" adds `cnt_lag_14` (15 features, 340 training rows after the longer warm-up) and is exactly `reports/metrics.json` from `make all`.

| Model | Version | Valid $R^2$ | Valid MAE | Test $R^2$ | Test MAE |
|---|---|---|---|---|---|
| baseline_mean | before | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| baseline_mean | after | -0.8763 | 1974.7 | -2.0605 | 2806.6 |
| baseline_persistence | before | 0.4507 | 900.6 | 0.5099 | 840.1 |
| baseline_persistence | after | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | before | 0.6475 | 832.9 | 0.5833 | 947.1 |
| ridge | after | 0.643 | 840.0 | 0.5772 | 956.4 |
| hgb | before | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| hgb | after | 0.2135 | 1295.2 | -0.5191 | 1999.7 |

The mean baseline moved although it uses no features: the longer warm-up drops seven more low-ridership January days, which raises the training mean. Persistence is unchanged because it is not fitted. The "before" test numbers are the starter's; I made no decision using test numbers except the single Exercise 4 run described below. The leaky model (`python -m src.leakage_demo`, features `casual`, `registered`, `temp`, `hum`) scores test $R^2$ = 1.000000 and is excluded as invalid.

## Headline finding

The naive evaluation gives test $R^2$ = 1.000 and is invalid: `casual + registered == cnt` on every row, so that model learns addition, not demand. Under a temporal split, the mean baseline scores -2.0605 on test, because mean daily ridership grew 58.0% from 2011 to 2012 and the 2011 average is far below late-2012 demand. Ridge reaches 0.5772 against a persistence baseline of 0.5099, a margin of 0.067, and loses to persistence on MAE (956.4 vs 840.1 rides). Gradient boosting scores -0.5191: it still beats the mean baseline, but a tree cannot predict above the largest target it saw, and 104 of 184 test days (56.5%) exceed the 2011 maximum of 6,043 rides. The range problem also reaches validation, where 71 of 182 days are above that maximum. My added feature, `cnt_lag_14`, changed validation $R^2$ by -0.0045 for ridge and -0.0050 for hgb, inside the 0.01 noise floor; I kept it so the Exercise 1 feature stays in the pipeline, and would drop it in production.

## Exercise 4: target transformation

I trained ridge and hgb on the ratio $\text{cnt}_d / \text{cnt\_roll\_7}_d$ (today relative to the mean of the seven days before it) and multiplied predictions back to rides (`python -m src.train --target-ratio`; the default pipeline is unchanged and `reports/metrics.json` is untransformed). The denominator uses only days before $d$, so the transformation is available at prediction time. I tried one transformation, chosen because it removes the diagnosed cause: the validation ratio range [0.16, 1.69] lies inside the training range [0.158, 1.878], so the tree no longer has to predict outside what it has seen. Predicting a ratio of 1 every day is itself a baseline ("last week's average"), reported as `baseline_roll7`.

| Model | Valid $R^2$ | Valid MAE | Test $R^2$ | Test MAE |
|---|---|---|---|---|
| hgb (untransformed) | 0.2135 | 1295.2 | -0.5191 | 1999.7 |
| ridge (untransformed) | 0.643 | 840.0 | 0.5772 | 956.4 |
| baseline_roll7 | 0.5945 | 823.9 | 0.4796 | 871.6 |
| ridge_ratio | 0.3906 | 1088.1 | 0.0156 | 1501.5 |
| hgb_ratio | 0.7409 | 691.8 | 0.6552 | 783.6 |

I judged the transformation on validation only and committed that verdict ("decision recorded before test") before running the test split once with `--final`. On test, hgb on the ratio target is the best model in this report and the only one that beats persistence on MAE (783.6 vs 840.1). Ridge got worse under the same transformation (test 0.0156): its inputs are still raw counts, which are furthest outside the 2011 range in H2 2012, so the extrapolation problem moves from its target to its features.

I consider the transformation defensible, with caveats. It is standard to model a growing series in relative terms, it introduces no leakage, it targets the specific mechanism behind the failure, and it beats the matching baseline (`baseline_roll7`) by 0.15 $R^2$ on validation, not only the weak ones. Against it: I chose it after seeing hgb fail and evaluated it on the same validation set, which I mitigated by trying one transformation and touching test once; it helps hgb and hurts ridge, so the fair statement is "hgb on a ratio target" rather than "hgb wins"; and after a sudden shock such as Hurricane Sandy (2012-10-29) the denominator stays high for a week, so ratio predictions will run high for that week.

## Exercise 5: ridge vs persistence

Ridge should not be deployed over persistence in its current form. On the test set it leads on $R^2$ (0.5772 vs 0.5099) but trails on MAE by 116 rides per day (956.4 vs 840.1), and a post-hoc diagnostic on the finished models shows why. Ridge’s advantage is real but narrow: on the 67 test days where weather severity changed from the day before, its MAE is 1055 against persistence’s 1141, because its standardized coefficients put nearly as much weight on same-day weather (severity −284, temperature +214, wind −157) as on lagged demand (7-day mean +539, yesterday +353). On the other 117 days persistence wins by 232 rides (668 vs 900), and the cause is bias, not noise: trained only on 2011, ridge under-predicts H2 2012 by 572 rides per day on average, while persistence, which inherits the growth through yesterday’s count, is off by +16. Its edge also depends on observing that day’s weather, which in deployment would be a forecast, so the real-world advantage would be smaller than measured here. The weather signal is worth keeping, but in a model that removes the trend: gradient boosting on the cnt / cnt_roll_7 target from Exercise 4 beats persistence on both metrics (test $R^2$ 0.6552, MAE 783.6).

## Attribution

**Model:** qwen3.5:9b via Ollama, built locally as `bikeshare-coder` with a 64k context, on an RTX 4060 Laptop GPU (8 GB) under WSL2; about 36% of the model ran on CPU. I did not switch models. Separately, I used Claude (claude.ai chat) as a tutor throughout: it explained the starter, proposed test code, shell commands and drafts that I reviewed and ran, and drafted parts of this report from numbers I produced.

**Generated by the agent (Claude Code driving qwen3.5:9b):**
- `src/features.py`: the `cnt_lag_14` line; the change to literal window sizes for `cnt_lag_7` and `cnt_roll_7` (made on my instruction after I diagnosed the bug).
- Two first drafts of the `cnt_lag_14` test, both of which would have failed for the wrong reason; replaced.
- The edits that seeded faults 1-3 (via `/seed-fault`), and the fixes for faults 1-3. I discarded an off-by-one comment from the fault 2 fix and reverted a dropped `sorted()` in the fault 3 fix.
- The first version of the fault 1 section of `docs/SEEDED_FAULTS.md` (its hook stderr was summarized; I replaced it with the full output).
- `reports/audit.jsonl` (`agent/audit.py` on my repo: 35 minutes, about 33 model requests, no findings).
- The Exercise 3 review of my teammate's repo (`leakage-auditor` subagent, 20 minutes; output verbatim in `docs/AUDIT.md`).

**Written by me (with tutor drafts I reviewed, edited and ran):** the five tests I added (`test_cnt_lag_14_on_named_date`, `test_cnt_lag_7_on_named_date`, `test_weather_encoder_rejects_unknown_labels`, `test_raw_data_is_unmodified`, `test_every_raw_column_is_classified`), with expected values looked up in the CSV by me; `RAW_SHA256` and the raw-column classes in `src/config.py`; the `--target-ratio` flag in `src/train.py`; the `/seed-fault` skill and its two revisions; the `CLAUDE.md` changes; fault sections 2-4 of `docs/SEEDED_FAULTS.md`; the verdict and verified findings in `docs/AUDIT.md`; `docs/DECISIONS.md`; this report.

**Decisions that were mine** (each has a dated line in `docs/DECISIONS.md`):
- What to predict, what counts as leakage, how to split, which baselines: I adopted the starter's choices and verified the leakage identity and the leaky $R^2$ = 1.000 myself.
- Adding `cnt_lag_14`, and keeping it despite a null validation delta.
- Diagnosing the `LAG_WARMUP` coupling, fixing it with literal windows, and adding a `cnt_lag_7` test that no starter test provided.
- Adding the unknown-label encoder test after fault 3 passed the starter suite unnoticed.
- Catching the fourth fault with a raw-data hash check rather than a PreToolUse hook or file permissions.
- Classifying every raw column (detection over switching to an allowlist; `season`, `mnth`, `weekday` treated as superseded).
- The ratio target for Exercise 4, its "defensible with caveats" verdict, and committing that verdict before the single test run.
- The Exercise 5 verdict.
- Treating the auditor's "no findings, high confidence" as a screen, not as evidence: the evidence is the tests I watched fail.
- Whether the answer is believable: the qualitative result is (leakage, the extrapolation failure and its fix all have direct evidence), but ridge's margin over persistence is small and partly an artefact of bias, so I do not present it as a strong win.
- The `CLAUDE.md` changes and the two tightenings of `/seed-fault`.
- My verdict on the teammate audit: checking each auditor claim against their code (one false positive, one misdiagnosis) and the five number-changing findings I verified that it missed.

### Working with a 9B model

- It frequently announced a plan and then ended its turn without making the tool call; I had to push it on several times.
- It wrote tests that would fail for the wrong reason (a missing fixture argument, `.date` instead of `.dt.date`).
- When raising `LAG_WARMUP` broke `cnt_roll_7`, the hook fired correctly, but the agent flipped the constant between 7 and 14 without diagnosing the cause, then hung for over eight minutes.
- It summarized output it was told to paste verbatim, marked a seeded fault with a `# FAULT` comment, rationalized past a skill precondition, and wrote explanations that were wrong while its code was right.
- Requests the model took too long to answer were closed by Claude Code after about 5m50s; Ollama logged them as HTTP 500 ("aborting completion request due to client closing the connection"), and Claude Code then backed off before retrying. I first attributed these to malformed tool calls and corrected that from the Ollama logs. Claude Code also does not recognize `bikeshare-coder` and assumes a 1M-token context, so it does not compact before the real 64k limit; I used `/clear` between tasks to compensate.
- Setup: `make model` failed with my Ollama version (piping the Modelfile to `ollama create -f -`), so I ran the create step manually with a temporary Modelfile.
