# Seeded faults

Exercise 2. Each fault: the change, the hook stderr, the red test output, and the turns the agent needed to fix it.

## Make feature_columns() in src/features.py also return "casual", so a leaked column reaches the model.

### Change
```diff
 def feature_columns(df: pd.DataFrame) -> list[str]:
     """Numeric columns that are safe to feed the model."""
     numeric = df.select_dtypes(include="number").columns
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN or c == "casual"]
```

### Hook stderr
```
[${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh]: Leakage tests failed after editing /home/owenm/bikeshare-open/src/features.py:

feats =         dteday  instant  season      yr mnth  ... weather_severity cnt_lag_1  cnt_lag_7   cnt_roll_7  cnt_lag_14
0   2...731.0  WINTER  2012.0  DEC  ...              1.0    1796.0      920.0  1531.428571      4585.0

[706 rows x 29 columns]

    def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_14', 'cnt_lag_7', 'cnt_roll_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:31: AssertionError
=============================================== short test summary info ================================================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: forbidden columns in feature set: {'casual'}
1 failed, 8 passed in 0.42s
```

### Test output
```
Exit code 2
python -m pytest tests/ -q
.....F..........                                                         [100%]
=================================== FAILURES ===================================
__________________________ test_no_forbidden_features __________________________

feats =         dteday  instant  season  ...  cnt_lag_7   cnt_roll_7 cnt_lag_14
0   2011-01-17     17.0  WINTER  ...     1321....1530.142857     3786.0
705 2012-12-31    731.0  WINTER  ...      920.0  1531.428571     4585.0

[706 rows x 29 columns]

    def test_no_forbidden_features(feats):
        """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
        construction, so this cannot discover a *new* leaky column. What it
        catches is someone editing feature_columns() or FORBIDDEN -- which is
        exactly the fault the Step 7 hook demonstrates."""
        cols = set(feature_columns(feats))
>       assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_14', 'cnt_lag_7', 'cnt_roll_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests/test_leakage.py:31: AssertionError
=========================== short test summary ============================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 15 passed in 1.25s
make: *** [Makefile:16: test] Error 1
```


### Turns to fix
1 (counted as my messages to a fresh session, given only "make test is failing"). The agent located the line correctly but misexplained its logic, claiming the condition kept only `casual`; it actually keeps every allowed column plus `casual`.

## Fault 2: `.shift(1)` deleted from `cnt_roll_7`

Seeded by the agent via /seed-fault (Edit tool). The run ended in an Ollama HTTP 500 before step 4, so this section was assembled by hand from the real outputs below.

### Change
```diff
diff --git a/src/features.py b/src/features.py
index 88cdf17..f303a67 100644
--- a/src/features.py
+++ b/src/features.py
@@ -73,7 +73,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     # If you add a longer lag, raise LAG_WARMUP in src/config.py.
     out["cnt_lag_1"] = out[TARGET].shift(1)
     out["cnt_lag_7"] = out[TARGET].shift(7)
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(7).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(7).mean()
 
     # 14-day calendar lag: total rides exactly 14 days prior to date d.
     out["cnt_lag_14"] = out[TARGET].shift(14)
```

### Hook stderr
The hook pipes pytest through `tail -20`, so this is its complete stderr.
```
[${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh]: Leakage tests failed after editing /home/owenm/bikeshare-open/src/features.py:
[706 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4482.142857142857) == 4477.142857142857 ± 0.00447714
E
E             comparison failed
E             Obtained: 4482.142857142857
E             Expected: 4477.142857142857 ± 0.00447714

tests/test_leakage.py:57: AssertionError
=================================================== short test summary info ===================================================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.float64(4482.142857142857) == 4477.142857142857 ±...
1 failed, 8 passed in 0.56s
```

### make test output
```
python -m pytest tests/ -q
........F.......                                                         [100%]
=================================== FAILURES ===================================
_______________________ test_rolling_mean_excludes_today _______________________

feats =         dteday  instant  season  ...  cnt_lag_7   cnt_roll_7 cnt_lag_14
0   2011-01-17     17.0  WINTER  ...     1321....1531.428571     3786.0
705 2012-12-31    731.0  WINTER  ...      920.0  1789.857143     4585.0

[706 rows x 29 columns]

    def test_rolling_mean_excludes_today(feats):
        """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
        by_date = feats.set_index("dteday")[TARGET]
        date = feats.dteday.iloc[200]
        window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
        if all(d in by_date.index for d in window):
            expected = sum(by_date[d] for d in window) / 7
>           assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)
E           assert np.float64(4482.142857142857) == 4477.142857142857 ± 0.00447714
E             
E             comparison failed
E             Obtained: 4482.142857142857
E             Expected: 4477.142857142857 ± 0.00447714

tests/test_leakage.py:57: AssertionError
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 15 passed in 1.24s
make: *** [Makefile:16: test] Error 1
```

### Turns to fix
1 (my messages to a /clear-ed session, given only "make test is failing"). Measured on a re-seed: my first fix attempt ran in the same session as the seeding, so I reverted, re-applied the identical one-line change with sed, cleared, and counted again. The agent diagnosed correctly from the test output, but its fix added a comment claiming the window covers t-1 through t-6 (it covers t-1 through t-7); I discarded the comment and kept the original line.

## Fault 3: weather encoder `raise` removed

Seeded by the agent via /seed-fault (Edit tool). The agent then tried to re-invoke the skill from inside itself (looking for hook output that did not exist); I stopped it and assembled this section by hand.

### Change
```diff
diff --git a/src/features.py b/src/features.py
index 88cdf17..93e1cf9 100644
--- a/src/features.py
+++ b/src/features.py
@@ -26,8 +26,6 @@ def encode_weather(series: pd.Series) -> pd.Series:
     """
     known = series.dropna()
     unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
 
 
```

### Hook stderr
None. The hook ran after the edit, but tests/test_leakage.py passed, so it exited 0 and printed nothing.

### make test output (fault in place, starter suite)
```
python -m pytest tests/ -q
................                                                         [100%]
16 passed in 1.20s
```

**Finding: the starter suite cannot see this fault.** Every weather label in the data (GOOD, MISTY, BAD) is in WEATHER_ORDER, so .map() produces no NaNs and the row-loss bound is never approached. The code can now fail silently, but nothing on this data makes it fail. I added test_weather_encoder_rejects_unknown_labels, which feeds the encoder an unseen label directly.

### New test, run with the fault still in place
```
.........F                                                               [100%]
=================================== FAILURES ===================================
_________________ test_weather_encoder_rejects_unknown_labels __________________

    def test_weather_encoder_rejects_unknown_labels():
        """A silent .map() turns an unseen label into NaN, and dropna() then
        deletes the row. Every label in the current data is known, so no
        data-driven test can see this; feed the encoder a label directly."""
>       with pytest.raises(ValueError, match="unmapped weather labels"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

tests/test_leakage.py:86: Failed
=========================== short test summary info ============================
FAILED tests/test_leakage.py::test_weather_encoder_rejects_unknown_labels - F...
1 failed, 9 passed in 0.40s
```

### Turns to fix
2 (my messages to a /clear-ed session, given only "make test is failing"). Turn 1: correct diagnosis, no edit. Turn 2: "apply it now". Its fix dropped sorted() from the error message, which still passes the test (it matches only the prefix) but makes the message order nondeterministic for multiple unknown labels; I restored the original line. This fault was only fixable once my new test existed: under the starter suite there was nothing failing to fix.

## Fault 4: Bash `cp` into data/raw/ (invisible to the hook)

I asked the agent to run `cp ~/ma7-notes/bike_daily_tampered.csv data/raw/bike_daily.csv` with the Bash tool. The tampered file differs in one row: 2012-07-04 (a test-set day) has temp 0 instead of 29.090849.

### What happened
- The deny rule `Edit(./data/raw/**)` did not block it: it binds the Edit tool, not shell commands. The only gate was my approval click, because `cp` is not on the allow list. In accept-edits mode it would have run unprompted.
- The PostToolUse hook did not fire: its matcher is `Edit|Write`, and `cp` is a Bash call.
- sha256 changed from a6bed1577f6e4de5... to 61836817fa0b863a...; `git diff --stat` showed 1 line changed.
- The full suite stayed green:
```
python -m pytest tests/ -q
.................                                                        [100%]
17 passed in 1.14s
```

### What would catch it
1. **A hash check on the raw file** (implemented below). Catches any route (cp, an editor, a changed download), but only when `make test` runs.
2. **A Stop hook** that runs `make test` and exits 2 on failure. Combined with (1), the agent could not end a turn after tampering.
3. **A PreToolUse hook on Bash** that rejects commands mentioning `data/raw/`. Prevents the write rather than detecting it, but an indirect path (a variable, `cd data/raw && cp ...`) can slip past a text match.
4. **`chmod -w data/raw/`**: a filesystem guarantee no tool call bypasses without an explicit chmod first.

### Implemented: test_raw_data_is_unmodified
```diff
diff --git a/src/config.py b/src/config.py
index 3a53c3b..5b657bb 100644
--- a/src/config.py
+++ b/src/config.py
@@ -53,3 +53,8 @@ VALID_END = "2012-06-30"
 # have no complete history and are dropped; tests/test_contract.py uses this
 # to bound how many rows may legitimately disappear.
 LAG_WARMUP = 14
+
+# sha256 of data/raw/bike_daily.csv as shipped. data/raw/ is immutable:
+# the deny rules only stop the agent's Edit tool, so a Bash `cp` can still
+# overwrite it. tests/test_contract.py checks this on every `make test`.
+RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"
diff --git a/tests/test_contract.py b/tests/test_contract.py
index 4f7a1b8..c2f7e18 100644
--- a/tests/test_contract.py
+++ b/tests/test_contract.py
@@ -1,9 +1,11 @@
 """Guard rails on the ingest contract and the trained model's behaviour."""
+import hashlib
+
 import pandas as pd
 import pytest
 from sklearn.metrics import r2_score
 
-from src.config import LAG_WARMUP, TARGET
+from src.config import LAG_WARMUP, RAW, RAW_SHA256, TARGET
 from src.features import build_features, feature_columns, reindex_to_calendar
 from src.ingest import load_raw
 from src.train import MODELS, temporal_split
@@ -56,3 +58,11 @@ def test_persistence_baseline_is_reported():
     """The baseline that matters for a lag-driven model is 'yesterday again'.
     It must be in the comparison, whatever its score."""
     assert "baseline_persistence" in MODELS
+
+
+def test_raw_data_is_unmodified():
+    """data/raw/ is immutable. The permission deny rules bind only the Edit
+    tool, and the PostToolUse hook only fires on Edit/Write, so a Bash `cp`
+    into data/raw/ passes both. Hashing the file catches any route."""
+    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
+    assert digest == RAW_SHA256, f"{RAW.name} was modified: sha256 {digest[:16]}..."
```

Run against the tampered file:
```
....F                                                                    [100%]
=================================== FAILURES ===================================
_________________________ test_raw_data_is_unmodified __________________________

    def test_raw_data_is_unmodified():
        """data/raw/ is immutable. The permission deny rules bind only the Edit
        tool, and the PostToolUse hook only fires on Edit/Write, so a Bash `cp`
        into data/raw/ passes both. Hashing the file catches any route."""
        digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
>       assert digest == RAW_SHA256, f"{RAW.name} was modified: sha256 {digest[:16]}..."
E       AssertionError: bike_daily.csv was modified: sha256 61836817fa0b863a...
E       assert '61836817fa0b...6259f852c9574' == 'a6bed1577f6e...0c86a0b5ad9dc'
E         
E         - a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc
E         + 61836817fa0b863ae5a3200e9ed76b38358495b115b90eedc636259f852c9574

tests/test_contract.py:68: AssertionError
=========================== short test summary info ============================
FAILED tests/test_contract.py::test_raw_data_is_unmodified - AssertionError: ...
1 failed, 4 passed in 1.19s
```
