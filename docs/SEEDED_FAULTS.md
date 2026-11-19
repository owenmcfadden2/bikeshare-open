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
