"""Guard rails on the ingest contract and the trained model's behaviour."""
import hashlib

import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.config import (
    FORBIDDEN, INDEX_COL, LAG_WARMUP, RAW, RAW_FEATURE_COLS, RAW_SHA256,
    RAW_SOURCE_COLS, RAW_SUPERSEDED_COLS, TARGET,
)
from src.features import build_features, feature_columns, reindex_to_calendar
from src.ingest import load_raw
from src.train import MODELS, temporal_split


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: the first LAG_WARMUP days have no complete history. Each
    absent calendar day m costs the LAG_WARMUP rows after it (cnt_lag_1 is
    NaN on m+1, cnt_roll_7 is NaN on m+1..m+7, cnt_lag_7 on m+7; their
    union is 7 rows). Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_beats_the_mean_baseline(feats):
    """A model that cannot beat predicting the training mean is not a result."""
    cols = feature_columns(feats)
    train, _, test = temporal_split(feats)
    ridge = MODELS["ridge"]
    ridge.fit(train[cols], train[TARGET])
    assert r2_score(test[TARGET], ridge.predict(test[cols])) > 0.4


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS


def test_raw_data_is_unmodified():
    """data/raw/ is immutable. The permission deny rules bind only the Edit
    tool, and the PostToolUse hook only fires on Edit/Write, so a Bash `cp`
    into data/raw/ passes both. Hashing the file catches any route."""
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    assert digest == RAW_SHA256, f"{RAW.name} was modified: sha256 {digest[:16]}..."


def test_every_raw_column_is_classified(raw):
    """A new column in the raw CSV must not enter the model silently.
    feature_columns() would pass any new numeric column straight through,
    and every leakage test would stay green. So every raw column must be
    named in exactly one set in src/config.py."""
    groups = [FORBIDDEN, RAW_FEATURE_COLS, RAW_SOURCE_COLS,
              RAW_SUPERSEDED_COLS, {INDEX_COL}]
    classified = set().union(*groups)
    assert sum(len(g) for g in groups) == len(classified), "a column is in two classes"
    unclassified = set(raw.columns) - classified
    assert not unclassified, f"unclassified raw columns: {sorted(unclassified)}"
