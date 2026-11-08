import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sentinelml import drift


def test_psi_identical_is_near_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000); b = rng.normal(size=5000)
    assert drift.psi(a, b) < 0.05
    assert drift.psi(a, b + 1.5) > 0.25


def test_ks_wasserstein_js_behave():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000); b = rng.normal(0.5, 1, 2000)
    assert 0 < drift.ks_statistic(a, b) < 1
    assert drift.wasserstein_norm(a, b) == pytest.approx(0.5, abs=0.1)
    js = drift.js_divergence(a, b)
    assert 0 <= js <= 1
    assert drift.js_divergence(a, a) == pytest.approx(0, abs=1e-9)


def test_fast_ks_wd_match_scipy():
    rng = np.random.default_rng(2)
    ref = np.sort(rng.normal(size=300)); cur = rng.normal(0.3, 1.2, 100)
    ks, wd = drift.ks_wd_fast(ref, cur)
    assert ks == pytest.approx(stats.ks_2samp(ref, cur).statistic, abs=1e-9)
    assert wd == pytest.approx(stats.wasserstein_distance(ref, cur), rel=1e-6)


def test_nan_handling_and_tiny_samples():
    assert np.isnan(drift.psi(np.array([1.0, 2]), np.array([1.0, 2])))
    a = np.r_[np.random.default_rng(0).normal(size=100), np.nan]
    assert np.isfinite(drift.ks_statistic(a, a))


def _profile():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    return drift.RefProfile.build(X, np.array([.5, .3, .2])), X


def test_feature_stats_flags_shifted_and_missing_features():
    prof, X = _profile()
    rng = np.random.default_rng(5)
    W = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    W["a"] += 3
    W.loc[rng.random(200) < 0.5, "b"] = np.nan
    fs = drift.feature_stats(prof, W)
    assert fs.loc["a", "ks"] > 0.8 and fs.loc["a", "psi"] > 0.25
    assert fs.loc["b", "miss_delta"] > 0.3
    assert fs.loc["c", "ks"] < 0.2


def test_data_quality_reports_schema_mismatch_and_duplicates():
    prof, X = _profile()
    q = drift.data_quality(prof, X.drop(columns=["c"]))
    assert not q["schema_ok"] and q["missing_columns"] == ["c"]
    dup = pd.concat([X.iloc[:50], X.iloc[:50]])
    assert drift.data_quality(prof, dup)["duplicate_rate"] == pytest.approx(0.5)
