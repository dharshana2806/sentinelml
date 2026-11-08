import numpy as np
import pandas as pd
import pytest

from sentinelml.monitor import build_features, feature_group, BASE_SIGNALS
from sentinelml.predictor import FailurePredictor, ece, episode_metrics, episodes, make_model, best_f1_threshold


def _sig(n=80, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(size=n) for c in BASE_SIGNALS})


def test_features_are_causal():
    """Changing the future must not change features computed for earlier times."""
    s = _sig(); f1 = build_features(s)
    s2 = s.copy(); s2.iloc[50:] += 100
    f2 = build_features(s2)
    pd.testing.assert_frame_equal(f1.iloc[:50], f2.iloc[:50])


def test_feature_groups_and_ablation():
    s = _sig()
    f = build_features(s, drop_groups=("drift",))
    assert not any(feature_group(c) == "drift" for c in f.columns)
    f0 = build_features(s, temporal=False)
    assert list(f0.columns) == list(build_features(s).columns[: len(f0.columns)]) or len(f0.columns) == len(BASE_SIGNALS)


def test_episodes_and_episode_metrics():
    deg = np.zeros(40, bool); deg[20:26] = True
    assert episodes(deg) == [(20, 26)]
    t = pd.DataFrame({"stream": "s", "t": np.arange(40), "seg": "test", "degraded": deg})
    t["alert"] = False; t.loc[16, "alert"] = True            # alert 4 windows before onset
    m = episode_metrics(t, "alert", H=6)
    assert m["episodes"] == 1 and m["warned_rate"] == 1.0 and m["mean_lead"] == 4
    t["alert"] = False; t.loc[22, "alert"] = True            # only after onset -> detection delay 2
    m = episode_metrics(t, "alert", H=6)
    assert m["warned_rate"] == 0 and m["mean_delay_unwarned"] == 2


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1]); assert ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0
    assert ece(y, np.array([1.0, 1.0, 0.0, 0.0])) == 1


def test_predictor_learns_a_leading_indicator_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=n); y = (x + rng.normal(0, 0.5, n) > 1).astype(float)
    tbl = pd.DataFrame({"f1": x, "f2": rng.normal(size=n), "y": y, "healthy": True})
    tr, va, te = tbl.iloc[:900], tbl.iloc[900:1200], tbl.iloc[1200:]
    fp = FailurePredictor("hgb", ["f1", "f2"], make_model("hgb")).fit(tr, va)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(te.y, fp.score(te)) > 0.85
    p = tmp_path / "fp.joblib"; fp.save(str(p))
    assert np.allclose(FailurePredictor.load(str(p)).score(te), fp.score(te))


def test_all_model_factories_fit():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 5)), columns=list("abcde")); y = (X.a > 0.5).astype(int)
    for name in ["logreg", "rf", "hgb", "xgb", "lgbm"]:
        make_model(name).fit(X, y)


def test_threshold_selection_handles_no_positives():
    assert best_f1_threshold(np.zeros(50, int), np.random.rand(50)) == 0.5
