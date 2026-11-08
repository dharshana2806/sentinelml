import numpy as np
import pandas as pd
import pytest

from sentinelml.config import SimConfig
from sentinelml.rca import quantile_restore, feature_diagnosis, top_k
from sentinelml.recovery import InvalidTransition, RecoveryProposal, simulate_recovery, validate_candidate
from sentinelml.simulate import build_context


@pytest.fixture(scope="module")
def ctx():
    return build_context("synth_binary", 0, SimConfig())


def _window(ctx, n, seed, shift_feature=None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ctx.td.X_pool), n, replace=False)
    X = ctx.td.X_pool.iloc[idx].reset_index(drop=True); y = ctx.td.y_pool[idx]
    for f in ([shift_feature] if isinstance(shift_feature, str) else (shift_feature or [])):
        X[f] = X[f] * 50 + 20
    return X, y


def test_quantile_restore_maps_back_to_reference(ctx):
    f = ctx.td.feature_names[0]
    x = ctx.td.X_pool[f].to_numpy()[:200] * 40 + 10
    r = quantile_restore(ctx.profile, x, f)
    assert abs(r.mean() - ctx.profile.mean[f]) < 0.5 * ctx.profile.std[f]
    x[:5] = np.nan
    assert not np.isnan(quantile_restore(ctx.profile, x, f)).any()


def _top3(ctx):
    return [ctx.td.feature_names[i] for i in np.argsort(-ctx.profile.importance)[:3]]


def test_rca_ranks_the_corrupted_features_first(ctx):
    fs = _top3(ctx)
    X, y = _window(ctx, 300, 1, shift_feature=fs)
    df = feature_diagnosis(ctx, X, y, use_shap=False)
    assert set(top_k(df, "combined", 3)) == set(fs) and set(top_k(df, "ks", 3)) == set(fs)
    assert df.loc[fs, "restore_gain"].max() > 0.0
    clean = feature_diagnosis(ctx, *_window(ctx, 300, 2), use_shap=False)
    assert clean.ks.max() < 0.25 and clean.restore_gain.sum() < df.restore_gain.sum()


def test_recovery_simulation_ranks_a_fix_and_guards(ctx):
    f = _top3(ctx)
    fit = [_window(ctx, 100, s, f) for s in range(4)]; ev = [_window(ctx, 100, 10 + s, f) for s in range(3)]
    out = simulate_recovery(ctx, fit, ev, f)
    names = [c.name for c in out["candidates"]]
    assert {"no_action", "rollback_previous_version", "retrain_recent", "mask_suspect_features", "align_suspect_features"} <= set(names)
    best = out["recommended"]; assert best != "no_action"
    c = {c.name: c for c in out["candidates"]}
    assert c[best].gain > 0 and c[best].feasible and c["no_action"].gain == 0
    hold = [_window(ctx, 100, 20 + s, f) for s in range(3)]
    v = validate_candidate(ctx, c[best].predictor, hold)
    assert v["passed"] and v["holdout_loss_candidate"] < v["holdout_loss_current"]
    # a candidate that does nothing must not pass validation
    assert not validate_candidate(ctx, c["no_action"].predictor, hold)["passed"]


def test_no_suspects_means_no_feature_candidates(ctx):
    w = [_window(ctx, 100, s) for s in range(4)]
    names = {c.name for c in simulate_recovery(ctx, w, w[:3], [])["candidates"]}
    assert "mask_suspect_features" not in names


def test_approval_state_machine():
    p = RecoveryProposal("retrain_recent")
    with pytest.raises(InvalidTransition):
        p.approve("user:1")                    # cannot approve before validation
    p.record_validation(True)
    with pytest.raises(InvalidTransition):
        p.approve("system")                    # approval needs a human
    with pytest.raises(InvalidTransition):
        p.apply()                              # cannot apply before approval
    p.approve("user:1"); p.apply(); p.rollback("user:1")
    assert p.state == "rolled_back" and [h["to"] for h in p.history] == ["validated", "approved", "applied", "rolled_back"]
    q = RecoveryProposal("x"); q.record_validation(False)
    with pytest.raises(InvalidTransition):
        q.approve("user:1")                    # failed validation can never be approved
