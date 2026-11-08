import numpy as np

from sentinelml.config import SimConfig
from sentinelml.pipeline import process_stream
from sentinelml.predictor import build_table, episodes
from sentinelml.simulate import build_context, hysteresis_state, simulate_stream
from types import SimpleNamespace

CFG = SimConfig(n_windows=160, n_events=3, warmup=20)


def test_hysteresis_prevents_flicker():
    sm = np.array([0, .2, .6, .4, .6, .2, 0])
    assert hysteresis_state(sm, hi=.5, lo=.3).tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_stream_is_reproducible_and_has_ground_truth():
    ctx = build_context("synth_binary", 5, CFG)
    a = simulate_stream(ctx, 5, keep_windows=False); b = simulate_stream(ctx, 5, keep_windows=False)
    assert np.allclose(a.loss, b.loss) and a.signals.equals(b.signals)
    assert (a.signals.event_id >= 0).sum() == sum(e.duration for e in a.events)
    assert a.signals.loc[:CFG.warmup - 1, "event_type"].eq("none").all()
    assert a.signals.loss_lag.isna().sum() == CFG.label_delay           # labels arrive with a delay


def test_labels_do_not_cross_split_boundaries():
    ctx = build_context("synth_binary", 5, CFG)
    r = simulate_stream(ctx, 5, keep_windows=False)
    s = SimpleNamespace(name=r.name, seed=r.seed, signals=r.signals, degraded=r.degraded, events=r.events)
    t = build_table([s], CFG)
    H, T = CFG.horizon, CFG.n_windows
    tr = t[t.seg == "train"]; va = t[t.seg == "val"]; te = t[t.seg == "test"]
    assert tr.t.max() + H < va.t.min() and va.t.max() + H < te.t.min()   # purge gaps >= horizon
    assert t[t.t > T - H - 1].y.isna().all()                              # no label without a full look-ahead


def test_process_stream_end_to_end():
    out = process_stream("synth_regression", 7, CFG)
    assert set(out) >= {"signals", "degraded", "events", "diag", "recovery"}
    for d in out["diag"]:
        assert set(d["ranks"]) >= {"ks", "combined", "shap_shift"}
    for r in out["recovery"]:
        assert "no_action" in r["candidates"] and r["recommended"] in r["candidates"]
