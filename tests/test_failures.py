import numpy as np
import pandas as pd

from sentinelml.config import FAILURE_TYPES
from sentinelml.drift import RefProfile
from sentinelml.failures import FailureEvent, apply_event, schedule_events


def _setup():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = (X.a > 0).astype(int).to_numpy()
    return X, y, RefProfile.build(X, np.ones(4))


def ev(ftype, shape="sudden", **kw):
    return FailureEvent(0, ftype, 1.0, 10, 20, shape, 10, kw.pop("features", ["a"]), kw)


def test_intensity_shapes():
    g = FailureEvent(0, "covariate_shift", 1.0, 10, 20, "gradual", 10, ["a"])
    assert g.intensity(9) == 0 and g.intensity(10) == 0.1 and g.intensity(19) == 1.0 and g.intensity(30) == 0
    s = FailureEvent(0, "covariate_shift", 1.0, 10, 20, "sudden", 10, ["a"])
    assert s.intensity(10) == 1.0


def test_each_failure_type_only_touches_what_it_should():
    X, y, prof = _setup()
    rng = np.random.default_rng(1)
    Xs, _ = apply_event(ev("covariate_shift"), X, y, 15, prof, rng, "binary", 2)
    assert Xs.a.mean() - X.a.mean() > 2 and np.allclose(Xs.b, X.b)
    Xm, _ = apply_event(ev("missing_values"), X, y, 15, prof, rng, "binary", 2)
    assert Xm.a.isna().mean() > 0.5 and Xm.b.isna().sum() == 0
    Xp, _ = apply_event(ev("pipeline_mismatch"), X, y, 15, prof, rng, "binary", 2)
    assert Xp.a.std() > 50 * X.a.std()
    Xo, _ = apply_event(ev("outliers"), X, y, 15, prof, rng, "binary", 2)
    assert (Xo.a.abs() > 6).mean() > 0.05
    _, yc = apply_event(ev("concept_drift", features=[], concept_feature="a"), X, y, 15, prof, rng, "binary", 2)
    assert (yc != y).mean() > 0.2
    Xl, yl = apply_event(ev("label_shift", features=[], target_class=1), X, y, 15, prof, rng, "binary", 2)
    assert yl.mean() > y.mean() + 0.2
    Xn, yn = apply_event(ev("covariate_shift"), X, y, 5, prof, rng, "binary", 2)   # inactive
    assert Xn is X


def test_inputs_not_mutated_and_reproducible():
    X, y, prof = _setup(); X0 = X.copy()
    a, _ = apply_event(ev("feature_corruption"), X, y, 15, prof, np.random.default_rng(3), "binary", 2)
    b, _ = apply_event(ev("feature_corruption"), X, y, 15, prof, np.random.default_rng(3), "binary", 2)
    assert X.equals(X0) and a.equals(b)


def test_schedule_has_no_overlap_and_ground_truth():
    rng = np.random.default_rng(0)
    evs = schedule_events(600, 10, rng, list("abcdef"), np.arange(6, dtype=float), "binary")
    assert len(evs) == 10
    for e1, e2 in zip(evs, evs[1:]):
        assert e1.end + 10 <= e2.start
    assert all(e.ftype in FAILURE_TYPES and 0 < e.severity <= 1 for e in evs)
    assert all(e.features for e in evs if e.ftype in {"covariate_shift", "missing_values"})
