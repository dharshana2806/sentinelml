"""Counterfactual recovery simulator + human-approval workflow.

For a diagnosed incident we build candidate recovery actions, *simulate each offline* on labelled recent
data that was not used to fit it, guard against regressions on clean reference data, price its compute
cost, and rank. Nothing is deployed automatically: a proposal must be validated on a separate holdout
and approved by a human (see `RecoveryProposal`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .model import TaskModel


# ---- candidate predictors ----------------------------------------------------------------------
class _Wrapped:
    def __init__(self, model: TaskModel):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


class MaskSuspects(_Wrapped):
    """Serve-time fix: replace suspected features by reference medians (treated as 'missing')."""
    def __init__(self, model, suspects, medians):
        super().__init__(model); self.suspects = suspects; self.medians = medians

    def predict(self, X):
        X = X.copy()
        for f in self.suspects:
            X[f] = self.medians[f]
        return self.model.predict(X)


class AlignSuspects(_Wrapped):
    """Preprocessing update: map suspect features from the observed (fit-window) distribution back onto
    the reference distribution (quantile mapping learned on labelled recent data)."""
    def __init__(self, model, suspects, fit_X, profile):
        super().__init__(model); self.suspects = suspects; self.profile = profile
        self.src = {f: np.sort(fit_X[f].dropna().to_numpy()) for f in suspects}

    def predict(self, X):
        X = X.copy()
        for f in self.suspects:
            x = X[f].to_numpy(float).copy(); nan = np.isnan(x)
            src = self.src[f]
            u = np.clip(np.searchsorted(src, x[~nan], side="right") / max(len(src), 1), 0.0, 1.0)
            x[~nan] = np.quantile(self.profile.ref_sorted[f], u)
            x[nan] = self.profile.median[f]
            X[f] = x
        return self.model.predict(X)


class Recalibrated(_Wrapped):
    """Classification: prior-shift correction of class probabilities. Regression: linear a + b*pred."""
    def __init__(self, model, fit_X, fit_y, y_train):
        super().__init__(model)
        self.task = model.task
        if self.task == "regression":
            p = model.predict(fit_X)
            b, a = np.polyfit(p, fit_y, 1)
            self.a, self.b = float(a), float(np.clip(b, 0.2, 3.0))
        else:
            k = len(model.classes_)
            cur = np.clip(np.bincount(np.searchsorted(model.classes_, fit_y), minlength=k) / len(fit_y), 1e-3, None)
            tr = np.clip(np.bincount(np.searchsorted(model.classes_, y_train), minlength=k) / len(y_train), 1e-3, None)
            self.w = cur / tr

    def predict(self, X):
        if self.task == "regression":
            return self.a + self.b * self.model.predict(X)
        proba = self.model.est.predict_proba(self.model._prep(X)) * self.w
        return self.model.classes_[proba.argmax(1)]


# ---- simulation --------------------------------------------------------------------------------
@dataclass
class CandidateResult:
    name: str
    description: str
    eval_loss: float
    guard_loss: float
    cost_s: float
    feasible: bool
    gain: float                      # loss reduction vs no action on the evaluation windows
    utility: float
    predictor: object = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "predictor"}


def _cat(windows):
    return pd.concat([w[0] for w in windows], ignore_index=True), np.concatenate([w[1] for w in windows])


def simulate_recovery(ctx, fit_windows: list, eval_windows: list, suspects: list,
                      guard_abs: float = 0.05, guard_rel: float = 0.25, cost_weight: float = 0.002,
                      min_gain_rel: float = 0.05, min_gain_abs: float = 0.01) -> dict:
    """Build, simulate and rank candidate recoveries.

    fit_windows   labelled recent windows used to *fit* candidates
    eval_windows  later labelled windows used to *score* candidates (never used for fitting)
    Returns dict(candidates=[CandidateResult sorted by utility], recommended=name).
    """
    td, model, prof = ctx.td, ctx.model, ctx.profile
    Xf, yf = _cat(fit_windows)
    Xe, ye = _cat(eval_windows)
    ref_loss0 = model.loss(td.y_ref, model.predict(td.X_ref))
    keep = [c for c in td.feature_names if c not in suspects]
    specs = [("no_action", "Keep the current model", lambda: _Wrapped(model))]
    specs.append(("rollback_previous_version", "Roll back to the previously registered model version",
                  lambda: _Wrapped(ctx.prev_model)))
    specs.append(("recalibrate", "Prior-shift / linear recalibration fitted on recent labels",
                  lambda: Recalibrated(model, Xf, yf, td.y_train)))
    specs.append(("retrain_recent", "Retrain on reference training data + recent labelled windows (weight x3)",
                  lambda: _Wrapped(TaskModel(td.task, seed=ctx.cfg.window_size).fit(
                      pd.concat([td.X_train, Xf], ignore_index=True), np.concatenate([td.y_train, yf]),
                      sample_weight=np.r_[np.ones(len(td.X_train)), 3 * np.ones(len(Xf))]))))
    if suspects and len(keep) >= 2:
        specs.append(("mask_suspect_features", f"Serve-time: replace {suspects} by reference medians",
                      lambda: MaskSuspects(model, suspects, prof.median)))
        specs.append(("align_suspect_features", f"Preprocessing update: quantile-align {suspects} to reference",
                      lambda: AlignSuspects(model, suspects, Xf, prof)))
        specs.append(("retrain_without_suspects", f"Retrain on reference data without {suspects}",
                      lambda: _Wrapped(TaskModel(td.task, seed=ctx.cfg.window_size + 1).fit(
                          td.X_train, td.y_train, columns=keep))))
    results = []
    base_eval = model.loss(ye, model.predict(Xe))
    for name, desc, build in specs:
        t0 = time.perf_counter()
        pred = build()
        cost = time.perf_counter() - t0
        e_loss = model.loss(ye, pred.predict(Xe))
        g_loss = model.loss(td.y_ref, pred.predict(td.X_ref))
        feasible = g_loss <= ref_loss0 + max(guard_abs, guard_rel * ref_loss0)
        gain = base_eval - e_loss
        results.append(CandidateResult(name, desc, e_loss, g_loss, cost, bool(feasible), gain,
                                       gain - cost_weight * cost, pred))
    results.sort(key=lambda r: -r.utility)
    min_gain = max(min_gain_abs, min_gain_rel * base_eval)
    rec = next((r for r in results if r.feasible and r.name != "no_action" and r.gain >= min_gain), None)
    return {"candidates": results, "recommended": rec.name if rec else "no_action",
            "baseline_eval_loss": base_eval}


def validate_candidate(ctx, predictor, holdout_windows: list, min_improvement_rel: float = 0.05) -> dict:
    """Validation gate on windows that were used neither for fitting nor for ranking."""
    X, y = _cat(holdout_windows)
    cur = ctx.model.loss(y, ctx.model.predict(X))
    new = ctx.model.loss(y, predictor.predict(X))
    ref0 = ctx.model.loss(ctx.td.y_ref, ctx.model.predict(ctx.td.X_ref))
    ref1 = ctx.model.loss(ctx.td.y_ref, predictor.predict(ctx.td.X_ref))
    ok_gain = new <= cur * (1 - min_improvement_rel)
    ok_guard = ref1 <= ref0 + max(0.05, 0.25 * ref0)
    return {"passed": bool(ok_gain and ok_guard), "holdout_loss_current": cur, "holdout_loss_candidate": new,
            "reference_loss_current": ref0, "reference_loss_candidate": ref1,
            "checks": {"improves_holdout": bool(ok_gain), "no_regression_on_reference": bool(ok_guard)}}


# ---- human approval workflow -------------------------------------------------------------------
class InvalidTransition(Exception):
    pass


_TRANSITIONS = {
    "proposed": {"validated", "validation_failed", "rejected"},
    "validated": {"approved", "rejected"},
    "validation_failed": {"rejected"},
    "approved": {"applied", "rejected"},
    "applied": {"rolled_back"},
    "rejected": set(), "rolled_back": set(),
}


@dataclass
class RecoveryProposal:
    """State machine: proposed -> validated -> approved -> applied (-> rolled_back).
    A proposal can never be approved before validation passes, nor applied before human approval."""
    action: str
    state: str = "proposed"
    history: list = field(default_factory=list)

    def _go(self, new: str, actor: str, note: str = ""):
        if new not in _TRANSITIONS[self.state]:
            raise InvalidTransition(f"cannot go from {self.state!r} to {new!r}")
        self.history.append({"from": self.state, "to": new, "actor": actor, "note": note,
                             "at": datetime.now(timezone.utc).isoformat()})
        self.state = new

    def record_validation(self, passed: bool, actor: str = "system"):
        self._go("validated" if passed else "validation_failed", actor)

    def approve(self, actor: str, note: str = ""):
        if not actor or actor == "system":
            raise InvalidTransition("approval requires a human actor")
        self._go("approved", actor, note)

    def reject(self, actor: str, note: str = ""):
        self._go("rejected", actor, note)

    def apply(self, actor: str = "system"):
        self._go("applied", actor)

    def rollback(self, actor: str, note: str = ""):
        self._go("rolled_back", actor, note)
