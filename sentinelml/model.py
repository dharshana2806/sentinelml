"""The *monitored* model: imputer + random forest with uncertainty outputs.

Kept deliberately simple: SentinelML monitors whatever model you deploy; this wrapper is what the
benchmark deploys, and what the API trains for uploaded datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer


class TaskModel:
    """Median-imputer + RandomForest with `outputs()` returning prediction, score and uncertainty."""

    def __init__(self, task: str, n_estimators: int = 60, seed: int = 0, max_depth: int | None = None,
                 min_samples_leaf: int = 2):
        self.task = task
        self.n_estimators = n_estimators
        self.seed = seed
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    # ---- training ---------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, sample_weight=None, columns: list | None = None) -> "TaskModel":
        self.columns = list(columns) if columns is not None else list(X.columns)
        self.imputer = SimpleImputer(strategy="median").fit(X[self.columns])
        Xi = self.imputer.transform(X[self.columns])
        kw = dict(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1,
                  min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth)
        if self.task == "regression":
            self.est = RandomForestRegressor(**kw)
            self.y_scale = float(np.std(y)) or 1.0
        else:
            self.est = RandomForestClassifier(**kw)
            self.y_scale = 1.0
        self.est.fit(Xi, y, sample_weight=sample_weight)
        self.classes_ = getattr(self.est, "classes_", None)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.est.feature_importances_

    # ---- inference --------------------------------------------------------------------------
    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        return self.imputer.transform(X[self.columns])

    def outputs(self, X: pd.DataFrame) -> dict:
        """Return dict(pred, score, unc): `score` is a 1-D summary used for prediction-drift tests,
        `unc` is per-row uncertainty (normalised entropy / tree-ensemble std)."""
        Xi = self._prep(X)
        if self.task == "regression":
            tree_preds = np.stack([t.predict(Xi) for t in self.est.estimators_])
            pred = tree_preds.mean(0)
            return {"pred": pred, "score": pred, "unc": tree_preds.std(0) / self.y_scale}
        proba = self.est.predict_proba(Xi)
        pred = self.est.classes_[proba.argmax(1)]
        k = proba.shape[1]
        ent = -(proba * np.log(np.clip(proba, 1e-12, 1))).sum(1) / np.log(max(k, 2))
        score = proba[:, 1] if k == 2 else proba.max(1)
        return {"pred": pred, "score": score, "unc": ent, "proba": proba}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xi = self._prep(X)
        return self.est.predict(Xi)

    def loss(self, y, pred) -> float:
        """Task loss, comparable across datasets: error-rate or RMSE / std(y_train)."""
        y = np.asarray(y)
        if self.task == "regression":
            return float(np.sqrt(np.mean((y - pred) ** 2)) / self.y_scale)
        return float(np.mean(y != pred))
