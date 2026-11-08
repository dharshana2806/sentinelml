"""Benchmark datasets. All are loaded offline (scikit-learn built-ins or synthetic).

Real datasets (scikit-learn bundled copies of UCI data):
  * breast_cancer  - binary classification, 569 x 30   (UCI WDBC, CC BY 4.0)
  * wine           - 3-class classification, 178 x 13  (UCI Wine, CC BY 4.0)
  * diabetes       - regression, 442 x 10              (Efron et al. 2004, bundled with sklearn)
Synthetic datasets (generated, no license issue):
  * synth_binary, synth_multiclass, synth_regression
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split

DATASETS = ["breast_cancer", "wine", "diabetes", "synth_binary", "synth_multiclass", "synth_regression"]


@dataclass
class TaskData:
    name: str
    task: str                      # binary | multiclass | regression
    feature_names: list
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_ref: pd.DataFrame            # reference-validation split: calibrates monitoring thresholds
    y_ref: np.ndarray
    X_pool: pd.DataFrame           # production pool: windows of the simulated stream are drawn from it
    y_pool: np.ndarray

    @property
    def is_classification(self) -> bool:
        return self.task in ("binary", "multiclass")


def _raw(name: str, seed: int):
    if name == "breast_cancer":
        d = skd.load_breast_cancer(as_frame=True)
        return d.data, d.target.to_numpy(), "binary"
    if name == "wine":
        d = skd.load_wine(as_frame=True)
        return d.data, d.target.to_numpy(), "multiclass"
    if name == "diabetes":
        d = skd.load_diabetes(as_frame=True)
        return d.data, d.target.to_numpy().astype(float), "regression"
    if name == "synth_binary":
        X, y = skd.make_classification(6000, 20, n_informative=8, n_redundant=4, class_sep=1.2,
                                       flip_y=0.02, random_state=seed)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]), y, "binary"
    if name == "synth_multiclass":
        X, y = skd.make_classification(6000, 16, n_informative=8, n_redundant=3, n_classes=4,
                                       n_clusters_per_class=1, class_sep=1.3, flip_y=0.02, random_state=seed)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(16)]), y, "multiclass"
    if name == "synth_regression":
        X, y = skd.make_friedman1(6000, n_features=10, noise=1.0, random_state=seed)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), y.astype(float), "regression"
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


def load_dataset(name: str, seed: int = 0) -> TaskData:
    """Load a dataset and split it 40/20/40 into train / reference-validation / production pool.

    The production pool is *never* used for training or threshold calibration.
    """
    X, y, task = _raw(name, seed)
    X = X.astype(float).reset_index(drop=True)
    strat = y if task != "regression" else None
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, train_size=0.4, random_state=seed, stratify=strat)
    strat2 = y_rest if task != "regression" else None
    X_ref, X_pool, y_ref, y_pool = train_test_split(X_rest, y_rest, train_size=1 / 3, random_state=seed,
                                                     stratify=strat2)
    rs = lambda d: d.reset_index(drop=True)  # noqa: E731
    return TaskData(name, task, list(X.columns), rs(X_tr), y_tr, rs(X_ref), y_ref, rs(X_pool), y_pool)
