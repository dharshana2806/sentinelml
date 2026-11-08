"""Business logic for the API: dataset validation, training, monitoring snapshots, serving contexts."""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score

from ..config import SimConfig
from ..drift import RefProfile, feature_stats
from ..model import TaskModel
from ..monitor import window_signals
from ..recovery import _Wrapped
from ..simulate import calibrate_threshold

MIN_ROWS = 60


def py(o):
    """Make numpy / pandas scalars JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [py(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---- datasets --------------------------------------------------------------------------------
def infer_task(y: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(y) or y.dtype == bool:
        return "binary" if y.nunique() == 2 else "multiclass"
    if y.nunique() <= 10 and np.allclose(y.dropna(), y.dropna().round()):
        return "binary" if y.nunique() == 2 else "multiclass"
    return "regression"


def read_and_validate(raw: bytes, target: str, task: str | None) -> tuple:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"could not parse CSV: {e}")
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found; columns: {list(df.columns)[:30]}")
    if len(df) < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} rows, got {len(df)}")
    n0 = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    feats = [c for c in df.columns if c != target]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]
    dropped_non_numeric = [c for c in feats if c not in numeric]
    all_null = [c for c in numeric if df[c].isna().all()]
    constant = [c for c in numeric if c not in all_null and df[c].nunique(dropna=True) <= 1]
    used = [c for c in numeric if c not in all_null]
    if not used:
        raise ValueError("no usable numeric feature columns")
    task = task or infer_task(df[target])
    if task not in ("binary", "multiclass", "regression"):
        raise ValueError("task must be binary, multiclass or regression")
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("regression target must be numeric")
    quality = {
        "rows": int(len(df)), "rows_dropped_missing_target": int(n0 - len(df)),
        "features_used": used, "dropped_non_numeric": dropped_non_numeric,
        "dropped_all_null": all_null, "constant_features": constant,
        "duplicate_row_rate": round(float(df.duplicated().mean()), 4),
        "max_feature_missing_rate": round(float(df[used].isna().mean().max()), 4),
        "target_distribution": (df[target].value_counts(normalize=True).round(4).to_dict() if task != "regression" else
                                {"mean": float(df[target].mean()), "std": float(df[target].std())}),
    }
    quality = py({k: ({str(a): b for a, b in v.items()} if k == "target_distribution" else v) for k, v in quality.items()})
    return df[used + [target]], task, quality


# ---- training -----------------------------------------------------------------------------------
def train_bundle(df: pd.DataFrame, target: str, task: str, n_estimators: int = 100, chronological: bool = True,
                 seed: int = 0) -> tuple:
    feats = [c for c in df.columns if c != target]
    n = len(df)
    if chronological:
        cut = int(0.7 * n); tr, ref = df.iloc[:cut], df.iloc[cut:]
    else:
        idx = np.random.default_rng(seed).permutation(n); cut = int(0.7 * n)
        tr, ref = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    classes = None
    if task == "regression":
        ytr, yref = tr[target].to_numpy(float), ref[target].to_numpy(float)
    else:
        classes = sorted(df[target].unique().tolist())
        enc = {c: i for i, c in enumerate(classes)}
        ytr, yref = tr[target].map(enc).to_numpy(int), ref[target].map(enc).to_numpy(int)
        if len(np.unique(ytr)) < len(classes):
            raise ValueError("training split does not contain every class; set chronological=false")
    Xtr, Xref = tr[feats].reset_index(drop=True), ref[feats].reset_index(drop=True)
    model = TaskModel(task, n_estimators=n_estimators, seed=seed).fit(Xtr, ytr)
    out_ref = model.outputs(Xref)
    profile = RefProfile.build(Xref, model.feature_importances_, out_ref, seed=seed)
    pred = model.predict(Xref)
    if task == "regression":
        metrics = {"rmse": float(np.sqrt(np.mean((yref - pred) ** 2))), "mae": float(mean_absolute_error(yref, pred)),
                   "r2": float(r2_score(yref, pred))}
    else:
        metrics = {"accuracy": float(accuracy_score(yref, pred)), "f1_macro": float(f1_score(yref, pred, average="macro"))}
        if task == "binary" and len(np.unique(yref)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(yref, out_ref["proba"][:, 1]))
    cfg = SimConfig(window_size=max(10, min(100, len(Xref) // 2)))
    base, thr = calibrate_threshold(model, Xref, yref, cfg, seed)
    cap = lambda X, y: (X.iloc[:5000], y[:5000])  # noqa: E731  (bound artifact size)
    Xt, yt = cap(Xtr, ytr)
    bundle = {"model": model, "serving": _Wrapped(model), "profile": profile, "classes": classes, "features": feats,
              "task": task, "target": target, "X_train": Xt, "y_train": yt, "X_ref": Xref, "y_ref": yref,
              "base_loss": base, "thr": thr, "n_classes": len(classes) if classes else 0}
    metrics["baseline_window_loss"] = base
    metrics["degradation_threshold"] = thr
    return bundle, py(metrics)


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


@lru_cache(maxsize=16)
def load_bundle(path: str) -> dict:
    return joblib.load(path)


def ctx_from_bundle(b: dict, prev_model: TaskModel | None = None) -> SimpleNamespace:
    td = SimpleNamespace(X_train=b["X_train"], y_train=b["y_train"], X_ref=b["X_ref"], y_ref=b["y_ref"],
                         feature_names=b["features"], task=b["task"], is_classification=b["task"] != "regression")
    if prev_model is None:                       # cheap stand-in "older version" for the rollback candidate
        n = len(b["X_train"]); sub = np.random.default_rng(0).choice(n, int(0.6 * n), replace=False)
        prev_model = TaskModel(b["task"], n_estimators=30, seed=7, max_depth=8).fit(b["X_train"].iloc[sub], b["y_train"][sub])
    return SimpleNamespace(td=td, model=b["model"], prev_model=prev_model, profile=b["profile"], base_loss=b["base_loss"],
                           thr=b["thr"], n_classes=b["n_classes"], cfg=SimpleNamespace(window_size=100))


# ---- serving / monitoring -------------------------------------------------------------------------
def rows_to_df(b: dict, rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    missing = [c for c in b["features"] if c not in df.columns]
    if len(missing) == len(b["features"]):
        raise ValueError(f"rows contain none of the expected features {b['features'][:10]}")
    for c in missing:
        df[c] = np.nan                                   # schema mismatch is *monitored*, not fatal
    return df[b["features"]].apply(pd.to_numeric, errors="coerce")


def decode(b: dict, enc: np.ndarray) -> list:
    if b["classes"] is None:
        return [float(v) for v in enc]
    return [py(b["classes"][int(i)]) for i in enc]


def encode_labels(b: dict, labels: list) -> np.ndarray:
    if b["classes"] is None:
        return np.asarray(labels, float)
    m = {c: i for i, c in enumerate(b["classes"])}
    try:
        return np.array([m[l] for l in labels])
    except KeyError as e:
        raise ValueError(f"unknown label {e}; known classes: {b['classes']}")


def snapshot(b: dict, X: pd.DataFrame, y_enc: np.ndarray | None = None) -> tuple:
    if len(X) < 10:
        raise ValueError("a monitoring window needs at least 10 rows")
    out = b["model"].outputs(X)
    sig = window_signals(b["profile"], X, out)
    sig["loss_labeled"] = float(b["model"].loss(y_enc, b["serving"].predict(X))) if y_enc is not None else None
    fd = feature_stats(b["profile"], X)[["psi", "ks", "wd", "miss"]].round(4)
    return py(sig), py(fd.to_dict("index"))
