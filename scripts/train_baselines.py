"""Phase 4: baseline models with leakage-safe model selection and experiment tracking.

Model selection uses ONLY the training split (time-ordered CV); the held-out test split is scored once,
for the selected model, at the end.

    python scripts/train_baselines.py --dataset breast_cancer
    SENTINEL_MLFLOW=1 MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/train_baselines.py --dataset synth_regression
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from sentinelml.datasets import load_dataset
from sentinelml.tracking import log_run


def candidates(task):
    imp = SimpleImputer(strategy="median")
    if task == "regression":
        return {
            "ridge": (make_pipeline(imp, StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10]}),
            "random_forest": (make_pipeline(imp, RandomForestRegressor(random_state=0)), {"randomforestregressor__n_estimators": [100], "randomforestregressor__min_samples_leaf": [1, 5]}),
            "xgboost": (XGBRegressor(verbosity=0, random_state=0), {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
            "lightgbm": (LGBMRegressor(verbose=-1, random_state=0), {"n_estimators": [100, 300], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}),
        }, "neg_root_mean_squared_error"
    return {
        "logistic_regression": (make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=2000)), {"logisticregression__C": [0.1, 1, 10]}),
        "random_forest": (make_pipeline(imp, RandomForestClassifier(random_state=0)), {"randomforestclassifier__n_estimators": [100], "randomforestclassifier__min_samples_leaf": [1, 5]}),
        "xgboost": (XGBClassifier(verbosity=0, random_state=0), {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}),
        "lightgbm": (LGBMClassifier(verbose=-1, random_state=0), {"n_estimators": [100, 300], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}),
    }, "f1_macro"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="breast_cancer"); ap.add_argument("--out", default="artifacts/baselines")
    a = ap.parse_args()
    td = load_dataset(a.dataset, 0)
    cands, scoring = candidates(td.task)
    best = (None, -np.inf, None)
    for name, (est, grid) in cands.items():
        t = time.time()
        gs = GridSearchCV(est, grid, scoring=scoring, cv=TimeSeriesSplit(4), n_jobs=1).fit(td.X_train, td.y_train)
        cv_time = time.time() - t
        pv = gs.predict(td.X_ref)                                  # validation split (never the stream pool)
        m = ({"val_rmse": float(np.sqrt(mean_squared_error(td.y_ref, pv))), "val_r2": float(r2_score(td.y_ref, pv))} if td.task == "regression"
             else {"val_accuracy": float(accuracy_score(td.y_ref, pv)), "val_f1_macro": float(f1_score(td.y_ref, pv, average="macro"))})
        m.update(cv_score=float(gs.best_score_), train_seconds=cv_time)
        log_run(f"baselines/{a.dataset}", {"model": name, **gs.best_params_, "dataset": a.dataset, "n_train": len(td.X_train)}, m)
        print(f"{name:22s} cv={gs.best_score_:.4f} {m}")
        if gs.best_score_ > best[1]:
            best = (name, gs.best_score_, gs.best_estimator_)
    os.makedirs(a.out, exist_ok=True)
    joblib.dump(best[2], f"{a.out}/{a.dataset}_{best[0]}.joblib")
    print("selected by CV on the training split:", best[0])
    # one-time pool evaluation of the SELECTED model only
    pp = best[2].predict(td.X_pool)
    print("pool (test) score of selected model:", f1_score(td.y_pool, pp, average="macro") if td.task != "regression" else float(np.sqrt(mean_squared_error(td.y_pool, pp))))


if __name__ == "__main__":
    main()
