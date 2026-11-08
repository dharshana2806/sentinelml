"""Train the deployable failure-onset predictor + failure-type classifier from the cached benchmark.

    python scripts/train_failure_predictor.py --cache results/cache --out artifacts
The artifacts are used by the API (/failure-predictions, /diagnosis). They are trained on SIMULATED failures;
see docs/LIMITATIONS.md before trusting their probabilities on real systems.
"""
import argparse, glob, os, pickle, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pandas as pd

from sentinelml.config import SimConfig
from sentinelml.failures import FailureEvent
from sentinelml.predictor import FailurePredictor, build_table, feature_columns, make_model
from sentinelml.rca import FailureTypeClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="results/cache"); ap.add_argument("--out", default="artifacts")
a = ap.parse_args()
R = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(f"{a.cache}/*.pkl"))]
streams = [SimpleNamespace(name=r["name"], seed=r["seed"], signals=r["signals"], degraded=r["degraded"],
                           events=[FailureEvent(**e) for e in r["events"]]) for r in R]
cfg = SimConfig()
tbl = build_table(streams, cfg)
fp = FailurePredictor("hgb", feature_columns(tbl), make_model("hgb"), horizon=cfg.horizon)
fp.fit(tbl[tbl.seg == "train"], tbl[tbl.seg == "val"])
os.makedirs(a.out, exist_ok=True)
fp.save(f"{a.out}/failure_predictor.joblib")
D = pd.DataFrame([d for r in R for d in r["diag"]])
sig = D[[c for c in D.columns if c.startswith("sig_")]].rename(columns=lambda c: c[4:])
joblib.dump(FailureTypeClassifier().fit(sig, D.ftype.to_numpy()), f"{a.out}/failure_type_clf.joblib")
print(f"saved failure predictor ({len(fp.columns)} features, threshold={fp.threshold:.3f}) and type classifier ({len(D)} incidents) to {a.out}")
