"""Walk through the whole SentinelML workflow against a RUNNING API server.

    python scripts/demo_api_flow.py [http://localhost:8000]
Registers a user, uploads a synthetic dataset, trains a model, logs clean then corrupted traffic, and runs
monitoring -> failure prediction -> diagnosis -> recovery simulation -> validation -> human approval -> rollback.
"""
import sys, uuid

import httpx
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/") + "/api/v1"
c = httpx.Client(timeout=300)


def step(title):
    print(f"\n=== {title}")


def call(method, path, ok=(200, 201), **kw):
    r = c.request(method, BASE + path, **kw)
    if r.status_code not in ok:
        sys.exit(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()


email = f"demo-{uuid.uuid4().hex[:6]}@example.com"
step("register + login")
call("POST", "/auth/register", json={"email": email, "password": "demo-password-1"})
tok = call("POST", "/auth/login", json={"email": email, "password": "demo-password-1"})["access_token"]
H = {"Authorization": f"Bearer {tok}"}
print("logged in as", email)

step("upload dataset + train model")
X, y = make_classification(3000, 8, n_informative=6, n_redundant=0, class_sep=1.5, shuffle=False, random_state=0)
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)]); df["label"] = y
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
ds = call("POST", "/datasets/upload", headers=H, data={"target": "label", "name": "demo"},
          files={"file": ("demo.csv", df.iloc[:1500].to_csv(index=False).encode(), "text/csv")})
model = call("POST", "/models/train", headers=H, json={"dataset_id": ds["id"], "name": "demo-model", "n_estimators": 40, "chronological": False})
mid = model["id"]
print("model", mid, "validation metrics:", {k: round(v, 3) for k, v in model["active_version"]["metrics"].items()})

step("log production traffic: 7 clean batches, then 3 batches with 3 corrupted features")
prod = df.iloc[1500:].reset_index(drop=True); feats = [f"x{i}" for i in range(8)]
for k in range(10):
    chunk = prod.iloc[k * 100:(k + 1) * 100]; Xb = chunk[feats].copy()
    if k >= 7:
        for f in ("x0", "x1", "x2"):
            Xb[f] = Xb[f] * 30 + 15
    call("POST", "/predictions", headers=H, json={"model_id": mid, "rows": Xb.to_dict("records"), "labels": chunk["label"].tolist()})
print("1000 predictions logged")

step("monitoring (10 windows)")
snaps = call("POST", "/monitoring/run", headers=H, json={"model_id": mid, "window_size": 100, "backfill": True})["snapshots"]
for s in snaps:
    print(f"window {s['window_index']}: psi_max={s['signals']['psi_max']:.2f}  labelled_loss={s['signals']['loss_labeled']:.2f}  alerts={len(s['alerts'])}")

step("failure prediction")
fp = call("POST", "/failure-predictions", headers=H, json={"model_id": mid})
print("method:", fp["method"], "| probability:", round(fp["failure_probability"], 3), "| alert:", fp["alert"])

step("diagnosis (suspected features, NOT proven causes)")
dg = call("POST", "/diagnosis", headers=H, json={"model_id": mid, "window_size": 150})
print("top suspects:", [s["feature"] for s in dg["suspect_features"]])
print("failure type probabilities:", dict(list(dg.get("failure_type_probabilities", {}).items())[:3]))

step("recovery: simulate -> validate -> human approve -> rollback")
sim = call("POST", "/recovery/simulate", headers=H, json={"model_id": mid, "diagnosis_id": dg["id"]})
for cand in sim["candidates"][:4]:
    print(f"  {cand['name']:28s} eval_loss={cand['eval_loss']:.3f} gain={cand['gain']:+.3f} feasible={cand['feasible']}")
print("recommended:", sim["recommended"])
if sim["recommended"] != "no_action":
    v = call("POST", "/recovery/validate", headers=H, json={"simulation_id": sim["id"]})
    print("validation:", v["state"], v["validation"]["checks"])
    if v["state"] == "validated":
        a = call("POST", "/recovery/approve", headers=H, json={"simulation_id": sim["id"], "note": "approved in demo"})
        print("approved & applied -> new production version", a["new_version"])
        rb = call("POST", "/recovery/rollback", headers=H, json={"simulation_id": sim["id"], "note": "demo rollback"})
        print("rolled back ->", rb["state"])
print(f"\nDone. Browse {BASE[:-len('/api/v1')]}/docs to explore the same endpoints (log in with the demo email above and password demo-password-1, then use Authorize).")
