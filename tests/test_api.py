"""End-to-end API tests: register -> upload -> train -> predict -> drift -> monitor -> failure prediction ->
diagnose -> simulate recovery -> validate -> approve -> rollback."""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification

from sentinelml.api import db, settings
from sentinelml.api.main import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "FAILURE_PREDICTOR_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(settings, "FAILURE_TYPE_PATH", tmp_path / "none2.joblib")
    return TestClient(create_app(db.make_engine("sqlite:///:memory:")))


def auth(client, email="a@example.com"):
    client.post("/api/v1/auth/register", json={"email": email, "password": "supersecret1"})
    tok = client.post("/api/v1/auth/login", json={"email": email, "password": "supersecret1"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def make_csv(n=1500, seed=0):
    X, y = make_classification(n, 8, n_informative=6, n_redundant=0, class_sep=1.5, shuffle=False, random_state=seed)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)]); df["label"] = y
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)   # shuffle=False sorts rows by class


def upload(client, h, df, target="label"):
    return client.post("/api/v1/datasets/upload", headers=h, data={"target": target, "name": "demo"},
                       files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")})


def test_auth_flow_and_protection(client):
    assert client.get("/health").json() == {"status": "ok"} and client.get("/ready").status_code == 200
    assert client.get("/api/v1/models").status_code == 401
    assert client.post("/api/v1/auth/register", json={"email": "bad", "password": "supersecret1"}).status_code == 422
    assert client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "short"}).status_code == 422
    h = auth(client)
    assert client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "supersecret1"}).status_code == 409
    assert client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "wrongpassword"}).status_code == 401
    assert client.get("/api/v1/auth/me", headers=h).json()["email"] == "a@example.com"
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert "x-request-id" in client.get("/health").headers


def test_dataset_validation_and_ownership(client):
    h = auth(client); h2 = auth(client, "b@example.com")
    assert upload(client, h, make_csv(), target="nope").status_code == 422
    assert upload(client, h, make_csv(20)).status_code == 422
    r = upload(client, h, make_csv()); assert r.status_code == 201
    body = r.json(); assert body["task"] == "binary" and body["quality"]["rows"] == 1500
    did = body["id"]
    assert client.get(f"/api/v1/datasets/{did}", headers=h2).status_code == 404
    assert len(client.get("/api/v1/datasets", headers=h).json()) == 1
    assert client.delete(f"/api/v1/datasets/{did}", headers=h).status_code == 204
    assert client.get(f"/api/v1/datasets/{did}", headers=h).status_code == 404


def test_full_workflow(client):
    h = auth(client)
    df = make_csv(3000)
    did = upload(client, h, df.iloc[:1500]).json()["id"]
    m = client.post("/api/v1/models/train", headers=h, json={"dataset_id": did, "name": "demo-model", "n_estimators": 40, "chronological": False})
    assert m.status_code == 201, m.text
    mid = m.json()["id"]; assert m.json()["active_version"]["metrics"]["accuracy"] > 0.8
    assert client.post("/api/v1/auth/register", json={"email": "c@example.com", "password": "supersecret1"})
    h_other = auth(client, "c@example.com")
    assert client.get(f"/api/v1/models/{mid}", headers=h_other).status_code == 404            # tenant isolation
    assert client.get(f"/api/v1/models/{mid}/versions", headers=h).json()[0]["stage"] == "staging"
    assert len(client.get("/api/v1/experiments", headers=h).json()) == 1

    # --- clean traffic with delayed labels --------------------------------------------------------------
    prod = df.iloc[1500:].reset_index(drop=True)
    feats = [f"x{i}" for i in range(8)]

    def send(chunk, corrupt=False):
        X = chunk[feats].copy()
        if corrupt:
            for f in ("x0", "x1", "x2"):                       # break the 3 most informative features
                X[f] = X[f] * 30 + 15
        r = client.post("/api/v1/predictions", headers=h, json={"model_id": mid, "rows": X.to_dict("records"), "labels": chunk["label"].tolist()})
        assert r.status_code == 201, r.text
        return r.json()

    for i in range(0, 700, 100):
        send(prod.iloc[i:i + 100])
    for i in range(700, 1000, 100):
        send(prod.iloc[i:i + 100], corrupt=True)
    hist = client.get(f"/api/v1/predictions/history?model_id={mid}&limit=5", headers=h).json()
    assert len(hist) == 5 and hist[0]["label"] is not None

    # --- monitoring -------------------------------------------------------------------------------------------
    assert client.get(f"/api/v1/monitoring/drift?model_id={mid}", headers=h).status_code == 404       # nothing yet
    r = client.post("/api/v1/monitoring/run", headers=h, json={"model_id": mid, "window_size": 100, "backfill": True})
    assert r.status_code == 200 and len(r.json()["snapshots"]) == 10
    s = r.json()["snapshots"]
    # NB: null PSI of the worst of 8 features on a 100-row window is already ~0.2 (small-sample noise), so
    # compare clean vs corrupted windows by a wide margin instead of against the 0.2 rule of thumb.
    assert s[0]["signals"]["psi_max"] < 0.4 and s[-1]["signals"]["psi_max"] > 1.0
    assert s[-1]["signals"]["loss_labeled"] > s[0]["signals"]["loss_labeled"]               # and it hurts performance
    d = client.get(f"/api/v1/monitoring/drift?model_id={mid}", headers=h).json()
    assert d["latest_feature_drift"]["x0"]["ks"] > 0.5 and len(d["history"]) == 10
    perf = client.get(f"/api/v1/monitoring/performance?model_id={mid}", headers=h).json()
    assert perf["n_labelled_predictions"] == 1000
    al = client.get("/api/v1/monitoring/alerts", headers=h).json()
    assert {"data_drift", "performance_degradation"} <= {a["kind"] for a in al}
    ov = client.get("/api/v1/monitoring/overview", headers=h).json()
    assert ov["total_models"] == 1 and ov["degraded"] == 1

    # --- failure prediction (fallback rule: no trained artifact in this test) ----------------------------------------
    fp = client.post("/api/v1/failure-predictions", headers=h, json={"model_id": mid})
    assert fp.status_code == 200 and fp.json()["method"] == "fallback_threshold_rule" and fp.json()["alert"] is True
    assert len(client.get(f"/api/v1/failure-predictions/history?model_id={mid}", headers=h).json()) == 1

    # --- diagnosis -----------------------------------------------------------------------------------------------------
    dg = client.post("/api/v1/diagnosis", headers=h, json={"model_id": mid, "window_size": 150})
    assert dg.status_code == 201, dg.text
    top = [x["feature"] for x in dg.json()["suspect_features"][:3]]
    assert set(top) == {"x0", "x1", "x2"} and "not causal" in dg.json()["caveat"]
    assert client.get(f"/api/v1/diagnosis/{dg.json()['id']}", headers=h).status_code == 200
    assert client.get(f"/api/v1/diagnosis/{dg.json()['id']}", headers=h_other).status_code == 404

    # --- recovery: simulate -> validate -> approve -> (rollback) ---------------------------------------------------------------
    sim = client.post("/api/v1/recovery/simulate", headers=h, json={"model_id": mid, "diagnosis_id": dg.json()["id"]})
    assert sim.status_code == 201, sim.text
    sid = sim.json()["id"]; assert sim.json()["state"] == "proposed"
    assert client.post("/api/v1/recovery/approve", headers=h, json={"simulation_id": sid}).status_code == 409   # not validated yet
    cands = {c["name"] for c in sim.json()["candidates"]}
    assert {"no_action", "retrain_recent", "align_suspect_features"} <= cands
    # an action that does not help must fail validation and then be un-approvable
    bad = client.post("/api/v1/recovery/validate", headers=h, json={"simulation_id": sid, "action": "rollback_previous_version"})
    assert bad.status_code == 200
    if not bad.json()["validation"]["passed"]:
        assert client.post("/api/v1/recovery/approve", headers=h, json={"simulation_id": sid}).status_code == 409
        sim = client.post("/api/v1/recovery/simulate", headers=h, json={"model_id": mid, "diagnosis_id": dg.json()["id"]}); sid = sim.json()["id"]
    act = sim.json()["recommended"]
    assert act != "no_action"
    v = client.post("/api/v1/recovery/validate", headers=h, json={"simulation_id": sid, "action": act})
    assert v.status_code == 200 and v.json()["state"] in ("validated", "validation_failed")
    if v.json()["state"] == "validated":
        a = client.post("/api/v1/recovery/approve", headers=h, json={"simulation_id": sid, "note": "looks good"})
        assert a.status_code == 200 and a.json()["state"] == "applied" and a.json()["new_version"] == 2
        vers = client.get(f"/api/v1/models/{mid}/versions", headers=h).json()
        assert [x["stage"] for x in vers] == ["staging", "production"]
        rb = client.post("/api/v1/recovery/rollback", headers=h, json={"simulation_id": sid, "note": "test rollback"})
        assert rb.status_code == 200 and rb.json()["state"] == "rolled_back"
        vers = client.get(f"/api/v1/models/{mid}/versions", headers=h).json()
        assert [x["stage"] for x in vers] == ["production", "archived"]
        assert client.post("/api/v1/recovery/rollback", headers=h, json={"simulation_id": sid}).status_code == 409


def test_prediction_edge_cases(client):
    h = auth(client); did = upload(client, h, make_csv()).json()["id"]
    mid = client.post("/api/v1/models/train", headers=h, json={"dataset_id": did, "name": "m", "n_estimators": 20, "chronological": False}).json()["id"]
    assert client.post("/api/v1/predictions", headers=h, json={"model_id": mid, "rows": []}).status_code == 422
    assert client.post("/api/v1/predictions", headers=h, json={"model_id": mid, "rows": [{"unrelated": 1}]}).status_code == 422
    assert client.post("/api/v1/predictions", headers=h, json={"model_id": mid, "rows": [{"x0": 1.0}], "labels": [0, 1]}).status_code == 422
    assert client.post("/api/v1/predictions", headers=h, json={"model_id": mid, "rows": [{"x0": 1.0}], "labels": [7]}).status_code == 422
    r = client.post("/api/v1/predictions", headers=h, json={"model_id": mid, "rows": [{"x0": 1.0, "x1": None}]})
    assert r.status_code == 201                                  # missing features are tolerated (and monitored)
    assert client.post("/api/v1/predictions", headers=h, json={"model_id": 9999, "rows": [{"x0": 1}]}).status_code == 404
    assert client.post("/api/v1/failure-predictions", headers=h, json={"model_id": mid}).status_code == 409
    assert client.post("/api/v1/diagnosis", headers=h, json={"model_id": mid}).status_code == 409
    assert client.get("/api/v1/settings/thresholds", headers=h).json()["psi_max"] == 0.2
    assert client.put("/api/v1/settings/thresholds", headers=h, json={"psi_max": 0.3, "missing_max": 0.1, "failure_probability": 0.4}).json()["psi_max"] == 0.3


ROOT_ART = __import__("pathlib").Path(__file__).resolve().parent.parent / "artifacts"


@pytest.mark.skipif(not (ROOT_ART / "failure_predictor.joblib").exists(), reason="run scripts/train_failure_predictor.py first")
def test_failure_prediction_uses_learned_artifact(client, monkeypatch):
    monkeypatch.setattr(settings, "FAILURE_PREDICTOR_PATH", ROOT_ART / "failure_predictor.joblib")
    monkeypatch.setattr(settings, "FAILURE_TYPE_PATH", ROOT_ART / "failure_type_clf.joblib")
    h = auth(client); df = make_csv(3000)
    did = upload(client, h, df.iloc[:1500]).json()["id"]
    mid = client.post("/api/v1/models/train", headers=h, json={"dataset_id": did, "name": "m", "n_estimators": 30, "chronological": False}).json()["id"]
    prod = df.iloc[1500:].reset_index(drop=True); feats = [f"x{i}" for i in range(8)]
    for k in range(12):
        X = prod.iloc[k * 100:(k + 1) * 100][feats].copy()
        if k >= 9:
            for f in ("x0", "x1", "x2"):
                X[f] = X[f] * 30 + 15
        client.post("/api/v1/predictions", headers=h, json={"model_id": mid, "rows": X.to_dict("records"), "labels": prod.iloc[k * 100:(k + 1) * 100]["label"].tolist()})
    assert client.post("/api/v1/monitoring/run", headers=h, json={"model_id": mid, "window_size": 100, "backfill": True}).status_code == 200
    r = client.post("/api/v1/failure-predictions", headers=h, json={"model_id": mid}).json()
    assert r["method"] == "learned_failure_predictor" and 0 <= r["failure_probability"] <= 1
    assert "SIMULATED" in r["calibration_note"] and r["warning_signals"]
    dg = client.post("/api/v1/diagnosis", headers=h, json={"model_id": mid, "window_size": 150}).json()
    assert "failure_type_probabilities" in dg and sum(dg["failure_type_probabilities"].values()) == pytest.approx(1, abs=0.01)
