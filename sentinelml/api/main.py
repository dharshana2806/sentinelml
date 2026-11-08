"""SentinelML FastAPI application. Run: uvicorn sentinelml.api.main:app --reload"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.orm import Session, sessionmaker

from .. import tracking
from ..monitor import build_features
from ..predictor import FailurePredictor
from ..rca import FailureTypeClassifier, diagnose
from ..recovery import (InvalidTransition, RecoveryProposal, simulate_recovery, validate_candidate)
from . import db, security, services, settings

log = logging.getLogger("sentinelml")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def jlog(**kw):
    log.info(json.dumps(kw, default=str))


# ---- schemas --------------------------------------------------------------------------------------
class Credentials(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)


class TrainIn(BaseModel):
    dataset_id: int
    name: str = Field(min_length=1, max_length=200)
    n_estimators: int = Field(100, ge=10, le=500)
    chronological: bool = True
    model_id: int | None = Field(None, description="if set, registers a new version of an existing model")


class PredictIn(BaseModel):
    model_id: int
    rows: list[dict] = Field(min_length=1, max_length=1000)
    labels: list | None = None


class LabelsIn(BaseModel):
    labels: list[dict] = Field(min_length=1, max_length=5000, description="[{prediction_id, label}]")


class MonitorRunIn(BaseModel):
    model_id: int
    window_size: int = Field(100, ge=10, le=5000)
    backfill: bool = False


class FailureIn(BaseModel):
    model_id: int
    horizon: int = 6


class DiagnosisIn(BaseModel):
    model_id: int
    window_size: int = Field(150, ge=30, le=5000)


class SimIn(BaseModel):
    model_id: int
    diagnosis_id: int | None = None


class SimActionIn(BaseModel):
    simulation_id: int
    action: str | None = None
    note: str = ""


class ThresholdsIn(BaseModel):
    psi_max: float = Field(0.2, gt=0)
    missing_max: float = Field(0.05, gt=0, le=1)
    failure_probability: float = Field(0.5, gt=0, lt=1)


# ---- app factory ------------------------------------------------------------------------------------
def create_app(engine=None) -> FastAPI:
    eng = engine or db.engine
    db.init_db(eng)
    Session_ = sessionmaker(bind=eng, expire_on_commit=False)
    app = FastAPI(title="SentinelML", version="0.1.0",
                  description="Predictive reliability, root-cause diagnosis and counterfactual recovery for deployed ML models")
    app.state.sessions = Session_
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        t = time.perf_counter()
        try:
            resp = await call_next(request)
        except Exception:
            jlog(event="unhandled_error", request_id=rid, path=request.url.path)
            raise
        resp.headers["x-request-id"] = rid
        jlog(event="request", request_id=rid, method=request.method, path=request.url.path,
             status=resp.status_code, ms=round(1000 * (time.perf_counter() - t), 1))
        return resp

    def get_db():
        s = app.state.sessions()
        try:
            yield s
        finally:
            s.close()

    bearer = HTTPBearer(auto_error=False, description="Paste the access_token returned by POST /api/v1/auth/login")

    def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer), s: Session = Depends(get_db)) -> db.User:
        if creds is None:
            raise HTTPException(401, "missing bearer token")
        try:
            uid = security.decode_token(creds.credentials)
        except Exception:
            raise HTTPException(401, "invalid or expired token")
        u = s.get(db.User, uid)
        if not u:
            raise HTTPException(401, "unknown user")
        return u

    def audit(s, user, action, **detail):
        s.add(db.AuditLog(user_id=user.id if user else None, action=action, detail=services.py(detail)))

    def own_model(s: Session, user: db.User, model_id: int) -> db.MLModel:
        m = s.get(db.MLModel, model_id)
        if not m or m.owner_id != user.id:
            raise HTTPException(404, "model not found")
        return m

    def active_version(m: db.MLModel) -> db.ModelVersion:
        prod = [v for v in m.versions if v.stage == "production"]
        return prod[-1] if prod else m.versions[-1]

    def bundle_of(v: db.ModelVersion) -> dict:
        return services.load_bundle(v.artifact_path)

    def thresholds(user):
        return {**settings.DEFAULT_THRESHOLDS, **(user.thresholds or {})}

    # ---- health ----------------------------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(s: Session = Depends(get_db)):
        try:
            s.execute(select(1))
        except Exception:
            raise HTTPException(503, "database unavailable")
        return {"status": "ready"}

    # ---- auth ----------------------------------------------------------------------------------------------
    @app.post("/api/v1/auth/register", status_code=201)
    def register(c: Credentials, s: Session = Depends(get_db)):
        email = c.email.lower()
        if s.scalar(select(db.User).where(db.User.email == email)):
            raise HTTPException(409, "email already registered")
        u = db.User(email=email, password_hash=security.hash_password(c.password))
        s.add(u); s.flush(); audit(s, u, "register"); s.commit()
        return {"id": u.id, "email": u.email}

    @app.post("/api/v1/auth/login")
    def login(c: Credentials, s: Session = Depends(get_db)):
        u = s.scalar(select(db.User).where(db.User.email == c.email.lower()))
        if not u or not security.verify_password(c.password, u.password_hash):
            raise HTTPException(401, "invalid credentials")
        return {"access_token": security.create_token(u.id), "token_type": "bearer"}

    @app.get("/api/v1/auth/me")
    def me(u: db.User = Depends(current_user)):
        return {"id": u.id, "email": u.email, "created_at": u.created_at}

    @app.get("/api/v1/settings/thresholds")
    def get_thr(u: db.User = Depends(current_user)):
        return thresholds(u)

    @app.put("/api/v1/settings/thresholds")
    def put_thr(t: ThresholdsIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        u.thresholds = t.model_dump(); s.merge(u); audit(s, u, "update_thresholds", **t.model_dump()); s.commit()
        return u.thresholds

    # ---- datasets ----------------------------------------------------------------------------------------------
    @app.post("/api/v1/datasets/upload", status_code=201)
    async def upload(file: UploadFile = File(...), target: str = Form(...), name: str | None = Form(None),
                     task: str | None = Form(None), u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        raw = await file.read()
        if len(raw) > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"file larger than {settings.MAX_UPLOAD_MB} MB")
        try:
            df, task_, quality = services.read_and_validate(raw, target, task)
        except ValueError as e:
            raise HTTPException(422, str(e))
        d = db.Dataset(owner_id=u.id, name=name or file.filename or "dataset", target=target, task=task_,
                       n_rows=len(df), columns=list(df.columns), quality=quality, path="")
        s.add(d); s.flush()
        path = settings.ARTIFACT_DIR / "datasets" / f"{d.id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True); df.to_csv(path, index=False)
        d.path = str(path); audit(s, u, "upload_dataset", dataset_id=d.id); s.commit()
        return _ds(d)

    def _ds(d):
        return {"id": d.id, "name": d.name, "target": d.target, "task": d.task, "n_rows": d.n_rows,
                "columns": d.columns, "quality": d.quality, "created_at": d.created_at}

    def own_ds(s, u, i):
        d = s.get(db.Dataset, i)
        if not d or d.owner_id != u.id:
            raise HTTPException(404, "dataset not found")
        return d

    @app.get("/api/v1/datasets")
    def list_ds(u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        return [_ds(d) for d in s.scalars(select(db.Dataset).where(db.Dataset.owner_id == u.id).order_by(db.Dataset.id))]

    @app.get("/api/v1/datasets/{dataset_id}")
    def get_ds(dataset_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        return _ds(own_ds(s, u, dataset_id))

    @app.delete("/api/v1/datasets/{dataset_id}", status_code=204)
    def del_ds(dataset_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        d = own_ds(s, u, dataset_id)
        Path(d.path).unlink(missing_ok=True)
        for m in s.scalars(select(db.MLModel).where(db.MLModel.dataset_id == d.id)):
            m.dataset_id = None
        s.delete(d); audit(s, u, "delete_dataset", dataset_id=dataset_id); s.commit()

    # ---- models ---------------------------------------------------------------------------------------------------
    def _ver(v):
        return {"id": v.id, "version": v.version, "stage": v.stage, "metrics": v.metrics, "params": v.params,
                "origin": v.origin, "created_at": v.created_at}

    def _mod(m):
        v = active_version(m) if m.versions else None
        return {"id": m.id, "name": m.name, "task": m.task, "target": m.target, "features": m.features,
                "dataset_id": m.dataset_id, "created_at": m.created_at,
                "active_version": _ver(v) if v else None, "n_versions": len(m.versions)}

    @app.post("/api/v1/models/train", status_code=201)
    def train(t: TrainIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        d = own_ds(s, u, t.dataset_id)
        df = pd.read_csv(d.path)
        t0 = time.perf_counter()
        try:
            bundle, metrics = services.train_bundle(df, d.target, d.task, t.n_estimators, t.chronological)
        except ValueError as e:
            raise HTTPException(422, str(e))
        metrics["train_seconds"] = round(time.perf_counter() - t0, 3)
        if t.model_id:
            m = own_model(s, u, t.model_id)
            if m.features != bundle["features"]:
                raise HTTPException(422, "feature set differs from existing model")
        else:
            m = db.MLModel(owner_id=u.id, dataset_id=d.id, name=t.name, task=d.task, target=d.target, features=bundle["features"])
            s.add(m); s.flush()
        n = (max([v.version for v in m.versions]) + 1) if m.versions else 1
        path = settings.ARTIFACT_DIR / "models" / str(m.id) / f"v{n}.joblib"
        services.save_bundle(bundle, path)
        v = db.ModelVersion(model_id=m.id, version=n, stage="staging", artifact_path=str(path), metrics=metrics,
                            params={"n_estimators": t.n_estimators, "chronological": t.chronological, "dataset_id": d.id})
        s.add(v)
        s.add(db.Experiment(owner_id=u.id, model_id=m.id, name=f"train:{m.name}:v{n}", params=v.params, metrics=metrics))
        audit(s, u, "train_model", model_id=m.id, version=n)
        s.commit(); s.refresh(m)
        tracking.log_run(f"sentinelml/{m.name}", v.params, metrics, {"version": n})
        return _mod(m)

    @app.get("/api/v1/models")
    def list_models(u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        return [_mod(m) for m in s.scalars(select(db.MLModel).where(db.MLModel.owner_id == u.id).order_by(db.MLModel.id))]

    @app.get("/api/v1/models/{model_id}")
    def get_model(model_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        return _mod(own_model(s, u, model_id))

    @app.get("/api/v1/models/{model_id}/versions")
    def versions(model_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        return [_ver(v) for v in own_model(s, u, model_id).versions]

    @app.post("/api/v1/models/{model_id}/versions/{version}/promote")
    def promote(model_id: int, version: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, model_id)
        v = next((x for x in m.versions if x.version == version), None)
        if not v:
            raise HTTPException(404, "version not found")
        for x in m.versions:
            if x.stage == "production":
                x.stage = "archived"
        v.stage = "production"; audit(s, u, "promote_version", model_id=m.id, version=version); s.commit()
        return _ver(v)

    # ---- predictions -----------------------------------------------------------------------------------------------
    @app.post("/api/v1/predictions", status_code=201)
    def predict(p: PredictIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, p.model_id); v = active_version(m); b = bundle_of(v)
        try:
            X = services.rows_to_df(b, p.rows)
            y = services.encode_labels(b, p.labels) if p.labels is not None else None
            if y is not None and len(y) != len(X):
                raise ValueError("labels length must match rows")
        except ValueError as e:
            raise HTTPException(422, str(e))
        preds = services.decode(b, b["serving"].predict(X))
        recs = []
        for i, (row, pr) in enumerate(zip(X.to_dict("records"), preds)):
            r = db.Prediction(model_id=m.id, version_id=v.id, features=services.py(row), prediction=pr,
                              label=(p.labels[i] if p.labels is not None else None))
            s.add(r); recs.append(r)
        s.commit()
        return {"model_id": m.id, "version": v.version, "predictions": [{"id": r.id, "prediction": r.prediction} for r in recs]}

    @app.post("/api/v1/predictions/labels")
    def add_labels(body: LabelsIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        n = 0
        for item in body.labels:
            r = s.get(db.Prediction, item.get("prediction_id"))
            if r and (m := s.get(db.MLModel, r.model_id)) and m.owner_id == u.id:
                r.label = item.get("label"); n += 1
        s.commit()
        return {"updated": n}

    @app.get("/api/v1/predictions/history")
    def pred_hist(model_id: int, limit: int = Query(100, ge=1, le=1000), u: db.User = Depends(current_user),
                  s: Session = Depends(get_db)):
        own_model(s, u, model_id)
        rows = s.scalars(select(db.Prediction).where(db.Prediction.model_id == model_id).order_by(db.Prediction.id.desc()).limit(limit))
        return [{"id": r.id, "version_id": r.version_id, "prediction": r.prediction, "label": r.label, "created_at": r.created_at} for r in rows]

    # ---- monitoring ---------------------------------------------------------------------------------------------------
    def logged(s, m, labeled_only=False, limit=None):
        q = select(db.Prediction).where(db.Prediction.model_id == m.id)
        if labeled_only:
            q = q.where(db.Prediction.label.is_not(None))
        q = q.order_by(db.Prediction.id.desc())
        if limit:
            q = q.limit(limit)
        return list(reversed(list(s.scalars(q))))

    def frame(b, recs):
        X = pd.DataFrame([r.features for r in recs])[b["features"]].astype(float)
        y = None
        if recs and all(r.label is not None for r in recs):
            y = services.encode_labels(b, [r.label for r in recs])
        return X, y

    def raise_alerts(s, m, sig, thr, loss_thr):
        new = []
        if sig["psi_max"] is not None and sig["psi_max"] > thr["psi_max"]:
            new.append(("data_drift", "critical" if sig["psi_max"] > 0.25 else "warning", f"max feature PSI {sig['psi_max']:.3f} > {thr['psi_max']}"))
        if sig["missing_max"] is not None and sig["missing_max"] > thr["missing_max"]:
            new.append(("data_quality", "warning", f"missing-value rate {sig['missing_max']:.3f} > {thr['missing_max']}"))
        if sig.get("loss_labeled") is not None and sig["loss_labeled"] > loss_thr:
            new.append(("performance_degradation", "critical", f"labelled loss {sig['loss_labeled']:.3f} > degradation threshold {loss_thr:.3f}"))
        for k, sev, msg in new:
            s.add(db.Alert(model_id=m.id, kind=k, severity=sev, message=msg))
        return new

    @app.post("/api/v1/monitoring/run")
    def run_monitor(body: MonitorRunIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, body.model_id); v = active_version(m); b = bundle_of(v)
        recs = [r for r in logged(s, m) if r.version_id == v.id]
        w = body.window_size
        chunks = [recs[i:i + w] for i in range(0, len(recs) - w + 1, w)] if body.backfill else [recs[-w:]]
        chunks = [c for c in chunks if len(c) >= 10]
        if not chunks:
            raise HTTPException(409, f"need at least 10 logged predictions for the active model version (have {len(recs)})")
        start = s.scalar(select(func.count()).select_from(db.MonitoringMetric).where(db.MonitoringMetric.model_id == m.id)) or 0
        out = []
        for i, c in enumerate(chunks):
            X, y = frame(b, c)
            sig, fd = services.snapshot(b, X, y)
            s.add(db.MonitoringMetric(model_id=m.id, version_id=v.id, window_index=start + i, n_rows=len(c), signals=sig, feature_drift=fd))
            alerts = raise_alerts(s, m, sig, thresholds(u), b["thr"])
            out.append({"window_index": start + i, "n_rows": len(c), "signals": sig, "alerts": [a[2] for a in alerts]})
        s.commit()
        return {"snapshots": out}

    def snaps(s, m, n=60):
        return list(reversed(list(s.scalars(select(db.MonitoringMetric).where(db.MonitoringMetric.model_id == m.id)
                                            .order_by(db.MonitoringMetric.id.desc()).limit(n)))))

    @app.get("/api/v1/monitoring/drift")
    def drift(model_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, model_id); sn = snaps(s, m)
        if not sn:
            raise HTTPException(404, "no monitoring snapshots yet; POST /api/v1/monitoring/run first")
        keys = ("psi_max", "psi_mean", "ks_max", "wd_max", "js_mean", "missing_max", "outlier_rate", "pred_psi")
        return {"latest_feature_drift": sn[-1].feature_drift,
                "history": [{"window_index": x.window_index, "created_at": x.created_at, **{k: x.signals.get(k) for k in keys}} for x in sn],
                "note": "Drift is evidence, not proof, of degradation: see /monitoring/performance for labelled loss."}

    @app.get("/api/v1/monitoring/performance")
    def performance(model_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, model_id); sn = snaps(s, m); b = bundle_of(active_version(m))
        hist = [{"window_index": x.window_index, "loss": x.signals.get("loss_labeled"), "unc_mean": x.signals.get("unc_mean")} for x in sn]
        lab = logged(s, m, labeled_only=True)
        return {"loss_definition": "error rate (classification) or RMSE/std(y_train) (regression)",
                "baseline_loss": b["base_loss"], "degradation_threshold": b["thr"], "n_labelled_predictions": len(lab), "history": hist}

    @app.get("/api/v1/monitoring/alerts")
    def alerts(model_id: int | None = None, limit: int = Query(50, ge=1, le=500), u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        ids = [m.id for m in s.scalars(select(db.MLModel).where(db.MLModel.owner_id == u.id))]
        q = select(db.Alert).where(db.Alert.model_id.in_(ids))
        if model_id:
            q = q.where(db.Alert.model_id == model_id)
        return [{"id": a.id, "model_id": a.model_id, "kind": a.kind, "severity": a.severity, "message": a.message, "created_at": a.created_at}
                for a in s.scalars(q.order_by(db.Alert.id.desc()).limit(limit))]

    @app.get("/api/v1/monitoring/overview")
    def overview(u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        models = list(s.scalars(select(db.MLModel).where(db.MLModel.owner_id == u.id)))
        rows, counts = [], {"healthy": 0, "at_risk": 0, "degraded": 0, "unmonitored": 0}
        for m in models:
            sn = snaps(s, m, 1); fp = s.scalar(select(db.FailurePredictionRec).where(db.FailurePredictionRec.model_id == m.id).order_by(db.FailurePredictionRec.id.desc()))
            b = bundle_of(active_version(m)); status = "unmonitored"
            if sn:
                loss = sn[-1].signals.get("loss_labeled")
                status = "degraded" if (loss is not None and loss > b["thr"]) else ("at_risk" if (fp and fp.alert) else "healthy")
            counts[status] += 1
            rows.append({"model_id": m.id, "name": m.name, "status": status, "predicted_failure_probability": fp.probability if fp else None})
        n_alerts = s.scalar(select(func.count()).select_from(db.Alert).where(db.Alert.model_id.in_([m.id for m in models] or [-1]), db.Alert.acknowledged == False)) or 0  # noqa: E712
        return {"total_models": len(models), **counts, "open_alerts": n_alerts, "models": rows}

    # ---- failure prediction --------------------------------------------------------------------------------------------------
    @app.post("/api/v1/failure-predictions")
    def failure_prediction(body: FailureIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, body.model_id); sn = snaps(s, m, 60)
        if len(sn) < 3:
            raise HTTPException(409, f"need >=3 monitoring snapshots (have {len(sn)}); POST /api/v1/monitoring/run with backfill=true")
        sig = pd.DataFrame([x.signals for x in sn])
        sig["loss_lag"] = pd.to_numeric(sig.get("loss_labeled"), errors="coerce").shift(1)
        feats = build_features(sig)
        detail = {}
        if settings.FAILURE_PREDICTOR_PATH.exists():
            fp = FailurePredictor.load(str(settings.FAILURE_PREDICTOR_PATH))
            row = feats.reindex(columns=fp.columns).iloc[[-1]]
            raw = float(fp.score(row)[0]); prob = float(fp.calibrated(np.array([raw]))[0])
            alert = raw >= fp.threshold
            detail.update(method="learned_failure_predictor", model=fp.name, trained_horizon=fp.horizon, raw_score=raw,
                          alert_threshold=fp.threshold, calibrated=fp.calibrator is not None,
                          calibration_note="Calibrated on a SIMULATED failure benchmark; real-world calibration is not validated.")
        else:
            raw = float(sig["psi_max"].iloc[-1]); prob = float(min(1.0, raw / 0.25)); alert = raw > thresholds(u)["psi_max"]
            detail.update(method="fallback_threshold_rule", calibrated=False,
                          note="No trained failure predictor artifact found; using a PSI threshold rule. Run scripts/train_failure_predictor.py.")
        cpz = feats.iloc[-1][[c for c in feats.columns if c.endswith("_cpz")]].dropna().sort_values(ascending=False).head(5)
        detail["warning_signals"] = [{"signal": k[:-4], "change_z": float(v), "value": float(sig[k[:-4]].iloc[-1]) if sig[k[:-4]].iloc[-1] == sig[k[:-4]].iloc[-1] else None} for k, v in cpz.items()]
        detail["history_windows_used"] = len(sn)
        rec = db.FailurePredictionRec(model_id=m.id, horizon=body.horizon, probability=prob, alert=bool(alert), detail=services.py(detail))
        s.add(rec)
        if alert:
            s.add(db.Alert(model_id=m.id, kind="predicted_failure", severity="warning", message=f"failure-onset score {raw:.3f} above alert threshold (p={prob:.2f})"))
        s.commit()
        return {"id": rec.id, "model_id": m.id, "horizon_windows": body.horizon, "failure_probability": prob, "alert": bool(alert), **detail}

    @app.get("/api/v1/failure-predictions/history")
    def fp_hist(model_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        own_model(s, u, model_id)
        return [{"id": r.id, "probability": r.probability, "alert": r.alert, "horizon": r.horizon, "created_at": r.created_at, "detail": r.detail}
                for r in s.scalars(select(db.FailurePredictionRec).where(db.FailurePredictionRec.model_id == model_id).order_by(db.FailurePredictionRec.id.desc()).limit(100))]

    # ---- diagnosis --------------------------------------------------------------------------------------------------------------
    @app.post("/api/v1/diagnosis", status_code=201)
    def diagnosis(body: DiagnosisIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, body.model_id); v = active_version(m); b = bundle_of(v)
        recs = [r for r in logged(s, m, labeled_only=True, limit=body.window_size) if r.version_id == v.id]
        if len(recs) < 30:
            raise HTTPException(409, f"diagnosis needs >=30 labelled predictions for the active version (have {len(recs)})")
        X, y = frame(b, recs)
        sig_rows = pd.DataFrame([services.snapshot(b, X, y)[0]])
        clf = joblib.load(settings.FAILURE_TYPE_PATH) if settings.FAILURE_TYPE_PATH.exists() else None
        res = services.py(diagnose(services.ctx_from_bundle(b), X, y, sig_rows, clf))
        res["n_rows"] = len(recs)
        d = db.Diagnosis(model_id=m.id, result=res); s.add(d); s.commit()
        return {"id": d.id, "model_id": m.id, **res}

    @app.get("/api/v1/diagnosis/{diagnosis_id}")
    def get_diag(diagnosis_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        d = s.get(db.Diagnosis, diagnosis_id)
        if not d:
            raise HTTPException(404, "diagnosis not found")
        own_model(s, u, d.model_id)
        return {"id": d.id, "model_id": d.model_id, "created_at": d.created_at, **d.result}

    # ---- recovery --------------------------------------------------------------------------------------------------------------------
    def split_windows(X, y, n=10):
        size = len(X) // n
        return [(X.iloc[i * size:(i + 1) * size].reset_index(drop=True), y[i * size:(i + 1) * size]) for i in range(n)]

    def recent_windows(s, m, v, b):
        recs = [r for r in logged(s, m, labeled_only=True, limit=2000) if r.version_id == v.id]
        if len(recs) < 100:
            raise HTTPException(409, f"recovery simulation needs >=100 labelled predictions for the active version (have {len(recs)})")
        X, y = frame(b, recs)
        w = split_windows(X, y)
        return w[:4], w[4:7], w[7:]                                 # fit / eval / holdout (chronological)

    def own_sim(s, u, i):
        r = s.get(db.RecoverySimulation, i)
        if not r:
            raise HTTPException(404, "simulation not found")
        own_model(s, u, r.model_id)
        return r

    def proposal_of(r):
        return RecoveryProposal(r.action or "", r.state, list(r.history or []))

    def save_state(r, prop):
        r.state = prop.state; r.history = services.py(prop.history)

    @app.post("/api/v1/recovery/simulate", status_code=201)
    def rec_simulate(body: SimIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        m = own_model(s, u, body.model_id); v = active_version(m); b = bundle_of(v)
        fit_w, eval_w, _ = recent_windows(s, m, v, b)
        suspects = []
        if body.diagnosis_id:
            d = s.get(db.Diagnosis, body.diagnosis_id)
            if not d or d.model_id != m.id:
                raise HTTPException(404, "diagnosis not found for this model")
            suspects = [f["feature"] for f in d.result.get("suspect_features", []) if (f.get("combined") or 0) > 0.5][:3]
        prev = None
        older = [x for x in m.versions if x.version < v.version]
        if older:
            prev = bundle_of(older[-1])["model"]
        ctx = services.ctx_from_bundle(b, prev)
        out = simulate_recovery(ctx, fit_w, eval_w, suspects)
        r = db.RecoverySimulation(model_id=m.id, diagnosis_id=body.diagnosis_id, action=out["recommended"],
                                  results=services.py({"suspects": suspects, "recommended": out["recommended"],
                                                       "baseline_eval_loss": out["baseline_eval_loss"],
                                                       "candidates": [c.to_dict() for c in out["candidates"]]}),
                                  previous_version_id=v.id, history=[])
        s.add(r); s.flush()
        path = settings.ARTIFACT_DIR / "proposals" / f"{r.id}.joblib"; path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({c.name: c.predictor for c in out["candidates"]}, path)
        audit(s, u, "recovery_simulate", simulation_id=r.id); s.commit()
        return {"id": r.id, "state": r.state, **r.results,
                "next_step": "POST /api/v1/recovery/validate, then a human must POST /api/v1/recovery/approve"}

    @app.post("/api/v1/recovery/validate")
    def rec_validate(body: SimActionIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        r = own_sim(s, u, body.simulation_id); m = s.get(db.MLModel, r.model_id); v = s.get(db.ModelVersion, r.previous_version_id)
        action = body.action or r.action
        preds = joblib.load(settings.ARTIFACT_DIR / "proposals" / f"{r.id}.joblib")
        if action not in preds:
            raise HTTPException(422, f"unknown action {action!r}; options: {list(preds)}")
        if action == "no_action":
            raise HTTPException(422, "no_action needs no validation")
        b = bundle_of(v)
        _, _, hold = recent_windows(s, m, v, b)
        val = services.py(validate_candidate(services.ctx_from_bundle(b), preds[action], hold))
        r.action = action; r.validation = val
        prop = proposal_of(r)
        try:
            prop.record_validation(val["passed"], actor=f"user:{u.id}")
        except InvalidTransition as e:
            raise HTTPException(409, str(e))
        save_state(r, prop); audit(s, u, "recovery_validate", simulation_id=r.id, passed=val["passed"]); s.commit()
        return {"id": r.id, "state": r.state, "action": action, "validation": val}

    @app.post("/api/v1/recovery/approve")
    def rec_approve(body: SimActionIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        """Human approval AND deployment: validated proposal -> approved -> applied as a new production version."""
        r = own_sim(s, u, body.simulation_id); m = s.get(db.MLModel, r.model_id)
        prop = proposal_of(r)
        try:
            prop.approve(f"user:{u.id}", body.note)
            pred = joblib.load(settings.ARTIFACT_DIR / "proposals" / f"{r.id}.joblib")[r.action]
            old = s.get(db.ModelVersion, r.previous_version_id); ob = bundle_of(old)
            nb = dict(ob); nb["serving"] = pred; nb["model"] = getattr(pred, "model", ob["model"])
            n = max(x.version for x in m.versions) + 1
            path = settings.ARTIFACT_DIR / "models" / str(m.id) / f"v{n}.joblib"; services.save_bundle(nb, path)
            for x in m.versions:
                if x.stage == "production":
                    x.stage = "archived"
            nv = db.ModelVersion(model_id=m.id, version=n, stage="production", artifact_path=str(path), metrics=old.metrics,
                                 params={"recovery_simulation_id": r.id, "action": r.action}, origin=f"recovery:{r.action}")
            s.add(nv); s.flush(); r.new_version_id = nv.id
            prop.apply(f"user:{u.id}")
        except InvalidTransition as e:
            raise HTTPException(409, str(e))
        save_state(r, prop); audit(s, u, "recovery_approve_apply", simulation_id=r.id, new_version=n); s.commit()
        return {"id": r.id, "state": r.state, "new_version": n, "history": r.history}

    @app.post("/api/v1/recovery/reject")
    def rec_reject(body: SimActionIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        r = own_sim(s, u, body.simulation_id); prop = proposal_of(r)
        try:
            prop.reject(f"user:{u.id}", body.note)
        except InvalidTransition as e:
            raise HTTPException(409, str(e))
        save_state(r, prop); audit(s, u, "recovery_reject", simulation_id=r.id); s.commit()
        return {"id": r.id, "state": r.state}

    @app.post("/api/v1/recovery/rollback")
    def rec_rollback(body: SimActionIn, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        r = own_sim(s, u, body.simulation_id); m = s.get(db.MLModel, r.model_id); prop = proposal_of(r)
        try:
            prop.rollback(f"user:{u.id}", body.note)
        except InvalidTransition as e:
            raise HTTPException(409, str(e))
        for x in m.versions:
            if x.id == r.new_version_id:
                x.stage = "archived"
            if x.id == r.previous_version_id:
                x.stage = "production"
        save_state(r, prop); audit(s, u, "recovery_rollback", simulation_id=r.id); s.commit()
        return {"id": r.id, "state": r.state, "restored_version_id": r.previous_version_id}

    @app.get("/api/v1/recovery/{simulation_id}")
    def rec_get(simulation_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        r = own_sim(s, u, simulation_id)
        return {"id": r.id, "state": r.state, "action": r.action, "validation": r.validation, "history": r.history, **r.results}

    # ---- experiments -----------------------------------------------------------------------------------------------------------------
    @app.get("/api/v1/experiments")
    def experiments(u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        return [{"id": e.id, "name": e.name, "model_id": e.model_id, "params": e.params, "metrics": e.metrics, "created_at": e.created_at}
                for e in s.scalars(select(db.Experiment).where(db.Experiment.owner_id == u.id).order_by(db.Experiment.id.desc()))]

    @app.get("/api/v1/experiments/{experiment_id}")
    def experiment(experiment_id: int, u: db.User = Depends(current_user), s: Session = Depends(get_db)):
        e = s.get(db.Experiment, experiment_id)
        if not e or e.owner_id != u.id:
            raise HTTPException(404, "experiment not found")
        return {"id": e.id, "name": e.name, "model_id": e.model_id, "params": e.params, "metrics": e.metrics, "created_at": e.created_at}

    return app


app = create_app()
