# Architecture

## Components
```mermaid
flowchart LR
  U[User / client] -->|JWT| API[FastAPI]
  API --> DB[(PostgreSQL / SQLite)]
  API --> ART[(Artifacts: joblib bundles, proposals - disk or S3)]
  API --> MON[monitor.py + drift.py]
  MON --> FP[predictor.py - failure onset]
  MON --> RCA[rca.py - diagnosis]
  RCA --> REC[recovery.py - counterfactual simulator]
  REC --> APPR{{Human approval state machine}}
  APPR -->|approved| API
  API -. optional .-> MLF[MLflow]
  SIM[failures.py + simulate.py + pipeline.py] -->|benchmark cache| TRAIN[scripts/train_failure_predictor.py] --> FP
```
## Serving data flow
```mermaid
sequenceDiagram
  participant C as Client
  participant A as API
  participant D as DB
  C->>A: POST /predictions (rows [+labels])
  A->>D: log predictions
  C->>A: POST /monitoring/run (window, backfill)
  A->>A: window_signals(profile, X, outputs)
  A->>D: MonitoringMetric + Alerts
  C->>A: POST /failure-predictions
  A->>D: read last <=60 snapshots
  A->>A: build_features -> HGB -> calibrate
  C->>A: POST /diagnosis (>=30 labelled rows)
  C->>A: POST /recovery/simulate -> /validate -> /approve (human) -> /rollback
```
## Recovery state machine
`proposed -> validated | validation_failed -> approved -> applied -> rolled_back`; `rejected` reachable before apply.
Approval requires a human actor and a passed validation on holdout windows not used for fitting or ranking.
In this API `approve` = approve **and** deploy as a new production version; the previous version is archived and can be restored with `rollback`.

## Data model (tables in `api/db.py`)
```mermaid
erDiagram
  users ||--o{ datasets : owns
  users ||--o{ models : owns
  datasets ||--o{ models : trains
  models ||--|{ model_versions : has
  models ||--o{ predictions : logs
  models ||--o{ monitoring_metrics : has
  models ||--o{ alerts : raises
  models ||--o{ failure_predictions : has
  models ||--o{ diagnoses : has
  models ||--o{ recovery_simulations : has
  users ||--o{ experiments : runs
  users ||--o{ audit_logs : generates
```
`organizations` from the original spec are **not implemented** (single-owner tenancy; every resource is scoped by `owner_id`
and returns 404 to other users - tested). Alembic migrations are **not** provided: tables are created with `create_all`.

## Technology choices
FastAPI+Pydantic (typed validation, OpenAPI) | SQLAlchemy 2 (SQLite for dev/tests, PostgreSQL in prod) | scikit-learn RF as the
monitored model (tree ensembles give cheap uncertainty and TreeSHAP) | HistGradientBoosting as predictor (handles NaN natively, CPU-fast) |
PyJWT + PBKDF2 (no extra native deps) | MLflow optional behind `tracking.py` | Prefect/Airflow **not** used: monitoring is triggered by
API calls / cron; adding an orchestrator was not justified at this scale | TimescaleDB not used (plain tables suffice for the tested scale).
