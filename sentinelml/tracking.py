"""Experiment tracking: MLflow when enabled (SENTINEL_MLFLOW=1 and mlflow installed), else JSON lines."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def log_run(experiment: str, params: dict, metrics: dict, tags: dict | None = None) -> dict:
    rec = {"experiment": experiment, "params": params, "metrics": metrics, "tags": tags or {}, "ts": time.time()}
    if os.getenv("SENTINEL_MLFLOW") == "1":
        try:
            import mlflow
            mlflow.set_experiment(experiment)
            with mlflow.start_run():
                mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
                if tags:
                    mlflow.set_tags({k: str(v) for k, v in tags.items()})
            rec["backend"] = "mlflow"
            return rec
        except Exception as e:                      # never let tracking break the pipeline
            rec["tracking_error"] = str(e)
    path = Path(os.getenv("SENTINEL_RUNS_FILE", "runs.jsonl"))
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    rec["backend"] = "jsonl"
    return rec
