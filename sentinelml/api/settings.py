"""Environment-based configuration (12-factor). Secrets come from the environment, never from code."""
import os
import secrets
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sentinelml.db")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "./artifacts"))
# In production SECRET_KEY must be provided (e.g. from AWS Secrets Manager). The random fallback only
# lives for the process lifetime, so tokens would not survive a restart - safe-by-default for dev.
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(48)
TOKEN_MINUTES = int(os.getenv("TOKEN_MINUTES", "60"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8501").split(",") if o]
FAILURE_PREDICTOR_PATH = Path(os.getenv("FAILURE_PREDICTOR_PATH", "./artifacts/failure_predictor.joblib"))
FAILURE_TYPE_PATH = Path(os.getenv("FAILURE_TYPE_PATH", "./artifacts/failure_type_clf.joblib"))
DEFAULT_THRESHOLDS = {"psi_max": 0.2, "missing_max": 0.05, "failure_probability": 0.5}
