# Verification log

| Check | Environment | Result |
|---|---|---|
| Unit + API tests (31) | Linux, Python 3.12 (build) and **Windows 10, Python 3.11 (project owner)** | 31 passed (190 s on Windows) |
| Guided API demo | Windows 10, Python 3.11, uvicorn on port 8001 | all stages ran; recovery proposal correctly refused by the validation gate (see `demo_output.txt`) |
| Swagger UI with bearer auth | Windows 10, browser | `/models`, `/monitoring/alerts`, `/monitoring/overview`, `/experiments` returned 200 (screenshots in `docs/screenshots`) |
| Research reproduction | Windows 10 | `experiments/analyze.py` reproduced the pooled PR-AUC 0.420 [0.32, 0.51] vs 0.112 [0.08, 0.17] |
| Docker build / compose, GitHub Actions, AWS | not run | written only |
| React frontend | not built | - |

Issues found and fixed during verification: (1) Pydantic 1.x installed in the user's venv -> `pyproject.toml` now pins `pydantic>=2`; (2) Swagger had no Authorize button because auth was read from raw headers -> now uses `HTTPBearer`;
(3) small-window PSI false alarms and single-window degradation alerts are known behaviours (see LIMITATIONS).
