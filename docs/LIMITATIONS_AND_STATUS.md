# Limitations and project status

## Built and verified (executed in the build environment)
Core library; benchmark generation (40 streams); experiments E1-E8; FastAPI backend with 31 passing pytest tests (84% line coverage measured on the first 27 tests);
`train_baselines.py` run on breast_cancer; failure-predictor artifacts; the learned artifact is exercised by an API test.

## NOT done / NOT verified
| Item from the master prompt | Status |
|---|---|
| React/TypeScript frontend | **Not built.** A Streamlit prototype exists (`dashboard/app.py`), syntax-checked only |
| AWS deployment, ECR/ECS/RDS/CloudWatch | **Not performed** (no AWS access). `deploy/aws/README.md` is an untested plan |
| Docker images / compose | Files written, **never built or run** (no Docker daemon in sandbox) |
| GitHub Actions | Workflow written, never executed |
| Alembic migrations, organizations table, TimescaleDB, Prefect/Airflow scheduler | Not implemented |
| Survival analysis, temporal neural nets for failure prediction | Not implemented (binary onset classification only) |
| LLM explanation layer (Phase 9) | Not implemented, deliberately: the ML system must stand alone and nothing here needed it |
| Systematic literature review, DOIs | Not done; see LITERATURE_AND_GAPS.md provenance note |
| Screenshots, demo video, performance/load testing, Playwright tests | Not done |
| Production monitoring of SentinelML itself | Only request-ID JSON logs, `/health`, `/ready`; no metrics/alerting |

## Scientific limitations
1. **Simulated failures.** I wrote both the failure injector and the predictor's signals; the predictor may exploit simulator regularities. Validate on real
   shifts (e.g. WILDS-style temporal splits) before claiming real-world benefit.
2. **Small and partly bootstrap data.** wine/breast_cancer pools are tiny, so windows overlap heavily; per-dataset CIs are wide.
3. **Ground-truth gating.** Failure prediction is evaluated on windows where the model is truly healthy; in deployment health is known only through delayed labels.
4. **Baselines.** The conventional baseline is one signal (max PSI). No multivariate drift test, NannyML-style estimation or tuned multi-signal rule was compared.
5. **Lead time on sudden failures** is mostly an artifact of loss smoothing in the degradation definition.
6. **Calibration** is fitted on the simulated benchmark and degrades on unseen datasets (ECE up to 0.14); do not present probabilities as real-world risk.
7. **RCA/E6 samples are not independent** (same models/streams); no CIs were computed for E5/E6. E6's "oracle" is hindsight-optimistic.
8. The monitored model is a random forest on tabular data; nothing was tested on deep models, images or text.
9. Reference profile is not refreshed after a recovery is applied (new versions keep the original reference).
10. API: unlimited-size prediction logs are scanned in Python (fine for demos, not for high throughput); no rate limiting; no token revocation.

11. **Pooled vs per-dataset.** Pooled PR-AUC is influenced by prevalence differences across datasets; per-dataset gains over PSI are directionally consistent but not individually significant (overlapping CIs); per-dataset false-alarm rates reach 53/100.
12. In `RESULTS.md` the ablation rows `no_performance` and `unlabelled_only` are the same variant (duplicate).
13. PSI's ROC-AUC is below 0.5 at H=12 (0.415): drift magnitude can be anti-informative when many drifts are harmless.
