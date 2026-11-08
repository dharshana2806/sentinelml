# Resume entry and pitch (claims limited to what was built and measured)

## Resume (pick 3-4 bullets)
**SentinelML - Predictive reliability monitoring for deployed ML models** (Python, scikit-learn, XGBoost/LightGBM, SHAP, FastAPI, SQLAlchemy, pytest)
- Built a failure-injection benchmark (7 failure types, sudden/gradual, ground-truth labelled; 5 datasets, 40 streams, 400 events) and an end-to-end pipeline for drift monitoring, failure-onset prediction, root-cause ranking and offline recovery simulation.
- Showed on this simulated benchmark that a temporal gradient-boosting predictor reached PR-AUC 0.42 (95% CI 0.32-0.51) vs 0.11 for a PSI>0.2 rule, with 11 vs 84 false alarms per 100 healthy windows (pooled); evaluated with chronological splits, purge gaps and stream-level bootstrap CIs; reported per-dataset results where gains were not individually significant.
- Implemented a FastAPI service (JWT auth, tenant isolation, model versioning/rollback, prediction logging, monitoring, diagnosis, human-approval recovery workflow) with 31 automated tests; verified end to end on Windows.
- Designed a recovery policy that matched "always retrain" on average improvement (34% vs 33% loss reduction) while cutting harmful actions from 5.5% to 0.8%.
Do **not** write: "deployed on AWS", "React dashboard", "production-ready", "novel", or real-world accuracy claims.

## 30-second pitch
"SentinelML predicts when a deployed model is about to degrade instead of just alerting on drift. On my simulated benchmark, drift size alone was no better than chance at predicting failures, while a temporal model over drift, uncertainty and delayed-label signals ranked them much better. It also diagnoses likely causes and simulates recovery options, and a human has to approve before anything is deployed."

## Likely follow-ups
- *What's the weakness?* Simulated failures designed by me; per-dataset confidence intervals overlap; thresholds transfer poorly to unseen datasets; no real deployment.
- *What would you do next?* Validate on real temporal-shift data, add stronger baselines (NannyML-style performance estimation), smooth the API's degradation alert, then containerise and deploy.
