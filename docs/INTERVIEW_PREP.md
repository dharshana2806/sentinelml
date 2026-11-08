# Interview preparation (grounded in what was actually built and measured)

Rule: only claim what is in README/RESULTS. If you did not run it yourself, say "I reproduced it with `experiments/analyze.py`" only after you have.

## Pitches
**30 s.** "SentinelML monitors deployed ML models. Instead of alerting on drift, which I showed is almost uninformative about imminent failure on my benchmark
(ROC-AUC 0.50), it predicts whether a healthy model will start degrading in the next few windows, diagnoses likely causes, and simulates recovery options offline
before a human approves one."
**2 min.** Add: benchmark of 7 injected failure types over 5 datasets; only 58% of events actually hurt the model; chronological evaluation with purge gaps;
PR-AUC 0.42 vs 0.11 for PSI>0.2 and 0.19 for a lagged-loss rule; false alarms 11 vs 84 per 100 healthy windows (pooled; per dataset SentinelML ranged 10-53, and per-dataset CIs overlap); recovery policy matched always-retrain on average gain
but cut harmful actions from 5.5% to 0.8%; FastAPI service with JWT, versioning, rollback and a human-approval state machine; 31 tests.
**5 min.** Add architecture (simulate -> signals -> causal features -> HGB -> calibrated alert; RCA layers; recovery guard + validation holdout), the negative results
(SHAP no better than PSI; no severity effect; ablations within CI), and the limitations (simulated failures, same author for simulator and detector, no real deployment).

## Questions you should be able to answer
**ML fundamentals / advanced**
- *Why chronological splits with purge gaps?* Labels look H windows ahead; without a gap a training label overlaps validation time -> leakage (tested in `test_labels_do_not_cross_split_boundaries`).
- *Data drift vs concept drift vs degradation?* P(x) vs P(y|x) vs measured loss; concept drift is invisible to unlabeled signals (it's 100% harmful in the benchmark and only detectable through delayed labels).
- *Why is PSI's ROC-AUC 0.5?* Many drifts are harmless and small windows make PSI noisy (null max-PSI ~0.2 for 100 rows); drift magnitude != harm.
- *Why PR-AUC rather than ROC-AUC?* Positives are ~11% of healthy windows.
- *Calibration?* ECE 0.03 in-distribution, up to 0.14 leave-one-dataset-out; isotonic fitted on validation; calibration breaks under dataset change.
- *Is SHAP causal?* No - associational; in E5 it was not better than KS/PSI. Restoration is an intervention on the model input only.
- *Why not survival analysis?* Not implemented; binary onset-within-H is simpler to evaluate, but a discrete-time hazard model would give time-to-failure (future work).
**MLOps / system design**
- Versioning & rollback (`model_versions.stage`, approval state machine, `/recovery/rollback`); why auto-deploy is unsafe (recovery estimates are noisy; guard + holdout + human).
- Scaling: prediction log scanned in Python today; move windows to a streaming/aggregation job and TimescaleDB/columnar store; store sketches (like whylogs profiles) instead of rows (privacy too).
- Label delay: the whole design separates label-free signals from lagged-loss signals; ablation shows lagged loss helps only modestly.
**Defence of weaknesses (expect these)**
- "Is the pooled PR-AUC inflated?" Partly: prevalence differs by dataset (16%-48% degraded windows). Say so, and point to per-dataset tables and leave-one-dataset-out ROC-AUC (0.68-0.80 vs PSI 0.56-0.63).
- "Your simulator favours your method." True; mitigation = leave-one-dataset-out and ablations; real next step = real temporal shift data.
- "Baseline is weak." Agreed: single-signal PSI; a stronger baseline (tuned multi-signal, NannyML-style estimation) is future work.
- "Is it novel?" Don't claim so; claim a reproducible empirical study and a safe recovery workflow.
- "Why did sudden failures get warned?" Smoothing lag in the degradation label, not foresight.
