# SentinelML: predicting, diagnosing and recovering from ML model degradation - technical report

## 1. Abstract
Monitoring tools usually alert when input data drifts, but drift often does not hurt a model. SentinelML asks whether temporal failure prediction, root-cause diagnosis and counterfactual recovery evaluation improve reliability compared with threshold monitoring. On a reproducible simulated benchmark (5 datasets, 40 streams, 400 injected failures) only 58% of injected events degraded the model, and max-PSI>0.2 had ROC-AUC 0.50 for predicting imminent degradation. A gradient-boosting predictor over rolling drift/quality/uncertainty/delayed-loss features reached pooled PR-AUC 0.42 [0.32, 0.51] vs 0.11 [0.08, 0.17], with calibrated outputs (ECE 0.03). Per-dataset gains were directionally consistent but not individually significant. Plain PSI/KS ranked corrupted features as well as SHAP or counterfactual restoration; a recovery policy matched always-retrain on mean gain while reducing harmful actions. All results are from simulation.

## 2. Problem and gap
Deployed models degrade through covariate shift, concept drift, pipeline bugs and data-quality failures. Existing tools (verified: Evidently, NannyML; WhyLabs hosted service discontinued) detect drift, track performance and, in NannyML's case, estimate performance without labels. The gap examined here is *forecasting onset* of degradation and *ranking recovery actions offline with guards and human approval*. No novelty claim is made; the literature check was not systematic (see `LITERATURE_AND_GAPS.md`).

## 3. Method
Failure injector -> windowed signals (PSI, KS, Wasserstein, JS, missing/outlier/duplicate rates, correlation shift, uncertainty, prediction drift, delayed loss) -> 127 causal rolling features -> onset-within-H classifier (H=6 windows) for windows where the model is healthy. Degradation = causal 3-window mean loss above a clean-simulation threshold with hysteresis; labels delayed 2 windows. Chronological split (train <50%, validation 50-65%, test >65%) with purge gaps; thresholds tuned on validation; isotonic calibration; bootstrap CIs over whole streams. Root cause: KS/PSI/missing-delta evidence, SHAP-shift, counterfactual quantile restoration, learned failure-type classifier. Recovery: 7 candidates simulated offline on held-out labelled windows with a regression guard, cost term and a validation gate; deployment only through a human-approval state machine.

## 4. Data
Breast cancer (binary, 569 rows), wine (3-class, 178), synthetic binary/multiclass/regression (6,000 rows each). Datasets are bundled with scikit-learn or generated; diabetes was excluded (baseline model barely beat the mean, giving 2-3 degradation episodes per stream).

## 5. Results (full tables: `results/RESULTS.md`)
| Experiment | Main finding |
|---|---|
| E1 predictive vs threshold | PR-AUC 0.42 vs 0.11 (PSI>0.2), 0.19 (lagged loss); 10.9 vs 84.0 false alarms/100 healthy windows; 80% of episodes warned, mean lead 5.1/6 windows |
| E2 algorithms | LR 0.32, RF 0.40, HGB 0.42, XGB 0.43, LightGBM 0.42 (overlapping CIs) |
| E3 leave-one-dataset-out | PR-AUC above PSI on all 5; ROC-AUC 0.68-0.80 vs 0.56-0.63; ECE up to 0.14; thresholds transfer poorly (wine recall 0.06) |
| E4 horizon/shape | H=3: 0.32; H=6: 0.42; H=12: 0.53; warned: gradual 85%, sudden 70% (partly a smoothing-lag artifact) |
| E5 root cause | any-hit@3: combined 0.975, PSI 0.969, KS 0.926, SHAP-shift 0.920, restoration 0.895, random 0.463; type classifier accuracy 0.969 (majority 0.184) |
| E6 recovery | mean loss reduction 33.6% (recommendation) vs 33.3% (always retrain); harm 0.8% vs 5.5%; oracle 34.7%; concept drift unrecoverable (+1.8%) |
| E7 ablation | removing any group stays within CI; temporal features matter most (PR-AUC 0.36, ROC 0.75 without them) |
| E8 severity | recall flat (0.53/0.56/0.57): no severity effect shown |
Live-service check (Windows): clean 100-row windows already raised PSI alerts (0.14-0.25) while corrupted windows reached PSI 6-7 and the diagnosis ranked exactly the three corrupted features first.

## 6. Discussion and limitations
Drift is evidence, not proof, of failure. The predictor's advantage is largest pooled; per dataset the intervals overlap, pooled PR-AUC is helped by prevalence differences (16-48% degraded windows), and per-dataset false-alarm rates reach 53/100. Failures are simulated by the same author as the predictor; the baseline is a single-signal rule; RCA/recovery samples are not independent. See `LIMITATIONS_AND_STATUS.md`.

## 7. Future work
Real temporal-shift validation (e.g. WILDS-style), stronger baselines (multivariate tests, NannyML-style estimation), survival/hazard models for time-to-failure, smoothing of the API degradation alert, S3 artifact storage, React UI, Docker/AWS deployment, drift-aware reference refresh after recovery.

## 8. References (from memory; verify details)
Gama et al. 2014 (ACM CSUR); Lu et al. 2019 (IEEE TKDE); Rabanser et al. 2019 (NeurIPS); Sculley et al. 2015 (NeurIPS); Breck et al. 2017 (IEEE Big Data); Garg et al. 2022 (ICLR); Lipton et al. 2018 (ICML); Saerens et al. 2002 (Neural Computation); Ovadia et al. 2019 (NeurIPS); Lundberg & Lee 2017 (NeurIPS); Vela et al. 2022 (Scientific Reports); Koh et al. 2021 (ICML). Competitor facts: NannyML docs, Evidently GitHub, GeekWire/ZenML (WhyLabs).
