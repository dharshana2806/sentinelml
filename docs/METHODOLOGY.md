# Methodology

**Stream simulation.** A model is trained on 40% of a dataset; 20% calibrates monitoring references; 40% is the production pool.
A stream = 600 windows x 100 rows (windows are drawn without replacement from the pool, or with replacement when the pool has <100 rows, as for wine's 72-row pool). 10 failure events per stream
are placed in non-overlapping slots (>=12 windows recovery), type/severity/shape/features randomised and logged as ground truth.
**Degraded state.** Causal 3-window mean of task loss (error rate or RMSE/std(y_train)) exceeds `mean + max(4 sd, 20% rel, 0.04 abs)` of a clean
simulation; hysteresis (exit at the midpoint to baseline) prevents flicker. Labels reach the monitor with delay 2 windows.
**Prediction task.** For windows where the model is currently healthy, y=1 if degradation starts within the next H windows.
Features: 21 base signals (drift, quality, uncertainty/prediction drift, delayed loss) x 5-6 causal rolling transforms = 127 features.
**Splits.** Per stream: train <50%, validation 50-65%, test >65% of time, with purge gaps of H windows; threshold = F1-optimal on validation; isotonic
calibration on validation. CIs: percentile bootstrap resampling whole streams (100 resamples).
**Alert metrics.** Window-level precision/recall/F1/false-alarm rate on healthy test windows; episode-level "warned" = alert within H windows before onset,
lead = windows between first such alert and onset, delay = first alert after onset for unwarned episodes.
**Root cause.** Evidence per feature: KS, PSI, missing delta; SHAP shift = |mean|SHAP| (window) - mean|SHAP| (reference)|; restoration = loss drop when the
feature is quantile-mapped back to reference. Diagnosis uses the 3 latest *labelled* windows (label delay) 8 windows after onset.
**Recovery.** Candidates fitted on 4 labelled windows after onset, scored on the next 3, evaluated on up to 8 later windows (not used in selection).
Feasible = loss on clean reference data not worse than baseline + max(0.05, 25%). Recommended = best utility (gain - 0.002 x seconds) with gain >= max(0.01, 5%).
**Statistical caveats of drift metrics:** see `sentinelml/drift.py` docstring. Notably, null PSI of the worst of k features on small windows already exceeds the 0.2 rule of thumb (observed during API testing).
