# SentinelML experimental results (auto-generated)

Streams: 40 (breast_cancer, synth_binary, synth_multiclass, synth_regression, wine); horizon H=6 windows; 127 predictive features. All numbers are computed from the cached benchmark by `experiments/analyze.py`.

## Benchmark characteristics

```
{
 "events_total": 400,
 "events_causing_degradation": 231,
 "degradation_rate_by_type": {
  "concept_drift": 1.0,
  "covariate_shift": 0.698,
  "feature_corruption": 0.688,
  "label_shift": 0.416,
  "missing_values": 0.439,
  "outliers": 0.275,
  "pipeline_mismatch": 0.691
 },
 "degradation_rate_by_dataset": {
  "breast_cancer": 0.312,
  "synth_binary": 0.462,
  "synth_multiclass": 0.7,
  "synth_regression": 0.912,
  "wine": 0.5
 },
 "episodes_by_dataset": {
  "breast_cancer": 30,
  "synth_binary": 45,
  "synth_multiclass": 77,
  "synth_regression": 112,
  "wine": 52
 },
 "degraded_fraction_by_dataset": {
  "breast_cancer": 0.161,
  "synth_binary": 0.174,
  "synth_multiclass": 0.279,
  "synth_regression": 0.476,
  "wine": 0.231
 }
}
```

## E1 - conventional threshold monitoring vs predictive monitoring

### ALL (pooled)

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB) | 0.420 [0.32, 0.51] | 0.834 | 0.390 | 0.558 | 0.460 | 10.9 | 0.80 | 5.09 | 4.33 | 0.030 |
| conventional_psi>0.2 | 0.112 [0.08, 0.17] | 0.497 | 0.109 | 0.825 | 0.193 | 84.0 | 1.00 | 5.70 | n/a | n/a |
| psi_max_tuned | 0.112 [0.08, 0.17] | 0.497 | 0.111 | 0.393 | 0.173 | 39.4 | 0.55 | 5.18 | 3.86 | n/a |
| lagged_loss_tuned | 0.192 [0.14, 0.24] | 0.656 | 0.213 | 0.263 | 0.235 | 12.2 | 0.27 | 6.00 | 4.52 | n/a |

### breast_cancer

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB) | 0.118 [0.07, 0.22] | 0.783 | 0.116 | 0.271 | 0.162 | 9.6 | 0.55 | 4.67 | 0.00 | 0.017 |
| conventional_psi>0.2 | 0.061 [0.03, 0.12] | 0.592 | 0.044 | 1.000 | 0.085 | 100.0 | 1.00 | 6.00 | n/a | n/a |
| psi_max_tuned | 0.061 [0.03, 0.12] | 0.592 | 0.084 | 0.492 | 0.144 | 24.9 | 0.64 | 5.29 | 0.00 | n/a |
| lagged_loss_tuned | 0.101 [0.04, 0.18] | 0.597 | 0.049 | 0.763 | 0.092 | 68.8 | 1.00 | 5.36 | n/a | n/a |

### synth_binary

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB) | 0.231 [0.12, 0.55] | 0.741 | 0.141 | 0.791 | 0.239 | 48.1 | 1.00 | 5.70 | n/a | 0.028 |
| conventional_psi>0.2 | 0.143 [0.07, 0.28] | 0.627 | 0.100 | 0.864 | 0.179 | 77.7 | 1.00 | 5.75 | n/a | n/a |
| psi_max_tuned | 0.143 [0.07, 0.28] | 0.627 | 0.132 | 0.445 | 0.204 | 29.2 | 0.75 | 4.67 | 0.50 | n/a |
| lagged_loss_tuned | 0.107 [0.07, 0.18] | 0.568 | 0.056 | 0.009 | 0.016 | 1.5 | 0.10 | 3.50 | 2.64 | n/a |

### synth_multiclass

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB) | 0.345 [0.26, 0.52] | 0.737 | 0.229 | 0.793 | 0.356 | 53.2 | 0.93 | 5.64 | 16.00 | 0.063 |
| conventional_psi>0.2 | 0.216 [0.13, 0.35] | 0.557 | 0.174 | 0.738 | 0.282 | 69.8 | 1.00 | 5.57 | n/a | n/a |
| psi_max_tuned | 0.216 [0.13, 0.35] | 0.557 | 0.237 | 0.134 | 0.171 | 8.6 | 0.23 | 5.00 | 3.25 | n/a |
| lagged_loss_tuned | 0.190 [0.16, 0.23] | 0.573 | 0.205 | 0.463 | 0.285 | 35.8 | 0.97 | 4.72 | 0.00 | n/a |

### synth_regression

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB) | 0.531 [0.37, 0.69] | 0.826 | 0.431 | 0.767 | 0.552 | 27.5 | 1.00 | 5.19 | n/a | 0.079 |
| conventional_psi>0.2 | 0.261 [0.14, 0.60] | 0.600 | 0.267 | 0.704 | 0.387 | 52.6 | 1.00 | 5.48 | n/a | n/a |
| psi_max_tuned | 0.261 [0.14, 0.60] | 0.600 | 0.367 | 0.340 | 0.353 | 15.9 | 0.58 | 4.44 | 3.56 | n/a |
| lagged_loss_tuned | 0.272 [0.19, 0.36] | 0.508 | 0.214 | 0.535 | 0.306 | 53.3 | 1.00 | 5.06 | n/a | n/a |

### wine

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB) | 0.293 [0.16, 0.52] | 0.772 | 0.207 | 0.694 | 0.319 | 27.7 | 0.81 | 5.65 | 8.00 | 0.065 |
| conventional_psi>0.2 | 0.168 [0.06, 0.34] | 0.577 | 0.094 | 1.000 | 0.173 | 100.0 | 1.00 | 6.00 | n/a | n/a |
| psi_max_tuned | 0.168 [0.06, 0.34] | 0.577 | 0.103 | 0.611 | 0.177 | 55.4 | 1.00 | 4.95 | n/a | n/a |
| lagged_loss_tuned | 0.190 [0.08, 0.30] | 0.643 | 0.114 | 0.759 | 0.198 | 61.6 | 0.95 | 5.90 | 2.00 | n/a |

## E2 - algorithm comparison (pooled)

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| logreg | 0.322 [0.23, 0.39] | 0.777 | 0.302 | 0.463 | 0.365 | 13.4 | 0.66 | 4.89 | 5.17 | 0.016 |
| rf | 0.401 [0.29, 0.49] | 0.816 | 0.331 | 0.600 | 0.427 | 15.1 | 0.76 | 5.35 | 6.86 | 0.035 |
| hgb | 0.420 [0.32, 0.51] | 0.834 | 0.390 | 0.558 | 0.460 | 10.9 | 0.80 | 5.09 | 4.33 | 0.030 |
| xgb | 0.425 [0.33, 0.52] | 0.833 | 0.360 | 0.635 | 0.459 | 14.1 | 0.83 | 5.24 | 3.13 | 0.035 |
| lgbm | 0.421 [0.32, 0.52] | 0.838 | 0.371 | 0.555 | 0.445 | 11.7 | 0.82 | 5.00 | 3.36 | 0.030 |

## E3 - leave-one-dataset-out generalisation

### held-out: breast_cancer

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB, unseen dataset) | 0.257 [0.07, 0.41] | 0.800 | 0.115 | 0.627 | 0.195 | 22.4 | 0.91 | 5.50 | n/a | 0.126 |
| conventional_psi>0.2 | 0.061 [0.03, 0.12] | 0.592 | 0.044 | 1.000 | 0.085 | 100.0 | 1.00 | 6.00 | n/a | n/a |

### held-out: synth_binary

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB, unseen dataset) | 0.222 [0.12, 0.37] | 0.703 | 0.115 | 0.909 | 0.204 | 69.7 | 1.00 | 5.65 | n/a | 0.083 |
| conventional_psi>0.2 | 0.143 [0.07, 0.28] | 0.627 | 0.100 | 0.864 | 0.179 | 77.7 | 1.00 | 5.75 | n/a | n/a |

### held-out: synth_multiclass

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB, unseen dataset) | 0.354 [0.26, 0.52] | 0.747 | 0.458 | 0.165 | 0.242 | 3.9 | 0.37 | 4.64 | 5.70 | 0.077 |
| conventional_psi>0.2 | 0.216 [0.13, 0.35] | 0.557 | 0.174 | 0.738 | 0.282 | 69.8 | 1.00 | 5.57 | n/a | n/a |

### held-out: synth_regression

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB, unseen dataset) | 0.380 [0.21, 0.62] | 0.726 | 0.330 | 0.566 | 0.417 | 31.3 | 0.87 | 5.37 | 0.67 | 0.140 |
| conventional_psi>0.2 | 0.261 [0.14, 0.60] | 0.600 | 0.267 | 0.704 | 0.387 | 52.6 | 1.00 | 5.48 | n/a | n/a |

### held-out: wine

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML (HGB, unseen dataset) | 0.287 [0.12, 0.49] | 0.680 | 0.400 | 0.056 | 0.098 | 0.9 | 0.19 | 2.00 | 2.67 | 0.054 |
| conventional_psi>0.2 | 0.168 [0.06, 0.34] | 0.577 | 0.094 | 1.000 | 0.173 | 100.0 | 1.00 | 6.00 | n/a | n/a |

## E4 - horizon and failure shape

### H=3

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML | 0.317 [0.22, 0.42] | 0.835 | 0.393 | 0.235 | 0.294 | 2.3 | 0.41 | 2.20 | 5.10 | 0.021 |
| conventional_psi>0.2 | 0.075 [0.05, 0.12] | 0.552 | 0.059 | 0.838 | 0.111 | 83.5 | 0.98 | 2.86 | 0.00 | n/a |

### H=6

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML | 0.420 [0.32, 0.51] | 0.834 | 0.390 | 0.558 | 0.460 | 10.9 | 0.80 | 5.09 | 4.33 | 0.030 |
| conventional_psi>0.2 | 0.112 [0.08, 0.17] | 0.497 | 0.109 | 0.825 | 0.193 | 84.0 | 1.00 | 5.70 | n/a | n/a |

### H=12

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| SentinelML | 0.528 [0.45, 0.61] | 0.794 | 0.381 | 0.784 | 0.513 | 35.0 | 0.96 | 10.92 | 2.00 | 0.033 |
| conventional_psi>0.2 | 0.180 [0.14, 0.25] | 0.415 | 0.201 | 0.786 | 0.320 | 86.0 | 1.00 | 11.64 | n/a | n/a |

```
{
 "sudden": {
  "episodes": 40,
  "warned_rate_sentinel": 0.7,
  "warned_rate_psi0.2": 1.0
 },
 "gradual": {
  "episodes": 73,
  "warned_rate_sentinel": 0.8493150684931506,
  "warned_rate_psi0.2": 1.0
 }
}
```

## E5 - root-cause analysis
```
{
 "n_diagnosed_incidents": 255,
 "n_feature_incidents": 162,
 "feature_ranking": {
  "ks": {
   "recall@1": 0.388,
   "any_hit@1": 0.84,
   "recall@3": 0.766,
   "any_hit@3": 0.926,
   "recall@5": 0.866,
   "any_hit@5": 0.969
  },
  "psi": {
   "recall@1": 0.407,
   "any_hit@1": 0.907,
   "recall@3": 0.803,
   "any_hit@3": 0.969,
   "recall@5": 0.911,
   "any_hit@5": 0.994
  },
  "imp_weighted": {
   "recall@1": 0.357,
   "any_hit@1": 0.809,
   "recall@3": 0.739,
   "any_hit@3": 0.975,
   "recall@5": 0.881,
   "any_hit@5": 1.0
  },
  "shap_shift": {
   "recall@1": 0.342,
   "any_hit@1": 0.772,
   "recall@3": 0.623,
   "any_hit@3": 0.92,
   "recall@5": 0.772,
   "any_hit@5": 0.969
  },
  "restore_gain": {
   "recall@1": 0.363,
   "any_hit@1": 0.802,
   "recall@3": 0.616,
   "any_hit@3": 0.895,
   "recall@5": 0.712,
   "any_hit@5": 0.938
  },
  "combined": {
   "recall@1": 0.377,
   "any_hit@1": 0.821,
   "recall@3": 0.789,
   "any_hit@3": 0.975,
   "recall@5": 0.926,
   "any_hit@5": 1.0
  }
 },
 "random_reference_any_hit@3": 0.463,
 "failure_type": {
  "accuracy": 0.969,
  "macro_f1": 0.964,
  "majority_baseline_accuracy": 0.184,
  "support": {
   "concept_drift": 47,
   "label_shift": 46,
   "covariate_shift": 45,
   "pipeline_mismatch": 45,
   "feature_corruption": 35,
   "outliers": 19,
   "missing_values": 18
  },
  "per_type_recall": {
   "concept_drift": 1.0,
   "covariate_shift": 0.978,
   "feature_corruption": 0.914,
   "label_shift": 0.978,
   "missing_values": 1.0,
   "outliers": 0.842,
   "pipeline_mismatch": 1.0
  }
 }
}
```

## E6 - counterfactual recovery
```
{
 "n_incidents": 255,
 "sentinel": {
  "mean_rel_loss_reduction": 0.336,
  "success_rate(>=10% better)": 0.698,
  "harm_rate(>5% worse)": 0.008
 },
 "always_retrain": {
  "mean_rel_loss_reduction": 0.333,
  "success_rate(>=10% better)": 0.667,
  "harm_rate(>5% worse)": 0.055
 },
 "always_rollback": {
  "mean_rel_loss_reduction": -0.124,
  "success_rate(>=10% better)": 0.063,
  "harm_rate(>5% worse)": 0.522
 },
 "always_recalibrate": {
  "mean_rel_loss_reduction": 0.067,
  "success_rate(>=10% better)": 0.29,
  "harm_rate(>5% worse)": 0.078
 },
 "oracle": {
  "mean_rel_loss_reduction": 0.347,
  "success_rate(>=10% better)": 0.722,
  "harm_rate(>5% worse)": 0.0
 },
 "sentinel_recommendation_counts": {
  "retrain_recent": 164,
  "no_action": 46,
  "recalibrate": 29,
  "retrain_without_suspects": 11,
  "rollback_previous_version": 2,
  "align_suspect_features": 2,
  "mask_suspect_features": 1
 },
 "sentinel_by_failure_type": {
  "concept_drift": {
   "n": 47,
   "mean_rel_loss_reduction": 0.018
  },
  "covariate_shift": {
   "n": 45,
   "mean_rel_loss_reduction": 0.622
  },
  "feature_corruption": {
   "n": 35,
   "mean_rel_loss_reduction": 0.457
  },
  "label_shift": {
   "n": 46,
   "mean_rel_loss_reduction": 0.211
  },
  "missing_values": {
   "n": 18,
   "mean_rel_loss_reduction": 0.337
  },
  "outliers": {
   "n": 19,
   "mean_rel_loss_reduction": 0.138
  },
  "pipeline_mismatch": {
   "n": 45,
   "mean_rel_loss_reduction": 0.496
  }
 }
}
```

## E7 - ablation

| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | episodes warned | mean lead (windows) | delay if unwarned | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 0.420 [0.32, 0.51] | 0.834 | 0.390 | 0.558 | 0.460 | 10.9 | 0.80 | 5.09 | 4.33 | 0.030 |
| no_drift | 0.385 [0.27, 0.48] | 0.815 | 0.312 | 0.630 | 0.417 | 17.3 | 0.86 | 5.30 | 1.27 | 0.019 |
| no_quality | 0.414 [0.31, 0.51] | 0.822 | 0.378 | 0.510 | 0.434 | 10.5 | 0.73 | 5.01 | 2.95 | 0.024 |
| no_uncertainty | 0.397 [0.31, 0.48] | 0.815 | 0.417 | 0.378 | 0.397 | 6.6 | 0.59 | 4.79 | 7.48 | 0.015 |
| no_performance(lagged loss) | 0.389 [0.29, 0.49] | 0.814 | 0.418 | 0.388 | 0.402 | 6.8 | 0.64 | 4.49 | 2.83 | 0.025 |
| no_temporal_features (instantaneous only) | 0.364 [0.24, 0.45] | 0.747 | 0.252 | 0.627 | 0.359 | 23.2 | 0.84 | 5.24 | 1.82 | 0.024 |
| unlabelled_only (drift+quality+uncertainty) | 0.389 [0.29, 0.49] | 0.814 | 0.418 | 0.388 | 0.402 | 6.8 | 0.64 | 4.49 | 2.83 | 0.025 |

## E8 - severity
```
{
 "low(<0.55)": {
  "positive_windows": 137,
  "recall_sentinel": 0.5328467153284672,
  "recall_psi0.2": 0.8540145985401459
 },
 "mid(0.55-0.75)": {
  "positive_windows": 193,
  "recall_sentinel": 0.5647668393782384,
  "recall_psi0.2": 0.7823834196891192
 },
 "high(>0.75)": {
  "positive_windows": 270,
  "recall_sentinel": 0.5666666666666667,
  "recall_psi0.2": 0.8407407407407408
 }
}
```
