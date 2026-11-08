# Literature, research gaps and recommended contribution

> **Provenance.** The entries below are cited **from memory** (no systematic database search was run, no DOIs retrieved).
> Verify titles, years and DOIs before citing. Competitor facts are separated in `COMPETITORS.md` where web verification was done.

| # | Paper (authors, year, venue) | Addresses | Relevance / limitation for SentinelML |
|---|---|---|---|
| 1 | Gama, Zliobaite, Bifet, Pechenizkiy, Bouchachia, *A survey on concept drift adaptation*, ACM Computing Surveys 2014 | taxonomy of drift & adaptation | defines drift types used in `failures.py`; mostly detection/adaptation, not forecasting of degradation |
| 2 | Lu, Liu, Dong, Gu, Gama, Zhang, *Learning under concept drift: a review*, IEEE TKDE 2019 | drift detection/understanding/adaptation | same; "understanding" (where/when/how) motivates RCA |
| 3 | Rabanser, Guennemann, Lipton, *Failing loudly*, NeurIPS 2019 | empirical comparison of shift detectors (univariate tests, dimensionality reduction) | supports our univariate-test signals; shows detection != harm |
| 4 | Sculley et al., *Hidden technical debt in ML systems*, NeurIPS 2015 | ML systems risk | motivation |
| 5 | Breck et al., *The ML Test Score*, IEEE Big Data 2017 | production-readiness rubric incl. monitoring | motivation for tests/monitoring checklist |
| 6 | Garg, Balakrishnan, Lipton, Neyshabur, Sedghi, *Leveraging unlabeled data to predict OOD performance*, ICLR 2022 | estimating *current* accuracy under shift from confidence | closest to label-free performance estimation; estimates present, not future, performance |
| 7 | Lipton, Wang, Smola, *Detecting and correcting for label shift with black box predictors*, ICML 2018 | label-shift detection/correction | basis for our prior-shift recalibration candidate |
| 8 | Saerens, Latinne, Decaestecker, *Adjusting the outputs of a classifier to new a priori probabilities*, Neural Computation 2002 | prior-shift correction (EM) | same |
| 9 | Ovadia et al., *Can you trust your model's uncertainty? ... under dataset shift*, NeurIPS 2019 | uncertainty degrades under shift | justifies uncertainty features |
| 10 | Lundberg & Lee, *A unified approach to interpreting model predictions*, NeurIPS 2017 | SHAP | attribution is associational, not causal (stated in `rca.py`) |
| 11 | Gretton et al., *A kernel two-sample test*, JMLR 2012 | MMD test | multivariate alternative to our univariate tests (not implemented) |
| 12 | Vela et al., *Temporal quality degradation in AI models*, Scientific Reports 2022 | empirical study of performance decay over time | evidence that degradation is common and model-dependent |
| 13 | Koh et al., *WILDS*, ICML 2021 | real distribution-shift benchmarks | future work: validate on real shifts instead of injected ones |
| 14 | Peters, Janzing, Schoelkopf, *Elements of causal inference*, MIT Press 2017 | causal inference foundations | why "restoration" is an intervention on the model input, not causal identification |

## Three candidate research gaps
| Gap | Claim | Achievability | Risk |
|---|---|---|---|
| **G1 Onset forecasting** | existing monitors *detect* drift or *estimate current* performance; forecasting *when a healthy model will start degrading*, from monitoring-signal history, is less explored in the tools/papers I checked | high - benchmark + temporal features | results depend on simulator realism |
| G2 Unified diagnosis | link failure *type* classification with feature-level cause ranking and counterfactual restoration | medium-high | injected failures are easy to separate |
| G3 Recovery as offline counterfactual evaluation | rank recovery actions by simulated outcome with guards and approval, rather than "retrain on drift" | medium | small recent-label budget makes estimates noisy |

**Recommended contribution: G1, supported by G3** as an empirical comparison of *recovery policies*. G2 is implemented but its results
(E5) are the least surprising. **I do not claim novelty**: my literature check was not systematic, and NannyML in particular already
offers label-free performance estimation and drift-to-feature linking (see `COMPETITORS.md`). The defensible claim is the
*empirical evidence on a reproducible benchmark*, nothing more.
