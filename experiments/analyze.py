"""Run research experiments 1-8 on the cached benchmark. Produces results/results.json, RESULTS.md, figures.

Usage: python experiments/analyze.py [--cache results/cache] [--out results]
Nothing here is hard-coded: every number in RESULTS.md is computed from the cached benchmark.
"""
import argparse, glob, json, os, pickle, sys, warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score
from sklearn.model_selection import GroupKFold

from sentinelml.config import SimConfig, FAILURE_TYPES
from sentinelml.failures import FailureEvent
from sentinelml.predictor import (FailurePredictor, MODEL_NAMES, best_f1_threshold, bootstrap_ci, build_table,
                                  episode_metrics, episodes, feature_columns, make_model, window_metrics)
from sentinelml.rca import FailureTypeClassifier, RANKERS

CFG = SimConfig()
N_BOOT = 100


def load(cache):
    out = []
    for p in sorted(glob.glob(f"{cache}/*.pkl")):
        r = pickle.load(open(p, "rb"))
        r["events_obj"] = [FailureEvent(**e) for e in r["events"]]
        out.append(r)
    return out


def as_stream(r):
    return SimpleNamespace(name=r["name"], seed=r["seed"], signals=r["signals"], degraded=r["degraded"],
                           events=r["events_obj"])


def cfg_h(H):
    c = SimConfig(); c.horizon = H
    return c


def pr_auc_fn(sub):
    d = sub[sub.healthy & sub.y.notna()]
    return average_precision_score(d.y, d.score) if 0 < d.y.sum() < len(d) else float("nan")


def eval_method(te, score_col, thr, H, calibrated_col=None, ci=True):
    """te: test rows (all windows). Returns window + episode metrics (+ bootstrap CI of PR-AUC)."""
    te = te.copy()
    te["score"] = te[score_col]
    te["alert"] = te.score >= thr
    d = te[te.healthy & te.y.notna()]
    wm = window_metrics(d.y.to_numpy(), d.score.to_numpy(), d.alert.to_numpy(),
                        d[calibrated_col].to_numpy() if calibrated_col else None)
    em = episode_metrics(te, "alert", H)
    out = {**wm, **em, "threshold": float(thr)}
    if ci:
        lo, hi = bootstrap_ci(te, pr_auc_fn, N_BOOT)
        out["pr_auc_ci"] = [lo, hi]
    return out


def fit_predict(tbl, model="hgb", cols=None, train_mask=None):
    cols = cols or feature_columns(tbl)
    tr = tbl[(tbl.seg == "train")] if train_mask is None else tbl[train_mask & (tbl.seg == "train")]
    va = tbl[(tbl.seg == "val")] if train_mask is None else tbl[train_mask & (tbl.seg == "val")]
    fp = FailurePredictor(model, cols, make_model(model)).fit(tr, va)
    return fp


def score_table(fp, tbl):
    t = tbl.copy()
    t["fp_score"] = fp.score(t)
    t["fp_cal"] = fp.calibrated(t["fp_score"].to_numpy())
    return t


def baselines(tbl, H):
    """Conventional monitors evaluated on the test segment of `tbl`."""
    va = tbl[(tbl.seg == "val") & tbl.healthy & tbl.y.notna()]
    te = tbl[tbl.seg == "test"]
    res = {}
    res["conventional_psi>0.2"] = eval_method(te, "psi_max", 0.2, H)
    thr = best_f1_threshold(va.y.to_numpy(int), va.psi_max.to_numpy())
    res["psi_max_tuned"] = eval_method(te, "psi_max", thr, H)
    thr = best_f1_threshold(va.y.to_numpy(int), va.loss_lag.fillna(0).to_numpy())
    t2 = te.assign(loss_lag=te.loss_lag.fillna(0))
    res["lagged_loss_tuned"] = eval_method(t2, "loss_lag", thr, H)
    return res


def run(cache, out):
    os.makedirs(f"{out}/figures", exist_ok=True)
    R = load(cache)
    streams = [as_stream(r) for r in R]
    names = sorted({r["name"] for r in R})
    H = CFG.horizon
    results = {"n_streams": len(R), "datasets": names, "horizon": H, "config": CFG.to_dict()}

    # ---- benchmark description -------------------------------------------------------------
    ev_rows = []
    for r in R:
        deg = r["degraded"]
        for e in r["events_obj"]:
            ev_rows.append(dict(dataset=r["name"], ftype=e.ftype, shape=e.shape,
                                degrades=bool(deg[e.start: e.end + 5].any())))
    ev = pd.DataFrame(ev_rows)
    results["benchmark"] = {
        "events_total": len(ev), "events_causing_degradation": int(ev.degrades.sum()),
        "degradation_rate_by_type": ev.groupby("ftype").degrades.mean().round(3).to_dict(),
        "degradation_rate_by_dataset": ev.groupby("dataset").degrades.mean().round(3).to_dict(),
        "episodes_by_dataset": {n: int(sum(len(episodes(r["degraded"])) for r in R if r["name"] == n)) for n in names},
        "degraded_fraction_by_dataset": {n: round(float(np.mean([r["degraded"].mean() for r in R if r["name"] == n])), 3) for n in names},
    }

    # ---- E1 + E2: pooled tables --------------------------------------------------------------
    tbl = build_table(streams, cfg_h(H))
    cols = feature_columns(tbl)
    results["n_features"] = len(cols)
    e1 = {}
    fp = fit_predict(tbl, "hgb", cols)
    tbl_s = score_table(fp, tbl)
    te = tbl_s[tbl_s.seg == "test"]
    e1["ALL (pooled)"] = {"SentinelML (HGB)": eval_method(te, "fp_score", fp.threshold, H, "fp_cal"), **baselines(tbl, H)}
    for n in names:                                     # per-dataset models
        sub = tbl[tbl.dataset == n]
        f = fit_predict(sub, "hgb", cols); ts = score_table(f, sub); tt = ts[ts.seg == "test"]
        e1[n] = {"SentinelML (HGB)": eval_method(tt, "fp_score", f.threshold, H, "fp_cal"), **baselines(sub, H)}
    results["E1_threshold_vs_predictive"] = e1

    e2 = {}
    for m in MODEL_NAMES:
        f = fit_predict(tbl, m, cols); ts = score_table(f, tbl); tt = ts[ts.seg == "test"]
        e2[m] = eval_method(tt, "fp_score", f.threshold, H, "fp_cal")
    results["E2_algorithms"] = e2

    # ---- E3: leave-one-dataset-out -----------------------------------------------------------
    e3 = {}
    for n in names:
        mask = tbl.dataset != n
        f = fit_predict(tbl, "hgb", cols, train_mask=mask)
        held = tbl[tbl.dataset == n]; ts = score_table(f, held); tt = ts[ts.seg == "test"]
        e3[n] = {"SentinelML (HGB, unseen dataset)": eval_method(tt, "fp_score", f.threshold, H, "fp_cal"),
                 "conventional_psi>0.2": e1[n]["conventional_psi>0.2"]}
    results["E3_leave_one_dataset_out"] = e3

    # ---- E4: horizon + shape ------------------------------------------------------------------
    e4 = {"by_horizon": {}, "by_shape": {}}
    for h in (3, 6, 12):
        t_h = build_table(streams, cfg_h(h))
        f = fit_predict(t_h, "hgb", cols); ts = score_table(f, t_h); tt = ts[ts.seg == "test"]
        e4["by_horizon"][h] = {"SentinelML": eval_method(tt, "fp_score", f.threshold, h, "fp_cal"),
                               "conventional_psi>0.2": eval_method(tt, "psi_max", 0.2, h)}
    te2 = te.copy(); te2["alert"] = te2.fp_score >= fp.threshold
    te2["alert_psi"] = te2.psi_max >= 0.2
    for shp in ("sudden", "gradual"):
        rows = []
        for _, g in te2.groupby("stream"):
            g = g[g.seg == "test"].sort_values("t")
            deg = g.degraded.to_numpy()
            for o, e in episodes(deg):
                if o < H or g.onset_shape.iloc[o - 1] != shp:
                    continue
                rows.append(dict(sentinel=bool(g.alert.to_numpy()[o - H:o].any()), psi=bool(g.alert_psi.to_numpy()[o - H:o].any())))
        d = pd.DataFrame(rows)
        e4["by_shape"][shp] = {"episodes": len(d), "warned_rate_sentinel": float(d.sentinel.mean()) if len(d) else None,
                               "warned_rate_psi0.2": float(d.psi.mean()) if len(d) else None}
    results["E4_horizon_and_shape"] = e4

    # ---- E7: ablations --------------------------------------------------------------------------
    e7 = {}
    variants = {"full": dict(), "no_drift": dict(drop_groups=("drift",)), "no_quality": dict(drop_groups=("quality",)),
                "no_uncertainty": dict(drop_groups=("uncertainty",)), "no_performance(lagged loss)": dict(drop_groups=("performance",)),
                "no_temporal_features (instantaneous only)": dict(temporal=False),
                "unlabelled_only (drift+quality+uncertainty)": dict(drop_groups=("performance",))}
    for k, kw in variants.items():
        t_a = build_table(streams, CFG, **kw)
        f = fit_predict(t_a, "hgb"); ts = score_table(f, t_a); tt = ts[ts.seg == "test"]
        e7[k] = eval_method(tt, "fp_score", f.threshold, H, "fp_cal")
    results["E7_ablation"] = e7

    # ---- E8: severity -----------------------------------------------------------------------------
    e8 = {}
    d = te2[te2.healthy & (te2.y == 1)]
    bins = pd.cut(d.onset_severity, [0, 0.55, 0.75, 1.01], labels=["low(<0.55)", "mid(0.55-0.75)", "high(>0.75)"])
    for b, g in d.groupby(bins, observed=True):
        e8[str(b)] = {"positive_windows": len(g), "recall_sentinel": float(g.alert.mean()), "recall_psi0.2": float(g.alert_psi.mean())}
    results["E8_severity"] = e8

    # ---- E5: root cause ------------------------------------------------------------------------------
    D = pd.DataFrame([d_ for r in R for d_ in r["diag"]])
    e5 = {"n_diagnosed_incidents": len(D)}
    if len(D):
        feat = D[D.true_features.map(len) > 0]
        e5["n_feature_incidents"] = len(feat)
        e5["feature_ranking"] = {}
        for rk in RANKERS:
            row = {}
            for k in (1, 3, 5):
                rec = [len(set(r.ranks[rk][:k]) & set(r.true_features)) / len(r.true_features) for r in feat.itertuples()]
                anyh = [bool(set(r.ranks[rk][:k]) & set(r.true_features)) for r in feat.itertuples()]
                row[f"recall@{k}"] = round(float(np.mean(rec)), 3); row[f"any_hit@{k}"] = round(float(np.mean(anyh)), 3)
            e5["feature_ranking"][rk] = row
        rng = np.random.default_rng(0)                                   # random-ranking reference
        e5["random_reference_any_hit@3"] = round(float(np.mean([
            bool(set(rng.choice(CFG_FEATS(r.dataset), 3, replace=False)) & set(r.true_features)) for r in feat.itertuples()])), 3)
        # failure-type classification, grouped CV over streams
        sigc = [c for c in D.columns if c.startswith("sig_")]
        X = D[sigc].rename(columns=lambda c: c[4:]); yv = D.ftype.to_numpy(); grp = D.dataset + "#" + D.seed.astype(str)
        pred = np.empty(len(D), dtype=object)
        for tr, te_ in GroupKFold(n_splits=min(5, grp.nunique())).split(X, yv, grp):
            clf = FailureTypeClassifier().fit(X.iloc[tr], yv[tr])
            pred[te_] = clf.predict_proba(X.iloc[te_]).idxmax(axis=1).to_numpy()
        e5["failure_type"] = {"accuracy": round(float((pred == yv).mean()), 3),
                              "macro_f1": round(float(f1_score(yv, pred.astype(str), average="macro")), 3),
                              "majority_baseline_accuracy": round(float(pd.Series(yv).value_counts(normalize=True).iloc[0]), 3),
                              "support": pd.Series(yv).value_counts().to_dict(),
                              "per_type_recall": {t: round(float((pred[yv == t] == t).mean()), 3) for t in sorted(set(yv))}}
    results["E5_root_cause"] = e5

    # ---- E6: recovery ------------------------------------------------------------------------------------
    rows = []
    for r in R:
        for x in r["recovery"]:
            c = x["candidates"]; no = c["no_action"]["future_loss"]
            def pol(name):
                return c[name]["future_loss"] if name in c else no
            feas = {k: v["future_loss"] for k, v in c.items() if v["feasible"]}
            rows.append(dict(dataset=x["dataset"], ftype=x["ftype"], noaction=no,
                             sentinel=pol(x["recommended"]), recommended=x["recommended"],
                             always_retrain=pol("retrain_recent"), always_rollback=pol("rollback_previous_version"),
                             always_recalibrate=pol("recalibrate"), oracle=min(feas.values())))
    P = pd.DataFrame(rows)
    e6 = {"n_incidents": len(P)}
    if len(P):
        for pn in ("sentinel", "always_retrain", "always_rollback", "always_recalibrate", "oracle"):
            rel = (P.noaction - P[pn]) / P.noaction.clip(lower=1e-6)
            e6[pn] = {"mean_rel_loss_reduction": round(float(rel.mean()), 3),
                      "success_rate(>=10% better)": round(float((P[pn] <= 0.9 * P.noaction).mean()), 3),
                      "harm_rate(>5% worse)": round(float((P[pn] > 1.05 * P.noaction).mean()), 3)}
        e6["sentinel_recommendation_counts"] = P.recommended.value_counts().to_dict()
        e6["sentinel_by_failure_type"] = {t: {"n": int(len(g)),
                                              "mean_rel_loss_reduction": round(float(((g.noaction - g.sentinel) / g.noaction.clip(lower=1e-6)).mean()), 3)}
                                          for t, g in P.groupby("ftype")}
    results["E6_recovery"] = e6

    json.dump(results, open(f"{out}/results.json", "w"), indent=1, default=float)
    figures(tbl_s, te, fp, e1, results, out)
    write_md(results, out)
    return results


def CFG_FEATS(ds):
    from sentinelml.datasets import load_dataset
    return load_dataset(ds, 0).feature_names


def figures(tbl_s, te, fp, e1, res, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = te[te.healthy & te.y.notna()]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for col, lab in (("fp_score", "SentinelML (HGB)"), ("psi_max", "PSI max (conventional)"), ("loss_lag", "Lagged loss (reactive)")):
        s = d[col].fillna(0)
        p, r, _ = precision_recall_curve(d.y, s)
        ax.plot(r, p, label=f"{lab} AP={average_precision_score(d.y, s):.2f}")
    ax.axhline(d.y.mean(), ls=":", c="gray", label="prevalence")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Failure-onset prediction, pooled test set"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/figures/pr_curves.png", dpi=130); plt.close(fig)

    names = [k for k in e1 if k != "ALL (pooled)"]
    fig, ax = plt.subplots(figsize=(7, 3.8)); w = 0.27
    for i, (m, lab) in enumerate((("SentinelML (HGB)", "SentinelML"), ("conventional_psi>0.2", "PSI>0.2"), ("lagged_loss_tuned", "Lagged loss"))):
        ax.bar(np.arange(len(names)) + i * w, [e1[n][m]["pr_auc"] for n in names], w, label=lab)
    ax.set_xticks(np.arange(len(names)) + w); ax.set_xticklabels(names, rotation=20, fontsize=8); ax.set_ylabel("PR-AUC"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/figures/pr_auc_by_dataset.png", dpi=130); plt.close(fig)

    e5 = res["E5_root_cause"]
    if "feature_ranking" in e5:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ks = list(e5["feature_ranking"]); ax.bar(ks, [e5["feature_ranking"][k]["any_hit@3"] for k in ks])
        ax.axhline(e5["random_reference_any_hit@3"], ls=":", c="gray", label="random"); ax.set_ylabel("any-hit@3"); ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8); ax.set_title("Root-cause feature ranking")
        fig.tight_layout(); fig.savefig(f"{out}/figures/rca_ranking.png", dpi=130); plt.close(fig)
    e6 = res["E6_recovery"]
    if e6.get("n_incidents"):
        pols = ["sentinel", "always_retrain", "always_rollback", "always_recalibrate", "oracle"]
        fig, ax = plt.subplots(figsize=(6.5, 3.8)); ax.bar(pols, [e6[p]["mean_rel_loss_reduction"] for p in pols])
        ax.set_ylabel("mean relative loss reduction vs no action"); plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out}/figures/recovery_policies.png", dpi=130); plt.close(fig)


def f(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def row(name, m):
    ci = m.get("pr_auc_ci")
    return (f"| {name} | {f(m['pr_auc'])}" + (f" [{f(ci[0],2)}, {f(ci[1],2)}]" if ci else "") +
            f" | {f(m['roc_auc'])} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['false_alarm_per_100'],1)} | "
            f"{f(m['warned_rate'],2)} | {f(m['mean_lead'],2)} | {f(m['mean_delay_unwarned'],2)} | {f(m.get('ece'))} |")


HDR = ("| method | PR-AUC [95% CI over streams] | ROC-AUC | precision | recall | F1 | false alarms /100 healthy windows | "
       "episodes warned | mean lead (windows) | delay if unwarned | ECE |\n|---|---|---|---|---|---|---|---|---|---|---|\n")


def write_md(res, out):
    L = [f"# SentinelML experimental results (auto-generated)\n",
         f"Streams: {res['n_streams']} ({', '.join(res['datasets'])}); horizon H={res['horizon']} windows; "
         f"{res['n_features']} predictive features. All numbers are computed from the cached benchmark by `experiments/analyze.py`.\n",
         "## Benchmark characteristics\n", f"```\n{json.dumps(res['benchmark'], indent=1)}\n```\n",
         "## E1 - conventional threshold monitoring vs predictive monitoring\n"]
    for k, v in res["E1_threshold_vs_predictive"].items():
        L.append(f"### {k}\n\n" + HDR + "\n".join(row(n, m) for n, m in v.items()) + "\n")
    L += ["## E2 - algorithm comparison (pooled)\n\n" + HDR + "\n".join(row(n, m) for n, m in res["E2_algorithms"].items()) + "\n",
          "## E3 - leave-one-dataset-out generalisation\n"]
    for k, v in res["E3_leave_one_dataset_out"].items():
        L.append(f"### held-out: {k}\n\n" + HDR + "\n".join(row(n, m) for n, m in v.items()) + "\n")
    L.append("## E4 - horizon and failure shape\n")
    for h, v in res["E4_horizon_and_shape"]["by_horizon"].items():
        L.append(f"### H={h}\n\n" + HDR + "\n".join(row(n, m) for n, m in v.items()) + "\n")
    L.append(f"```\n{json.dumps(res['E4_horizon_and_shape']['by_shape'], indent=1)}\n```\n")
    L.append("## E5 - root-cause analysis\n```\n" + json.dumps(res["E5_root_cause"], indent=1, default=str) + "\n```\n")
    L.append("## E6 - counterfactual recovery\n```\n" + json.dumps(res["E6_recovery"], indent=1) + "\n```\n")
    L.append("## E7 - ablation\n\n" + HDR + "\n".join(row(n, m) for n, m in res["E7_ablation"].items()) + "\n")
    L.append("## E8 - severity\n```\n" + json.dumps(res["E8_severity"], indent=1) + "\n```\n")
    open(f"{out}/RESULTS.md", "w").write("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="results/cache"); ap.add_argument("--out", default="results")
    a = ap.parse_args(); run(a.cache, a.out); print("done")
