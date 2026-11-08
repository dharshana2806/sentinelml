"""End-to-end benchmark pipeline for ONE stream: simulate -> detect episodes -> diagnose -> recover."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimConfig
from .predictor import episodes
from .rca import feature_diagnosis, signature, RANKERS, top_k
from .recovery import simulate_recovery
from .simulate import build_context, simulate_stream

ONSET_TO_DIAG = 8      # diagnose / recover this many windows after the onset


def process_stream(name: str, seed: int, cfg: SimConfig, do_rca: bool = True, do_recovery: bool = True) -> dict:
    ctx = build_context(name, seed, cfg)
    res = simulate_stream(ctx, seed)
    d = cfg.label_delay
    diag, rec = [], []
    if do_rca or do_recovery:
        for o, e_end in episodes(res.degraded):
            evs = [e for e in res.events if e.start <= o <= e.end + 3]
            t0 = o + ONSET_TO_DIAG
            if not evs or t0 + 4 >= min(evs[-1].end, cfg.n_windows - 1):
                continue
            ev = evs[-1]
            dw = res.windows[t0 - d - 2: t0 - d + 1]                       # last 3 *labelled* windows
            X = pd.concat([w[0] for w in dw], ignore_index=True); y = np.concatenate([w[1] for w in dw])
            sig_rows = res.signals.iloc[t0 - d - 2: t0 - d + 1]
            fd = feature_diagnosis(ctx, X, y)
            base = dict(dataset=name, seed=seed, event_id=ev.event_id, ftype=ev.ftype, shape=ev.shape,
                        severity=ev.severity, true_features=[str(f) for f in ev.features], onset=int(o))
            if do_rca:
                diag.append({**base, **{f"sig_{k}": v for k, v in signature(ctx, sig_rows, X, y).items()},
                             "ranks": {r: top_k(fd, r, 5) for r in RANKERS}})
            if do_recovery:
                suspects = [f for f in top_k(fd, "combined", 3) if fd.loc[f, "combined"] > 0.5]
                fit_w = res.windows[o: o + 4]
                eval_w = res.windows[o + 4: o + 7]
                out = simulate_recovery(ctx, fit_w, eval_w, suspects)
                fut = res.windows[t0 + 1: min(t0 + 9, ev.end)]
                Xf = pd.concat([w[0] for w in fut], ignore_index=True); yf = np.concatenate([w[1] for w in fut])
                cands = {}
                for c in out["candidates"]:
                    cands[c.name] = {**c.to_dict(), "future_loss": ctx.model.loss(yf, c.predictor.predict(Xf))}
                rec.append({**base, "suspects": suspects, "recommended": out["recommended"], "candidates": cands,
                            "baseline_eval_loss": out["baseline_eval_loss"], "n_future": len(fut)})
    sig = res.signals.copy()
    sig["degraded"] = res.degraded
    return {"name": name, "seed": seed, "signals": res.signals, "degraded": res.degraded, "loss": res.loss,
            "events": [e.to_dict() for e in res.events], "diag": diag, "recovery": rec,
            "thr": ctx.thr, "base_loss": ctx.base_loss}
