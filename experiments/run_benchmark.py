"""Generate the failure benchmark: simulate streams + diagnose + recovery, cached per stream.

Usage:  python experiments/run_benchmark.py [--streams 8] [--out results/cache]
Runtime on 1 CPU: ~20-25 s per stream (5 datasets x 8 streams ~ 15-20 min).
"""
import argparse, os, pickle, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sentinelml.config import SimConfig
from sentinelml.pipeline import process_stream

DATASETS = ["breast_cancer", "wine", "synth_binary", "synth_multiclass", "synth_regression"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--out", default="results/cache")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = SimConfig()
    for name in a.datasets:
        for k in range(a.streams):
            seed = 100 + k
            path = f"{a.out}/{name}_{seed}.pkl"
            if os.path.exists(path):
                continue
            t = time.time()
            r = process_stream(name, seed, cfg)
            with open(path, "wb") as f:
                pickle.dump(r, f)
            print(f"{name} seed={seed} done in {time.time()-t:.1f}s  episodes_diag={len(r['diag'])}", flush=True)
