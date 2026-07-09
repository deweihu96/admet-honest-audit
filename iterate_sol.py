"""ITERATION HARNESS (regression) -- solubility_aqsoldb.

Loads models/<version>.py, trains on train, scores VALIDATION MAE across the
5-seed budget, prints MAE mean +/- 95% CI. MAE is LOWER-is-better.

TEST WALL: imports data_split_sol (train/valid only); never references test.
    uv run python iterate_sol.py --version sol_v01_morgan_lgbm
"""
import argparse
import importlib
import time
import numpy as np
from scipy import stats
from tdc import Evaluator
import data_split_sol as D
from features import build_matrix


def ci_hw(x):
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def run(version):
    t0 = time.time()
    mod = importlib.import_module(f"models.{version}")
    ev = Evaluator(name=D.METRIC)
    sp = D.splits()
    scores = []
    for s in D.SEEDS:
        tr, va = sp[s]
        Xtr = build_matrix(tr["Drug"].tolist(), mod.FEATURES); ytr = tr["Y"].to_numpy()
        Xva = build_matrix(va["Drug"].tolist(), mod.FEATURES); yva = va["Y"].to_numpy()
        est = mod.build(s).fit(Xtr, ytr)
        scores.append(float(ev(yva, est.predict(Xva))))     # regression: predict()
    x = np.array(scores)
    m, hw = float(x.mean()), ci_hw(x)
    dt = time.time() - t0
    print(f"version={version}")
    print(f"  FEATURES={mod.FEATURES}")
    print(f"  VALIDATION MAE (lower better): {m:.4f} +/- {hw:.4f}  "
          f"(95% CI [{m-hw:.4f}, {m+hw:.4f}], {len(x)} seeds)")
    print(f"  per-seed std={x.std(ddof=1):.4f}   wall-clock={dt:.1f}s")
    return m, hw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    run(ap.parse_args().version)
