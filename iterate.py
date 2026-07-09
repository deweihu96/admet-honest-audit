"""ITERATION HARNESS -- the only thing the research loop calls each iteration.

Loads a model version from models/<version>.py, trains on train, scores on
VALIDATION across the 25-seed budget, prints validation metric mean +/- 95% CI.

TEST WALL: this file imports data_split (train/valid only) and never references
the test set. There is no code path from here to test. Run as:
    uv run python iterate.py --version v01_desc_rf
"""
import argparse
import importlib
import numpy as np
from scipy import stats
from tdc import Evaluator
import data_split as D
from features import build_matrix


def ci_hw(x):
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def run(version):
    mod = importlib.import_module(f"models.{version}")
    ev = Evaluator(name=D.METRIC)
    sp = D.splits()
    scores = []
    for s in D.SEEDS:
        tr, va = sp[s]
        Xtr = build_matrix(tr["Drug"].tolist(), mod.FEATURES); ytr = tr["Y"].to_numpy()
        Xva = build_matrix(va["Drug"].tolist(), mod.FEATURES); yva = va["Y"].to_numpy()
        est = mod.build(s).fit(Xtr, ytr)
        scores.append(float(ev(yva, est.predict_proba(Xva)[:, 1])))
    x = np.array(scores)
    m, hw = float(x.mean()), ci_hw(x)
    print(f"version={version}")
    print(f"  FEATURES={mod.FEATURES}")
    print(f"  VALIDATION {D.METRIC}: {m:.4f} +/- {hw:.4f}  "
          f"(95% CI [{m-hw:.4f}, {m+hw:.4f}], {len(x)} seeds)")
    print(f"  per-seed std={x.std(ddof=1):.4f}   floor(prevalence)={D.naive_floor():.4f}")
    return m, hw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    run(ap.parse_args().version)
