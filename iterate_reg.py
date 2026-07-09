"""GENERIC regression iteration harness -- task-parameterized.

Loads a train/valid data module (--task, e.g. data_split_caco) and a model
version (--version), trains on train, scores VALIDATION MAE across the task's
seed budget. MAE is LOWER-is-better.

TEST WALL: imports only the given data module (train/valid); never the test set.
    uv run python iterate_reg.py --task data_split_caco --version caco_v01_morgan_lgbm
"""
import argparse
import importlib
import time
import numpy as np
from scipy import stats
from tdc import Evaluator
from features import build_matrix


def ci_hw(x):
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def run(task, version):
    t0 = time.time()
    D = importlib.import_module(task)
    mod = importlib.import_module(f"models.{version}")
    ev = Evaluator(name=D.METRIC)
    sp = D.splits()
    scores = []
    for s in D.SEEDS:
        tr, va = sp[s]
        Xtr = build_matrix(tr["Drug"].tolist(), mod.FEATURES); ytr = tr["Y"].to_numpy()
        Xva = build_matrix(va["Drug"].tolist(), mod.FEATURES); yva = va["Y"].to_numpy()
        est = mod.build(s).fit(Xtr, ytr)
        scores.append(float(ev(yva, est.predict(Xva))))
    x = np.array(scores)
    m, hw = float(x.mean()), ci_hw(x)
    print(f"task={task}  version={version}  FEATURES={mod.FEATURES}")
    print(f"  VALIDATION {D.METRIC} (lower better): {m:.4f} +/- {hw:.4f}  "
          f"(95% CI [{m-hw:.4f}, {m+hw:.4f}], {len(x)} seeds)")
    print(f"  per-seed std={x.std(ddof=1):.4f}   wall-clock={time.time()-t0:.1f}s")
    return m, hw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--version", required=True)
    a = ap.parse_args()
    run(a.task, a.version)
