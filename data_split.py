"""TRAIN/VALIDATION data access for the iteration loop.

STRUCTURAL TEST WALL: this module reaches the data ONLY through
group.get_train_valid_split(), which returns (train, valid) and CANNOT return
the test set. It never calls group.get()[...]['test'] and never reads test.csv,
so no code path that imports this module can obtain test at all. Test lives
exclusively in final_test_eval.py.
"""
import pandas as pd
from tdc.benchmark_group import admet_group

ENDPOINT = "cyp2d6_substrate_carbonmangels"
METRIC = "pr-auc"
SEEDS = list(range(1, 26))          # 25-seed budget for this small/high-variance endpoint

_group = None
_splits = None


def _grp():
    global _group
    if _group is None:
        _group = admet_group(path="data/")
    return _group


def splits():
    """{seed: (train_df, valid_df)} via get_train_valid_split only (no test)."""
    global _splits
    if _splits is None:
        g = _grp()
        _splits = {s: g.get_train_valid_split(benchmark=ENDPOINT, split_type="default",
                                              seed=s) for s in SEEDS}
    return _splits


def naive_floor():
    """PR-AUC naive baseline = positive prevalence, from train_val (train+valid)."""
    tr, va = splits()[SEEDS[0]]
    y = pd.concat([tr["Y"], va["Y"]]).to_numpy()
    return float((y == 1).mean())
