"""TRAIN/VALIDATION data access for the caco2_wang regression loop.

STRUCTURAL TEST WALL (same discipline): data reached ONLY via
get_train_valid_split -> (train, valid); cannot return test. Test lives only in
the final-eval script. caco2 is small (728 train_val / 182 test), so it gets the
25-seed validation budget (like cyp2d6), not the 5-seed large-endpoint budget.
"""
from tdc.benchmark_group import admet_group

ENDPOINT = "caco2_wang"
METRIC = "mae"                      # LOWER is better
DIRECTION = "lower"
SEEDS = list(range(1, 26))         # 25 seeds: small endpoint, tighter validation CIs

_group = None
_splits = None


def _grp():
    global _group
    if _group is None:
        _group = admet_group(path="data/")
    return _group


def splits():
    global _splits
    if _splits is None:
        g = _grp()
        _splits = {s: g.get_train_valid_split(benchmark=ENDPOINT, split_type="default",
                                              seed=s) for s in SEEDS}
    return _splits
