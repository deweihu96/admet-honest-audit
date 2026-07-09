"""TRAIN/VALIDATION data access for the solubility_aqsoldb regression loop.

STRUCTURAL TEST WALL (same as data_split.py): reaches data ONLY through
get_train_valid_split(), which returns (train, valid) and cannot return test.
No group.get()[...]['test'], no test.csv. Test lives only in
final_test_eval_sol.py.
"""
from tdc.benchmark_group import admet_group

ENDPOINT = "solubility_aqsoldb"
METRIC = "mae"                      # LOWER is better (harness-verified)
DIRECTION = "lower"
SEEDS = [1, 2, 3, 4, 5]             # 5-seed budget for a large/balanced endpoint

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
