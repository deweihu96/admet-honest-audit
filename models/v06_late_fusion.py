"""v06 -- designed LATE-FUSION architecture with a validated blend weight.

COMPOSITION (novel, known primitives):
  two per-view heads:
    head_desc   : RandomForest(balanced) on physicochemical descriptors
    head_morgan : RandomForest(balanced) on Morgan r2 substructures
  fusion: p = w*p_desc + (1-w)*p_morgan, where w is chosen by internal
  StratifiedKFold on the TRAIN fold to maximize out-of-fold PR-AUC (grid 0..1).

RATIONALE: v03/v04 concatenated desc+morgan into ONE RF (early fusion). This is
LATE fusion: separate models per view, blended by a data-chosen weight rather
than by concatenation or a flat 50/50 average. Tests whether learning HOW MUCH
to trust each view (per train fold, no leakage) beats mixing the raw features.
Expectation per the ceiling: likely TIES v04; kept only if CI-separably better.
"""
from sklearn.ensemble import RandomForestClassifier
from features import DIMS
from model_utils import LateFusionWeighted

FEATURES = ["desc", "morgan_r2"]
_D = DIMS["desc"]
_M = _D + DIMS["morgan_r2"]


def build(seed):
    def rf():
        return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                      class_weight="balanced_subsample")
    heads = [("desc", (0, _D), rf), ("morgan", (_D, _M), rf)]
    return LateFusionWeighted(heads, seed=seed, cv=5)
