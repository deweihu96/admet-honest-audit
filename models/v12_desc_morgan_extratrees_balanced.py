"""v12 -- (descriptors + Morgan r2) + ExtraTrees, balanced.

RATIONALE: base-learner lever. ExtraTrees (extremely randomized trees) uses random
split thresholds rather than optimized ones, trading a little bias for lower
variance -- a standard alternative to RandomForest that can win on noisy, small
tabular problems. Same features and imbalance handling as v04, swapping only the
tree ensemble.
"""
from sklearn.ensemble import ExtraTreesClassifier

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                class_weight="balanced_subsample")
