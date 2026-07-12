"""v13 -- (descriptors + Morgan r2) + HistGradientBoosting, balanced.

RATIONALE: base-learner lever within the boosting family. sklearn's
HistGradientBoosting is a LightGBM-style histogram gradient booster with a
different implementation and regularization defaults; class_weight='balanced'
addresses the minority class that PR-AUC rewards. Tests whether a second,
independent boosting implementation separates where LightGBM did not.
"""
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=seed)
