"""sol_v13 -- (descriptors + Morgan r2) + HistGradientBoosting (regression).

RATIONALE: base-learner lever within the boosting family. sklearn's
HistGradientBoosting is an independent histogram-GBM implementation with different
regularization defaults than LightGBM. Tests whether a second boosting engine
separates further on solubility.
"""
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         random_state=seed)
