"""sol_v12 -- (descriptors + Morgan r2) + ExtraTrees (regression).

RATIONALE: base-learner lever. Extremely randomized trees trade a little bias for
lower variance vs RandomForest, a reasonable alternative on noisy tabular data.
Same features as sol_v11, swapping only the tree ensemble.
"""
from sklearn.ensemble import ExtraTreesRegressor

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return ExtraTreesRegressor(n_estimators=400, n_jobs=-1, random_state=seed)
