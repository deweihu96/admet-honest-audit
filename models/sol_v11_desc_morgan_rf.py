"""sol_v11 -- (descriptors + Morgan r2) + RandomForest (regression).

RATIONALE: base-learner lever. The solubility stream is boosting-heavy; a
RandomForest regressor (bagging) has a different bias/variance profile and is a
standard strong tabular baseline. Same descriptor+fingerprint union.
"""
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=seed)
