"""v08 -- Morgan r3 (ECFP6) + RandomForest, balanced.

RATIONALE: radius lever. v03/v04 used Morgan r2 (ECFP4). ECFP6 (radius 3) encodes
larger substructural environments, which can help endpoints where bigger
pharmacophores drive activity. Same balanced-RF learner as the best incumbent,
isolating the fingerprint radius.
"""
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["morgan_r3"]


def build(seed):
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                  class_weight="balanced_subsample")
