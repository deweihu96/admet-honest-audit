"""v03 -- (descriptors + Morgan r2) + RandomForest.

RATIONALE: v01 (desc+RF, 0.609) beat v02 (desc+LGBM, 0.584), so keep RF. So far
only physicochemical descriptors were used. Add Morgan r2 (ECFP4) substructural
bits alongside descriptors to give RF both a physchem and a substructure view.
Test whether a richer feature union lifts validation on this endpoint.
"""
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)
