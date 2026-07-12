"""v10 -- (descriptors + Morgan r3/ECFP6) + RandomForest, balanced.

RATIONALE: the v04 union recipe (physchem + substructure + balanced RF) but with
the ECFP6 radius instead of ECFP4. Tests whether pairing descriptors with the
larger-radius fingerprint improves on v04's descriptor+ECFP4 union. Direct
apples-to-apples radius test at the best-performing topology.
"""
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["desc", "morgan_r3"]


def build(seed):
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                  class_weight="balanced_subsample")
