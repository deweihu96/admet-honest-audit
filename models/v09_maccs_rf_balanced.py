"""v09 -- MACCS keys + RandomForest, balanced.

RATIONALE: fingerprint-family lever. MACCS is 166 curated structural keys, a
fundamentally different (expert-defined, low-dimensional) representation than the
hashed Morgan bits. A reasonable modeler tries it because low-dimensional keyed
fingerprints sometimes generalize better on small assays. Same balanced-RF learner.
"""
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["maccs"]


def build(seed):
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                  class_weight="balanced_subsample")
