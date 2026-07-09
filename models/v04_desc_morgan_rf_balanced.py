"""v04 -- (descriptors + Morgan r2) + RandomForest, class_weight balanced.

RATIONALE: v03 (desc+morgan+RF) is the best incumbent (0.616) but every version
so far ignores the 28% class imbalance, and PR-AUC specifically rewards ranking
the minority positives. Add class_weight='balanced_subsample' to upweight
positives in RF's split criterion. Imbalance handling is a stated lever of the
family-1 SOTA leaders. Same best features, isolate the imbalance change.
"""
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                  class_weight="balanced_subsample")
