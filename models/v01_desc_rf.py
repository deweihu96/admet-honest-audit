"""v01 -- descriptors + RandomForest.

RATIONALE: anchor the loop on the strongest single recipe found in prior
diagnostics (RDKit 2D descriptors + RandomForest scored validation ~0.61 on this
endpoint, the best of the earlier grid). Family 1 (descriptor + tree). No
imbalance handling yet -- establish the honest baseline first, then reason from
its validation result.
"""
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["desc"]


def build(seed):
    return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed)
