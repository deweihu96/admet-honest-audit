"""v07 -- Morgan r2 (ECFP4) + LightGBM.

RATIONALE: a fingerprint-only gradient-boosting baseline. Prior specs paired
descriptors with trees; this tests whether ECFP4 substructure bits alone, fed to a
boosted-tree learner (the family-1 SOTA workhorse), compete without physicochemical
descriptors. A standard, defensible ADMET recipe.
"""
import lightgbm as lgb

FEATURES = ["morgan_r2"]


def build(seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
