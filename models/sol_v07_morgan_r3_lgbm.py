"""sol_v07 -- Morgan r3 (ECFP6) + LightGBM (regression).

RATIONALE: radius lever. sol_v01 used Morgan r2 (ECFP4) + LGBM; this swaps to the
larger ECFP6 radius, holding the boosted-tree learner fixed. Tests whether bigger
substructural environments help the solubility fit.
"""
import lightgbm as lgb

FEATURES = ["morgan_r3"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
