"""sol_v10 -- (descriptors + Morgan r2 @ 1024 bits) + LightGBM (regression).

RATIONALE: fingerprint bit-size lever. Folds ECFP4 into 1024 bits (denser, more
collisions) instead of 2048, a mild regularizer on the fingerprint view; paired
with descriptors and boosting. Tests representation width on solubility.
"""
import lightgbm as lgb

FEATURES = ["desc", "morgan_r2_1024"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
