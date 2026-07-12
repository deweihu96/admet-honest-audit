"""sol_v09 -- (descriptors + Morgan r3/ECFP6) + LightGBM (regression).

RATIONALE: the sol_v03 descriptor+fingerprint union recipe but with the ECFP6
radius instead of ECFP4. Apples-to-apples radius test at the union topology that
carried the solubility signal.
"""
import lightgbm as lgb

FEATURES = ["desc", "morgan_r3"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
