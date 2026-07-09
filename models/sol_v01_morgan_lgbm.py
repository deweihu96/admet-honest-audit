"""sol_v01 -- Morgan r2 + LightGBM regressor. Tier 1 (element switching).

RATIONALE: family-1 fingerprint-GBM baseline (MapLight-style) to anchor the
regression loop. Substructural fingerprint view only -- deliberately the weaker
starting point per SOTA_PRIOR_REG, since aqueous solubility is theory-driven by
physicochemical descriptors that a Morgan-only model does not see directly.
"""
import lightgbm as lgb

FEATURES = ["morgan_r2"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
