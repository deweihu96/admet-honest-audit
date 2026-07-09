"""caco_v01 -- Morgan r2 + LightGBM regressor. Tier 1 (element switching).

RATIONALE: family-1 fingerprint-GBM baseline. Gradient boosting tops the Caco2
board (CaliciBoost/XGBoost/MapLight per the frozen prior -- methods, not their
numbers). Start from a substructural-only fingerprint model; deliberately the
weaker start, since Caco2 permeability tracks lipophilicity/PSA that descriptors
capture directly.
"""
import lightgbm as lgb

FEATURES = ["morgan_r2"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
