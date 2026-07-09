"""sol_v02 -- descriptors + LightGBM regressor. Tier 1 (element switching).

RATIONALE: v01 (Morgan+LGBM) validated MAE 1.261. Aqueous solubility is driven by
physicochemical properties (logP, TPSA, MW, H-bonding) that RDKit 2D descriptors
encode directly and Morgan bits do not. Swap fingerprint view -> descriptor view,
same LightGBM. Expect a CI-separable MAE reduction (lower is better) -- the first
real promotion test on this endpoint.
"""
import lightgbm as lgb

FEATURES = ["desc"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
