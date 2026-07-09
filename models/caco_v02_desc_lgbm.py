"""caco_v02 -- descriptors + LightGBM regressor. Tier 1.

RATIONALE: Caco2 permeability is driven by lipophilicity (logP), polar surface
area, H-bonding and size -- exactly the RDKit 2D descriptors. Swap the Morgan
view for the descriptor view (same LightGBM). Mirrors the solubility finding
where descriptors CI-separably beat fingerprints; expect a promotion here too.
"""
import lightgbm as lgb

FEATURES = ["desc"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
