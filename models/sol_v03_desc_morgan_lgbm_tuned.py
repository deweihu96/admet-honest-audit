"""sol_v03 -- (descriptors + Morgan r2) + tuned LightGBM. Tier 1.

RATIONALE: v02 (desc+LGBM) is the lock at MAE 0.826. Descriptors carry the
solubility signal; add Morgan r2 for orthogonal substructural information and
raise model capacity (more trees, lower LR, more leaves) since 500 trees may
underfit ~6400 training rows. Either CI-separably improves on v02 (promote) or
ties it (parsimony keeps the simpler v02).
"""
import lightgbm as lgb

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=63,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
