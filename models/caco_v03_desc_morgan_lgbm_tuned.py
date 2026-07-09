"""caco_v03 -- (descriptors + Morgan r2) + tuned LightGBM. Tier 1.

RATIONALE: v02 (desc+LGBM) is the lock at MAE 0.365. Add Morgan r2 for
orthogonal substructural signal and raise capacity (more trees, lower LR). Either
CI-separably beats v02 (promote) or ties it (parsimony keeps the simpler v02).
"""
import lightgbm as lgb

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, num_leaves=63,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
