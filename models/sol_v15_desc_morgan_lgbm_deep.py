"""sol_v15 -- (descriptors + Morgan r2) + higher-capacity LightGBM (regression).

RATIONALE: capacity/hyperparameter lever. More trees, more leaves, and a lower
learning rate give the booster more capacity to exploit any residual signal a
reasonable modeler would try this on a large endpoint like solubility (~9k train).
Same features as sol_v03; tests whether more model capacity separates further.
"""
import lightgbm as lgb

FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=800, learning_rate=0.02, num_leaves=63,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_samples=10,
                             random_state=seed, n_jobs=-1, verbose=-1)
