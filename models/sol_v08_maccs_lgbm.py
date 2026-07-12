"""sol_v08 -- MACCS keys + LightGBM (regression).

RATIONALE: fingerprint-family lever. MACCS 166 structural keys are a different,
low-dimensional, expert-defined representation than hashed Morgan bits. A
reasonable modeler tries it on solubility because keyed fingerprints sometimes
generalize better; boosted-tree learner held fixed.
"""
import lightgbm as lgb

FEATURES = ["maccs"]


def build(seed):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)
