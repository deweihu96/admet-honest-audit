"""v11 -- (descriptors + Morgan r2 @ 1024 bits) + LightGBM.

RATIONALE: fingerprint bit-size lever. A 1024-bit ECFP4 folds more substructures
into each bit (denser, more hash collisions) than the 2048-bit default -- sometimes
a useful regularizer on small datasets, and cheaper. Paired with boosting on the
descriptor+fingerprint union. Tests representation width, holding learner fixed.
"""
import lightgbm as lgb

FEATURES = ["desc", "morgan_r2_1024"]


def build(seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
