"""v02 -- descriptors + LightGBM.

RATIONALE: v01 (desc+RF) validated 0.609 +/- 0.053. RF is bagging; the family-1
SOTA leaders (MapLight, CaliciBoost) are gradient-boosted. Swap RF -> LightGBM on
the SAME descriptor features to test whether boosting's bias/variance profile
beats bagging on this endpoint. Still no imbalance handling -- isolate the
estimator change.
"""
import lightgbm as lgb

FEATURES = ["desc"]


def build(seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)
