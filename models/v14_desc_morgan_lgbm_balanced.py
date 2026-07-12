"""v14 -- (descriptors + Morgan r2) + LightGBM with scale_pos_weight.

RATIONALE: imbalance-handling lever ON the boosting family. v02/v07 used LightGBM
with NO class weighting; v04 added imbalance handling only to RF. PR-AUC rewards
ranking minority positives, so give the booster the negative/positive ratio via
scale_pos_weight. Tests whether imbalance-aware boosting (not bagging) separates.
"""
import lightgbm as lgb
import numpy as np


class _BalancedLGBM:
    """LGBMClassifier that sets scale_pos_weight from the training class ratio at
    fit time (n_neg/n_pos), the standard imbalance knob for boosted trees."""
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        n_pos = max(int((y == 1).sum()), 1)
        n_neg = int((y == 0).sum())
        self.model_ = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg / n_pos,
            random_state=self.seed, n_jobs=-1, verbose=-1).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


FEATURES = ["desc", "morgan_r2"]


def build(seed):
    return _BalancedLGBM(seed)
