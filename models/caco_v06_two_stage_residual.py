"""caco_v06 -- designed TWO-STAGE residual regression (LightGBM). Tier 2.

COMPOSITION: stage-1 LightGBM on descriptors predicts logPapp; stage-2 LightGBM
on Morgan r2 predicts stage-1's RESIDUALS; output = stage1 + stage2. View-wise
manual boosting: the substructural model only explains what physchem misses.
Targets orthogonal residual signal that plain concatenation (v03) did not
isolate. Kept only if CI-separably better than the lock (v02).
"""
import lightgbm as lgb
from features import DIMS
from model_utils import TwoStageResidual

FEATURES = ["desc", "morgan_r2"]
_D = DIMS["desc"]
_M = _D + DIMS["morgan_r2"]


def build(seed):
    def stage1():
        return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=-1, verbose=-1)

    def stage2():
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=-1, verbose=-1)
    return TwoStageResidual((0, _D), stage1, (_D, _M), stage2)
