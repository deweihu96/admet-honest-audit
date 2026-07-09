"""sol_v05 -- designed LATE-FUSION regression with validated blend weight. Tier 2.

COMPOSITION: two per-view LightGBM heads (desc, morgan) blended as
p = w*p_desc + (1-w)*p_morgan, where w is chosen by internal KFold on the TRAIN
fold to MINIMIZE out-of-fold MAE (LateFusionWeightedReg).

RATIONALE: v03 concatenated desc+morgan (early fusion) and tied v02. This is LATE
fusion: separate per-view regressors blended by a data-chosen weight instead of
raw concatenation. Tests whether learning how much to trust each view lowers MAE.
Kept only if CI-separably better than the lock (v02).
"""
import lightgbm as lgb
from features import DIMS
from model_utils import LateFusionWeightedReg

FEATURES = ["desc", "morgan_r2"]
_D = DIMS["desc"]
_M = _D + DIMS["morgan_r2"]


def build(seed):
    def lgbm():
        return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=-1, verbose=-1)
    heads = [("desc", (0, _D), lgbm), ("morgan", (_D, _M), lgbm)]
    return LateFusionWeightedReg(heads, seed=seed, cv=5)
