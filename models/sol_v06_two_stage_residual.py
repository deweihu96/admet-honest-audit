"""sol_v06 -- designed TWO-STAGE residual regression. Tier 2 (composition).

COMPOSITION: stage-1 LightGBM on descriptors predicts logS; stage-2 LightGBM on
Morgan r2 predicts stage-1's RESIDUALS; output = stage1 + stage2. A view-wise
manual boosting: the substructural model only has to explain what the physchem
model misses.

RATIONALE: descriptors dominate solubility (v02), but Morgan substructures might
capture systematic residual error (e.g. specific functional-group effects) that
descriptors miss. Two-stage residual targets exactly that orthogonal signal,
which plain concatenation (v03) did not isolate. Kept only if CI-separably better
than the lock (v02).
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

    def stage2():   # smaller: it only models the residual
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=seed, n_jobs=-1, verbose=-1)
    return TwoStageResidual((0, _D), stage1, (_D, _M), stage2)
