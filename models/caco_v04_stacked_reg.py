"""caco_v04 -- designed STACKED regression topology (LightGBM bases). Tier 2.

COMPOSITION: StackingRegressor over 3 heterogeneous base heads on distinct views
    - desc   -> LightGBM
    - morgan -> LightGBM
    - desc   -> Ridge (standardized)     # linear base for diversity
  meta-learner: Ridge over base out-of-fold predictions (cv=5).

All GBM/linear bases (no RandomForest -- RF-in-stacking was the 442s cost driver
on solubility). RATIONALE: test whether a learned meta-combination of diverse
learners/views lowers MAE below the single desc+LGBM lock (v02). Kept only if
CI-separably better (parsimony).
"""
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from features import DIMS
from model_utils import ColumnSlice

FEATURES = ["desc", "morgan_r2"]
_D = DIMS["desc"]
_M = _D + DIMS["morgan_r2"]


def _lgbm(seed):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbose=-1)


def build(seed):
    base = [
        ("desc_lgbm", make_pipeline(ColumnSlice(0, _D), _lgbm(seed))),
        ("morgan_lgbm", make_pipeline(ColumnSlice(_D, _M), _lgbm(seed))),
        ("desc_ridge", make_pipeline(ColumnSlice(0, _D), StandardScaler(), Ridge())),
    ]
    return StackingRegressor(estimators=base, final_estimator=Ridge(),
                             cv=5, n_jobs=1)
