"""sol_v04 -- designed STACKED regression topology. Tier 2 (composition).

COMPOSITION: StackingRegressor over 3 heterogeneous base heads on distinct views
    - desc   -> LightGBM
    - desc   -> RandomForest
    - morgan -> LightGBM
  meta-learner: Ridge over base out-of-fold predictions (cv=5).

RATIONALE: v02 (single desc+LGBM) is the lock at MAE 0.826. Test whether a
learned meta-combination of diverse learners/views lowers MAE below a single
model. Same compose-from-known-parts boundary. Kept only if CI-separably better
than v02 (parsimony).
"""
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
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
        ("desc_rf", make_pipeline(ColumnSlice(0, _D),
            RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed))),
        ("morgan_lgbm", make_pipeline(ColumnSlice(_D, _M), _lgbm(seed))),
    ]
    return StackingRegressor(estimators=base, final_estimator=Ridge(),
                             cv=5, n_jobs=1)
