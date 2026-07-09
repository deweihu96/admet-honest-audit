"""v05 -- designed STACKED topology (meta-learner over heterogeneous base heads).

COMPOSITION (novel topology, known primitives):
  base heads, each on a DIFFERENT feature view:
    - desc    -> RandomForest(balanced)
    - morgan  -> RandomForest(balanced)
    - desc    -> LightGBM
  meta-learner: LogisticRegression over the base heads' out-of-fold predict_proba
  (sklearn StackingClassifier, cv=5, stack_method='predict_proba').

RATIONALE: v04 (flat features + one RF) plateaued at ~0.626. This is not a
component swap: it is a designed 2-level topology where diverse base learners see
distinct views and a meta-learner learns how to weight them, using internal OOF
to avoid leakage. Tests whether a learned combination of views beats a single
model on the SAME information. Expectation per the endpoint ceiling: likely TIES
v04; kept only if CI-separably better (parsimony rule).
"""
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from features import DIMS
from model_utils import ColumnSlice

FEATURES = ["desc", "morgan_r2"]
_D = DIMS["desc"]                       # desc cols [0:_D), morgan cols [_D:_D+2048)
_M = _D + DIMS["morgan_r2"]


def build(seed):
    base = [
        ("desc_rf", make_pipeline(ColumnSlice(0, _D),
            RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                   class_weight="balanced_subsample"))),
        ("morgan_rf", make_pipeline(ColumnSlice(_D, _M),
            RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=seed,
                                   class_weight="balanced_subsample"))),
        ("desc_lgbm", make_pipeline(ColumnSlice(0, _D),
            lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=seed, n_jobs=-1, verbose=-1))),
    ]
    return StackingClassifier(estimators=base,
                              final_estimator=LogisticRegression(max_iter=1000),
                              stack_method="predict_proba", cv=5, n_jobs=1)
