"""Small helpers for model version definitions."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score, mean_absolute_error


class AveragingEnsemble:
    """Average predict_proba across a list of fitted estimators (soft voting)."""
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        for e in self.estimators:
            e.fit(X, y)
        return self

    def predict_proba(self, X):
        p = np.mean([e.predict_proba(X)[:, 1] for e in self.estimators], axis=0)
        return np.vstack([1 - p, p]).T


class ColumnSlice(BaseEstimator, TransformerMixin):
    """Select columns [start:stop] -- routes a feature block to a base head so
    heterogeneous learners see different views of a concatenated matrix."""
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.start:self.stop]


class LateFusionWeighted:
    """Late fusion of exactly two per-view heads with a VALIDATED blend weight.

    Unlike plain concatenation (v03/v04) or a flat average, the blend weight w
    for head0 (and 1-w for head1) is chosen by internal StratifiedKFold on the
    TRAINING fold only: fit each head out-of-fold, then pick the w on a grid that
    maximizes out-of-fold PR-AUC. Heads are then refit on the full train. No
    validation/test leakage -- w is a property of the train fold.
    """
    def __init__(self, heads, seed=0, cv=5, grid=None):
        self.heads = heads            # [(name, (start, stop), factory), ...] len 2
        self.seed = seed
        self.cv = cv
        self.grid = np.linspace(0, 1, 21) if grid is None else grid

    def fit(self, X, y):
        assert len(self.heads) == 2, "designed for two heads"
        n = len(y)
        oof = {name: np.zeros(n) for name, _, _ in self.heads}
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(X, y):
            for name, (a, b), fac in self.heads:
                m = fac().fit(X[tr, a:b], y[tr])
                oof[name][va] = m.predict_proba(X[va, a:b])[:, 1]
        (n0, _, _), (n1, _, _) = self.heads
        self.w_, best = 0.5, -1.0
        for w in self.grid:
            s = average_precision_score(y, w * oof[n0] + (1 - w) * oof[n1])
            if s > best:
                best, self.w_ = s, w
        self.fitted_ = {name: fac().fit(X[:, a:b], y) for name, (a, b), fac in self.heads}
        return self

    def predict_proba(self, X):
        (n0, (a0, b0), _), (n1, (a1, b1), _) = self.heads
        p = (self.w_ * self.fitted_[n0].predict_proba(X[:, a0:b0])[:, 1]
             + (1 - self.w_) * self.fitted_[n1].predict_proba(X[:, a1:b1])[:, 1])
        return np.vstack([1 - p, p]).T


class LateFusionWeightedReg:
    """Regression late fusion of two per-view heads with a validated blend weight.
    w for head0 (1-w for head1) chosen by internal KFold on the TRAIN fold to
    MINIMIZE out-of-fold MAE (lower is better). Heads refit on full train. The
    regression analogue of LateFusionWeighted."""
    def __init__(self, heads, seed=0, cv=5, grid=None):
        self.heads = heads
        self.seed = seed
        self.cv = cv
        self.grid = np.linspace(0, 1, 21) if grid is None else grid

    def fit(self, X, y):
        assert len(self.heads) == 2
        n = len(y)
        oof = {name: np.zeros(n) for name, _, _ in self.heads}
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            for name, (a, b), fac in self.heads:
                m = fac().fit(X[tr, a:b], y[tr])
                oof[name][va] = m.predict(X[va, a:b])
        (n0, _, _), (n1, _, _) = self.heads
        self.w_, best = 0.5, np.inf
        for w in self.grid:
            e = mean_absolute_error(y, w * oof[n0] + (1 - w) * oof[n1])
            if e < best:
                best, self.w_ = e, w
        self.fitted_ = {name: fac().fit(X[:, a:b], y) for name, (a, b), fac in self.heads}
        return self

    def predict(self, X):
        (n0, (a0, b0), _), (n1, (a1, b1), _) = self.heads
        return (self.w_ * self.fitted_[n0].predict(X[:, a0:b0])
                + (1 - self.w_) * self.fitted_[n1].predict(X[:, a1:b1]))


class TwoStageResidual:
    """Two-stage residual regression: stage-1 regressor on view A predicts y;
    stage-2 regressor on view B predicts stage-1's residuals; output = sum. A
    designed composition (manual view-wise boosting), not a component swap."""
    def __init__(self, view_a, make_a, view_b, make_b):
        self.view_a = view_a          # (start, stop)
        self.make_a = make_a
        self.view_b = view_b
        self.make_b = make_b

    def fit(self, X, y):
        a0, a1 = self.view_a; b0, b1 = self.view_b
        self.m1_ = self.make_a().fit(X[:, a0:a1], y)
        resid = y - self.m1_.predict(X[:, a0:a1])
        self.m2_ = self.make_b().fit(X[:, b0:b1], resid)
        return self

    def predict(self, X):
        a0, a1 = self.view_a; b0, b1 = self.view_b
        return self.m1_.predict(X[:, a0:a1]) + self.m2_.predict(X[:, b0:b1])
