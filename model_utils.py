"""Small helpers for model version definitions."""
import numpy as np


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
