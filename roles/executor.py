"""Executor role: train on train, score on VALIDATION, across seeds.

VALIDATION-ONLY BY CONSTRUCTION. This module reaches data ONLY through
adapter.load_train_valid(seed). It never imports or calls load_test_molecules /
load_test_labeled -- there is no code path from here to test. (grep-verified in
roles/verify_stage2.py.)

Featurization is reused from features.build_matrix (Morgan r2 for modeling comes
from the model_spec's FEATURES; the r3 leakage radius belongs to the auditor and
is not touched here). The model_spec is the same shape as models/*.py version
files: an object exposing FEATURES (list of feature blocks) and build(seed) ->
estimator with fit/predict (regression) or fit/predict_proba (classification).
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error

from features import build_matrix

# Validation metrics. These are byte-identical to tdc.Evaluator(name)(y, p) for the
# metrics ADMET uses (tdc dispatches roc-auc->roc_auc_score, pr-auc->
# average_precision_score, mae->mean_absolute_error, spearman->spearmanr[0]), so the
# validation numbers are unchanged. Kept local to avoid a runtime tdc dependency and
# to keep the iteration path from importing testwall (the wall).
_METRIC_FN = {
    "pr-auc": average_precision_score,
    "roc-auc": roc_auc_score,
    "mae": mean_absolute_error,
    "spearman": lambda y, p: spearmanr(y, p).correlation,
}


@dataclass(frozen=True)
class ScoreResult:
    """Frozen validation result. No test information, by construction."""
    metric_name: str
    higher_is_better: bool
    mean: float
    std: float                     # per-seed std, ddof=1 (matches iterate*.py)
    ci_halfwidth: float            # 95% t-CI half-width over seeds
    per_seed: tuple
    n_seeds: int

    @property
    def ci(self):
        return (self.mean - self.ci_halfwidth, self.mean + self.ci_halfwidth)


def _ci_halfwidth(x):
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def train_and_score(model_spec, adapter, seeds=None) -> ScoreResult:
    """Train model_spec on train, score on validation, over `seeds`.

    seeds: list of seeds, or an int count (-> 1..count), or None (->
    adapter.seed_budget). Uses adapter.load_train_valid ONLY; never touches test.
    """
    if seeds is None:
        seeds = adapter.seed_budget
    if isinstance(seeds, int):
        seeds = list(range(1, seeds + 1))

    features = model_spec.FEATURES
    build = model_spec.build
    is_clf = adapter.task_type == "classification"
    ev = _METRIC_FN[adapter.metric_name]

    scores = []
    for s in seeds:
        train, valid = adapter.load_train_valid(s)         # train/valid ONLY
        X_tr = build_matrix(train["Drug"].tolist(), features); y_tr = train["Y"].to_numpy()
        X_va = build_matrix(valid["Drug"].tolist(), features); y_va = valid["Y"].to_numpy()
        est = build(s).fit(X_tr, y_tr)
        pred = est.predict_proba(X_va)[:, 1] if is_clf else est.predict(X_va)
        scores.append(float(ev(y_va, pred)))

    x = np.array(scores)
    return ScoreResult(
        metric_name=adapter.metric_name,
        higher_is_better=adapter.higher_is_better,
        mean=float(x.mean()),
        std=float(x.std(ddof=1)),
        ci_halfwidth=_ci_halfwidth(x),
        per_seed=tuple(scores),
        n_seeds=len(x),
    )
