"""TestReadLog: the test-read ledger with the bootstrap-the-max selection
correction and the adaptive-disclosure flag.

Lives INSIDE the wall (testwall/): it holds test labels/predictions only to
compute the correction, and returns ONLY a summary ReadSummary. Every test read
goes through here. When multiple reads reuse the same fixed test set, the
'corrected' CI bootstraps the SELECTED (best-per-direction) statistic across all
session reads -- so taking several looks at the reused test set cannot launder an
optimistic max into a tight CI. Validation always stays the headline.
"""
from dataclasses import dataclass

import numpy as np
from numpy.random import RandomState
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error

METRIC_FN = {
    "pr-auc": average_precision_score,
    "roc-auc": roc_auc_score,
    "mae": mean_absolute_error,
    "spearman": lambda y, p: spearmanr(y, p).correlation,
}
CLASSIFICATION = {"pr-auc", "roc-auc"}


@dataclass(frozen=True)
class ReadSummary:
    endpoint: str
    read_number: int
    adaptive: bool
    validation_headline: str
    test_point: float
    bootstrap_ci: tuple           # this read's own 95% CI
    corrected_ci: tuple           # bootstrap-max over all session reads (reused test)
    disclosure: str


def _bootstrap(y, preds, metric_fn, is_clf, higher, n=5000, seed=0):
    rng = RandomState(seed); N = len(y); out = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if is_clf and not (0 < y[idx].sum() < len(idx)):
            continue
        vals = [metric_fn(y[idx], p[idx]) for p in preds]
        out.append(max(vals) if higher else min(vals))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


class TestReadLog:
    def __init__(self):
        self._reads = {}                 # endpoint -> list of (y, pred)
        self._adaptive_pending = False

    def mark_adaptive(self):
        """Flag that the NEXT read is adaptive (e.g. an idea was injected after a
        prior test read on the same reused test set)."""
        self._adaptive_pending = True

    def n_reads(self, endpoint):
        return len(self._reads.get(endpoint, []))

    def record_read(self, endpoint, y_true, pred, metric_name, higher_is_better,
                    validation_headline) -> ReadSummary:
        metric_fn = METRIC_FN[metric_name]
        is_clf = metric_name in CLASSIFICATION
        self._reads.setdefault(endpoint, []).append((np.asarray(y_true), np.asarray(pred)))
        reads = self._reads[endpoint]
        n = len(reads)
        adaptive = self._adaptive_pending or n > 1
        self._adaptive_pending = False

        point = float(metric_fn(y_true, pred))
        own_ci = _bootstrap(y_true, [pred], metric_fn, is_clf, higher_is_better)
        corrected = _bootstrap(reads[0][0], [p for _, p in reads],
                               metric_fn, is_clf, higher_is_better)

        disc = (f"Test read #{n} on {endpoint} (fixed test set"
                f"{', REUSED' if n > 1 else ''}). "
                f"{'ADAPTIVE: an idea entered after a prior read -> selection-corrected. ' if adaptive else ''}"
                f"Corrected CI is the bootstrap-max over {n} session read(s). "
                f"Validation ({validation_headline}) remains the headline; this "
                f"test read is a bounded one-shot confirmation, not the headline.")
        return ReadSummary(endpoint=endpoint, read_number=n, adaptive=adaptive,
                           validation_headline=validation_headline, test_point=point,
                           bootstrap_ci=own_ci, corrected_ci=corrected, disclosure=disc)
