"""Arbiter role: CI-overlap comparison, parsimony, and plateau logic.

PURE, DETERMINISTIC CODE. No model, no LLM, no randomness, no test access. It
operates only on validation ScoreResult objects (Stage 2), which carry
mean, ci_halfwidth, and higher_is_better. This is the component that holds the
project's selection discipline, so it must be exact.

Lock policy (reproduces the validated runs), two phases:
  * BEFORE any CI-separable promotion, the lock is a SOFT best-mean incumbent:
    a tied candidate with a better mean becomes the incumbent (this is why
    cyp2d6, where nothing ever separated, locked v04 = highest mean).
  * A CI-separable win (beats) COMMITS the lock. A committed lock is held
    through ties by parsimony -- a tied candidate cannot dislodge it even with a
    better mean (this is why solubility kept v02 over the better-mean, more
    complex, tied v04). Only another beats() moves a committed lock.
"""
from dataclasses import dataclass

from roles.executor import ScoreResult


# ---- primitive comparisons (direction-aware) ----
def _better_mean(a: ScoreResult, b: ScoreResult) -> bool:
    return a.mean > b.mean if a.higher_is_better else a.mean < b.mean


def ci_overlap(a: ScoreResult, b: ScoreResult) -> bool:
    """Do the two 95% CIs overlap? (Interval overlap; symmetric.)"""
    return a.ci[0] <= b.ci[1] and b.ci[0] <= a.ci[1]


def beats(a: ScoreResult, b: ScoreResult) -> bool:
    """Is `a` CI-separably better than `b`: non-overlapping CIs AND on the
    better side per the metric's direction."""
    if a.higher_is_better:
        return a.ci[0] > b.ci[1]      # a's whole CI above b's
    return a.ci[1] < b.ci[0]          # a's whole CI below b's (lower is better)


def compare(candidate: ScoreResult, current_lock: ScoreResult) -> str:
    """The significant-vs-tied verdict used in the iteration tables: 'significant'
    iff the CIs do not overlap, else 'tied'."""
    return "significant" if not ci_overlap(candidate, current_lock) else "tied"


def parsimony_decision(candidate, current_lock, candidate_complexity,
                       lock_complexity) -> str:
    """'promote' or 'keep_lock'. Promote ONLY if candidate CI-separably beats the
    lock; on a tie the SIMPLER model wins (a tie promotes only if the candidate
    is strictly simpler). Reproduces keeping v04 over the tied, more-complex
    v05/v06 compositions, and v02 over the tied v03-v06 on solubility."""
    if beats(candidate, current_lock):
        return "promote"
    if compare(candidate, current_lock) == "tied" and candidate_complexity < lock_complexity:
        return "promote"
    return "keep_lock"


# ---- plateau ----
@dataclass(frozen=True)
class PlateauResult:
    plateau_index: int          # -1 if never reached
    pre_plateau_best_index: int  # best-by-mean incumbent at the plateau point
    consecutive_at_end: int


def detect_plateau(history, k: int = 3) -> PlateauResult:
    """First index at which there have been k consecutive iterations with no
    CI-separable improvement over the running best-by-mean incumbent."""
    if not history:
        return PlateauResult(-1, -1, 0)
    best_idx, counter, plateau_at = 0, 0, -1
    best_at_plateau = 0
    for i in range(1, len(history)):
        counter = 0 if beats(history[i], history[best_idx]) else counter + 1
        if _better_mean(history[i], history[best_idx]):
            best_idx = i
        if plateau_at == -1 and counter >= k:
            plateau_at = i
            best_at_plateau = best_idx
    return PlateauResult(plateau_at, best_at_plateau if plateau_at != -1 else -1, counter)


# ---- full selection replay + loop control ----
@dataclass(frozen=True)
class LoopOutcome:
    locked_index: int
    compare_labels: tuple        # per version i>=1: 'significant'|'tied'
    lock_trajectory: tuple       # locked index after processing each version
    committed: bool
    plateau_index: int
    stopped_index: int           # last version processed under the loop knobs
    post_plateau_gain: bool      # any post-plateau result beats the pre-plateau best


def run_selection(history, k: int = 3, max_loops=None, force_full: bool = False) -> LoopOutcome:
    """Replay the selection policy over a history of ScoreResults.

    plateau ALWAYS runs and is ALWAYS recorded. force_full=False: the loop STOPS
    at the plateau. force_full=True: the loop runs to max_loops but the plateau
    point is still recorded, and post_plateau_gain flags whether any post-plateau
    result actually beats() the pre-plateau best (guards against laundering a
    post-plateau noise-max into a claimed win)."""
    plateau = detect_plateau(history, k=k)

    lock_idx, committed = 0, False
    labels, trajectory = [], [0]
    for i in range(1, len(history)):
        cand, lock = history[i], history[lock_idx]
        labels.append(compare(cand, lock))
        if beats(cand, lock):
            lock_idx, committed = i, True
        elif ci_overlap(cand, lock):          # tie
            if not committed and _better_mean(cand, lock):
                lock_idx = i                  # soft best-mean incumbent
            # committed lock: held through ties (parsimony)
        # CI-separably worse: keep lock
        trajectory.append(lock_idx)

    n = len(history)
    if not force_full and plateau.plateau_index != -1:
        stopped = plateau.plateau_index
    else:
        stopped = (min(max_loops, n) - 1) if max_loops else n - 1

    locked_at_stop = trajectory[stopped]
    post_gain = False
    if plateau.plateau_index != -1:
        best = history[plateau.pre_plateau_best_index]
        post_gain = any(beats(history[j], best)
                        for j in range(plateau.plateau_index + 1, stopped + 1))

    return LoopOutcome(
        locked_index=locked_at_stop,
        compare_labels=tuple(labels[:stopped]),
        lock_trajectory=tuple(trajectory[:stopped + 1]),
        committed=committed,
        plateau_index=plateau.plateau_index,
        stopped_index=stopped,
        post_plateau_gain=post_gain,
    )
