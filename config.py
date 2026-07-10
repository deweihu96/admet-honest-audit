"""Run configuration + the data-sufficiency gate.

The gate is the project's discipline made executable: 'high' novelty tier (a
bespoke architecture) is REFUSED when the test set is too small to statistically
support such a claim. It reads only the test SIZE (never labels) and estimates
minority-class support from the train prevalence, so it stays wall-clean.
"""
from dataclasses import dataclass

# High-tier thresholds. Rationale: a bespoke-architecture WIN must be provable on
# the fixed test set. cyp2d6 (135 test, ~38 minority positives) yields a test
# bootstrap CI of ~+/-0.12 -- far too wide to credit a bespoke claim -> refuse.
HIGH_TIER_MIN_TEST_SIZE = 300
HIGH_TIER_MIN_MINORITY = 100


@dataclass
class Config:
    endpoint: str
    novelty_tier: str = "middle"     # 'low' | 'middle' | 'high'
    max_loops: int = 12
    force_full: bool = False
    plateau_k: int = 3

    def check_data_sufficiency(self, adapter):
        """Return (allowed: bool, message: str). Only the 'high' tier is gated."""
        if self.novelty_tier != "high":
            return True, f"tier={self.novelty_tier}: no high-tier data-sufficiency gate"
        n_test = len(adapter.load_test_molecules())      # SIZE only, no labels
        reasons = []
        if n_test < HIGH_TIER_MIN_TEST_SIZE:
            reasons.append(f"test size {n_test} < {HIGH_TIER_MIN_TEST_SIZE}")
        if adapter.task_type == "classification":
            y = adapter.load_train_reference()["Y"].to_numpy()
            prev = float((y == 1).mean())
            est_minority = int(min(prev, 1 - prev) * n_test)
            if est_minority < HIGH_TIER_MIN_MINORITY:
                reasons.append(f"est. minority test positives ~{est_minority} "
                               f"< {HIGH_TIER_MIN_MINORITY}")
        if reasons:
            return False, "HIGH TIER REFUSED (" + "; ".join(reasons) + \
                "): test set too small to support a bespoke-architecture claim"
        return True, "high tier OK: test set large enough to support the claim"
