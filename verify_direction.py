"""Empirically lock metric DIRECTION to TDC's actual scorer.

For one endpoint of each metric type, build a deliberately-GOOD and a
deliberately-BAD prediction and score both with the SAME scorer TDC's
group.evaluate uses internally: Evaluator(name=<metric>)(y_true, y_pred).

WALL: no test labels are touched. y_true is taken from the VALID split (which
we are allowed to see); good/bad predictions are derived from it. The real test
set is never read here.
"""
import numpy as np
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics

# Inferred rule we are checking against.
INFERRED = {"mae": "lower-is-better", "spearman": "higher-is-better",
            "roc-auc": "higher-is-better", "pr-auc": "higher-is-better"}

# One representative endpoint per metric type.
REPS = {
    "mae": "caco2_wang",
    "spearman": "vdss_lombardo",
    "roc-auc": "pgp_broccatelli",       # ~54% positive, well balanced
    "pr-auc": "cyp2d6_substrate_carbonmangels",  # small, imbalanced
}

rng = np.random.RandomState(0)
group = admet_group(path="data/")

rows = []
for metric, endpoint in REPS.items():
    # y_true from VALID labels only (seed 1). Never test.
    _, valid = group.get_train_valid_split(benchmark=endpoint, split_type="default", seed=1)
    y_true = valid["Y"].to_numpy().astype(float)

    if metric in ("roc-auc", "pr-auc"):
        # classification: predictions are SCORES/probabilities.
        # good: score tracks the label; bad: score anti-tracks it.
        noise = rng.normal(0, 0.15, size=len(y_true))
        good_pred = np.clip(0.5 + 0.4 * (2 * y_true - 1) + 0.05 * noise, 0, 1)
        bad_pred = np.clip(0.5 - 0.4 * (2 * y_true - 1) + 0.05 * noise, 0, 1)
    else:
        # regression: good tracks y_true (small noise), bad anti-correlates.
        spread = y_true.std() + 1e-9
        good_pred = y_true + rng.normal(0, 0.05 * spread, size=len(y_true))
        bad_pred = -y_true + rng.normal(0, 0.05 * spread, size=len(y_true))

    # EXACT TDC scorer for this metric (same object group.evaluate builds).
    evaluator = Evaluator(name=metric)
    good_score = float(evaluator(y_true, good_pred))
    bad_score = float(evaluator(y_true, bad_pred))

    implied = "lower-is-better" if good_score < bad_score else "higher-is-better"
    matches = implied == INFERRED[metric]
    rows.append((metric, endpoint, good_score, bad_score, implied,
                 "YES" if matches else "NO  <-- MISMATCH"))

# ---- table ----
hdr = ("metric", "endpoint", "good_pred", "bad_pred", "implied_dir", "matches_inferred?")
w = [max(len(str(r[i])) for r in [hdr] +
         [(m, e, f"{g:.4f}", f"{b:.4f}", d, x) for (m, e, g, b, d, x) in rows])
     for i in range(len(hdr))]
print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
print("-" * (sum(w) + 2 * (len(w) - 1)))
all_ok = True
for (m, e, g, b, d, x) in rows:
    if "NO" in x:
        all_ok = False
    cells = (m, e, f"{g:.4f}", f"{b:.4f}", d, x)
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(cells)))

print()
if all_ok:
    print("ALL FOUR DIRECTIONS MATCH the inferred rule. Safe to proceed.")
else:
    print("!!! DIRECTION MISMATCH -- STOP. Do not build climb code on this map.")
