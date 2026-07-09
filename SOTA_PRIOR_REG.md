# SOTA prior — solubility_aqsoldb (regression; read at loop start)

Endpoint: aqueous solubility (logS), AqSolDB. LARGE (~7985 train_val, 1997 test),
real-valued, metric MAE (LOWER is better). Scaffold split. 5-seed budget.

Direction (harness-verified, do NOT invert from the PR-AUC runs): MAE is an error
metric, so LOWER is better and "A beats B" means A's MAE is lower AND their
validation CIs do not overlap.

## Families and limitations (same three as before, regression framing)

1. **Descriptors/fingerprints + gradient boosting** (MapLight, CaliciBoost). For
   solubility this is especially strong: aqueous solubility is largely driven by
   physicochemical descriptors (logP, TPSA, MW, H-bonding, aromaticity), which
   RDKit 2D descriptors capture directly. Expect descriptor features to carry
   real signal here. *Limitation:* fixed representation; misses effects not
   encoded by the chosen descriptors (e.g. solid-state/crystal packing).

2. **Chemprop-style D-MPNN** — learned representations can help on large
   regression sets. *Limitation:* not installed offline; a torch MLP on features
   is only a weak stand-in and not worth it when descriptor-GBM is this strong.

3. **Pretrained embedding + light head** — no offline weights available.

## Expectation for THIS run (opposite of cyp2d6)

This endpoint has thousands of molecules and real signal, so models SHOULD
separate: a descriptor-based learner should CI-separably beat a fingerprint-only
one, and a well-tuned/richer model should separate from a weak baseline. A
genuine CI-separable improvement is the EXPECTED, healthy outcome and should
trigger a promotion (re-lock). If nothing ever separates across six iterations,
that is a WARNING (candidate space too narrow or a variance problem), not a
success -- flag it.

Starting recipe grounded in family 1: a fingerprint-GBM baseline (LightGBM on
Morgan r2), then move toward descriptors, which theory says should win for
solubility.
