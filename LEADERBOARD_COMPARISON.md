# Leaderboard comparison — honest framing (three endpoints)

Reference leaderboard numbers are QUARANTINED (LEADERBOARD_PRIOR.md): used here
only to *contextualize* our result after the fact, never as a selection signal.
Our headline is always VALIDATION; the one-shot test read is a bounded
confirmation. The claim is never "we beat/matched rank N" — it is "our test
bootstrap CI is statistically indistinguishable from ranks X–Y, so the
fine-grained ranking there is not supported by the test-set sampling error."

---
## solubility_aqsoldb — locked sol_v02 (desc+LGBM), MAE lower-better

- HONEST HEADLINE (validation): **MAE 0.826 ± 0.151**  → mid-table (≈ ranks 8–9:
  RDKit2D+MLP 0.827, Basic ML 0.828). This is our generalization estimate.
- one-shot test read: **0.741, bootstrap 95% CI [0.711, 0.773]** — a FAVORABLE
  draw relative to validation (gap −0.085, within validation noise).
- Ranks whose reported mean±std overlaps our test CI [0.711, 0.773]:
  r1 MiniMol 0.741±0.013 ✓ | r2 Chemprop-RDKit 0.761±0.025 ✓ |
  r3 DeepMol 0.775±0.006 ✓ | r4 AttentiveFP 0.776±0.008 ✓ | r5 MapLight+GNN 0.789 ✗
  → statistically indistinguishable from **ranks 1–4**, separable from rank 5 down.
- SELF-AUDIT: **FLAGGED** — InChIKey identity overlap 7, NN Tanimoto max 1.000.
- FRAMING: a top-cluster (ranks 1–4) statistical **tie on a CONTAMINATED split**,
  NOT "we matched SOTA." The point coincidence (0.741 = rank-1 MiniMol) is a
  favorable test draw, not our headline; the headline is validation 0.826 and the
  split is leaky.

---
## cyp2d6_substrate — locked v04 (desc+morgan RF-balanced), AUPRC higher-better

- HONEST HEADLINE (validation): **AUPRC 0.626 ± 0.053**  → ≈ rank 12 (Chemprop 0.632).
- one-shot test read: **0.686, bootstrap 95% CI [0.552, 0.804]** (135 test mols,
  43 positives → very wide CI).
- Ranks whose point estimate falls inside our test CI [0.552, 0.804]:
  r1 ContextPred 0.736 … through … r15 NeuralFP 0.572 — **all of ranks 1–15**;
  only ranks 16–18 (≤0.498) separate.
- SELF-AUDIT: CLEAN (NN max 0.845, InChIKey 0).
- FRAMING: the ±0.12 sampling error **swallows ranks 1–15** — essentially the
  entire competitive field. The leaderboard's fine-grained top-15 ordering is NOT
  supported by the test-set sampling error. Corroborated internally: honest
  iteration could not CI-separate any of our own six versions either.

---
## caco2_wang — locked caco_v02 (desc+LGBM), MAE lower-better

- HONEST HEADLINE (validation): **MAE 0.365 ± 0.017**  → mid-table (≈ rank 15,
  MiniMol 0.350). This is our generalization estimate.
- one-shot test read: **0.267, bootstrap 95% CI [0.236, 0.297]** — but the
  valid-vs-test gap is **−0.098, FLAGGED** (test far easier than validation).
- Ranks whose point estimate falls inside our test CI [0.236, 0.297]:
  r1 CaliciBoost 0.256 … r8 DeepMol 0.297 — **ranks 1–8**; rank 9 (0.321) separates.
- SELF-AUDIT: **CLEAN of #217 duplicates** — InChIKey identity overlap 0, NN max
  0.953 (<1.0). The issue-#217 documented train/test duplicates are ABSENT in
  PyTDC 1.1.15 (version-drift: a documented bug silently patched, no changelog).
- FRAMING: the molecular split is now clean, BUT the test read is a flagged
  FAVORABLE DRAW. Our honest number is validation 0.365 (mid-table ≈ rank 15); we
  do NOT claim the top-8 that the 0.267 test draw superficially suggests. The
  second audit axis (valid-vs-test gap) catches the inflated test read that the
  molecular Tanimoto/InChIKey check cannot see. (Deeper implication: since every
  leaderboard entry shares this same easy test set, all reported caco2 numbers
  are optimistic relative to a fair held-out estimate.)

---
## Quarantine pressure log (honest)

Two moments where fetched SOTA numbers tempted the loop toward them:
1. solubility test 0.741 coincides EXACTLY with rank-1 MiniMol. Tempting to
   headline "matched SOTA." Defused by headlining validation (0.826) and flagging
   the leaky split; selection never saw the leaderboard.
2. caco2 test 0.267 lands in the rank-1–8 cluster. Tempting to claim top-tier.
   Defused by the valid-vs-test flag (−0.098) → headline validation 0.365.
Neither loop used any leaderboard number for selection or stopping; all six
caco2 versions were selected purely on our validation CIs.
