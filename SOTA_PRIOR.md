# SOTA prior — cyp2d6_substrate (read at loop start; do NOT web-search mid-loop)

Endpoint: CYP2D6 substrate classification. Small (~532 train_val, 135 test),
imbalanced (~28% positive), metric PR-AUC (higher-better). Scaffold split.

## Three model families and their stated limitations

1. **Fingerprints/descriptors + gradient boosting** (MapLight, CaliciBoost — the
   current fingerprint-GBM leaders). Strength: strong, cheap, robust on small
   tabular molecular data; hard to beat on many ADMET tasks. *Limitation:* fixed
   representation — cannot learn task-specific features; bag-of-substructures
   misses higher-order/mechanistic structure; sensitive to descriptor choice.

2. **Chemprop-style D-MPNN GNN** (learned message-passing representation).
   NovoExpert-2 found Chemprop v2 wins on DILI, suggesting learned
   representations help on *mechanism-driven* endpoints. *Limitation:* data-
   hungry; on small imbalanced sets it overfits and its variance balloons; the
   win is endpoint-dependent, not universal. **Offline note:** Chemprop/D-MPNN
   is not installed in this env; a small torch MLP on fingerprints is the only
   available "learned-head" stand-in, and it is a weak proxy for a true GNN.

3. **Pretrained molecular embedding + light head** (e.g. ChemBERTa/MolFormer
   frozen features + small classifier). Strength: transfers from large unlabeled
   corpora; good default when labels are scarce. *Limitation:* domain shift from
   pretraining corpus; embedding may not encode the assay-relevant chemistry.
   **Offline note:** no pretrained weights available offline in this env.

## Prior that shapes expectations here

On small imbalanced endpoints like this one, the audit literature (Receptor.AI,
Feb 2026) shows top leaderboard positions are frequently **within sampling
noise** of each other. Honest gains should be expected to **plateau early**: once
a competent fingerprint/descriptor + GBM baseline is in place, further redesigns
are unlikely to produce CI-separable validation improvements. A plateau is a
legitimate finding ("converged at validation ~X, consistent with the endpoint's
signal ceiling"), not a failure of the search.

## What "improvement" means in this loop

Selection/stopping is on VALIDATION only (test wall). An iteration "improves"
only if its validation CI is separable-above the running best (lower bound of
new CI > upper bound of best CI). Mean-only wiggles inside overlapping CIs are
noise, not progress.
