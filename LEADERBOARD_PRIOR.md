# Frozen leaderboard prior (fetched once from tdcommons.ai, then FROZEN)

Fetched from the TDC ADMET Group leaderboards. Do NOT re-fetch inside the
iteration loop.

## QUARANTINE RULE (load-bearing)

The TEST SCORES below are REFERENCE ONLY. They must NOT be used as a selection
or stopping signal. Selection/stopping stays on OUR validation, always. Use the
methods/features/limitations to decide WHAT to try; never steer toward a
published number. Targeting a reported test score = test leakage through the
literature. If a design choice is justified by "gets us near rank N's number,"
STOP.

Usable (methods, not numbers): the top of every ADMET-regression board here is
gradient boosting on molecular descriptors/fingerprints (CaliciBoost, XGBoost,
MapLight). MapLight = RDKit descriptors + fingerprints + GBM. Pretrained
embeddings (MiniMol) and D-MPNN GNNs (Chemprop) appear but do not dominate the
small tabular endpoints. Stated limitation of the GBM leaders: fixed
representation; of GNN/pretrained: data-hungry, variance-prone on small sets.

---
## REFERENCE SCORES -- DO NOT OPTIMIZE TOWARD

### Caco2_Wang (MAE, lower better)
1 CaliciBoost 0.256±0.006 | 2 XG Boost 0.274±0.004 | 3 MapLight 0.276±0.005 |
4 BaseBoosting 0.285±0.005 | 5 MolMapNet-D 0.287±0.005 | 6 MapLight+GNN 0.287±0.005 |
7 XGBoost 0.289±0.011 | 8 DeepMol 0.297±0.008 | 9 Basic ML 0.321±0.005 |
10 ADMETrix 0.326±0.042 | 11 Chemprop-RDKit 0.330±0.024 | ... | 15 MiniMol 0.350±0.018

### Solubility_AqSolDB (MAE, lower better)
1 MiniMol 0.741±0.013 | 2 Chemprop-RDKit 0.761±0.025 | 3 DeepMol 0.775±0.006 |
4 AttentiveFP 0.776±0.008 | 5 MapLight+GNN 0.789±0.003 | 6 MapLight 0.792±0.002 |
7 CMPNN 0.796±0.038 | 8 RDKit2D+MLP 0.827±0.047 | 9 Basic ML 0.828±0.002 |
10 Chemprop 0.829±0.022 | ... | 18 Morgan+MLP 1.203±0.019

### CYP2D6_Substrate_CarbonMangels (AUPRC, higher better)
1 ContextPred 0.736±0.024 | 2 DeepMol 0.731±0.037 | 3 MapLight+GNN 0.720±0.002 |
4 MapLight 0.713±0.009 | 5 CFA 0.704±0.015 | 6 AttrMasking 0.704±0.028 |
7 MiniMol 0.695±0.032 | 8 Chemprop-RDKit 0.686±0.031 | 9 ZairaChem 0.685±0.029 |
10 RDKit2D+MLP 0.677±0.047 | 11 Morgan+MLP 0.671±0.066 | ... | 15 NeuralFP 0.572±0.062 |
16 Euclia 0.498±0.015 | 17 CNN 0.485±0.037 | 18 Basic ML 0.478±0.018

---
## Design implications (methods only, no number-chasing)

- Caco2 & solubility: family-1 (descriptor/fingerprint + GBM) tops the board and
  is what we build. For caco2, permeability tracks lipophilicity/PSA -> RDKit
  descriptors should carry signal, mirroring the solubility finding.
- The leaderboard std columns are ~0.002-0.05 and were computed on the SAME fixed
  test set every entry shares; several top entries are within each other's std.
  That, combined with our own bootstrap CIs, is the basis of the "ranking within
  sampling noise" claim -- not any comparison of our number to theirs for
  selection.
