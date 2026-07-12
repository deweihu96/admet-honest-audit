# Robustness: does a broader, forced search dissolve a plateau?

**Reviewer concern.** "You only ran 4-6 loops per endpoint; maybe more model
attempts would find a separating model." This note answers that directly and at
strength. It is a **robustness demonstration, not a climb**: we do not search for a
winner, we force a broad, honest search to run past its stopping point and show the
plateau holds. Every candidate is a legitimate modeling choice; every one goes
through the **same arbiter discipline** (CI-overlap, parsimony).

## The candidate stream: 15 legitimate models

The proposer stream was expanded from 6 to **15 distinct specs**, each a defensible
design a real modeler would try, spanning the standard levers:

| # | spec | lever |
|---|---|---|
| 1 | v01_desc_rf | descriptors + RandomForest (baseline) |
| 2 | v02_desc_lgbm | descriptors + LightGBM (boosting vs bagging) |
| 3 | v03_desc_morgan_rf | + Morgan r2/ECFP4 union |
| 4 | v04_desc_morgan_rf_balanced | + class-weight imbalance handling |
| 5 | v05_stacked_meta | stacking (RF/RF/LGBM heads + LogReg meta) |
| 6 | v06_late_fusion | late fusion, validated blend weight |
| 7 | v07_morgan_lgbm | fingerprint-only boosting |
| 8 | v08_morgan_r3_rf_balanced | **radius**: ECFP6 (r3) |
| 9 | v09_maccs_rf_balanced | **fingerprint type**: MACCS keys |
| 10 | v10_desc_morgan_r3_rf_balanced | descriptor + ECFP6 union |
| 11 | v11_desc_morgan1024_lgbm | **bit size**: 1024-bit ECFP4 |
| 12 | v12_desc_morgan_extratrees_balanced | **base learner**: ExtraTrees |
| 13 | v13_desc_morgan_histgb | **base learner**: HistGradientBoosting |
| 14 | v14_desc_morgan_lgbm_balanced | imbalance-aware boosting (scale_pos_weight) |
| 15 | v15_desc_logreg_balanced | **hypothesis class**: regularized linear (QuantileTransformer + LogReg) |

Run on the three endpoints that plateaued in the multi-endpoint run
(`cyp2d6_substrate_carbonmangels`, `cyp2c9_substrate_carbonmangels`, `dili`; small
classification, 25-seed budget) with the **unmodified** orchestrator,
`force_full=True` and a **non-binding** `max_loops=20`. The 15-candidate stream is
the real iteration count (not the cap): **all 15 ran** on each endpoint. Validation
only; test is never touched. Wall-clock on this Mac (arm64, 25 seeds): ~56 min for
the full run (3 plateaued + solubility positive control).

```bash
uv sync --extra figures
uv run --extra figures python robustness_forced_loops.py    # writes figures/ + JSON
```

## (a) 15 forced iterations produced NO CI-separable gain on any endpoint

For each endpoint, `post_plateau_gain` asks: does **any** post-plateau iteration
`beats()` (non-overlapping 95% CI, correct side) the pre-plateau best? On all three:
**No.** Plateau fires at iteration 4 each time, so iterations 5-15 (eleven further
models) are the post-plateau test. Full per-iteration trajectories
(`figures/plotA_plateau_holds.png`, CI bands shown):

#### cyp2d6_substrate (pr-auc, 25 seeds, 15 iterations; plateau at it 4)
| it | spec | mean | 95% CI | post-plateau | CI-sep vs pre-plateau best |
|---|---|---|---|---|---|
| 1 | v01_desc_rf | 0.6083 | [0.5559, 0.6607] | | no |
| 2 | v02_desc_lgbm | 0.5837 | [0.5224, 0.6451] | | no |
| 3 | v03_desc_morgan_rf | 0.6178 | [0.5678, 0.6678] | | no |
| 4 | v04_desc_morgan_rf_balanced | **0.6254** (pre-plateau best) | [0.5733, 0.6775] | | |
| 5 | v05_stacked_meta | 0.6109 | [0.5590, 0.6627] | yes | no |
| 6 | v06_late_fusion | 0.6198 | [0.5742, 0.6655] | yes | no |
| 7 | v07_morgan_lgbm | 0.5033 | [0.4628, 0.5439] | yes | no |
| 8 | v08_morgan_r3_rf_balanced | 0.5851 | [0.5456, 0.6246] | yes | no |
| 9 | v09_maccs_rf_balanced | 0.6026 | [0.5545, 0.6507] | yes | no |
| 10 | v10_desc_morgan_r3_rf_balanced | 0.6268 | [0.5757, 0.6780] | yes | no |
| 11 | v11_desc_morgan1024_lgbm | 0.5753 | [0.5175, 0.6332] | yes | no |
| 12 | v12_desc_morgan_extratrees_balanced | 0.6409 | [0.5934, 0.6884] | yes | no |
| 13 | v13_desc_morgan_histgb | 0.5985 | [0.5450, 0.6519] | yes | no |
| 14 | v14_desc_morgan_lgbm_balanced | 0.5870 | [0.5301, 0.6439] | yes | no |
| 15 | v15_desc_logreg_balanced | 0.4596 | [0.4087, 0.5104] | yes | no |

**post_plateau_gain = False.** Lock drifted to v12 (highest mean 0.6409).

#### cyp2c9_substrate (pr-auc, 25 seeds, 15 iterations; plateau at it 4)
| it | spec | mean | 95% CI | post-plateau | CI-sep vs pre-plateau best |
|---|---|---|---|---|---|
| 1 | v01_desc_rf | 0.4029 | [0.3585, 0.4473] | | no |
| 2 | v02_desc_lgbm | 0.3867 | [0.3395, 0.4340] | | no |
| 3 | v03_desc_morgan_rf | **0.4152** (pre-plateau best) | [0.3710, 0.4594] | | |
| 4 | v04_desc_morgan_rf_balanced | 0.4149 | [0.3680, 0.4618] | | no |
| 5 | v05_stacked_meta | 0.4276 | [0.3836, 0.4716] | yes | no |
| 6 | v06_late_fusion | 0.4257 | [0.3843, 0.4671] | yes | no |
| 7 | v07_morgan_lgbm | 0.4078 | [0.3597, 0.4559] | yes | no |
| 8 | v08_morgan_r3_rf_balanced | 0.4311 | [0.3861, 0.4761] | yes | no |
| 9 | v09_maccs_rf_balanced | 0.4049 | [0.3537, 0.4561] | yes | no |
| 10 | v10_desc_morgan_r3_rf_balanced | 0.4183 | [0.3683, 0.4682] | yes | no |
| 11 | v11_desc_morgan1024_lgbm | 0.3904 | [0.3439, 0.4369] | yes | no |
| 12 | v12_desc_morgan_extratrees_balanced | 0.4354 | [0.3909, 0.4799] | yes | no |
| 13 | v13_desc_morgan_histgb | 0.4029 | [0.3557, 0.4501] | yes | no |
| 14 | v14_desc_morgan_lgbm_balanced | 0.3875 | [0.3388, 0.4361] | yes | no |
| 15 | v15_desc_logreg_balanced | 0.3415 | [0.2906, 0.3925] | yes | no |

**post_plateau_gain = False.** Lock drifted to v12 (highest mean 0.4354).

#### dili (roc-auc, 25 seeds, 15 iterations; plateau at it 4)
| it | spec | mean | 95% CI | post-plateau | CI-sep vs pre-plateau best |
|---|---|---|---|---|---|
| 1 | v01_desc_rf | **0.8413** (pre-plateau best) | [0.8117, 0.8709] | | |
| 2 | v02_desc_lgbm | 0.8164 | [0.7898, 0.8430] | | no |
| 3 | v03_desc_morgan_rf | 0.8397 | [0.8085, 0.8708] | | no |
| 4 | v04_desc_morgan_rf_balanced | 0.8363 | [0.8035, 0.8690] | | no |
| 5 | v05_stacked_meta | 0.8406 | [0.8134, 0.8678] | yes | no |
| 6 | v06_late_fusion | 0.8375 | [0.8078, 0.8672] | yes | no |
| 7 | v07_morgan_lgbm | 0.7711 | [0.7339, 0.8083] | yes | no |
| 8 | v08_morgan_r3_rf_balanced | 0.7951 | [0.7632, 0.8270] | yes | no |
| 9 | v09_maccs_rf_balanced | 0.8512 | [0.8298, 0.8726] | yes | no |
| 10 | v10_desc_morgan_r3_rf_balanced | 0.8364 | [0.8033, 0.8694] | yes | no |
| 11 | v11_desc_morgan1024_lgbm | 0.8261 | [0.8014, 0.8507] | yes | no |
| 12 | v12_desc_morgan_extratrees_balanced | 0.8381 | [0.8084, 0.8679] | yes | no |
| 13 | v13_desc_morgan_histgb | 0.8107 | [0.7823, 0.8392] | yes | no |
| 14 | v14_desc_morgan_lgbm_balanced | 0.8225 | [0.7941, 0.8509] | yes | no |
| 15 | v15_desc_logreg_balanced | 0.7487 | [0.7196, 0.7779] | yes | no |

**post_plateau_gain = False.** Lock drifted to v09 (highest mean 0.8512).

### The critical honesty point: a higher mean is not a gain

On all three endpoints the committed lock moved to a post-plateau candidate with the
**highest mean** (v12 on cyp2d6 0.6409 and cyp2c9 0.4354; v09/MACCS on dili 0.8512).
A naive "highest mean wins" reading would call these improvements from the broader
search. They are not:

- cyp2d6 v12 CI **[0.593, 0.688]** overlaps the pre-plateau best v04 **[0.573, 0.678]** almost entirely.
- dili v09 CI **[0.830, 0.873]** overlaps v01 **[0.812, 0.871]**.

`beats()` is False in every case, so `post_plateau_gain = False`. The soft best-mean
incumbent drifts to the luckiest noise draw among 15 tries; the arbiter credits a
promotion only on a CI-separable win, and there is none. This is exactly the
noise-max the `post_plateau_gain` guard exists to refuse to launder into a claim.
With 15 tries and reward noise on ~35-90 minority positives, some candidate will post
the highest mean by chance the discipline correctly declines to call it signal. The
plot shows every one of those CIs overlapping the pre-plateau-best band.

## (b) Positive control: solubility separates early and holds over 15 candidates

The same expansion was applied to `solubility_aqsoldb` (which DID separate), with a
matching **15-spec regression stream** (`sol_v01..sol_v15`, same levers: radius,
bit-size, MACCS, RF/ExtraTrees/HistGB, robust Ridge, higher-capacity LGBM). Result
(`figures/plotB_signal_appears_early.png`):

- **First CI-separable promotion at iteration 2** (`sol_v02_desc_lgbm`): the
  descriptors-beat-fingerprints move, MAE 1.261 -> 0.826 with non-overlapping CIs.
- **Locked `sol_v02` through all 15 iterations** none of the 13 later candidates
  (including higher-capacity deep LGBM, RF, ExtraTrees, HistGB, robust Ridge) beats
  it. The descriptor models cluster at MAE ~0.82-0.85; the fingerprint-only models
  (sol_v01 morgan 1.26, sol_v07 ECFP6 1.28, sol_v08 MACCS 1.10) sit clearly worse.

So the expanded stream behaves correctly on **both** regimes: it finds the real
separation immediately where one exists, and it does not manufacture one where none
does. The extra candidates do not spuriously beat the descriptor model.

## (c) Conclusion, correctly scoped

- With **15 genuinely different, legitimate candidate models** forced past the
  plateau, **no CI-separable gain appeared on any of the three small endpoints**. The
  plateau is robust to a much broader search, not just to loop count.
- On a signal-bearing endpoint (solubility), separation still appears at **iteration
  2** and holds: real signal announces itself early and the extra models do not
  overturn it.
- Together, the headline finding "honest iteration climbs where there is real signal
  and plateaus where there is not" is **robust to the breadth of the search**, not an
  artifact of trying too few models.

**Scope / what this is NOT.** This is a result about the **breadth of this candidate
stream** fingerprints (Morgan r2/r3, 1024/2048-bit, MACCS), descriptors and their
unions, tree ensembles (RF, ExtraTrees), gradient boosting (LightGBM,
HistGradientBoosting), regularized linear models, and stacking/late-fusion
compositions. It is **not** a claim that the search was exhaustive, nor that a
**fundamentally different representation could not separate these endpoints**. In
particular, learned representations that were **not tested here** a supervised
message-passing GNN (Chemprop-style) or a frozen pretrained molecular embedding with
a small head could in principle find signal this stream does not. The honest,
narrow statement: across a broad sweep of the standard fingerprint/descriptor/
tree/boosting/linear/composition space, forcing the search well past its plateau does
not produce a CI-separable gain on cyp2d6 / cyp2c9 / dili. Had any forced iteration
posted one, it would be reported here as a real finding that broader search helps on
that endpoint; none did.
