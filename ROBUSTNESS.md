# Robustness: does forcing more iterations dissolve a plateau?

**Reviewer concern.** "You only ran 4-6 loops per endpoint; maybe more iterations
would find a separating model." This note answers that directly. It is a
**robustness demonstration, not a climb**: we do not search for better models, we
force the existing search to run past its stopping point and show the plateau holds.

## What was run

For three endpoints that plateaued in the multi-endpoint run
(`cyp2d6_substrate_carbonmangels`, `cyp2c9_substrate_carbonmangels`, `dili`, all
small classification tasks, 25-seed budget), the **unmodified** orchestrator was run
with `force_full=True, max_loops=15`. `force_full` disables the plateau's early stop
but the plateau detector still runs and its point is still recorded; every iteration
goes through the **same arbiter discipline** (CI-overlap, parsimony) as a normal run.

The proposer stream is the six model specs (`v01..v06`). With only six candidates,
the stream, not `max_loops`, is the binding cap: **each run executed 6 iterations**
(reported honestly; `max_loops=15` was never reached). The plateau fires at iteration
4 in all three, leaving iterations 5-6 as the post-plateau test.

Metric is validation only; test is never touched here. Reproduce with:

```bash
uv sync --extra figures
uv run --extra figures python robustness_forced_loops.py    # writes figures/ + JSON
```

## (a) Forcing loops past the plateau produced NO CI-separable gain

For each endpoint, `post_plateau_gain` = does **any** post-plateau iteration
`beats()` (non-overlapping 95% CI, correct side) the pre-plateau best? On all three:
**No.** Per-iteration validation means with 95% CI over seeds (see `figures/plotA_plateau_holds.png`):

### cyp2d6_substrate (pr-auc, 25 seeds, 6 iterations; plateau at it 4)
| it | spec | mean | 95% CI | post-plateau | CI-separable vs pre-plateau best |
|---|---|---|---|---|---|
| 1 | v01_desc_rf | 0.6083 | [0.5559, 0.6607] | | |
| 2 | v02_desc_lgbm | 0.5837 | [0.5224, 0.6451] | | no |
| 3 | v03_desc_morgan_rf | 0.6178 | [0.5678, 0.6678] | | no |
| 4 | v04_desc_morgan_rf_balanced | **0.6254** | [0.5733, 0.6775] | | (pre-plateau best) |
| 5 | v05_stacked_meta | 0.6109 | [0.5590, 0.6627] | yes | no |
| 6 | v06_late_fusion | 0.6198 | [0.5742, 0.6655] | yes | no |

Locked: `v04_desc_morgan_rf_balanced`. **post_plateau_gain = False.**

### cyp2c9_substrate (pr-auc, 25 seeds, 6 iterations; plateau at it 4)
| it | spec | mean | 95% CI | post-plateau | CI-separable vs pre-plateau best |
|---|---|---|---|---|---|
| 1 | v01_desc_rf | 0.4029 | [0.3585, 0.4473] | | |
| 2 | v02_desc_lgbm | 0.3867 | [0.3395, 0.4340] | | no |
| 3 | v03_desc_morgan_rf | **0.4152** | [0.3710, 0.4594] | | (pre-plateau best) |
| 4 | v04_desc_morgan_rf_balanced | 0.4149 | [0.3680, 0.4618] | | no |
| 5 | v05_stacked_meta | 0.4276 | [0.3836, 0.4716] | yes | no |
| 6 | v06_late_fusion | 0.4257 | [0.3843, 0.4671] | yes | no |

Locked: `v05_stacked_meta`. **post_plateau_gain = False.**

**Honest nuance worth stating.** Under `force_full=True` the lock moved to `v05`
(mean 0.4276) whereas the original `force_full=False` run stopped at the plateau and
locked `v04` (mean 0.4149). This is *not* evidence that more search helped: `v05`'s
CI [0.384, 0.472] overlaps `v03`'s [0.371, 0.459] almost entirely, so `post_plateau_gain`
is still False. The soft best-mean incumbent drifted by a hair of mean; the arbiter
credits a promotion only on a CI-separable win, and there is none. This is exactly the
noise-max the `post_plateau_gain` guard exists to refuse to launder into a claim. The
lock *label* changed; the *conclusion* (no real gain) did not.

### dili (roc-auc, 25 seeds, 6 iterations; plateau at it 4)
| it | spec | mean | 95% CI | post-plateau | CI-separable vs pre-plateau best |
|---|---|---|---|---|---|
| 1 | v01_desc_rf | **0.8413** | [0.8117, 0.8709] | | (pre-plateau best) |
| 2 | v02_desc_lgbm | 0.8164 | [0.7898, 0.8430] | | no |
| 3 | v03_desc_morgan_rf | 0.8397 | [0.8085, 0.8708] | | no |
| 4 | v04_desc_morgan_rf_balanced | 0.8363 | [0.8035, 0.8690] | | no |
| 5 | v05_stacked_meta | 0.8406 | [0.8134, 0.8678] | yes | no |
| 6 | v06_late_fusion | 0.8375 | [0.8078, 0.8672] | yes | no |

Locked: `v01_desc_rf`. **post_plateau_gain = False.**

In the plot, every post-plateau iteration's CI overlaps the pre-plateau best band. The
means wobble by a percent or two; the CIs say those wobbles are noise.

## (b) On endpoints that DID separate, separation appeared early

For the three endpoints that separated in the validated runs (regression, mae), the
**first CI-separable promotion** (`figures/plotB_signal_appears_early.png`):

| endpoint | metric | seeds | first separation | spec | locked |
|---|---|---|---|---|---|
| solubility_aqsoldb | mae | 5 | **iteration 2** | sol_v02_desc_lgbm | sol_v02_desc_lgbm |
| caco2_wang | mae | 25 | **iteration 2** | caco_v02_desc_lgbm | caco_v02_desc_lgbm |
| lipophilicity_astrazeneca | mae | 5 | **iteration 2** | caco_v02_desc_lgbm | caco_v02_desc_lgbm |

In all three, the real signal, the physically-motivated descriptors-beat-fingerprints
move, appears at the **first non-baseline candidate** (iteration 2): a large,
CI-separable MAE drop from the Morgan-fingerprint baseline. Nothing after iteration 2
separates further; the model that wins wins immediately.

## (c) Conclusion

- Forcing the loop past the plateau (to the full 6-candidate stream, with
  `max_loops=15` as a non-binding ceiling) produced **no CI-separable gain on any of
  the three plateaued endpoints**. The plateau is robust to loop count.
- Where signal exists, it **separates within the first couple of iterations**. A
  plateau reached after several iterations is therefore genuine, not premature: real
  gains announce themselves early, so their absence at iteration 4-6 is informative.
- Together: the headline finding "honest iteration climbs where there is real signal
  and plateaus where there is not" is **robust to how many loops we run**. More
  iterations do not summon a separation that was not there.

**Scope / what this is NOT.** This is a robustness result about *loop count*, not a
claim that the search was exhaustive or that no model could ever separate these
endpoints. The candidate stream is six specs; a fundamentally different feature space
or architecture is outside what was tested. The honest statement is narrow and exact:
within this proposer stream, running more iterations past the plateau does not change
the plateaus-vs-separates verdict on any endpoint tested. Had any forced iteration
posted a CI-separable gain, it would be reported here as a real finding that more
search helps on that endpoint; none did.
