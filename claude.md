LAUDE.md

Standing rules for this project. Read this at the start of every session. These
are invariants, not suggestions. If a task in a session prompt appears to
conflict with anything here, stop and flag it rather than proceeding.

## What this project is

An agentic ML project on the TDC (Therapeutics Data Commons) ADMET benchmark
group. The agent does two things in ONE loop:

1. **Climb**: propose and train models to score well on ADMET endpoints.
2. **Self-audit**: check every candidate for data leakage, so the climb is
   honest by construction.

Motivation: a Feb 2026 audit (Receptor.AI, bioRxiv) found widespread data
leakage and non-reproducibility in top TDC ADMET leaderboard entries. On this
board, naive climbing and cheating are nearly synonymous. Our contribution is an
agent that climbs *without* tripping the leakage traps that compromised the
board, and reports the honest-methods ceiling per endpoint.

## The load-bearing rule: the train / valid / test wall

This is the single most important rule in the project. Most of the audited
leaderboard failures come from breaking it.

- The agent optimizes on VALIDATION only. Every model attempt trains on `train`,
  is scored on `valid`. Selection among candidates uses validation only.
- The TEST set is touched exactly ONCE per endpoint, at the very end, to score
  the final chosen model (or ensemble). Never for tuning, never for selection,
  never for "just checking".
- If you are ever tempted to read, score against, or select on the test set
  during the search, STOP and flag it. Do not do it "helpfully".
- The official TDC protocol uses 5 seeds. `get_train_valid_split` reshuffles
  train/valid per seed while test stays fixed. Select on the MEAN over seeds,
  never cherry-pick the best seed. Penalize high-variance candidates rather than
  rewarding lucky ones; many ADMET tasks have <100 positives and reward noise.

## The self-audit runs on the DATA SPLIT, not on the agent's choices

The leakage check is a property of the train/test split, which the agent cannot
game through model selection. This is what keeps "climb + self-audit" honest
rather than grading its own homework.

Leakage check spec (use exactly this):
- Compute NEAREST-NEIGHBOR Tanimoto similarity: for each TEST molecule, its
  single most similar TRAIN molecule. Morgan fingerprints, radius 3, 2048 bits.
- Radius 3 (ECFP6) here is INTENTIONAL and separate from the modeling feature
  radius (2 / ECFP4). Leakage uses the audit spec's larger radius for a
  stricter near-duplicate check; modeling uses the standard radius. Two
  different radii on purpose, not an inconsistency.
- The reference set is an EXPLICIT choice, and one number must not serve two
  roles: the SPLIT audit ("is TDC's split clean?") uses `train_val` (seed-
  independent); MODEL-attribution ("did this model gain from overlap with what
  IT saw?") uses that seed's `train`. Default reported check is the split audit.
- Report median and max nearest-neighbor similarity across the test set.
- Do NOT use average similarity to the whole training set. Averages hide
  near-duplicates; the per-test-molecule nearest neighbor is the entire
  sensitivity of the check.
- A candidate that scores well *because of* high train/test overlap is flagged
  and down-weighted, not selected.
- Also flag scores that exceed the plausible experimental-noise ceiling of an
  assay: beating the noise floor is evidence of leakage, not skill.

## Reproducibility is a graded verdict, not a binary

TDC has no dataset versioning or checksums, so even clean baselines do not
reproduce exactly (the audit saw drift in known-good models). Report
reproducibility as: runs-and-matches / runs-but-differs / does-not-run. "Cannot
be exactly reproduced" is a valid finding. Do NOT report benign dataset drift as
fraud.

## Environment: use `uv`

We use **uv** for all Python environment and dependency management. Do not use
conda, venv, or bare pip.

- Create/activate the project env with uv. Pin Python explicitly (3.10+).
- Install packages with `uv add <pkg>` (or `uv pip install` inside the env).
- Run scripts with `uv run <script.py>` so the resolved env is always used.
- Keep `pyproject.toml` / `uv.lock` under version control; the lockfile is part
  of reproducibility and we hold ourselves to the standard we audit others by.
- Core deps: `PyTDC`, `rdkit`, `scikit-learn`, plus a gradient-boosting library
  (xgboost/lightgbm) and, later, torch + a molecular-embedding stack.
- If a package is missing, tell me; do not web-search mid-run to work around it.

## Compute: Esrum HPC (Slurm)

- Cluster: 1 head, 12 compute, 1 A100 node (2x A100 80GB), 3 H100 nodes
  (2x H100 80GB each). Wide open over summer.
- Request single GPUs (`--gres=gpu:1`) and let Slurm bin-pack; do NOT grab whole
  nodes. Parallelize ACROSS the 22 endpoints as a job array, not within one.
- Per-attempt wall-clock cap ~40 min by default (high hard ceiling as a runaway
  guard). A timeout counts as a failed attempt. The cap is about avoiding
  rabbit holes, not saving compute; compute is not the constraint here.
- Spend the surplus compute on RESAMPLING (more seeds for final re-scoring), not
  on longer single runs. Tight variance estimates are what make "it generalizes"
  credible on small, imbalanced ADMET tasks.

## Prior knowledge the agent may use

- The three audited-CLEAN baselines are CaliciBoost, MapLight, MapLight+GNN.
  Use these as honest-baseline references and starting priors.
- Known model families to explore: (1) fingerprints/descriptors + gradient
  boosting; (2) Chemprop-style supervised GNN; (3) frozen pretrained molecular
  embedding + small head (the lightest and a good default to explore).
- The prior seeds WHAT to try. It must NOT be a channel for test leakage: do not
  steer toward configurations because they match a paper's reported TEST score.
  Published numbers came from the same fixed test set; matching them is indirect
  test optimization. Select on OUR validation only.

## Working style

- Empirical and skeptical. Show numbers and code. Prefer strong simple baselines
  before complex ones. No overclaiming: a mid-table honest result is a real
  result and must be reported as-is.
- Avoid em-dashes in written output.
- End substantial sessions by stating what must be verified before the next step.
