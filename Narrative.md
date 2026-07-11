
nest AI: an agent that climbs the ADMET leaderboard without cheating, and audits it while doing so

## The question

Machine-learning leaderboards look precise. The Therapeutics Data Commons (TDC)
ADMET benchmark ranks models across 22 drug-property endpoints, and a top position
reads like evidence of a better model. But a February 2026 audit by Receptor.AI
(Nurislamov et al., "Critical Assessment of ML models for ADMET Prediction in TDC
leaderboards," bioRxiv 2026.02.26.708193) found that most top-ranked entries fail
on unavailable code, non-reproducible environments, or data leakage, and that
deliberately overfitting a model to the public test set moves it up the rankings.
On a leaderboard with a fully open test set, climbing and cheating can be the same
act.

That raised the question we set out to answer: can an AI agent iteratively improve
ADMET models *honestly*, without the test-set overfitting that compromises the
board, and what does honest iteration reveal about the benchmark itself?

## What we built

An agentic pipeline, driven by Claude Code, that does two things in one loop:

1. It **climbs**: an agent proposes model designs, trains them, and selects among
   them, iterating to improve validation performance.
2. It **audits**: the same loop checks every model and every data split for
   leakage, and reports its own results with honest uncertainty.

The design enforces honesty structurally rather than by good intentions. Four roles
are separated so that the component that touches the test set cannot influence the
component that designs models:

- a **proposer** that reasons about validation results and designs the next model;
- an **executor** that trains and scores on validation only;
- an **arbiter** (deterministic code, no model) that decides promotions by a
  confidence-interval-overlap rule, keeps simpler models on ties (parsimony), and
  detects plateaus;
- a **walled auditor** that is the only component allowed to touch test-set
  molecules, for computing train/test similarity, and that returns only a frozen
  verdict object carrying no test labels and no per-molecule data.

The test set is reachable only by the auditor (for leakage checks, never labels)
and by a single final-evaluation step, invoked once after a model is locked. The
iteration loop has no code path to the test set. The wall is a property of what
each module can import, not a promise.

Three further disciplines make the honesty real:

- **Validation is always the headline.** The single test read is reported with a
  test-set bootstrap confidence interval, and when a favorable test draw beats
  validation, that gap is flagged rather than celebrated.
- **The human is inside the loop, and subject to the same accounting as the
  agent.** A user can inject a design idea the agent did not reach, and can command
  when to test. But every test read is counted; repeated testing widens the
  reported uncertainty (a bootstrap-over-the-best correction), and adaptive testing
  is disclosed as a lower bound on true uncertainty. The tool tracks its own user's
  peeking more strictly than the leaderboard tracks anyone's.
- **The tool refuses ambition the data cannot support.** A high-novelty tier
  (bespoke architectures) is gated on test-set size: on a 43-positive endpoint, the
  tool declines, because no bespoke model could be statistically distinguished
  there.

## What we found

We ran the pipeline across 10 of the 22 ADMET endpoints. Three findings, each
demonstrated rather than asserted.

**Honest iteration climbs where there is signal and plateaus where there is not.**
In 3 of 10 endpoints, models CI-separably improved, and every one was the same
physically motivated move: physicochemical descriptors beating substructural
fingerprints on a property that descriptors essentially encode (solubility,
caco2 permeability, lipophilicity). In the other 7, the candidate models were
statistically indistinguishable, and the loop correctly reported "cannot separate
these" instead of manufacturing a winner from noise. On solubility, our honestly
selected model's test read coincides with the rank-1 leaderboard entry, but we do
not claim to have matched it: our honest estimate is the validation number, the
test read is a favorable draw, and the split itself is contaminated.

**On small endpoints, the leaderboard ranks noise.** For the small imbalanced
classification endpoints, the test-set bootstrap interval is so wide (roughly 0.25
in PR-AUC on ~135-molecule test sets) that it spans most of the leaderboard. The
fine-grained ranking of the top models on those endpoints is not supported by the
sampling error. Our own inability to separate our own models is the same fact the
leaderboard's narrow standard-deviation columns conceal.

**A documented benchmark bug drifted silently, and a single-axis check would have
missed real leakage.** The audit's own recommendation is dataset versioning; we
found concretely why. Across three endpoints with documented train/test duplicates,
two had been silently patched between data versions (with no changelog) and one
(bbb_martins) still carried the duplicates live. And the case for a two-axis
leakage check is not theoretical: on bbb_martins, two real duplicates were
salt-form pairs that structural (Tanimoto) similarity missed entirely and only a
desalted-identity (InChIKey) check caught, while on ld50 a Tanimoto collision
falsely flagged a clean split that the identity axis correctly cleared. Each axis
caught what the other missed.

## Why it matters

Drug-discovery ML is built on benchmarks like this one, and the field is
increasingly aware that leaderboard positions can reflect test-set overfitting
rather than predictive strength. Our contribution is not a better ADMET model; it
is a demonstration that an AI agent can do the iterative modeling work *without*
the overfitting, and can audit the benchmark's integrity as it goes, producing
results that are honest about what the data can and cannot support. The same
pipeline, run periodically, is a standing integrity check on a leaderboard rather
than another entry gaming it.

## How Claude got us there

Claude Code was the research agent, not a coding assistant. It ran the
iterate-audit-lock loop, held state across iterations, and enforced the test wall
by construction. What mattered most were the moments an ordinary pipeline would
have quietly cheated and this one did not: the agent caught its own ensemble
posting an inflated 0.70 that honest analysis showed was a favorable draw worth
~0.59; it stopped iterating at a pre-registered plateau rather than padding its
trail to look thorough; it flagged when reading real leaderboard scores tempted it
toward matching a published number; and its two-axis auditor independently caught
real leakage a single check would have missed. The research trail is committed to
git, one commit per model version with its design rationale, so the reasoning is
reproducible, not just the result.

## Where it goes

The pipeline is scoped to molecular (TDC-style) benchmarks now, with the leakage
check specialized to molecules. The architecture generalizes: a benchmark adapter,
a pluggable leakage interface per data type, user-selectable novelty tiers, and the
test-read accounting are all benchmark-agnostic. The natural next form is a plugin
a researcher points at any benchmark to improve a model honestly, which, run as new
models are submitted, is also a standing leaderboard auditor. One artifact, two
faces: the analysis it produced, and the reusable method it embodies.
