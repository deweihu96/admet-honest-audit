# How did you use Claude? (submission field)

## Short version (if the field is tight)

Claude Code was the research agent, not a coding assistant. It ran the entire
iterate-audit-lock loop: proposing model designs, reasoning about validation
results, and building the four-role pipeline that enforces the test wall
structurally. What mattered most were the moments an ordinary pipeline would have
quietly cheated and this one did not. Claude Code caught its own ensemble posting an
inflated 0.70 that honest analysis showed was a favorable test-set draw worth ~0.59.
It stopped iterating at a pre-registered plateau rather than padding its trail to
look thorough. When we forced 15 model attempts to test robustness, its
confidence-interval discipline correctly refused to credit the luckiest high-mean
model as a real gain, exactly the noise-max the guard exists to catch. And its
two-axis auditor independently caught real train/test leakage that a single
similarity check missed. The full research trail is committed to git, one commit per
model version with its rationale, so the reasoning is reproducible, not just the
numbers.

## Full version (if the field allows more)

**Which products.** Primarily **Claude Code**, used as an autonomous research agent
across the whole project, on an HPC cluster and locally on macOS. We also used Claude
in this chat interface for research design, methodology pressure-testing, and drafting.

**Where Claude Code mattered most.** This project is about doing machine-learning
science *honestly*, and the interesting thing is that Claude Code held that
discipline at the exact points where cutting a corner would have been easy and
invisible. Concretely:

- **It caught its own inflated result.** Early on, an ensemble posted a test PR-AUC
  of 0.70, competitive with the leaderboard. Rather than take the win, Claude Code
  diagnosed it: the ensemble added no validation lift, the same model beat its own
  validation by a systematic margin only on the fixed test set, and a test-set
  bootstrap put the honest value near 0.59. It reported the 0.70 as a favorable draw,
  not a result. This is the project's whole thesis, caught on its own output.

- **It refused to pad the search.** On a plateaued endpoint, it stopped at a
  pre-registered stopping rule rather than running more iterations to look thorough,
  explicitly noting that extending past the rule would be a form of search-length
  p-hacking.

- **Its discipline caught a noise-max under pressure.** When we forced 15 distinct
  model attempts on small endpoints to test whether more search would separate them,
  the best-mean model looked like a small improvement, but its confidence interval
  overlapped the incumbent's, so the deterministic arbiter refused to credit it. That
  is exactly the kind of "more tries found something better" illusion that inflates
  leaderboards, and the discipline held.

- **It built the test wall as structure, not intention.** Claude Code implemented the
  four-role separation so that the component allowed to touch the test set (the
  auditor, molecules only, returning a summary verdict with no labels) cannot pass
  information to the component that designs models. The wall is enforced by what each
  module can import, verified by grep, not by the agent choosing to behave.

- **Its auditor caught real leakage a single check would miss.** On one endpoint, two
  genuine train/test duplicates were salt-form pairs that structural similarity missed
  entirely and only a desalted-identity check caught; on another, a structural false
  positive was correctly cleared. The two-axis design earned its place on real data.

- **It fixed a broken candidate honestly rather than shipping a strawman.** During the
  robustness experiment, a linear model blew up on an outlier molecular descriptor.
  Claude Code diagnosed the cause and fixed it properly (rank-based feature
  transformation) rather than letting a broken candidate make the honest conclusion
  look stronger than it was.

**Reproducibility.** The research trail is the deliverable as much as the result: one
git commit per model version, each carrying the design change and its
validation-grounded rationale, so a reviewer can read how the reasoning evolved.
Claude Code also made the package portable, when it failed to install on macOS
because a transitive dependency dragged in an unused GPU/deep-learning tree, it
diagnosed the chain, made that dependency optional, and verified the whole pipeline
reproduces from a fresh clone on a standard laptop.

**The honest summary.** We did not use Claude to write code faster. We used it to do a
piece of science where the hard part was *not fooling ourselves*, and the most
valuable thing it did was repeatedly decline to.
