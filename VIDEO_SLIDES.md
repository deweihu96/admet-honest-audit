# 3-Minute Demo Video: Slide-by-Slide Spec

Total budget: ~3:00. Roughly 400-450 spoken words. Timing per slide is a
CEILING, not a target. If a rehearsal runs long, cut a sentence from each
slide before cutting a slide. Talk OVER the figures; don't read text off them.

Guiding rule: the video makes a judge (1) understand what we did, (2) believe
we found something real, (3) see the discipline that makes it credible. Depth
lives in the repo, not the video. Resist adding a fourth finding.

---

## SECTION 1 - ORIENTATION (~0:00-0:45)

### Slide 1 - Title + one-line what-it-is  (~0:10)
- Project name.
- One line: "An AI agent that improves drug-property prediction models
  honestly, and audits the benchmark's integrity while doing so."
- Your name / affiliation, Researcher track.
- (Optional: your face on camera for this slide only, judges connect with a
  person. If not, a clean title card.)

### Slide 2 - The problem, in plain terms  (~0:15)
- What ADMET / TDC is, in one sentence: a standard leaderboard ranking ML models
  on drug properties (solubility, toxicity, etc.).
- The problem, one sentence: a 2026 audit found most top entries fail
  reproducibility or leak test data, and overfitting the open test set is a
  reliable way to climb the ranking.
- Land the tension: "so a high leaderboard position may reflect overfitting, not
  a better model."

### Slide 3 - What we built (the overview)  (~0:20)
- One diagram or 3 labeled boxes: an agentic loop that does TWO things at once:
  CLIMBS (proposes and improves models) and AUDITS (checks every model and split
  for leakage, reports honest uncertainty).
- The one structural idea, stated simply: "the part of the system that can see
  the test set cannot influence the part that designs models, so it can't cheat,
  by construction."
- One line naming the driver: "built as an autonomous research agent with
  Claude Code."
- DO NOT go into all four roles here. One sentence of architecture, then move on.

---

## SECTION 2 - THE FINDING (~0:45-1:30)

### Slide 4 - Finding 1: honest iteration climbs only where there's signal  (~0:20)
- Show the "signal appears early" figure (Plot B): the sharp MAE drop at
  iteration 2, then flat.
- Narrate: across 10 endpoints, models measurably separated in only 3, always the
  same physically sensible move. Everywhere else they were statistically
  indistinguishable, and the agent reported that instead of inventing a winner.
- One line: "when there's real signal it shows up immediately; when there isn't,
  no amount of iterating summons it."

### Slide 5 - Finding 2: on small endpoints, the leaderboard ranks noise  (~0:25)
- THE headline figure. Show a small endpoint's test-set bootstrap CI, wide enough
  to span most of the leaderboard.
- Narrate: on the small endpoints, the honest error bar is so wide it covers the
  top ~15 models. The fine-grained ranking there is not supported by the data.
- The quotable line: "the top of the leaderboard on these endpoints is one
  statistical cluster wearing a ranking."
- (If you do the open/close-on-same-image trick: this is the transformed
  leaderboard image.)

---

## SECTION 3 - THE DISCIPLINE (~1:30-2:30)

### Slide 6 - It caught its own inflated result  (~0:30)
- The memorable moment. Show the number: 0.70 -> 0.59.
- Narrate: our agent's own model posted a test score of 0.70, competitive with
  rank one. Then it caught itself, that 0.70 was a lucky test-set draw; the honest
  value was ~0.59; and it reported the lower number and flagged the gap.
- One line: "the thing that sinks the leaderboard, a favorable test draw dressed
  as skill, our agent caught on its own output."

### Slide 7 - We stress-tested the discipline  (~0:30)
- Show the "plateau holds" figure (Plot A): 15 iterations, flat within CI bands.
- Narrate: the obvious challenge is "you only tried a few models." So we forced 15
  distinct, legitimate models past the plateau. None separated. And when the
  luckiest one looked slightly better on average, the confidence-interval rule
  correctly refused to credit it, exactly the noise-max that inflates leaderboards.
- One line: "more search didn't manufacture a win, because the discipline wouldn't
  let it."

---

## SECTION 4 - CLOSE (~2:30-3:00)

### Slide 8 - What it means + where it goes  (~0:20)
- The thesis: this isn't a better model. It's a demonstration that an AI agent can
  do the modeling work WITHOUT the overfitting that compromises the benchmark, and
  audit the benchmark's integrity as it goes.
- The vision, one line: the same pipeline, run as new models are submitted, is a
  standing integrity check on a leaderboard, not another entry gaming it. (Packages
  naturally as a reusable Claude Code tool.)

### Slide 9 - Reproducibility flourish + close  (~0:10)
- One line: "everything, the pipeline, the 10-endpoint findings, the data,
  reproduces from a fresh clone on a laptop."
- Repo link on screen.
- (Optional: close on the transformed leaderboard image from Slide 5, same picture
  you opened the finding on, now showing the statistical tie. Bookends the story.)

---

## CUT LIST (do NOT add these back; they live in the repo/writeup)
- The full four-role architecture diagram (one sentence in Slide 3 is enough).
- The full 10-endpoint results table (one endpoint's story is more vivid).
- The two-axis leakage / bbb_martins salt-form finding (great, but it's a third
  finding; no time. Mention in the writeup, not the video).
- The human-in-the-loop / read-counter machinery (fourth layer of nuance).
- The version-drift / #217 audit finding (belongs in the written submission).
- Live software running (a loop takes tens of minutes; can't show honestly in 3
  min. The figures ARE the demo.).

## ON-CAMERA HONESTY NOTES
- "In the top statistical cluster," never "we beat SOTA."
- "Honest iteration plateaus over this candidate space," never "unclimbable."
- Show CI bands on every figure; a mean line alone reads as a claim you can't make.
- Rehearse for time. If you're over 3:00, trim a sentence per slide, not a slide.
