# 3-Minute Demo Voiceover Script

Word-for-word narration matched to `video_deck.html` (9 slides). Delivery cues are
in [brackets] and are not spoken. Lines marked [slow] are the beats to land.
Total budget: <= 450 spoken words, <= 180 seconds.

---

**SLIDE 1 (10s) — Honest AI on the ADMET leaderboard**

[advance slide] This is Honest AI on the ADMET leaderboard. It's an AI agent that improves drug-property prediction models honestly, and audits the benchmark's integrity while it does.

---

**SLIDE 2 (17s) — Leaderboards look precise. Often they aren't.**

TDC ADMET ranks machine-learning models on drug properties. Leaderboards look precise, but a 2026 audit found most top entries fail reproducibility or leak test data, and overfitting the open test set climbs the board. So a high rank may reflect overfitting, not a better model.

---

**SLIDE 3 (17s) — One agentic loop that climbs and audits, the test set walled off**

[advance slide] We built one agentic loop that climbs and audits: it improves models, and checks every split for leakage. The key idea is structural. The part that can see the test set cannot influence the part that designs models. It can't cheat by construction. Built with Claude Code.

---

**SLIDE 4 (24s) — Where there's signal, it shows up at once**

The first finding. Across ten endpoints, models separated in only three, always the same sensible move: better descriptors beating fingerprints. When it did, it showed up immediately, at the second attempt, then went flat. Everywhere else they were statistically indistinguishable, and the agent said so instead of inventing a winner. Where there's signal, it shows up at once. Iterating doesn't summon it.

---

**SLIDE 5 (28s) — One statistical cluster, wearing a ranking**

The second finding is the headline. On a small endpoint, we measured the test-set sampling error directly, the spread you get from resampling a test set this small. That single interval is wide enough to cover the top ten leaderboard entries, while the leaderboard's own reported spread is about five times tighter. So the fine-grained ranking at the top sits inside sampling noise. [slow] It's one statistical cluster, wearing a ranking.

---

**SLIDE 6 (28s) — It caught its own inflated result**

[slow] Now, the discipline. Our agent's own model posted a test score of zero point seven zero, competitive with rank one. Then it caught itself. That number was a favorable test-set draw, not skill. The honest generalization estimate was about [slow] zero point five nine. It reported the lower number and flagged the gap. The exact move that inflates leaderboards, caught on its own output.

---

**SLIDE 7 (25s) — We tried to break it. It held.**

The obvious challenge is, you only tried a few models. So we tried to break it. We forced fifteen distinct, legitimate models past the plateau. None separated. When the luckiest one looked slightly better on average, the confidence-interval rule refused to credit it, correctly, as noise. Honest iteration plateaus over this candidate space. More search didn't manufacture a win, because the discipline wouldn't let it.

---

**SLIDE 8 (16s) — Not a better model, a way to model without fooling yourself**

This isn't a better model. It's a way to model without fooling yourself, and to audit the benchmark as you go. The vision: run it as new models arrive, and it becomes a standing integrity check on a leaderboard, not another entry gaming it.

---

**SLIDE 9 (11s) — Everything reproduces from a fresh clone**

[advance slide] And everything reproduces from a fresh clone. [slow] The pipeline, the findings, and the data, clone, install, and run on a standard laptop. It's all in the repo.

---

**Budget (verified):** 449 spoken words (<= 450), 176 seconds (<= 180). Per-slide
target seconds sum to 176s, leaving ~4s of headroom for the [slow] beats and slide
advances. Pacing runs 2.2–2.8 words/second; if a rehearsal runs long, cut one
sentence from slide 4 or 7 before touching a whole slide.
