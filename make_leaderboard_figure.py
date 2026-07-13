"""Headline figure: leaderboard scores vs the test-set sampling error WE measured.

HONEST CLAIM this figure supports (and must not overstate):
  "The test-set sampling error we measured for this endpoint is wider than the
   gaps between leaderboard ranks, so the fine-grained ranking is within noise."

What we DO and DO NOT draw:
  - We do NOT have the leaderboard models' predictions, so we CANNOT and DO NOT
    draw a separate confidence interval per leaderboard entry.
  - We plot the leaderboard's reported scores as POINTS, and overlay ONE band
    whose width is our measured 95% bootstrap-CI width for THIS endpoint's test
    set. The message is: an interval of this measured width, laid over the board,
    overlaps most of the ranked models -- the ranks sit inside sampling noise.

Run:  uv run --extra figures python make_leaderboard_figure.py
Writes figures/leaderboard_sampling_overlap.{png,pdf}
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# 1. OUR measured number (from FINDINGS_MULTI.md -- do not hardcode by guess).
#    cyp2d6_substrate | pr-auc | test point 0.686 | test bootstrap CI [0.552, 0.804]
# ----------------------------------------------------------------------------
ENDPOINT   = "cyp2d6_substrate"          # CYP2D6-Substrate (CarbonMangels)
METRIC     = "PR-AUC"
N_TEST     = 135
CI_LO, CI_HI = 0.552, 0.804              # our measured 95% bootstrap CI (FINDINGS_MULTI.md)
HALF_WIDTH = (CI_HI - CI_LO) / 2.0       # = 0.126
FULL_WIDTH = CI_HI - CI_LO               # = 0.252
OUR_TEST_POINT = 0.686                   # for reference only (not the band center)

# ----------------------------------------------------------------------------
# 2. Leaderboard scores.  <<< PLACEHOLDER -- REPLACE WITH REAL PUBLIC VALUES >>>
#    Read the top ~8-10 entries for CYP2D6-Substrate from the TDC ADMET
#    leaderboard and paste (rank, model_name, reported_PR_AUC) below, then set
#    PLACEHOLDER = False.  Until then the figure carries a visible watermark.
#    Source to read:
#      https://tdcommons.ai/benchmark/admet_group/cyp2d6_substrate_carbonmangels/
# ----------------------------------------------------------------------------
PLACEHOLDER = False
LEADERBOARD_SOURCE = ("TDC CYP2D6_Substrate_CarbonMangels leaderboard (all AUPRC, "
                      "single metric), read 2026-07-13")
LEADERBOARD = [
    # (rank, model_name, reported_AUPRC, reported_std)  -- real public values
    (1,  "ContextPred",      0.736, 0.024),
    (2,  "DeepMol (AutoML)", 0.731, 0.037),
    (3,  "MapLight + GNN",   0.720, 0.002),
    (4,  "MapLight",         0.713, 0.009),
    (5,  "CFA",              0.704, 0.015),
    (6,  "AttrMasking",      0.704, 0.028),
    (7,  "MiniMol",          0.695, 0.032),
    (8,  "Chemprop-RDKit",   0.686, 0.031),
    (9,  "ZairaChem",        0.685, 0.029),
    (10, "RDKit2D + MLP",    0.677, 0.047),
]

# ----------------------------------------------------------------------------
# 3. Band placement + honest count.
#    Center the measured-width interval on the MEDIAN of the shown leaderboard
#    scores (reads cleaner than centering on rank 1; symmetric over the cluster).
# ----------------------------------------------------------------------------
ranks  = np.array([r for r, _, _, _ in LEADERBOARD])
names  = [n for _, n, _, _ in LEADERBOARD]
scores = np.array([s for _, _, s, _ in LEADERBOARD])
stds   = np.array([sd for _, _, _, sd in LEADERBOARD])   # leaderboard's reported seed std
mean_std = float(stds.mean())
ratio    = HALF_WIDTH / mean_std                          # how many x wider our band is

center = float(np.median(scores))
band_lo, band_hi = center - HALF_WIDTH, center + HALF_WIDTH
inside = (scores >= band_lo) & (scores <= band_hi)
n_inside = int(inside.sum())
inside_ranks = ranks[inside]
# contiguous "top N" only if the covered ranks start at 1 and are consecutive
top_run = 0
for k, rk in enumerate(sorted(ranks)):
    if inside[list(ranks).index(rk)]:
        if rk == top_run + 1:
            top_run = rk
        else:
            break
    else:
        break

# ----------------------------------------------------------------------------
# 4. Style (warm-editorial palette; readability first).
# ----------------------------------------------------------------------------
PAPER, INK, SLATE = "#FBFAF7", "#2A2620", "#6B6459"
TEAL, AMBER, LINE = "#0E7C86", "#C0733A", "#E0D9CD"
plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 15,
    "font.family": "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(12.8, 7.8))
fig.subplots_adjust(left=0.205, right=0.97, top=0.715, bottom=0.12)

# the ONE measured band (vertical stripe across all ranks)
ax.axvspan(band_lo, band_hi, color=TEAL, alpha=0.13, zorder=0)
ax.axvline(band_lo, color=TEAL, lw=1.3, alpha=0.55, zorder=1)
ax.axvline(band_hi, color=TEAL, lw=1.3, alpha=0.55, zorder=1)

# leaderboard points, each with the leaderboard's OWN reported seed std as a
# small horizontal bar.  This is their reported precision -- NOT a test-set
# bootstrap CI (we cannot compute that without their predictions).  The visual
# contrast (tiny bars vs the wide band) is the whole point.
ax.errorbar(scores, ranks, xerr=stds, fmt="none", ecolor=INK, elinewidth=2.0,
            capsize=5, capthick=2.0, zorder=3, alpha=0.9)
ax.scatter(scores, ranks, s=150, color=INK, zorder=4,
           edgecolor=PAPER, linewidth=1.5)
for r, n, s, ins in zip(ranks, names, scores, inside):
    ax.annotate(f"{s:.3f}", (s, r), xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=12, color=(AMBER if ins else SLATE))

# y axis = rank, 1 at top
ax.set_yticks(ranks)
ax.set_yticklabels([f"#{r}  {n}" for r, n in zip(ranks, names)], fontsize=13.5)
ax.invert_yaxis()
ax.set_ylim(ranks.max() + 0.7, ranks.min() - 0.7)  # tidy headroom (no top arrow)
ax.set_xlabel(f"{METRIC}  (reported leaderboard score)", fontsize=16, labelpad=10)

pad = 0.02
ax.set_xlim(min(band_lo, (scores - stds).min()) - pad,
            max(band_hi, (scores + stds).max()) + pad)
ax.grid(axis="x", color=LINE, lw=0.8, alpha=0.7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# honest count callout -- placed in the empty lower-right of the plot
if top_run >= 2:
    msg = f"ranks 1–{top_run} lie within one\nmeasured test-set sampling interval"
else:
    msg = f"{n_inside} of {len(ranks)} ranked models lie within\none measured sampling interval"
ax.text(0.975, 0.05, msg, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=13.5, color=AMBER, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFFFFF", ec=AMBER, lw=1.4))

# legend contrasting the two intervals -- compact, in the empty upper-left
# (all points sit on the right, so nothing is covered)
band_handle = Patch(facecolor=TEAL, alpha=0.16, edgecolor=TEAL,
    label=f"our measured sampling error  ({FULL_WIDTH:.3f} wide)")
pt_handle = ax.errorbar([], [], xerr=[], fmt="o", color=INK, ecolor=INK,
    markerfacecolor=INK, markeredgecolor=PAPER, markersize=9, elinewidth=2.0,
    capsize=5, capthick=2.0,
    label="leaderboard AUPRC  ±  reported seed std")
leg = ax.legend(handles=[band_handle, pt_handle], loc="upper left",
    fontsize=11.5, frameon=True, framealpha=0.97, edgecolor=LINE,
    bbox_to_anchor=(0.012, 0.985), handletextpad=0.7)
leg.get_frame().set_facecolor("#FFFFFF")

# title (wrapped) + honest subtitle
span_word = f"top {top_run}" if top_run >= 2 else f"{n_inside} of the ranked"
fig.suptitle(
    f"On {ENDPOINT}, the measured test-set sampling error\n"
    f"spans the {span_word} leaderboard entries",
    x=0.02, y=0.985, ha="left", va="top", fontsize=17.5, fontweight="bold", color=INK)
fig.text(0.02, 0.855,
    f"Wide band = the test-set sampling error WE measured (95% bootstrap CI, n={N_TEST}, half-width {HALF_WIDTH:.3f}).\n"
    f"Small bars = each model's leaderboard-reported seed std (mean ±{mean_std:.3f}) — about {ratio:.0f}× tighter than our band.\n"
    f"So the gaps between ranks sit inside sampling noise. (Seed spread vs test-set resampling; we did NOT recompute a per-model CI.)",
    ha="left", va="top", fontsize=11.5, color=SLATE, linespacing=1.35)

# watermark until real leaderboard values are filled in
if PLACEHOLDER:
    ax.text(0.5, 0.5, "PLACEHOLDER\nLEADERBOARD SCORES", transform=ax.transAxes,
            ha="center", va="center", fontsize=38, color="#C0733A", alpha=0.16,
            rotation=18, fontweight="bold", zorder=6)

png = "figures/leaderboard_sampling_overlap.png"
pdf = "figures/leaderboard_sampling_overlap.pdf"
fig.savefig(png, dpi=200)
fig.savefig(pdf)
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. Print everything for the human check.
# ----------------------------------------------------------------------------
print("=" * 74)
print("HONEST LEADERBOARD vs SAMPLING-ERROR FIGURE")
print("=" * 74)
print(f"endpoint            : {ENDPOINT}  ({METRIC}, n_test={N_TEST})")
print(f"our measured CI     : [{CI_LO}, {CI_HI}]  (from FINDINGS_MULTI.md)")
print(f"  half-width        : {HALF_WIDTH:.3f}")
print(f"  full band width   : {FULL_WIDTH:.3f}")
print(f"band center (median of shown scores): {center:.3f}")
print(f"band span           : [{band_lo:.3f}, {band_hi:.3f}]")
print(f"leaderboard source  : {LEADERBOARD_SOURCE}")
print(f"PLACEHOLDER DATA    : {PLACEHOLDER}  "
      f"{'<-- REPLACE before use; figure is watermarked' if PLACEHOLDER else ''}")
print(f"reported seed std   : mean ±{mean_std:.3f}  (our measured half-width is "
      f"~{ratio:.1f}x wider)")
print("leaderboard scores used:")
for r, n, s, sd in LEADERBOARD:
    print(f"   #{r:<2} {n:<16} {s:.3f} ± {sd:.3f}  "
          f"{'INSIDE band' if band_lo <= s <= band_hi else 'outside'}")
print(f"ranks within band   : {n_inside} of {len(ranks)}  (ranks {sorted(inside_ranks.tolist())})")
print(f"contiguous top-run  : ranks 1–{top_run}" if top_run >= 2
      else f"contiguous top-run  : none (top model not adjacent-covered)")
print(f"saved               : {png}  and  {pdf}")
print("=" * 74)
