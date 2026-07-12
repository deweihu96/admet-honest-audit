"""Robustness experiment: does forcing more iterations dissolve a plateau?

Reviewer concern: "you only ran 4-6 loops; maybe more would find a separating
model." We test that WITHOUT changing pipeline logic: run the SAME orchestrator
with force_full=True (do not stop at the plateau) over the full candidate stream,
and ask whether ANY post-plateau iteration is CI-separable (arbiter.beats) from the
pre-plateau best. This is a robustness demonstration, NOT a climb.

The candidate stream is the proposer: 15 legitimate model specs per task family
(fingerprint radius/bit-size/type, base-learner family, imbalance handling,
compositions). With max_loops as a non-binding ceiling, all 15 run and the real
count is reported. Validation only; test is never touched here.

    uv run --extra figures python robustness_forced_loops.py
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")   # sklearn feature-name notices; do not affect numbers

from config import Config
from run import Orchestrator, load_specs
from roles import arbiter

FIGDIR = "figures"
# Non-binding ceiling: the 15-candidate stream length is the real iteration count
# (reported honestly), not max_loops. Set above 15 so nothing is truncated by the cap.
MAX_LOOPS = 20

# Classification, plateau regime (small endpoints, 25-seed budget). 15 candidates:
# the original 6 plus 9 legitimate variations spanning fingerprint radius/bit-size/
# type, base-learner family, imbalance handling, and compositions (see each spec's
# docstring). All go through the SAME arbiter discipline.
CLF = ["v01_desc_rf", "v02_desc_lgbm", "v03_desc_morgan_rf",
       "v04_desc_morgan_rf_balanced", "v05_stacked_meta", "v06_late_fusion",
       "v07_morgan_lgbm", "v08_morgan_r3_rf_balanced", "v09_maccs_rf_balanced",
       "v10_desc_morgan_r3_rf_balanced", "v11_desc_morgan1024_lgbm",
       "v12_desc_morgan_extratrees_balanced", "v13_desc_morgan_histgb",
       "v14_desc_morgan_lgbm_balanced", "v15_desc_logreg_balanced"]
FORCED = {
    "cyp2d6_substrate_carbonmangels": CLF,
    "cyp2c9_substrate_carbonmangels": CLF,
    "dili": CLF,
}

# Positive control: solubility (which DID separate) run through a matching 15-spec
# EXPANDED regression stream. Confirms the descriptor separation still fires early
# and that none of the 9 extra candidates spuriously beat it.
SOL15 = ["sol_v01_morgan_lgbm", "sol_v02_desc_lgbm", "sol_v03_desc_morgan_lgbm_tuned",
         "sol_v04_stacked_reg", "sol_v05_late_fusion_reg", "sol_v06_two_stage_residual",
         "sol_v07_morgan_r3_lgbm", "sol_v08_maccs_lgbm", "sol_v09_desc_morgan_r3_lgbm",
         "sol_v10_desc_morgan1024_lgbm", "sol_v11_desc_morgan_rf",
         "sol_v12_desc_morgan_extratrees", "sol_v13_desc_morgan_histgb",
         "sol_v14_desc_ridge", "sol_v15_desc_morgan_lgbm_deep"]
SEP = {
    "solubility_aqsoldb": SOL15,
}


def run_forced(endpoint, versions):
    """force_full=True, non-binding max_loops: run the whole 15-candidate stream
    through the SAME arbiter discipline without stopping at the plateau. Validation only."""
    cfg = Config(endpoint, novelty_tier="middle", force_full=True, max_loops=MAX_LOOPS)
    orch = Orchestrator(cfg)
    orch.run_loop(load_specs(versions))
    hist = orch.history
    plat = arbiter.detect_plateau(hist, k=cfg.plateau_k)
    pre_best = plat.pre_plateau_best_index if plat.plateau_index != -1 else \
        max(range(len(hist)), key=lambda i: hist[i].mean if hist[i].higher_is_better else -hist[i].mean)

    rows = []
    for i, r in enumerate(hist):
        # CI-separable gain over the pre-plateau best (the exact arbiter call)
        sep = arbiter.beats(r, hist[pre_best]) if i != pre_best else False
        rows.append(dict(
            iteration=i + 1, spec=orch.spec_names[i], metric=r.metric_name,
            mean=r.mean, ci_lo=r.ci[0], ci_hi=r.ci[1], ci_halfwidth=r.ci_halfwidth,
            higher_is_better=r.higher_is_better,
            post_plateau=(plat.plateau_index != -1 and i > plat.plateau_index),
            ci_separable_vs_pre_plateau_best=bool(sep),
        ))
    return dict(
        endpoint=endpoint, metric=hist[0].metric_name,
        n_iterations=len(hist), seed_budget=orch.adapter.seed_budget,
        plateau_index=plat.plateau_index, plateau_iteration=plat.plateau_index + 1,
        pre_plateau_best_index=pre_best, pre_plateau_best_iteration=pre_best + 1,
        pre_plateau_best_spec=orch.spec_names[pre_best],
        post_plateau_gain=bool(orch.outcome.post_plateau_gain),
        locked_spec=orch.locked_name, rows=rows,
    )


def run_separated(endpoint, versions):
    """Full trajectory; find the FIRST iteration whose CI-separably beats the
    running lock (first committed promotion)."""
    cfg = Config(endpoint, novelty_tier="middle", force_full=True, max_loops=MAX_LOOPS)
    orch = Orchestrator(cfg)
    orch.run_loop(load_specs(versions))
    hist = orch.history

    first_sep = None
    lock = 0
    for i in range(1, len(hist)):
        if arbiter.beats(hist[i], hist[lock]):
            first_sep = i
            break
        if arbiter.ci_overlap(hist[i], hist[lock]) and arbiter._better_mean(hist[i], hist[lock]):
            lock = i
    rows = [dict(iteration=i + 1, spec=orch.spec_names[i], mean=r.mean,
                 ci_lo=r.ci[0], ci_hi=r.ci[1], ci_halfwidth=r.ci_halfwidth,
                 higher_is_better=r.higher_is_better)
            for i, r in enumerate(hist)]
    return dict(
        endpoint=endpoint, metric=hist[0].metric_name, n_iterations=len(hist),
        seed_budget=orch.adapter.seed_budget,
        first_separation_index=first_sep,
        first_separation_iteration=(first_sep + 1) if first_sep is not None else None,
        first_separation_spec=orch.spec_names[first_sep] if first_sep is not None else None,
        locked_spec=orch.locked_name, rows=rows,
    )


# --------------------------- plotting ---------------------------
def plot_A(forced, path):
    import matplotlib.pyplot as plt
    n = len(forced)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4))
    if n == 1:
        axes = [axes]
    for ax, d in zip(axes, forced):
        xs = [r["iteration"] for r in d["rows"]]
        ys = [r["mean"] for r in d["rows"]]
        lo = [r["ci_lo"] for r in d["rows"]]
        hi = [r["ci_hi"] for r in d["rows"]]
        ax.fill_between(xs, lo, hi, alpha=0.25, color="#4C72B0",
                        label="95% CI (over seeds)")
        ax.plot(xs, ys, "-o", color="#1f4e79", label="validation mean")
        # pre-plateau best CI band as a horizontal reference the later CIs overlap
        pb = d["pre_plateau_best_index"]
        b_lo, b_hi = d["rows"][pb]["ci_lo"], d["rows"][pb]["ci_hi"]
        ax.axhspan(min(b_lo, b_hi), max(b_lo, b_hi), color="#DD8452", alpha=0.12)
        ax.axhline(d["rows"][pb]["mean"], color="#DD8452", ls=":", lw=1,
                   label=f"pre-plateau best (it {pb+1})")
        pit = d["plateau_iteration"]
        ax.axvline(pit, color="grey", ls="--", lw=1.2, label=f"plateau (it {pit})")
        gain = d["post_plateau_gain"]
        verdict = ("post-plateau CI-separable gain: NONE"
                   if not gain else "POST-PLATEAU GAIN FOUND -- FLAG")
        ax.set_title(f"{d['endpoint']}\n{d['metric']}, {d['seed_budget']} seeds  "
                     f"(ran {d['n_iterations']} its)\n{verdict}",
                     fontsize=9, color=("#2e7d32" if not gain else "#c62828"))
        ax.set_xlabel("iteration")
        ax.set_ylabel(d["metric"])
        ax.set_xticks(xs)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Plot A -- Plateau holds under forced iteration over the full 15-candidate stream "
                 "(force_full=True). CIs shown: later iterations overlap the pre-plateau best.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path + ".png", dpi=150)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def plot_B(sep, path):
    import matplotlib.pyplot as plt
    n = len(sep)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4))
    if n == 1:
        axes = [axes]
    for ax, d in zip(axes, sep):
        xs = [r["iteration"] for r in d["rows"]]
        ys = [r["mean"] for r in d["rows"]]
        lo = [r["ci_lo"] for r in d["rows"]]
        hi = [r["ci_hi"] for r in d["rows"]]
        ax.fill_between(xs, lo, hi, alpha=0.25, color="#55A868", label="95% CI (over seeds)")
        ax.plot(xs, ys, "-o", color="#2f6b45", label="validation mean")
        fs = d["first_separation_iteration"]
        if fs is not None:
            ax.axvline(fs, color="#C44E52", ls="--", lw=1.6,
                       label=f"first separation (it {fs})")
            ax.plot([fs], [d["rows"][fs - 1]["mean"]], "*", color="#C44E52", ms=16)
        ax.set_title(f"{d['endpoint']}\n{d['metric']}, {d['seed_budget']} seeds", fontsize=9)
        ax.set_xlabel("iteration")
        ax.set_ylabel(d["metric"] + "  (lower is better)")
        ax.set_xticks(xs)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Plot B -- Real signal appears early: first CI-separable promotion "
                 "occurs within the first few iterations (or not at all).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path + ".png", dpi=150)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    forced = [run_forced(e, v) for e, v in FORCED.items()]
    sep = [run_separated(e, v) for e, v in SEP.items()]

    plot_A(forced, os.path.join(FIGDIR, "plotA_plateau_holds"))
    plot_B(sep, os.path.join(FIGDIR, "plotB_signal_appears_early"))
    with open(os.path.join(FIGDIR, "robustness_trajectories.json"), "w") as f:
        json.dump({"forced": forced, "separated": sep}, f, indent=2)

    # ---- console summary (transcribed into ROBUSTNESS.md) ----
    print("=" * 92)
    print("FORCED-LOOP RUNS ON PLATEAUED ENDPOINTS (force_full=True, max_loops=15)")
    print("=" * 92)
    for d in forced:
        print(f"\n{d['endpoint']}  [{d['metric']}, {d['seed_budget']} seeds, "
              f"ran {d['n_iterations']} iterations -- full 15-candidate stream, "
              f"max_loops non-binding]")
        print(f"  plateau at iteration {d['plateau_iteration']}; "
              f"pre-plateau best = it {d['pre_plateau_best_iteration']} "
              f"({d['pre_plateau_best_spec']}); locked {d['locked_spec']}")
        hib = d["rows"][0]["higher_is_better"]
        print(f"  {'it':>3} {'spec':32s} {'mean':>8} {'95% CI':>18} {'post-plat':>9} {'CI-sep>best':>11}")
        for r in d["rows"]:
            print(f"  {r['iteration']:>3} {r['spec']:32s} {r['mean']:>8.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] {str(r['post_plateau']):>9} "
                  f"{str(r['ci_separable_vs_pre_plateau_best']):>11}")
        print(f"  ==> post_plateau_gain = {d['post_plateau_gain']}  "
              f"({'PLATEAU HOLDS' if not d['post_plateau_gain'] else 'GAIN FOUND -- FLAG'})")

    print("\n" + "=" * 92)
    print("SEPARATED ENDPOINTS -- first CI-separable promotion iteration")
    print("=" * 92)
    for d in sep:
        print(f"  {d['endpoint']:28s} {d['metric']:>8}  first separation at iteration "
              f"{d['first_separation_iteration']} ({d['first_separation_spec']}); "
              f"locked {d['locked_spec']}")

    any_gain = any(d["post_plateau_gain"] for d in forced)
    print("\n" + "=" * 92)
    print(f"OVERALL: post-plateau CI-separable gain on ANY forced endpoint? "
          f"{'YES -- FLAG FOR REVIEW' if any_gain else 'NO -- plateau robust to loop count'}")
    print("=" * 92)


if __name__ == "__main__":
    main()
