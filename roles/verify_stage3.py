"""Stage-3 regression check: replay the two validated iteration histories through
the extracted arbiter and confirm identical promote/tie/lock/plateau decisions.

    uv run python -m roles.verify_stage3

Feeds the validated ScoreResults (recorded means +/- 95% CI half-widths from the
per-version commits) in order. The arbiter is pure code, so this needs no model
runs -- it tests decision logic only. Any divergence -> reported, no commit.
"""
from roles.executor import ScoreResult
from roles import arbiter

# validated (mean, ci_halfwidth) per version, from the committed iteration tables
CYP = [  # PR-AUC, higher better, 25 seeds
    ("v01", 0.6090, 0.0527, 1), ("v02", 0.5837, 0.0614, 1),
    ("v03", 0.6161, 0.0502, 2), ("v04", 0.6256, 0.0527, 2),
    ("v05", 0.6102, 0.0526, 4), ("v06", 0.6185, 0.0464, 3),
]
SOL = [  # MAE, lower better, 5 seeds
    ("v01", 1.2606, 0.2179, 1), ("v02", 0.8263, 0.1509, 1),
    ("v03", 0.8281, 0.1453, 2), ("v04", 0.8172, 0.1498, 4),
    ("v05", 0.8185, 0.1531, 3), ("v06", 0.8232, 0.1522, 3),
]


def build(rows, higher_is_better, metric, n):
    results = [ScoreResult(metric, higher_is_better, m, 0.0, hw, (), n) for _, m, hw, _ in rows]
    names = [r[0] for r in rows]
    complexity = [r[3] for r in rows]
    return results, names, complexity


def check(tag, rows, higher_is_better, metric, n, expected):
    results, names, cx = build(rows, higher_is_better, metric, n)
    print("=" * 92)
    print(f"{tag}: replay through arbiter")
    print("=" * 92)

    # forced full run (validated ran all 6 -- climb then compositions)
    out = arbiter.run_selection(results, k=3, max_loops=len(results), force_full=True)
    labels = ["baseline"] + list(out.compare_labels)
    locks = [names[i] for i in out.lock_trajectory]
    print(f"{'version':8s} {'compare':12s} {'lock after':11s} {'validated':22s}")
    ok = True
    for i, name in enumerate(names):
        exp_cmp, exp_lock = expected["seq"][i]
        got_cmp = labels[i]
        got_lock = locks[i]
        row_ok = (got_cmp == exp_cmp and got_lock == exp_lock)
        ok &= row_ok
        print(f"{name:8s} {got_cmp:12s} {got_lock:11s} "
              f"{('cmp=' + exp_cmp + ' lock=' + exp_lock):22s} [{'ok' if row_ok else 'MISMATCH'}]")

    locked = names[out.locked_index]
    plateau_v = names[out.plateau_index] if out.plateau_index != -1 else "none"
    lock_ok = locked == expected["lock"]
    plat_ok = plateau_v == expected["plateau"]
    gain_ok = out.post_plateau_gain == expected["post_gain"]
    print(f"\n  locked (forced) : {locked:5s}  [{'ok' if lock_ok else 'MISMATCH'}, expected {expected['lock']}]")
    print(f"  plateau fired at: {plateau_v:5s}  [{'ok' if plat_ok else 'MISMATCH'}, expected {expected['plateau']}]")
    print(f"  post-plateau gain beats pre-plateau best: {out.post_plateau_gain}  "
          f"[{'ok' if gain_ok else 'MISMATCH'}, expected {expected['post_gain']}]")

    # non-forced: plateau STOPS the loop
    out_nf = arbiter.run_selection(results, k=3, force_full=False)
    nf_lock = names[out_nf.locked_index]
    nf_stop = names[out_nf.stopped_index]
    nf_ok = nf_stop == expected["nonforced_stop"] and nf_lock == expected["lock"]
    print(f"  non-forced: stops at {nf_stop}, lock {nf_lock}  "
          f"[{'ok' if nf_ok else 'MISMATCH'}, expected stop {expected['nonforced_stop']}]")

    # parsimony_decision spot-checks
    print("  parsimony_decision spot-checks:")
    pars_ok = True
    for ci, li, exp in expected["parsimony"]:
        d = arbiter.parsimony_decision(results[ci], results[li], cx[ci], cx[li])
        pok = d == exp
        pars_ok &= pok
        print(f"    {names[ci]} vs lock {names[li]} -> {d:9s} "
              f"[{'ok' if pok else 'MISMATCH'}, expected {exp}]")

    return ok and lock_ok and plat_ok and gain_ok and nf_ok and pars_ok


def main():
    cyp_expected = {
        "seq": [("baseline", "v01"), ("tied", "v01"), ("tied", "v03"),
                ("tied", "v04"), ("tied", "v04"), ("tied", "v04")],
        "lock": "v04", "plateau": "v04", "post_gain": False,
        "nonforced_stop": "v04",
        "parsimony": [(4, 3, "keep_lock"), (5, 3, "keep_lock")],  # v05,v06 vs v04
    }
    sol_expected = {
        "seq": [("baseline", "v01"), ("significant", "v02"), ("tied", "v02"),
                ("tied", "v02"), ("tied", "v02"), ("tied", "v02")],
        "lock": "v02", "plateau": "v05", "post_gain": False,
        "nonforced_stop": "v05",
        "parsimony": [(1, 0, "promote"), (2, 1, "keep_lock"),
                      (3, 1, "keep_lock"), (5, 1, "keep_lock")],
    }
    ok_cyp = check("CYP2D6 (v01-v06)", CYP, True, "pr-auc", 25, cyp_expected)
    print()
    ok_sol = check("SOLUBILITY (v01-v06)", SOL, False, "mae", 5, sol_expected)

    print("\n" + "=" * 92)
    ok = ok_cyp and ok_sol
    print(f"OVERALL: {'both decision sequences reproduce EXACTLY -> safe to commit' if ok else 'DIVERGENCE -> DO NOT COMMIT'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
