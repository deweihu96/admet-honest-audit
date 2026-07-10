"""Stage-5 end-to-end regression: run the orchestrator on the three validated
endpoints (v01-v06 as the candidate stream, force_full=False) and confirm it
reproduces the validated lock, decision sequence, one-shot test read, and leakage
verdict. Plus the high-tier data-sufficiency gate unit check and read-counter.

    uv run python verify_stage5.py
"""
from config import Config
from run import Orchestrator, load_specs

STREAMS = {
    "cyp2d6_substrate_carbonmangels": dict(
        versions=["v01_desc_rf", "v02_desc_lgbm", "v03_desc_morgan_rf",
                  "v04_desc_morgan_rf_balanced", "v05_stacked_meta", "v06_late_fusion"],
        lock="v04_desc_morgan_rf_balanced", test_point=0.686, ci=(0.552, 0.804),
        clean=True, n_exact=0, gap_flagged=None),
    "solubility_aqsoldb": dict(
        versions=["sol_v01_morgan_lgbm", "sol_v02_desc_lgbm",
                  "sol_v03_desc_morgan_lgbm_tuned", "sol_v04_stacked_reg",
                  "sol_v05_late_fusion_reg", "sol_v06_two_stage_residual"],
        lock="sol_v02_desc_lgbm", test_point=0.741, ci=(0.711, 0.773),
        clean=False, n_exact=7, gap_flagged=False),
    "caco2_wang": dict(
        versions=["caco_v01_morgan_lgbm", "caco_v02_desc_lgbm",
                  "caco_v03_desc_morgan_lgbm_tuned", "caco_v04_stacked_reg",
                  "caco_v05_late_fusion_reg", "caco_v06_two_stage_residual"],
        lock="caco_v02_desc_lgbm", test_point=0.267, ci=(0.236, 0.297),
        clean=True, n_exact=0, gap_flagged=True),
}


def run_endpoint(endpoint, exp):
    cfg = Config(endpoint, novelty_tier="middle", force_full=False)
    orch = Orchestrator(cfg)
    orch.run_loop(load_specs(exp["versions"]))
    read = orch.request_test()             # user-commanded test trigger
    return orch, read


def main():
    print("=" * 100)
    print("HIGH-TIER DATA-SUFFICIENCY GATE (unit check: cyp2d6, 135 test / ~38 positives)")
    print("=" * 100)
    gate_ok = False
    try:
        Orchestrator(Config("cyp2d6_substrate_carbonmangels", novelty_tier="high"))
        print("  gate did NOT refuse -> FAIL")
    except ValueError as e:
        gate_ok = True
        print(f"  REFUSED as expected: {e}")

    ok = gate_ok
    print("\n" + "=" * 100)
    print("END-TO-END ORCHESTRATOR RUNS vs VALIDATED")
    print("=" * 100)
    rows = []
    for endpoint, exp in STREAMS.items():
        orch, read = run_endpoint(endpoint, exp)
        lock_ok = orch.locked_name == exp["lock"]
        pt_ok = abs(read.read.test_point - exp["test_point"]) < 3e-3
        ci_ok = (abs(read.read.bootstrap_ci[0] - exp["ci"][0]) < 8e-3
                 and abs(read.read.bootstrap_ci[1] - exp["ci"][1]) < 8e-3)
        leak_ok = (read.leakage.clean == exp["clean"]
                   and read.leakage.n_exact_identity == exp["n_exact"])
        gap_ok = exp["gap_flagged"] is None or read.gap.flagged == exp["gap_flagged"]
        rc_ok = read.read.read_number == 1 and not read.read.adaptive
        row_ok = lock_ok and pt_ok and ci_ok and leak_ok and gap_ok and rc_ok
        ok &= row_ok
        rows.append((endpoint, orch, read, exp, row_ok, lock_ok, pt_ok, ci_ok, leak_ok, gap_ok))
        print(f"\n{endpoint}")
        print(f"  ran {len(orch.history)} candidates; decision seq: "
              f"{[(n, c) for n, c, _ in orch.decision_sequence()]}")
        print(f"  LOCK      : {orch.locked_name:32s} exp {exp['lock']:32s} "
              f"[{'ok' if lock_ok else 'MISMATCH'}]")
        print(f"  test point: {read.read.test_point:.3f}  bootstrap CI "
              f"[{read.read.bootstrap_ci[0]:.3f}, {read.read.bootstrap_ci[1]:.3f}]  "
              f"exp {exp['test_point']:.3f} {exp['ci']}  [{'ok' if pt_ok and ci_ok else 'MISMATCH'}]")
        print(f"  leakage   : clean={read.leakage.clean} n_exact={read.leakage.n_exact_identity}  "
              f"[{'ok' if leak_ok else 'MISMATCH'}]")
        print(f"  gap       : {read.gap.gap:+.3f} flagged={read.gap.flagged} "
              f"suspicious={read.gap.suspicious}  [{'ok' if gap_ok else 'MISMATCH'}]")
        print(f"  read_ctr  : read #{read.read.read_number}, adaptive={read.read.adaptive}  "
              f"[{'ok' if rc_ok else 'MISMATCH'}]")
        print(f"  headline  : {read.validation_headline}  (validation stays headline)")
        print(f"  -> {'REPRODUCED' if row_ok else 'DIVERGENCE'}")

    print("\n" + "=" * 100)
    print(f"OVERALL: {'all reproduced + gate + read-counter OK -> safe to commit' if ok else 'DIVERGENCE -> DO NOT COMMIT'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
