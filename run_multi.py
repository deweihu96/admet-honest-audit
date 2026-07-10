"""Evidence-generation run: the UNMODIFIED orchestrator across a spanning set of
endpoints. Each endpoint is an independent Orchestrator run (tier=middle,
force_full=False). Errors are flagged, not patched. Prints a structured block per
endpoint for transcription into FINDINGS_MULTI.md.

    uv run python run_multi.py
"""
import traceback

from config import Config
from run import Orchestrator, load_specs

# LightGBM-based regression stream (cheap; no RF-in-stacking)
REG = ["caco_v01_morgan_lgbm", "caco_v02_desc_lgbm", "caco_v03_desc_morgan_lgbm_tuned",
       "caco_v04_stacked_reg", "caco_v05_late_fusion_reg", "caco_v06_two_stage_residual"]
# Classifier stream (cyp2d6 version specs)
CLF = ["v01_desc_rf", "v02_desc_lgbm", "v03_desc_morgan_rf",
       "v04_desc_morgan_rf_balanced", "v05_stacked_meta", "v06_late_fusion"]

TARGETS = [
    ("lipophilicity_astrazeneca", REG),
    ("ppbr_az", REG),
    ("clearance_hepatocyte_az", REG),        # Spearman -> confirm direction handling
    ("dili", CLF),
    ("cyp2c9_substrate_carbonmangels", CLF),
    ("ld50_zhu", REG),                        # #217 target
    ("bbb_martins", CLF),                     # #217 target
]


def run_one(endpoint, stream):
    cfg = Config(endpoint, novelty_tier="middle", force_full=False)
    orch = Orchestrator(cfg)
    orch.run_loop(load_specs(stream))
    read = orch.request_test()
    lr = orch.history[orch.outcome.locked_index]
    seq = [(n, c) for n, c, _ in orch.decision_sequence()]
    separates = any(c == "significant" for _, c in seq)
    width = read.read.bootstrap_ci[1] - read.read.bootstrap_ci[0]
    print("=" * 96)
    print(f"ENDPOINT: {endpoint}")
    print("=" * 96)
    print(f"  metric={lr.metric_name} higher_is_better={lr.higher_is_better} "
          f"seed_budget={orch.adapter.seed_budget} n_test={read.leakage.n_test}")
    print(f"  ran {len(orch.history)} candidates; decision seq: {seq}")
    print(f"  regime: {'SEPARATES (promotion fired)' if separates else 'PLATEAUS (all tied)'}")
    print(f"  LOCK: {orch.locked_name}")
    print(f"  validation headline: {read.validation_headline}")
    print(f"  test read: point {read.read.test_point:.4f}  bootstrap CI "
          f"[{read.read.bootstrap_ci[0]:.4f}, {read.read.bootstrap_ci[1]:.4f}]  width {width:.4f}")
    print(f"  valid-vs-test gap: {read.gap.gap:+.4f} flagged={read.gap.flagged} "
          f"suspicious={read.gap.suspicious}")
    print(f"  leakage: clean={read.leakage.clean} n_exact_identity={read.leakage.n_exact_identity} "
          f"nn_max={read.leakage.nn_similarity_max:.3f} nn_median={read.leakage.nn_similarity_median:.3f}")
    print(f"  read_counter: read #{read.read.read_number} adaptive={read.read.adaptive}")
    print(f"[[ROW]] {endpoint}|{lr.metric_name}|{'separates' if separates else 'plateaus'}|"
          f"{lr.mean:.4f}|{lr.ci_halfwidth:.4f}|{orch.locked_name}|{read.read.test_point:.4f}|"
          f"{read.read.bootstrap_ci[0]:.4f}|{read.read.bootstrap_ci[1]:.4f}|{width:.4f}|"
          f"{read.leakage.clean}|{read.leakage.n_exact_identity}|{read.gap.flagged}|"
          f"{orch.adapter.seed_budget}|{read.leakage.n_test}")


def main():
    for endpoint, stream in TARGETS:
        try:
            run_one(endpoint, stream)
        except Exception:
            print("=" * 96)
            print(f"ENDPOINT: {endpoint} -- ERROR (FLAGGED, not patched)")
            print("=" * 96)
            traceback.print_exc()


if __name__ == "__main__":
    main()
