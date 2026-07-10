"""Stage-2 regression check: prove the executor reproduces validated VALIDATION
numbers. Run from repo root:

    uv run python -m roles.verify_stage2

Compares, per model version: OLD path (iterate*.py via data_split*.py + PyTDC)
vs NEW path (roles.executor via the Stage 1 adapter) vs the validated reference
from the prior sessions. Any difference beyond floating-point noise is a
regression -> reported, no commit.
"""
import importlib

import iterate            # OLD cyp2d6 harness (data_split, 25 seeds)
import iterate_sol        # OLD solubility harness (data_split_sol, 5 seeds)
from roles.executor import train_and_score
from benchmarks.tdc_admet import TDCAdmetAdapter

TOL = 1e-6

# (endpoint, version module, old-runner, validated mean, validated hw)
CASES = [
    ("cyp2d6_substrate_carbonmangels", "v04_desc_morgan_rf_balanced", iterate, 0.6256, 0.0527),
    ("solubility_aqsoldb", "sol_v01_morgan_lgbm", iterate_sol, 1.2606, 0.2179),
    ("solubility_aqsoldb", "sol_v02_desc_lgbm", iterate_sol, 0.8263, 0.1509),
]


def main():
    print("=" * 100)
    print("STAGE 2 EXECUTOR REPRODUCTION  (OLD iterate*.py  vs  NEW executor  vs  validated)")
    print("=" * 100)
    print(f"{'version':30s} {'old mean±hw':18s} {'new mean±hw':18s} {'validated':16s} "
          f"{'|d mean|':>9s} {'match':>7s}")
    print("-" * 100)

    results, all_ok = {}, True
    for endpoint, version, old_mod, ref_mean, ref_hw in CASES:
        mod = importlib.import_module(f"models.{version}")
        adapter = TDCAdmetAdapter(endpoint)
        old_mean, old_hw = old_mod.run(version)             # OLD path
        r = train_and_score(mod, adapter)                    # NEW path (default seeds)
        results[version] = r
        dmean = abs(old_mean - r.mean)
        ok = dmean < TOL and abs(old_hw - r.ci_halfwidth) < TOL
        all_ok &= ok
        print(f"{version:30s} {old_mean:.4f}±{old_hw:.4f}    {r.mean:.4f}±{r.ci_halfwidth:.4f}    "
              f"{ref_mean:.3f}±{ref_hw:.3f}   {dmean:>9.2e} {('OK' if ok else 'DIFF'):>7s}")

    # CI-separable gap must still hold on solubility (v02 upper < v01 lower)
    print("\n" + "=" * 100)
    print("CI-SEPARABLE GAP (solubility): v02 upper < v01 lower must still hold")
    print("=" * 100)
    v1 = results["sol_v01_morgan_lgbm"]
    v2 = results["sol_v02_desc_lgbm"]
    v1_lo = v1.ci[0]
    v2_hi = v2.ci[1]
    sep = v2_hi < v1_lo          # MAE lower-better: v02's whole CI below v01's
    print(f"  v01 CI = [{v1.ci[0]:.4f}, {v1.ci[1]:.4f}]   v02 CI = [{v2.ci[0]:.4f}, {v2.ci[1]:.4f}]")
    print(f"  v02 upper {v2_hi:.4f} < v01 lower {v1_lo:.4f} ? {sep}  "
          f"-> {'SEPARABLE (promotion holds)' if sep else 'NOT separable -- REGRESSION'}")

    print("\n" + "=" * 100)
    ok = all_ok and sep
    print(f"OVERALL: {'reproduced within tol + gap holds -> safe to commit' if ok else 'REGRESSION -> DO NOT COMMIT'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
