"""Stage-1 regression check: prove TDCAdmetAdapter reproduces the OLD code's
splits EXACTLY on the three validated endpoints. Run as a module from repo root:

    uv run python -m benchmarks.verify_stage1

Compares, per seed, the train/valid partition (and the test set) produced by the
OLD data_split*.py (via PyTDC) against the NEW adapter (vendored split on cached
CSVs). Any mismatch is a regression -> reported loudly, no commit.
"""
import importlib

import data_split          # cyp2d6 (SEEDS 1..25)
import data_split_sol      # solubility (SEEDS 1..5)
import data_split_caco     # caco2 (SEEDS 1..25)
from tdc.benchmark_group import admet_group
from benchmarks.tdc_admet import TDCAdmetAdapter, METRIC_MAP

# endpoint -> (old module, expected metric, expected higher_is_better, expected seed_budget)
CASES = [
    ("cyp2d6_substrate_carbonmangels", data_split, "pr-auc", True, 25),
    ("solubility_aqsoldb", data_split_sol, "mae", False, 5),
    ("caco2_wang", data_split_caco, "mae", False, 25),
]


def sig(df):
    """Order-independent, multiplicity-preserving signature of a split's rows."""
    return sorted((str(a), str(b), str(c))
                  for a, b, c in zip(df["Drug_ID"], df["Drug"], df["Y"]))


def main():
    group = admet_group(path="data/")     # reference: OLD test sets only
    all_ok = True

    print("=" * 82)
    print("STAGE 1 SPLIT REPRODUCTION  (OLD data_split*.py  vs  NEW TDCAdmetAdapter)")
    print("=" * 82)
    for endpoint, old_mod, exp_metric, exp_hib, exp_budget in CASES:
        adapter = TDCAdmetAdapter(endpoint)
        old_splits = old_mod.splits()
        seeds = old_mod.SEEDS
        n_train_ok = n_valid_ok = 0
        first_bad = None
        for s in seeds:
            tr_old, va_old = old_splits[s]
            tr_new, va_new = adapter.load_train_valid(s)
            t_match = sig(tr_old) == sig(tr_new)
            v_match = sig(va_old) == sig(va_new)
            n_train_ok += t_match
            n_valid_ok += v_match
            if not (t_match and v_match) and first_bad is None:
                first_bad = (s, len(tr_old), len(tr_new), len(va_old), len(va_new))
        # test set comparison (old via PyTDC vs adapter.load_test_labeled)
        test_old = group.get(endpoint)["test"]
        test_match = sig(test_old) == sig(adapter.load_test_labeled())

        ep_ok = (n_train_ok == len(seeds) and n_valid_ok == len(seeds) and test_match)
        all_ok &= ep_ok
        print(f"\n{endpoint}  ({len(seeds)} seeds)")
        print(f"  train splits identical: {n_train_ok}/{len(seeds)}   "
              f"valid splits identical: {n_valid_ok}/{len(seeds)}   "
              f"test set identical: {test_match}")
        print(f"  -> {'IDENTICAL' if ep_ok else 'MISMATCH'}")
        if first_bad:
            s, tro, trn, vao, van = first_bad
            print(f"  !! first mismatch seed={s}: "
                  f"train old/new={tro}/{trn}, valid old/new={vao}/{van}")

    print("\n" + "=" * 82)
    print("METRIC RESOLUTION  (adapter vs verified map)")
    print("=" * 82)
    metric_ok = True
    for endpoint, _, exp_metric, exp_hib, exp_budget in CASES:
        a = TDCAdmetAdapter(endpoint)
        row_ok = (a.metric_name == exp_metric and a.higher_is_better == exp_hib
                  and a.seed_budget == exp_budget)
        metric_ok &= row_ok
        print(f"  {endpoint:32s} metric={a.metric_name:8s} higher_is_better="
              f"{str(a.higher_is_better):5s} task={a.task_type:14s} "
              f"seed_budget={a.seed_budget:2d}  [{'ok' if row_ok else 'MISMATCH'}]")
    # spot-check: cyp3a4_substrate must resolve to roc-auc, not substrate-family pr-auc
    c3 = METRIC_MAP["cyp3a4_substrate_carbonmangels"]
    spot_ok = c3 == "roc-auc"
    print(f"  spot-check cyp3a4_substrate_carbonmangels -> {c3}  "
          f"[{'ok: not a family default' if spot_ok else 'MISMATCH'}]")

    print("\n" + "=" * 82)
    print("TEST-LABEL EXPOSURE  (load_test_molecules must drop Y)")
    print("=" * 82)
    exposure_ok = True
    for endpoint, _, _, _, _ in CASES:
        a = TDCAdmetAdapter(endpoint)
        mol_cols = list(a.load_test_molecules().columns)
        has_y = "Y" in mol_cols
        exposure_ok &= (not has_y)
        print(f"  {endpoint:32s} load_test_molecules cols={mol_cols}  "
              f"[{'ok: no Y' if not has_y else 'LEAK: Y exposed'}]")

    print("\n" + "=" * 82)
    ok = all_ok and metric_ok and spot_ok and exposure_ok
    print(f"OVERALL: {'ALL IDENTICAL + metrics/exposure OK -> safe to commit' if ok else 'REGRESSION -> DO NOT COMMIT'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
