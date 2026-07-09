"""Positive control for the leakage audit.

GitHub issue #217 (mims-harvard/TDC) documented REAL train/test molecular
overlap: caco2_wang (~12 shared) and ld50_zhu (~11 shared); cyp2d6_substrate
has 0. Our audit half must catch a known bug autonomously. Success criterion:
flag caco2_wang and ld50_zhu (near-1.0 max NN sim and/or nonzero exact overlap)
while leaving cyp2d6_substrate clean.

No modeling here -- this is a property of the SPLIT only. NN Tanimoto r=3, 2048
bits, train_val as reference (the split-audit reference). Plus exact-SMILES
overlap, computed both on raw strings AND RDKit-canonical strings (raw matching
misses duplicates written with different but equivalent SMILES).
"""
import numpy as np
from rdkit import Chem, RDLogger
from tdc.benchmark_group import admet_group
from run_dryrun import nn_tanimoto          # audited NN-Tanimoto (r=3) function

RDLogger.DisableLog("rdApp.*")


def canon(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m is not None else None


def audit(group, endpoint):
    b = group.get(endpoint)
    tv, test = b["train_val"], b["test"]

    # NN Tanimoto (test -> nearest train_val), r=3.
    nn = nn_tanimoto(test["Drug"].tolist(), tv["Drug"].tolist())
    med, mx = float(np.median(nn)), float(np.max(nn))
    n_ge99 = int((nn >= 0.99).sum())
    n_eq1 = int((nn >= 0.999).sum())

    # Exact overlap, raw strings vs canonicalized.
    raw_tv, raw_te = set(tv["Drug"]), set(test["Drug"])
    raw_overlap = len(raw_tv & raw_te)
    can_tv = {c for c in (canon(s) for s in tv["Drug"]) if c}
    can_te_list = [canon(s) for s in test["Drug"]]
    can_overlap = sum(1 for c in can_te_list if c in can_tv)

    flagged = (mx >= 0.99) or (can_overlap > 0)
    return dict(endpoint=endpoint, n_test=len(test), median=med, max=mx,
                n_ge99=n_ge99, n_eq1=n_eq1, raw_overlap=raw_overlap,
                can_overlap=can_overlap, flagged=flagged)


def main():
    group = admet_group(path="data/")
    # dirty (per #217) + clean negative control
    targets = [("caco2_wang", "DIRTY per #217 (~12)"),
               ("ld50_zhu", "DIRTY per #217 (~11)"),
               ("cyp2d6_substrate_carbonmangels", "CLEAN control (0)")]
    rows = [(audit(group, ep), note) for ep, note in targets]

    print("=" * 100)
    print("LEAKAGE AUDIT POSITIVE CONTROL  (NN Tanimoto r=3 + exact-SMILES overlap)")
    print("=" * 100)
    h = ("endpoint", "expect", "n_test", "NN med", "NN max", ">=.99", "==1.0",
         "raw dup", "canon dup", "FLAGGED?")
    print(f"{h[0]:32s} {h[1]:22s} {h[2]:>6s} {h[3]:>7s} {h[4]:>7s} {h[5]:>5s} "
          f"{h[6]:>5s} {h[7]:>7s} {h[8]:>9s} {h[9]:>9s}")
    print("-" * 100)
    for r, note in rows:
        print(f"{r['endpoint']:32s} {note:22s} {r['n_test']:>6d} {r['median']:>7.3f} "
              f"{r['max']:>7.3f} {r['n_ge99']:>5d} {r['n_eq1']:>5d} {r['raw_overlap']:>7d} "
              f"{r['can_overlap']:>9d} {'YES' if r['flagged'] else 'no':>9s}")

    print("\nInterpretation:")
    for r, note in rows:
        verdict = "FLAGGED as leaky" if r["flagged"] else "clean"
        print(f"  {r['endpoint']:32s}: {verdict}  "
              f"(max NN {r['max']:.3f}, canon dups {r['can_overlap']})")


if __name__ == "__main__":
    main()
