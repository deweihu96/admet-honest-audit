"""Synthetic positive control for the leakage audit.

The real #217-documented overlaps (caco2_wang, ld50_zhu) are NOT present in this
TDC version -- under every matching scheme (canonical, desalted, stereo-
insensitive, connectivity-InChIKey) train_val vs test overlap is 0. So we cannot
use them as a positive control on this data. Instead we PLANT known duplicates
and confirm the checker catches them: copy N test molecules into the reference
set, re-run the audit, and require it to flag them.

Also demonstrates a checker lesson: Morgan-FP Tanimoto == 1.0 can be a hash
COLLISION (distinct molecules, identical 2048-bit vector), so exact identity
must be confirmed by InChIKey, not by Tanimoto alone.
"""
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi
from tdc.benchmark_group import admet_group

RDLogger.DisableLog("rdApp.*")
N_BITS, R = 2048, 3


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, R, nBits=N_BITS)


def inchikey_conn(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    parent = max(frags, key=lambda f: f.GetNumHeavyAtoms()) if frags else m
    try:
        return inchi.MolToInchiKey(parent).split("-")[0]
    except Exception:
        return None


def audit(ref_smiles, test_smiles):
    """NN Tanimoto max/median + exact-identity overlap via connectivity InChIKey."""
    ref_fps = [f for f in (fp(s) for s in ref_smiles) if f is not None]
    nn = []
    for s in test_smiles:
        f = fp(s)
        if f is not None:
            nn.append(max(DataStructs.BulkTanimotoSimilarity(f, ref_fps)))
    nn = np.array(nn)
    ref_ik = {inchikey_conn(s) for s in ref_smiles}
    ref_ik.discard(None)
    ik_overlap = sum(1 for s in test_smiles if inchikey_conn(s) in ref_ik)
    return float(np.median(nn)), float(np.max(nn)), int((nn >= 0.999).sum()), ik_overlap


def main():
    group = admet_group(path="data/")
    b = group.get("cyp2d6_substrate_carbonmangels")   # verified-clean base
    ref = b["train_val"]["Drug"].tolist()
    test = b["test"]["Drug"].tolist()

    print("=" * 78)
    print("SYNTHETIC POSITIVE CONTROL  (base: cyp2d6_substrate, verified clean)")
    print("=" * 78)

    med0, max0, eq1_0, ik0 = audit(ref, test)
    print(f"\n[baseline, no injection]")
    print(f"  NN median={med0:.3f}  NN max={max0:.3f}  Tanimoto==1.0: {eq1_0}  "
          f"InChIKey identity overlap: {ik0}")

    # Inject: plant 10 test molecules verbatim into the reference set.
    n_inject = 10
    planted = test[:n_inject]
    ref_dirty = ref + planted
    med1, max1, eq1_1, ik1 = audit(ref_dirty, test)
    print(f"\n[after planting {n_inject} test molecules into the reference]")
    print(f"  NN median={med1:.3f}  NN max={max1:.3f}  Tanimoto==1.0: {eq1_1}  "
          f"InChIKey identity overlap: {ik1}")

    caught = (max1 >= 0.999) and (ik1 >= n_inject)
    print(f"\n  checker flags the planted overlap? "
          f"{'YES -- max sim -> 1.0 and InChIKey identity >= injected' if caught else 'NO -- BUG'}")
    print(f"  (identity overlap {ik1} >= injected {n_inject}; "
          f"Tanimoto==1.0 count {eq1_1})")


if __name__ == "__main__":
    main()
