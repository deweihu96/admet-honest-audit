"""Walled auditor: the ONLY component (besides final-eval) allowed to touch test.

It calls adapter.load_train_reference() and adapter.load_test_molecules()
(SMILES only, NO labels), computes split-level leakage similarity internally, and
returns ONLY a frozen LeakageVerdict of aggregate summaries. It never returns the
raw test molecules or the per-molecule similarity array -- those exist only as
locals inside audit(). This frozen verdict is the sole thing that crosses back to
the loop.

Leakage radius is r3 (2048 bits), deliberately distinct from the modeling r2
(features.py). The InChIKey identity axis (desalted connectivity) is kept because
it caught solubility's acyclic duplicates that scaffold-overlap missed.
"""
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi

RDLogger.DisableLog("rdApp.*")

LEAK_RADIUS, LEAK_NBITS = 3, 2048
IDENTITY_TANIMOTO = 0.99      # NN sim >= this looks like (near-)identity
NEAR_DUP_MASS_FLOOR = 0.02    # frac of test with NN >= 0.9 above this = suspicious mass


@dataclass(frozen=True)
class LeakageVerdict:
    """Split-level leakage summary. The ONLY thing that crosses back to the loop.

    Aggregate, order-invariant summaries ONLY -- carries NO test labels, NO test
    predictions, and NO per-molecule array. A recipient cannot reconstruct any
    test molecule, its identity, or its label from these fields.
    """
    endpoint: str
    clean: bool
    nn_similarity_median: float
    nn_similarity_max: float
    nn_similarity_p95: float
    n_exact_identity: int          # count of InChIKey-identity test/train overlaps
    n_test: int                    # size only (not molecules)
    noise_floor_ok: bool           # near-duplicate mass below the suspicious floor
    notes: tuple


def _fp3(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(
        m, LEAK_RADIUS, nBits=LEAK_NBITS)


def _ikey(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    parent = max(fr, key=lambda f: f.GetNumHeavyAtoms()) if fr else m
    try:
        return inchi.MolToInchiKey(parent).split("-")[0]   # connectivity block
    except Exception:
        return None


class Auditor:
    def __init__(self, adapter):
        self.adapter = adapter

    def audit(self) -> LeakageVerdict:
        # reference = train_val molecules; test = SMILES only (no labels)
        ref = self.adapter.load_train_reference()
        test = self.adapter.load_test_molecules()
        assert "Y" not in test.columns, "auditor must never see test labels"

        ref_fps = [f for f in (_fp3(s) for s in ref["Drug"]) if f is not None]
        nn = np.array([max(DataStructs.BulkTanimotoSimilarity(_fp3(s), ref_fps))
                       for s in test["Drug"] if _fp3(s) is not None])

        ref_ik = {_ikey(s) for s in ref["Drug"]}
        ref_ik.discard(None)
        n_exact = sum(1 for s in test["Drug"] if _ikey(s) in ref_ik)

        nn_max = float(nn.max())
        frac_high = float((nn >= 0.9).mean())
        noise_floor_ok = frac_high < NEAR_DUP_MASS_FLOOR
        clean = (nn_max < IDENTITY_TANIMOTO) and (n_exact == 0)

        notes = []
        if n_exact > 0:
            notes.append(f"{n_exact} InChIKey-identity train/test overlaps (real duplicates)")
        if nn_max >= IDENTITY_TANIMOTO:
            notes.append(f"NN Tanimoto max {nn_max:.3f} >= {IDENTITY_TANIMOTO} (near-identity)")
        if clean:
            notes.append("no exact identity and no near-identity neighbor: split clean")

        # per-molecule nn array and test SMILES are LOCALS -- never stored/returned
        return LeakageVerdict(
            endpoint=self.adapter.endpoint,
            clean=clean,
            nn_similarity_median=float(np.median(nn)),
            nn_similarity_max=nn_max,
            nn_similarity_p95=float(np.percentile(nn, 95)),
            n_exact_identity=int(n_exact),
            n_test=int(len(nn)),
            noise_floor_ok=noise_floor_ok,
            notes=tuple(notes),
        )


# ---- SECOND AUDIT AXIS (final-eval time; needs a locked model's test read) ----
@dataclass(frozen=True)
class GapVerdict:
    """Summary-only: valid-vs-test divergence. Carries no per-molecule data."""
    gap: float                     # test_point - valid_mean
    flagged: bool                  # |gap| exceeds the validation CI half-width
    suspicious: bool               # flagged AND test easier than validation
    note: str


def valid_vs_test_gap(valid_mean, valid_ci_halfwidth, test_point,
                      higher_is_better) -> GapVerdict:
    """Flag when a locked model's one-shot test read diverges systematically from
    validation. 'Test easier than validation' is the leakage/inflation-suspicious
    direction (higher-is-better: test>valid; lower-is-better: test<valid). This
    is the axis that flagged caco2's easy-but-clean test set."""
    gap = test_point - valid_mean
    flagged = abs(gap) > valid_ci_halfwidth
    test_easier = (gap > 0) if higher_is_better else (gap < 0)
    suspicious = flagged and test_easier
    if not flagged:
        note = "valid ~ test (within validation CI)"
    elif suspicious:
        note = "test EASIER than validation -- inflation/leakage-suspicious direction"
    else:
        note = "test HARDER than validation (out-of-distribution, expected)"
    return GapVerdict(gap=float(gap), flagged=bool(flagged),
                      suspicious=bool(suspicious), note=note)
