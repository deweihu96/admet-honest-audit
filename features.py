"""Featurization blocks shared by the iteration harness and final test eval.
Pure function of SMILES -> matrix; no train/test awareness. Cached per (smiles,
block) so repeated seeds are cheap."""
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")
_DESC_NAMES = [d[0] for d in Descriptors._descList]
_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
_cache = {}

DIMS = {"morgan_r2": 2048, "morgan_r3": 2048, "morgan_r2_1024": 1024,
        "maccs": 167, "desc": len(_DESC_NAMES)}


def _morgan(m, r, n):
    if m is None:
        return np.zeros(n, dtype=np.float32)
    v = AllChem.GetMorganFingerprintAsBitVect(m, r, nBits=n)
    a = np.zeros(n, dtype=np.float32); DataStructs.ConvertToNumpyArray(v, a)
    return a


def _block(smi, block):
    key = (smi, block)
    if key in _cache:
        return _cache[key]
    m = Chem.MolFromSmiles(smi)
    if block == "morgan_r2":
        v = _morgan(m, 2, 2048)
    elif block == "morgan_r3":
        v = _morgan(m, 3, 2048)
    elif block == "morgan_r2_1024":
        v = _morgan(m, 2, 1024)
    elif block == "maccs":
        v = (np.zeros(167, dtype=np.float32) if m is None
             else np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32))
    elif block == "desc":
        v = (np.zeros(len(_DESC_NAMES), dtype=np.float32) if m is None
             else np.nan_to_num(np.array(_CALC.CalcDescriptors(m), dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0))
    else:
        raise ValueError(f"unknown feature block: {block}")
    _cache[key] = v
    return v


def build_matrix(smiles, blocks):
    """Concatenate the named feature blocks column-wise for a list of SMILES."""
    return np.hstack([np.vstack([_block(s, b) for s in smiles]) for b in blocks])
