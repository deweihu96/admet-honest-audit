"""Scaffold train/valid split, VENDORED VERBATIM from PyTDC's
tdc.utils.split.create_scaffold_split (tdc 1.1.15).

Vendored (not imported) so the adapter reproduces splits from committed CSVs
without importing PyTDC at runtime (dodging the PyTDC/setuptools<81 pin). The
algorithm is byte-for-byte the same as TDC's, and benchmarks/verify_stage1.py
proves the resulting splits are identical to the validated code on 3 endpoints.
Do NOT "clean up" this function -- any change risks diverging from the splits
every validated number was computed on.
"""
from collections import defaultdict
from random import Random

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def create_scaffold_split(df, seed, frac, entity):
    """Identical to tdc.utils.split.create_scaffold_split. Returns
    {'train','valid','test'} DataFrames (test empty when frac[2]==0)."""
    random = Random(seed)

    s = df[entity].values
    scaffolds = defaultdict(set)
    idx2mol = dict(zip(list(range(len(s))), s))

    error_smiles = 0
    for i, smiles in enumerate(s):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False)
            scaffolds[scaffold].add(i)
        except Exception:
            error_smiles += 1

    train, val, test = [], [], []
    train_size = int((len(df) - error_smiles) * frac[0])
    val_size = int((len(df) - error_smiles) * frac[1])
    test_size = (len(df) - error_smiles) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return {
        "train": df.iloc[train].reset_index(drop=True),
        "valid": df.iloc[val].reset_index(drop=True),
        "test": df.iloc[test].reset_index(drop=True),
    }
