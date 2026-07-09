"""Single-endpoint dry run on Caco2_Wang.

Two independent pieces, timed end-to-end:
  (A) Honest baseline: Morgan FP + LightGBM, official 5-seed protocol.
      Train on `train`, tune (early-stop) on `valid`, score `test` ONCE per seed.
  (B) Leakage check on the SPLIT (not the model): nearest-neighbor Tanimoto
      similarity of every test molecule to its closest train molecule.

The train/valid/test wall is absolute. Test is only ever passed to
group.evaluate_many for final scoring. It is never read for selection/tuning.
"""
import time
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import lightgbm as lgb
from tdc.benchmark_group import admet_group

RDLogger.DisableLog("rdApp.*")

BENCH = "Caco2_Wang"
SEEDS = [1, 2, 3, 4, 5]           # official TDC protocol
# Two DIFFERENT radii on purpose (not a bug, see call sites + CLAUDE.md):
FP_RADIUS_MODEL = 2               # ECFP4, standard modeling feature radius
FP_RADIUS_LEAK = 3                # ECFP6, the Receptor.AI audit leakage spec
N_BITS = 2048


# ----------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------
def morgan_matrix(smiles, radius, n_bits):
    """SMILES list -> (N, n_bits) float32 numpy fingerprint matrix."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows)


def morgan_bitvects(smiles, radius, n_bits):
    """SMILES list -> list of RDKit ExplicitBitVect for fast BulkTanimoto."""
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append(None)
            continue
        out.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return out


# ----------------------------------------------------------------------
# (A) Honest baseline
# ----------------------------------------------------------------------
def run_baseline(group):
    print("=" * 70)
    print("(A) HONEST BASELINE  --  Morgan(r=2,2048) + LightGBM, 5-seed protocol")
    print("=" * 70)

    benchmark = group.get(BENCH)
    name = benchmark["name"]
    test = benchmark["test"]
    # radius 2 (ECFP4): standard modeling features. Deliberately different from
    # the leakage check's radius 3 -- see FP_RADIUS_* and CLAUDE.md.
    X_test = morgan_matrix(test["Drug"].tolist(), FP_RADIUS_MODEL, N_BITS)  # features only

    predictions_list = []
    per_seed_valid_mae = []
    for seed in SEEDS:
        train, valid = group.get_train_valid_split(
            benchmark=BENCH, split_type="default", seed=seed
        )
        # radius 2 (ECFP4) modeling features (cf. leakage check at radius 3).
        X_tr = morgan_matrix(train["Drug"].tolist(), FP_RADIUS_MODEL, N_BITS)
        y_tr = train["Y"].to_numpy()
        X_va = morgan_matrix(valid["Drug"].tolist(), FP_RADIUS_MODEL, N_BITS)
        y_va = valid["Y"].to_numpy()

        # Tune ONLY on valid: early stopping driven by validation MAE.
        model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        valid_mae = np.mean(np.abs(model.predict(X_va) - y_va))
        per_seed_valid_mae.append(valid_mae)

        # Score TEST exactly once here, only to hand predictions to the official
        # evaluator. No selection is made using these numbers.
        y_pred_test = model.predict(X_test)
        predictions_list.append({name: y_pred_test})
        print(f"  seed={seed}: best_iter={model.best_iteration_:4d}  "
              f"valid_MAE={valid_mae:.4f}")

    # Official scoring: evaluate_many computes the endpoint metric (MAE) on the
    # fixed test set for each seed and returns [mean, std].
    results = group.evaluate_many(predictions_list)
    mean, std = results[name]
    print(f"\n  valid MAE (selection signal): {np.mean(per_seed_valid_mae):.4f} "
          f"+/- {np.std(per_seed_valid_mae):.4f}")
    print(f"  OFFICIAL TEST MAE (5 seeds) : {mean:.4f} +/- {std:.4f}   "
          f"[lower is better]")
    return mean, std


# ----------------------------------------------------------------------
# (B) Leakage check -- nearest-neighbor Tanimoto, explicit reference set
# ----------------------------------------------------------------------
def nn_tanimoto(query_smiles, reference_smiles, radius=FP_RADIUS_LEAK, n_bits=N_BITS):
    """For each QUERY molecule, the max Tanimoto similarity to any REFERENCE
    molecule (its single nearest neighbor).

    The reference set is an EXPLICIT argument because it encodes the QUESTION
    being asked, and one number must never serve both roles:

      * SPLIT audit  -> reference = train_val (all training-available mols).
        "Is TDC's published split clean?" Seed-independent; a property of the
        TDC split itself, not of any model. This is the default leakage check.

      * MODEL attribution -> reference = that seed's train (637), the mols the
        model ACTUALLY saw. "Did this model's test score benefit from overlap
        with its own training data?" Seed-dependent; a property of the model.

    Radius is FP_RADIUS_LEAK=3 (ECFP6) per the Receptor.AI audit spec -- note
    this is deliberately NOT the modeling radius (FP_RADIUS_MODEL=2 / ECFP4).
    Returns a numpy array of length len(query_smiles) (skipping unparseable).
    """
    ref_fps = [f for f in morgan_bitvects(reference_smiles, radius, n_bits)
               if f is not None]
    q_fps = morgan_bitvects(query_smiles, radius, n_bits)
    nn = []
    for f in q_fps:
        if f is None:
            continue
        nn.append(max(DataStructs.BulkTanimotoSimilarity(f, ref_fps)))
    return np.array(nn)


def run_leakage_check(group):
    print("\n" + "=" * 70)
    print("(B) LEAKAGE CHECK  --  NN Tanimoto, Morgan(r=3,2048)  [audit spec radius]")
    print("=" * 70)

    benchmark = group.get(BENCH)
    train_val = benchmark["train_val"]      # all training-available molecules
    test = benchmark["test"]

    # SPLIT AUDIT: reference is train_val (728), seed-independent property of the
    # TDC split. This is the default check the project reports per endpoint.
    nn_sim = nn_tanimoto(test["Drug"].tolist(), train_val["Drug"].tolist())

    median = float(np.median(nn_sim))
    mx = float(np.max(nn_sim))
    frac_ge_09 = float(np.mean(nn_sim >= 0.9))
    frac_eq_1 = float(np.mean(nn_sim >= 0.999))
    print(f"  reference set               : train_val (n={len(train_val)})  [SPLIT audit]")
    print(f"  test molecules checked      : {len(nn_sim)}")
    print(f"  MEDIAN nearest-neighbor sim : {median:.4f}")
    print(f"  MAX    nearest-neighbor sim : {mx:.4f}")
    print(f"  frac test with NN sim >= 0.90: {frac_ge_09:.3f}  "
          f"({int(frac_ge_09*len(nn_sim))} molecules)")
    print(f"  frac test that are exact dup : {frac_eq_1:.3f}  "
          f"({int(frac_eq_1*len(nn_sim))} molecules, NN sim = 1.0)")

    # Chase the Drug_ID overlap flag from the loader inspection.
    tr_ids = set(train_val["Drug_ID"])
    shared = [i for i in test["Drug_ID"] if i in tr_ids]
    print(f"\n  Drug_ID shared train_val/test: {len(shared)} -> {shared[:10]}")
    tr_smiles = set(train_val["Drug"])
    exact_smiles_overlap = sum(1 for s in test["Drug"] if s in tr_smiles)
    print(f"  EXACT SMILES-string overlap  : {exact_smiles_overlap} test molecules "
          f"appear verbatim in train_val")
    return median, mx, nn_sim


def main():
    t0 = time.time()
    group = admet_group(path="data/")
    mean, std = run_baseline(group)
    median, mx, _ = run_leakage_check(group)
    wall = time.time() - t0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  endpoint            : {BENCH}  (metric: MAE, lower better)")
    print(f"  test MAE (5 seeds)  : {mean:.4f} +/- {std:.4f}")
    print(f"  NN Tanimoto median  : {median:.4f}")
    print(f"  NN Tanimoto max     : {mx:.4f}")
    print(f"  WALL-CLOCK (full run): {wall:.1f} s")


if __name__ == "__main__":
    main()
