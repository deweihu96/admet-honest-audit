"""Generalized honest baseline harness: regression AND classification.

Per-endpoint metric/direction resolve by EXACT endpoint name from TDC's
admet_metrics (never by assay family; cyp3a4_substrate -> roc-auc, not the
substrate-family default). Classification endpoints train a classifier and emit
PROBABILITIES (AUROC and PR-AUC both need scores, not labels).

Wall rule (CLAUDE.md): train on `train`, tune/select on `valid`, score `test`
exactly once per seed via group.evaluate_many. Official 5-seed protocol.
"""
import time
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import lightgbm as lgb
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics

RDLogger.DisableLog("rdApp.*")

SEEDS = [1, 2, 3, 4, 5]
FP_RADIUS_MODEL = 2      # ECFP4 modeling features (leakage check uses r=3; see CLAUDE.md)
N_BITS = 2048

# Direction per metric, empirically verified against TDC's scorer (verify_direction.py).
DIRECTION = {"mae": "lower", "spearman": "higher", "roc-auc": "higher", "pr-auc": "higher"}
# LightGBM early-stopping metric that matches each official metric as closely as possible.
EARLYSTOP_METRIC = {"mae": "l1", "spearman": "l2", "roc-auc": "auc", "pr-auc": "average_precision"}
CLASSIFICATION_METRICS = {"roc-auc", "pr-auc"}


def metric_for(endpoint):
    """Official metric for an EXACT endpoint name (per-endpoint, no family default)."""
    return admet_metrics[endpoint.lower()]


def morgan_matrix(smiles, radius=FP_RADIUS_MODEL, n_bits=N_BITS):
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32)); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows)


def run_endpoint(group, endpoint):
    metric = metric_for(endpoint)
    is_clf = metric in CLASSIFICATION_METRICS
    evaluator = Evaluator(name=metric)          # exact TDC scorer

    benchmark = group.get(endpoint)
    name = benchmark["name"]
    test = benchmark["test"]

    # For classification, confirm binary labels and that positive == Y==1.
    if is_clf:
        y_tv = benchmark["train_val"]["Y"].to_numpy()
        uniq = set(np.unique(y_tv).tolist())
        assert uniq <= {0.0, 1.0}, f"{endpoint}: labels not binary: {uniq}"
        n_pos = int((y_tv == 1).sum())
        print(f"  [{endpoint}] metric={metric} ({DIRECTION[metric]}-better), "
              f"binary=YES, positive=Y==1, train_val positives={n_pos}/{len(y_tv)} "
              f"({n_pos/len(y_tv):.1%})")
    else:
        print(f"  [{endpoint}] metric={metric} ({DIRECTION[metric]}-better), regression")

    X_test = morgan_matrix(test["Drug"].tolist())   # features only; no test labels read

    valid_scores, predictions_list = [], []
    for seed in SEEDS:
        train, valid = group.get_train_valid_split(benchmark=endpoint,
                                                    split_type="default", seed=seed)
        X_tr = morgan_matrix(train["Drug"].tolist()); y_tr = train["Y"].to_numpy()
        X_va = morgan_matrix(valid["Drug"].tolist()); y_va = valid["Y"].to_numpy()

        common = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, random_state=seed,
                      n_jobs=-1, verbose=-1)
        if is_clf:
            model = lgb.LGBMClassifier(**common)
        else:
            model = lgb.LGBMRegressor(**common)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  eval_metric=EARLYSTOP_METRIC[metric],
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # Predict PROBABILITIES for classification, values for regression.
        va_pred = model.predict_proba(X_va)[:, 1] if is_clf else model.predict(X_va)
        valid_scores.append(float(evaluator(y_va, va_pred)))   # valid = allowed to score

        te_pred = model.predict_proba(X_test)[:, 1] if is_clf else model.predict(X_test)
        predictions_list.append({name: te_pred})               # test scored once, below

    # Official test scoring: evaluate_many touches test once per seed, returns [mean, std].
    test_mean, test_std = group.evaluate_many(predictions_list)[name]
    v = np.array(valid_scores)
    return {"endpoint": endpoint, "metric": metric,
            "valid_mean": float(v.mean()), "valid_std": float(v.std()),
            "test_mean": test_mean, "test_std": test_std}


def main():
    t0 = time.time()
    group = admet_group(path="data/")

    # Validate on exactly two endpoints: one balanced ROC-AUC, one small/imbalanced PR-AUC.
    endpoints = ["pgp_broccatelli", "cyp2d6_substrate_carbonmangels"]
    results = []
    for ep in endpoints:
        print("=" * 78)
        results.append(run_endpoint(group, ep))
    wall = time.time() - t0

    print("\n" + "=" * 78)
    print("CLASSIFICATION HARNESS VALIDATION (5-seed, mean +/- std)")
    print("=" * 78)
    hdr = ("endpoint", "metric", "valid (mean+/-std)", "test (mean+/-std)")
    print(f"{hdr[0]:32s} {hdr[1]:8s} {hdr[2]:22s} {hdr[3]:22s}")
    print("-" * 88)
    for r in results:
        print(f"{r['endpoint']:32s} {r['metric']:8s} "
              f"{r['valid_mean']:.4f} +/- {r['valid_std']:.4f}      "
              f"{r['test_mean']:.4f} +/- {r['test_std']:.4f}")
    print(f"\nWALL-CLOCK (2 endpoints x 5 seeds): {wall:.1f} s")


if __name__ == "__main__":
    main()
