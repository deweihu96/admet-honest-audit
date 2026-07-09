"""Variance-regime check for a small imbalanced PR-AUC endpoint.

Question: is the seed-to-seed test-PR-AUC spread on cyp2d6_substrate SEED NOISE
(shrinks at sqrt(n) when we resample -> resampling fixes selection) or INTRINSIC
to the tiny positive count (CI plateaus -> more seeds won't save selection)?

Method: train one model per seed (each seed reshuffles train/valid), re-score
each on the FIXED test set. This is re-scoring across seeds, never selection
against test. Then for n in {5,10,20,40} report mean test PR-AUC and the 95% CI
half-width t(n-1)*std/sqrt(n), and check whether the half-width halves from
5->20 seeds (the sqrt(n) prediction).

Caveat stated in the report: this CI is over the train/valid seed distribution
on ONE fixed 135-molecule test set; it does NOT include test-set sampling error,
which no number of seeds removes.
"""
import numpy as np
from scipy import stats
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
import lightgbm as lgb
from tdc import Evaluator
from tdc.benchmark_group import admet_group

RDLogger.DisableLog("rdApp.*")

ENDPOINT = "cyp2d6_substrate_carbonmangels"
METRIC = "pr-auc"
MAX_SEEDS = 40
FP_RADIUS, N_BITS = 2, 2048
TARGET_HW = 0.02


def fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(N_BITS, dtype=np.float32)
    v = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=N_BITS)
    arr = np.zeros(N_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(v, arr)
    return arr


def main():
    group = admet_group(path="data/")
    evaluator = Evaluator(name=METRIC)
    benchmark = group.get(ENDPOINT)
    test = benchmark["test"]

    # Cache fingerprints once (molecules are stable; only split membership varies).
    cache = {s: fp(s) for s in set(benchmark["train_val"]["Drug"]).union(test["Drug"])}
    X_test = np.vstack([cache[s] for s in test["Drug"]])   # features only
    y_test = test["Y"].to_numpy()   # touched ONLY by the scorer below, never for selection

    # Score the SAME modeling recipe once per seed on the fixed test set.
    seed_scores = []
    for seed in range(1, MAX_SEEDS + 1):
        train, valid = group.get_train_valid_split(benchmark=ENDPOINT,
                                                    split_type="default", seed=seed)
        X_tr = np.vstack([cache[s] for s in train["Drug"]]); y_tr = train["Y"].to_numpy()
        X_va = np.vstack([cache[s] for s in valid["Drug"]]); y_va = valid["Y"].to_numpy()
        model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=seed, n_jobs=-1, verbose=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        te_pred = model.predict_proba(X_test)[:, 1]
        seed_scores.append(float(evaluator(y_test, te_pred)))   # re-score, no selection
    seed_scores = np.array(seed_scores)

    def ci_halfwidth(x):
        n = len(x)
        std = x.std(ddof=1)
        sem = std / np.sqrt(n)
        hw = stats.t.ppf(0.975, n - 1) * sem
        return x.mean(), std, hw

    print(f"endpoint: {ENDPOINT}   metric: {METRIC}   test size: {len(test)}")
    print(f"{'seeds':>6} {'mean':>8} {'std':>8} {'95% CI half-width':>18}")
    print("-" * 44)
    results = {}
    for n in [5, 10, 20, 40]:
        mean, std, hw = ci_halfwidth(seed_scores[:n])
        results[n] = (mean, std, hw)
        print(f"{n:>6} {mean:>8.4f} {std:>8.4f} {hw:>18.4f}")

    # sqrt(n) test: half-width should ~halve from 5 -> 20 seeds (sqrt(20/5)=2).
    hw5, hw20, hw40 = results[5][2], results[20][2], results[40][2]
    print("\nsqrt(n) check (CI half-width should scale ~1/sqrt(n)):")
    print(f"  hw(5)/hw(20)  = {hw5/hw20:.2f}   (sqrt(n) predicts 2.00)")
    print(f"  hw(5)/hw(40)  = {hw5/hw40:.2f}   (sqrt(n) predicts 2.83)")

    # Is std stable across n? (If std grows with n -> heavy tails -> not clean seed noise.)
    print("\nstd stability across seed counts (stable std => i.i.d. seed noise):")
    for n in [5, 10, 20, 40]:
        print(f"  std(first {n:>2}) = {results[n][1]:.4f}")

    # Seeds needed to reach target half-width, using the most stable std (n=40).
    std40 = results[40][1]
    n_needed = int(np.ceil((1.96 * std40 / TARGET_HW) ** 2))
    print(f"\nseeds to reach 95% CI half-width < {TARGET_HW} (using std@40={std40:.4f}): "
          f"~{n_needed}")


if __name__ == "__main__":
    main()
