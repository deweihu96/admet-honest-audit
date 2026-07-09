"""Single-endpoint climb + self-audit loop on cyp2d6_substrate (hard case:
small, imbalanced, PR-AUC). The point of this build is a CORRECT loop, not a
wide search. If the loop is honest here, it is honest anywhere.

Pipeline:
  1. Propose a modest candidate set (featurizer x model family x a few HPs).
  2. Each candidate: fit on `train` ONLY, score on `valid`, across the endpoint's
     budgeted seed count (25, per the variance analysis). Record validation
     mean + 95% CI. (No early stopping -> valid is a clean selection signal.)
  3. SELECTION = variance-aware CI-overlap (policy b):
       - FLOOR (first-class guard): candidate must beat the PR-AUC naive
         baseline (positive prevalence). Ineligible otherwise.
       - Among floor-passers, TOP = highest validation mean; TIED = every
         floor-passer whose 95% CI overlaps TOP's 95% CI.
       - CAP at 8; if more tie, keep highest validation MEAN. Mean is used ONLY
         to break ties inside a statistically-indistinguishable group, NEVER to
         make the primary cut.
  4. ENSEMBLE selected candidates (average predicted probabilities).
  5. TEST: score the ensemble on the fixed test set once, across the 25 seeds.
     Reported, never used for selection.
  6. SELF-AUDIT: split-level NN-Tanimoto leakage check (r=3, train_val ref).

WALL (CLAUDE.md): fit uses `train` only; selection uses `valid` only; test
predictions are computed during fitting but SCORED exactly once, at the end.
"""
import time
import numpy as np
from scipy import stats
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from run_dryrun import nn_tanimoto            # reuse the audited leakage function

RDLogger.DisableLog("rdApp.*")

ENDPOINT = "cyp2d6_substrate_carbonmangels"
METRIC = "pr-auc"
SEEDS = list(range(1, 26))                    # 25-seed budget for this hard endpoint
N_BITS = 2048
CAP = 8

# ---------------------------------------------------------------- featurizers
_DESC_NAMES = [d[0] for d in Descriptors._descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)
_fp_cache, _desc_cache = {}, {}


def _morgan(smi, radius):
    key = (smi, radius)
    if key not in _fp_cache:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            _fp_cache[key] = np.zeros(N_BITS, dtype=np.float32)
        else:
            v = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=N_BITS)
            arr = np.zeros(N_BITS, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(v, arr)
            _fp_cache[key] = arr
    return _fp_cache[key]


def _desc(smi):
    if smi not in _desc_cache:
        mol = Chem.MolFromSmiles(smi)
        vals = (np.zeros(len(_DESC_NAMES), dtype=np.float32) if mol is None
                else np.array(_DESC_CALC.CalcDescriptors(mol), dtype=np.float32))
        _desc_cache[smi] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return _desc_cache[smi]


def featurize(smiles, kind):
    if kind == "morgan_r2":
        return np.vstack([_morgan(s, 2) for s in smiles])
    if kind == "morgan_r3":
        return np.vstack([_morgan(s, 3) for s in smiles])
    if kind == "desc":
        return np.vstack([_desc(s) for s in smiles])
    raise ValueError(kind)


# ---------------------------------------------------------------- candidates
def lgbm(**kw):
    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    base.update(kw)
    return lambda seed: lgb.LGBMClassifier(random_state=seed, **base)


def rf(**kw):
    return lambda seed: RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                               random_state=seed, **kw)


def logreg_bits(C):
    return lambda seed: LogisticRegression(C=C, max_iter=2000, random_state=seed)


def logreg_desc(C):
    # descriptors need scaling for a linear model
    return lambda seed: make_pipeline(StandardScaler(),
                                      LogisticRegression(C=C, max_iter=2000,
                                                         random_state=seed))


CANDIDATES = [
    ("morgan_r2 + LGBM",        "morgan_r2", lgbm()),
    ("morgan_r3 + LGBM",        "morgan_r3", lgbm()),
    ("morgan_r2 + LGBM(reg)",   "morgan_r2", lgbm(n_estimators=200, num_leaves=15,
                                                  reg_lambda=1.0)),
    ("morgan_r2 + RF",          "morgan_r2", rf()),
    ("morgan_r3 + RF",          "morgan_r3", rf()),
    ("morgan_r2 + LogReg(C1)",  "morgan_r2", logreg_bits(1.0)),
    ("morgan_r2 + LogReg(C.1)", "morgan_r2", logreg_bits(0.1)),
    ("desc + LGBM",             "desc",      lgbm()),
    ("desc + RF",               "desc",      rf()),
    ("desc + LogReg",           "desc",      logreg_desc(1.0)),
]


def ci_halfwidth(x):
    n = len(x)
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def main():
    t0 = time.time()
    group = admet_group(path="data/")
    evaluator = Evaluator(name=METRIC)
    benchmark = group.get(ENDPOINT)
    test = benchmark["test"]
    y_test = test["Y"].to_numpy()               # scorer-only; never for selection

    # PR-AUC naive baseline = positive prevalence (from train_val, allowed).
    y_tv = benchmark["train_val"]["Y"].to_numpy()
    floor = float((y_tv == 1).mean())

    # Pre-split per seed (test features per featurizer cached implicitly).
    splits = {seed: group.get_train_valid_split(benchmark=ENDPOINT,
              split_type="default", seed=seed) for seed in SEEDS}

    n_fits = 0
    cand_val = {}                                # name -> array of 25 valid scores
    test_probs = {}                              # name -> {seed -> test prob vector}
    for name, kind, make in CANDIDATES:
        X_test = featurize(test["Drug"].tolist(), kind)
        vals, tp = [], {}
        for seed in SEEDS:
            train, valid = splits[seed]
            X_tr = featurize(train["Drug"].tolist(), kind); y_tr = train["Y"].to_numpy()
            X_va = featurize(valid["Drug"].tolist(), kind); y_va = valid["Y"].to_numpy()
            model = make(seed)
            model.fit(X_tr, y_tr)                # fit on TRAIN only (no valid, no test)
            n_fits += 1
            vals.append(float(evaluator(y_va, model.predict_proba(X_va)[:, 1])))
            tp[seed] = model.predict_proba(X_test)[:, 1]   # stashed, NOT scored yet
        cand_val[name] = np.array(vals)
        test_probs[name] = tp

    # -------- build candidate table + selection (validation only) --------
    stats_by = {}
    for name, _, _ in CANDIDATES:
        v = cand_val[name]
        m, hw = float(v.mean()), float(ci_halfwidth(v))
        stats_by[name] = dict(mean=m, hw=hw, ci=(m - hw, m + hw),
                              floor_pass=m > floor)

    eligible = [n for n, _, _ in CANDIDATES if stats_by[n]["floor_pass"]]
    top = max(eligible, key=lambda n: stats_by[n]["mean"])
    top_ci = stats_by[top]["ci"]
    tied = [n for n in eligible if overlaps(stats_by[n]["ci"], top_ci)]
    capped_note = ""
    if len(tied) > CAP:
        tied = sorted(tied, key=lambda n: stats_by[n]["mean"], reverse=True)[:CAP]
        capped_note = f" (capped from qualifying set to {CAP} by mean tiebreak)"
    selected = set(tied)

    # -------- ENSEMBLE on TEST: score once, across seeds --------
    ens_scores = []
    for seed in SEEDS:
        probs = np.mean([test_probs[n][seed] for n in selected], axis=0)
        ens_scores.append(float(evaluator(y_test, probs)))
    ens_scores = np.array(ens_scores)
    ens_mean, ens_hw = float(ens_scores.mean()), float(ci_halfwidth(ens_scores))

    # -------- SELF-AUDIT: split-level leakage check --------
    nn = nn_tanimoto(test["Drug"].tolist(), benchmark["train_val"]["Drug"].tolist())
    nn_median, nn_max = float(np.median(nn)), float(np.max(nn))

    wall = time.time() - t0

    # ---------------------------- report ----------------------------
    print("=" * 92)
    print(f"CLIMB + AUDIT  --  {ENDPOINT}  (metric {METRIC}, higher-better, "
          f"{len(SEEDS)} seeds)")
    print("=" * 92)
    print(f"PR-AUC naive baseline (positive prevalence): {floor:.4f}")
    print(f"TOP candidate (highest valid mean): {top}\n")
    hdr = ("candidate", "featurizer", "valid mean +/- 95%CI", "floor", "selected")
    print(f"{hdr[0]:24s} {hdr[1]:11s} {hdr[2]:24s} {hdr[3]:6s} {hdr[4]}")
    print("-" * 92)
    for name, kind, _ in sorted(CANDIDATES, key=lambda c: stats_by[c[0]]["mean"],
                                reverse=True):
        s = stats_by[name]
        print(f"{name:24s} {kind:11s} "
              f"{s['mean']:.4f} +/- {s['hw']:.4f}        "
              f"{'PASS' if s['floor_pass'] else 'FAIL':6s} "
              f"{'YES' if name in selected else 'no'}")

    print("\n" + "-" * 92)
    print(f"CI-overlap selection: {len(tied)} candidate(s) tied at the top{capped_note}")
    print(f"  selected: {sorted(selected)}")
    print(f"\nENSEMBLE TEST {METRIC} (scored once, {len(SEEDS)} seeds): "
          f"{ens_mean:.4f} +/- {ens_hw:.4f}   -> 95% CI [{ens_mean-ens_hw:.4f}, "
          f"{ens_mean+ens_hw:.4f}]")
    separated = (ens_mean - ens_hw) > floor
    print(f"  test CI lower bound {ens_mean-ens_hw:.4f} vs baseline {floor:.4f}: "
          f"{'SEPARATED from baseline' if separated else 'NOT separated -- overlaps baseline'}")

    print(f"\nSELF-AUDIT (split leakage, NN Tanimoto r=3, train_val ref):")
    print(f"  median NN sim = {nn_median:.4f}   max NN sim = {nn_max:.4f}")

    print(f"\nCOMPUTE: {len(CANDIDATES)} candidates x {len(SEEDS)} seeds = "
          f"{n_fits} fits (test predictions reused from those fits; no extra).")
    print(f"WALL-CLOCK: {wall:.1f} s")


if __name__ == "__main__":
    main()
