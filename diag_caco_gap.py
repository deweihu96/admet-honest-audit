"""DIAGNOSIS ONLY (caco2_wang): is the -0.098 valid-vs-test MAE gap a benign
easier-but-clean test set, or sub-threshold train/test similarity our InChIKey +
Tanimoto(r3)-max check misses?

No lock change, no re-iteration. Test labels/structures are used ONLY to
characterize the split; nothing here selects on test.

Four checks:
  1. Label (Y) distribution: train_val vs test (moments, range, KS test).
  2. NN-Tanimoto DISTRIBUTION (r3) test->train_val: percentiles + tail counts,
     for caco2 AND solubility/cyp2d6 as reference points.
  3. Gap across ALL caco2 versions (v01,v03-v06 vs v02): split-intrinsic (all
     models) or model-specific (only v02)?
  4. Murcko scaffold overlap: are test scaffolds held out or a subset of train?
"""
import importlib
import numpy as np
from scipy import stats
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from features import build_matrix
import data_split_caco as DC

RDLogger.DisableLog("rdApp.*")
GROUP = admet_group(path="data/")


def fp3(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)


def nn_dist(endpoint):
    b = GROUP.get(endpoint)
    ref = [f for f in (fp3(s) for s in b["train_val"]["Drug"]) if f is not None]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fp3(s), ref))
                   for s in b["test"]["Drug"] if fp3(s) is not None])
    return nn


def scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return None


# ---------- 1. label distribution ----------
print("=" * 84)
print("1. LABEL (Y) DISTRIBUTION  train_val vs test  (caco2_wang)")
print("=" * 84)
b = GROUP.get(DC.ENDPOINT)
ytv = b["train_val"]["Y"].to_numpy(); yte = b["test"]["Y"].to_numpy()
for name, y in [("train_val", ytv), ("test", yte)]:
    print(f"  {name:9s} n={len(y):4d}  mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  p25={np.percentile(y,25):.3f}  med={np.median(y):.3f}  "
          f"p75={np.percentile(y,75):.3f}  max={y.max():.3f}")
ks = stats.ks_2samp(ytv, yte)
print(f"  KS test train_val vs test: D={ks.statistic:.3f}, p={ks.pvalue:.3g}  "
      f"-> {'distributions DIFFER' if ks.pvalue < 0.05 else 'no significant difference'}")

# ---------- 2. NN-similarity distribution ----------
print("\n" + "=" * 84)
print("2. NN-TANIMOTO (r3) DISTRIBUTION  test -> nearest train_val")
print("=" * 84)
print(f"  {'endpoint':14s} {'n':>5s} {'med':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s} "
      f"{'max':>6s} {'>=.7':>5s} {'>=.8':>5s} {'>=.9':>5s}")
for ep, tag in [("caco2_wang", "caco2"), ("solubility_aqsoldb", "solubility"),
                ("cyp2d6_substrate_carbonmangels", "cyp2d6")]:
    nn = nn_dist(ep)
    print(f"  {tag:14s} {len(nn):>5d} {np.median(nn):>6.3f} {np.percentile(nn,90):>6.3f} "
          f"{np.percentile(nn,95):>6.3f} {np.percentile(nn,99):>6.3f} {nn.max():>6.3f} "
          f"{int((nn>=0.7).sum()):>5d} {int((nn>=0.8).sum()):>5d} {int((nn>=0.9).sum()):>5d}")

# ---------- 3. gap across all caco2 versions ----------
print("\n" + "=" * 84)
print("3. VALID-vs-TEST GAP across ALL caco2 versions (split-intrinsic vs model-specific)")
print("=" * 84)
ev = Evaluator(name=DC.METRIC)
sp = DC.splits()
test = b["test"]; y_test = yte
X_test = {}
versions = ["caco_v01_morgan_lgbm", "caco_v02_desc_lgbm",
            "caco_v03_desc_morgan_lgbm_tuned", "caco_v04_stacked_reg",
            "caco_v05_late_fusion_reg", "caco_v06_two_stage_residual"]
print(f"  {'version':32s} {'valid MAE':>9s} {'test MAE':>9s} {'gap':>8s}")
for v in versions:
    mod = importlib.import_module(f"models.{v}")
    key = tuple(mod.FEATURES)
    if key not in X_test:
        X_test[key] = build_matrix(test["Drug"].tolist(), mod.FEATURES)
    Xte = X_test[key]
    vscore, tscore = [], []
    for s in DC.SEEDS:
        tr, va = sp[s]
        Xtr = build_matrix(tr["Drug"].tolist(), mod.FEATURES); ytr = tr["Y"].to_numpy()
        Xva = build_matrix(va["Drug"].tolist(), mod.FEATURES); yva = va["Y"].to_numpy()
        est = mod.build(s).fit(Xtr, ytr)
        vscore.append(float(ev(yva, est.predict(Xva))))
        tscore.append(float(ev(y_test, est.predict(Xte))))
    vm, tm = np.mean(vscore), np.mean(tscore)
    print(f"  {v:32s} {vm:>9.4f} {tm:>9.4f} {tm-vm:>+8.4f}")

# ---------- 4. scaffold overlap ----------
print("\n" + "=" * 84)
print("4. MURCKO SCAFFOLD OVERLAP (is test held out or a subset of train?)")
print("=" * 84)
for ep, tag in [("caco2_wang", "caco2"), ("solubility_aqsoldb", "solubility"),
                ("cyp2d6_substrate_carbonmangels", "cyp2d6")]:
    bb = GROUP.get(ep)
    tv_scaf = {scaffold(s) for s in bb["train_val"]["Drug"]}; tv_scaf.discard(None); tv_scaf.discard("")
    te_scaf_list = [scaffold(s) for s in bb["test"]["Drug"]]
    te_scaf = {s for s in te_scaf_list if s}
    shared = te_scaf & tv_scaf
    # per-molecule: fraction of test molecules whose scaffold is in train
    mol_in_train = sum(1 for s in te_scaf_list if s and s in tv_scaf)
    print(f"  {tag:11s}: test unique scaffolds={len(te_scaf):4d}, shared with train="
          f"{len(shared):4d} ({len(shared)/max(len(te_scaf),1):.1%} of test scaffolds); "
          f"test MOLECULES whose scaffold is in train: {mol_in_train}/{len(te_scaf_list)} "
          f"({mol_in_train/len(te_scaf_list):.1%})")
