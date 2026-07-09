"""FINAL TEST EVALUATION -- the ONLY file that reads the test set.

Invoked ONCE, manually, after the research loop locks its final model version.
The iteration loop (iterate.py / data_split.py / models/) has no path here.

Reports the honest template: validation headline, test point + bootstrap CI,
valid-vs-test gap (flagged), and the split leakage self-audit (NN Tanimoto r=3 +
desalted connectivity-InChIKey identity, vs the PR-AUC prevalence floor).

    uv run python final_test_eval.py --version v05_...
"""
import argparse
import importlib
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from features import build_matrix
import data_split as D

RDLogger.DisableLog("rdApp.*")


def ci_hw(x):
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def bootstrap_ci(y, p, n=5000, seed=0):
    rng = np.random.RandomState(seed); N = len(y); out = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if 0 < y[idx].sum() < len(idx):
            out.append(average_precision_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def ikey(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fr = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    parent = max(fr, key=lambda f: f.GetNumHeavyAtoms()) if fr else m
    try:
        return inchi.MolToInchiKey(parent).split("-")[0]
    except Exception:
        return None


def fp3(smi):
    m = Chem.MolFromSmiles(smi)
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)


def main(version):
    mod = importlib.import_module(f"models.{version}")
    ev = Evaluator(name=D.METRIC)
    group = admet_group(path="data/")
    b = group.get(D.ENDPOINT)                 # <-- test enters memory ONLY here
    test = b["test"]; y_test = test["Y"].to_numpy()
    tv = b["train_val"]
    floor = float((tv["Y"].to_numpy() == 1).mean())

    sp = D.splits()
    X_test = build_matrix(test["Drug"].tolist(), mod.FEATURES)
    valid_scores, test_seed, test_pred_acc = [], [], []
    for s in D.SEEDS:
        tr, va = sp[s]
        Xtr = build_matrix(tr["Drug"].tolist(), mod.FEATURES); ytr = tr["Y"].to_numpy()
        Xva = build_matrix(va["Drug"].tolist(), mod.FEATURES); yva = va["Y"].to_numpy()
        est = mod.build(s).fit(Xtr, ytr)
        valid_scores.append(float(ev(yva, est.predict_proba(Xva)[:, 1])))
        tp = est.predict_proba(X_test)[:, 1]
        test_seed.append(float(ev(y_test, tp))); test_pred_acc.append(tp)
    vs = np.array(valid_scores); ts = np.array(test_seed)
    pred = np.mean(test_pred_acc, axis=0)
    point = float(average_precision_score(y_test, pred))
    blo, bhi = bootstrap_ci(y_test, pred)
    gap = point - float(vs.mean())

    # leakage self-audit (split-level)
    ref_fps = [f for f in (fp3(s) for s in tv["Drug"]) if f is not None]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fp3(s), ref_fps))
                   for s in test["Drug"] if fp3(s) is not None])
    tv_ik = {ikey(s) for s in tv["Drug"]}; tv_ik.discard(None)
    ik_overlap = sum(1 for s in test["Drug"] if ikey(s) in tv_ik)

    print("=" * 86)
    print(f"FINAL TEST EVAL  --  {version}  on {D.ENDPOINT}  ({D.METRIC})")
    print("=" * 86)
    print(f"test n={len(y_test)}, positives={int(y_test.sum())}, floor(prevalence)={floor:.3f}")
    print(f"\nHEADLINE validation {D.METRIC}: {vs.mean():.3f} +/- {ci_hw(vs):.3f}  "
          f"(primary generalization estimate)")
    print(f"test read (single shot)      : point {point:.3f}, bootstrap 95% CI "
          f"[{blo:.3f}, {bhi:.3f}]  (honest test error bar)")
    print(f"test seed CI (STABILITY ONLY): +/-{ci_hw(ts):.3f}  (training-seed robustness; "
          f"not a generalization bar)")
    flag = abs(gap) > ci_hw(vs)
    print(f"VALID-vs-TEST gap            : {gap:+.3f}  "
          f"[{'FLAGGED: test-draw favorable, trust validation' if flag else 'ok: valid~test'}]")
    print(f"\nSELF-AUDIT (split leakage):")
    print(f"  NN Tanimoto r=3: median={float(np.median(nn)):.3f}, max={float(np.max(nn)):.3f}")
    print(f"  exact-identity overlap (desalted connectivity InChIKey): {ik_overlap}")
    clean = float(np.max(nn)) < 0.99 and ik_overlap == 0
    print(f"  verdict: {'CLEAN (genuine scaffold split)' if clean else 'FLAG (possible overlap)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    main(ap.parse_args().version)
