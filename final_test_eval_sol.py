"""FINAL TEST EVALUATION (regression) -- the ONLY file that reads test for the
solubility loop. Invoked once, after the loop locks its final model.

Reports: validation MAE headline, test point MAE + bootstrap CI, valid-vs-test
gap (flagged), leakage self-audit (NN Tanimoto r=3 + desalted connectivity
InChIKey). MAE is LOWER-is-better; for MAE the LEAKAGE-suspicious direction is
test < valid (test easier), the opposite sign from PR-AUC.

    uv run python final_test_eval_sol.py --version sol_v02_desc_lgbm
"""
import argparse
import importlib
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, inchi
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from features import build_matrix
import data_split_sol as D

RDLogger.DisableLog("rdApp.*")


def ci_hw(x):
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def bootstrap_ci(y, pred, n=5000, seed=0):
    rng = np.random.RandomState(seed); N = len(y); out = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        out.append(mean_absolute_error(y[idx], pred[idx]))
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
    b = group.get(D.ENDPOINT)                     # <-- test enters memory ONLY here
    test = b["test"]; y_test = test["Y"].to_numpy()
    tv = b["train_val"]

    sp = D.splits()
    X_test = build_matrix(test["Drug"].tolist(), mod.FEATURES)
    valid_scores, test_seed, preds = [], [], []
    for s in D.SEEDS:
        tr, va = sp[s]
        Xtr = build_matrix(tr["Drug"].tolist(), mod.FEATURES); ytr = tr["Y"].to_numpy()
        Xva = build_matrix(va["Drug"].tolist(), mod.FEATURES); yva = va["Y"].to_numpy()
        est = mod.build(s).fit(Xtr, ytr)
        valid_scores.append(float(ev(yva, est.predict(Xva))))
        p = est.predict(X_test)
        test_seed.append(float(ev(y_test, p))); preds.append(p)
    vs = np.array(valid_scores); ts = np.array(test_seed)
    pred = np.mean(preds, axis=0)
    point = float(mean_absolute_error(y_test, pred))
    blo, bhi = bootstrap_ci(y_test, pred)
    gap = point - float(vs.mean())

    ref_fps = [f for f in (fp3(s) for s in tv["Drug"]) if f is not None]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fp3(s), ref_fps))
                   for s in test["Drug"] if fp3(s) is not None])
    tv_ik = {ikey(s) for s in tv["Drug"]}; tv_ik.discard(None)
    ik_overlap = sum(1 for s in test["Drug"] if ikey(s) in tv_ik)

    print("=" * 86)
    print(f"FINAL TEST EVAL (regression)  --  {version}  on {D.ENDPOINT}  (MAE, lower better)")
    print("=" * 86)
    print(f"test n={len(y_test)}")
    print(f"\nHEADLINE validation MAE: {vs.mean():.3f} +/- {ci_hw(vs):.3f}  "
          f"(primary generalization estimate)")
    print(f"test read (single shot): point {point:.3f}, bootstrap 95% CI "
          f"[{blo:.3f}, {bhi:.3f}]  (honest test error bar)")
    print(f"test seed CI (STABILITY): +/-{ci_hw(ts):.3f}  (training-seed robustness)")
    # MAE: test < valid (negative gap) is the leakage-suspicious direction.
    flag = abs(gap) > ci_hw(vs)
    suspicious = gap < -ci_hw(vs)
    print(f"VALID-vs-TEST gap       : {gap:+.3f}  "
          f"[{'FLAGGED' if flag else 'ok: valid~test'}"
          f"{'; test EASIER than valid -- LEAKAGE-suspicious direction' if suspicious else ''}]")
    print(f"\nSELF-AUDIT (split leakage):")
    print(f"  NN Tanimoto r=3: median={float(np.median(nn)):.3f}, max={float(np.max(nn)):.3f}")
    print(f"  exact-identity overlap (desalted connectivity InChIKey): {ik_overlap}")
    clean = float(np.max(nn)) < 0.99 and ik_overlap == 0
    print(f"  verdict: {'CLEAN (genuine scaffold split)' if clean else 'FLAG (possible overlap)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    main(ap.parse_args().version)
