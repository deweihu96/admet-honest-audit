"""Corrected endpoint report for cyp2d6_substrate -- the template as it will
appear when we scale. Load-bearing reporting fixes:

 (a) HEADLINE test error bar = test-set BOOTSTRAP CI (resample test molecules),
     NOT the seed CI. The seed CI is reported but labeled stability-only.
 (b) VALIDATION mean +/- CI is the primary generalization estimate; the one-shot
     test read is a bootstrap-bounded confirmation.
 (c) VALID-vs-TEST gap is a first-class reported quantity and a second audit
     axis (catches sampling-luck the Tanimoto check cannot see).
 Plus: report BOTH the ensemble AND the best single model (each with bootstrap
 CI). Whether ensembling helped is a reported finding, not a silent gate.

WALL: fit on train, select on valid; test scored/bootstrapped for reporting
only, never for selection.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from climb_cyp2d6 import CANDIDATES, featurize, ci_halfwidth, overlaps, ENDPOINT, METRIC, CAP
from run_dryrun import nn_tanimoto

RDLogger.DisableLog("rdApp.*")
SEEDS = list(range(1, 26))
evaluator = Evaluator(name=METRIC)


def bootstrap_ci(y_true, pred, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    N = len(y_true); out = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if 0 < y_true[idx].sum() < len(idx):
            out.append(average_precision_score(y_true[idx], pred[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def ikey(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    parent = max(frags, key=lambda f: f.GetNumHeavyAtoms()) if frags else m
    try:
        return inchi.MolToInchiKey(parent).split("-")[0]
    except Exception:
        return None


def main():
    group = admet_group(path="data/")
    b = group.get(ENDPOINT)
    test = b["test"]; y_test = test["Y"].to_numpy()
    tv = b["train_val"]
    floor = float((tv["Y"].to_numpy() == 1).mean())

    splits = {s: group.get_train_valid_split(benchmark=ENDPOINT, split_type="default",
                                             seed=s) for s in SEEDS}
    valid_y = {s: splits[s][1]["Y"].to_numpy() for s in SEEDS}
    # fit all candidates; stash valid scores (for selection) + valid/test probs
    valid_scores, valid_probs, test_probs = {}, {}, {}
    for name, kind, make in CANDIDATES:
        X_test = featurize(test["Drug"].tolist(), kind)
        vs, vp, tp = [], {}, {}
        for s in SEEDS:
            tr, va = splits[s]
            Xtr = featurize(tr["Drug"].tolist(), kind); ytr = tr["Y"].to_numpy()
            Xva = featurize(va["Drug"].tolist(), kind)
            m = make(s).fit(Xtr, ytr)
            vp[s] = m.predict_proba(Xva)[:, 1]
            vs.append(float(evaluator(valid_y[s], vp[s])))
            tp[s] = m.predict_proba(X_test)[:, 1]
        valid_scores[name] = np.array(vs); valid_probs[name] = vp; test_probs[name] = tp
    n_fits = len(CANDIDATES) * len(SEEDS)

    # selection (validation only)
    sb = {n: (valid_scores[n].mean(), ci_halfwidth(valid_scores[n])) for n, _, _ in CANDIDATES}
    sb = {n: (m, hw, (m - hw, m + hw)) for n, (m, hw) in sb.items()}
    elig = [n for n in sb if sb[n][0] > floor]
    top = max(elig, key=lambda n: sb[n][0])
    tied = [n for n in elig if overlaps(sb[n][2], sb[top][2])]
    if len(tied) > CAP:
        tied = sorted(tied, key=lambda n: sb[n][0], reverse=True)[:CAP]
    ensemble = set(tied)
    best_single = top

    # per-model reporting helper
    def report_model(members, label):
        # validation computed the SAME way as test: average member probabilities
        # per seed, then score (not a mean of member scores).
        vseed = np.array([float(evaluator(valid_y[s],
                          np.mean([valid_probs[n][s] for n in members], 0))) for s in SEEDS])
        vmean, vhw = float(vseed.mean()), float(ci_halfwidth(vseed))
        tseed = np.array([float(evaluator(y_test, np.mean([test_probs[n][s] for n in members], 0)))
                          for s in SEEDS])
        tmean, thw = float(tseed.mean()), float(ci_halfwidth(tseed))     # stability-only
        pred = np.mean([np.mean([test_probs[n][s] for n in members], 0) for s in SEEDS], 0)
        point = float(average_precision_score(y_test, pred))
        blo, bhi = bootstrap_ci(y_test, pred)
        gap = point - vmean
        return dict(label=label, n=len(members), vmean=vmean, vhw=vhw,
                    tseed_mean=tmean, tseed_hw=thw, point=point, blo=blo, bhi=bhi, gap=gap)

    ens = report_model(ensemble, f"ENSEMBLE ({len(ensemble)} models)")
    sng = report_model({best_single}, f"BEST SINGLE ({best_single})")

    # leakage self-audit (split-level)
    nn = nn_tanimoto(test["Drug"].tolist(), tv["Drug"].tolist())
    tv_ik = {ikey(s) for s in tv["Drug"]}; tv_ik.discard(None)
    ik_overlap = sum(1 for s in test["Drug"] if ikey(s) in tv_ik)

    # ---------------------------- report ----------------------------
    print("=" * 88)
    print(f"ENDPOINT REPORT  --  {ENDPOINT}   metric={METRIC} (higher-better)   "
          f"seeds={len(SEEDS)}")
    print("=" * 88)
    print(f"test n={len(y_test)}, positives={int(y_test.sum())};  "
          f"PR-AUC naive floor (prevalence)={floor:.3f}")

    for r in (ens, sng):
        flag = abs(r["gap"]) > r["vhw"]
        print(f"\n{r['label']}")
        print(f"  HEADLINE  validation {METRIC}      : {r['vmean']:.3f} +/- {r['vhw']:.3f}"
              f"   <- primary honest generalization estimate")
        print(f"  test read (single shot)         : point {r['point']:.3f}, "
              f"bootstrap 95% CI [{r['blo']:.3f}, {r['bhi']:.3f}]  <- honest test error bar")
        print(f"  test seed CI (STABILITY ONLY)   : +/-{r['tseed_hw']:.3f}  "
              f"(robustness to training seed on a FIXED test set; NOT a generalization bar)")
        print(f"  VALID-vs-TEST gap               : {r['gap']:+.3f}  "
              f"[{'FLAGGED: test-draw favorable, trust validation' if flag else 'ok: valid~test'}]")

    ens_helps = ens["vmean"] > sng["vmean"] + 1e-9
    print(f"\nENSEMBLE vs BEST-SINGLE (reported finding, not a gate):")
    print(f"  ensemble valid {ens['vmean']:.3f} vs best-single valid {sng['vmean']:.3f}  "
          f"-> ensembling {'helps' if ens_helps else 'does NOT help'} on validation "
          f"(delta {ens['vmean']-sng['vmean']:+.3f})")

    print(f"\nSELF-AUDIT (split leakage):")
    print(f"  NN Tanimoto r=3: median={float(np.median(nn)):.3f}, max={float(np.max(nn)):.3f}")
    print(f"  exact-identity overlap (desalted connectivity InChIKey): {ik_overlap}")
    print(f"  verdict: {'CLEAN' if (np.max(nn) < 0.99 and ik_overlap == 0) else 'FLAG'} "
          f"(genuine scaffold split, no train/test molecular overlap)")

    print(f"\nCOMPUTE: {n_fits} fits ({len(CANDIDATES)} candidates x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
