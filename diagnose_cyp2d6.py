"""GATE diagnosis: explain the ensemble valid<test inversion on cyp2d6_substrate
BEFORE any scaling. Four questions:

  1. Ensemble's OWN validation CI vs its test CI (same scoring as single models).
  2. LEVEL vs STABILITY: is test systematically above valid seed-by-seed (paired),
     i.e. a level shift, distinct from the tight-CI stability story?
  3. Test-set SAMPLING floor: bootstrap the ~135-molecule / ~38-positive test set
     to get the honest error bar on 0.70 that no number of seeds removes.
  4. Composition stability: re-run selection on disjoint seed blocks; does the
     ensemble membership (and the test number) wobble given the sharp tie edge?

WALL: fits use train only; selection uses valid only. Test is scored/bootstrapped
for CHARACTERIZATION only; nothing here selects on test.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from tdc import Evaluator
from tdc.benchmark_group import admet_group
from climb_cyp2d6 import (CANDIDATES, featurize, ci_halfwidth, overlaps,
                          ENDPOINT, METRIC, CAP)

SEEDS_ALL = list(range(1, 101))
BLOCKS = {"seeds 1-25": list(range(1, 26)), "seeds 26-50": list(range(26, 51)),
          "seeds 51-75": list(range(51, 76)), "seeds 76-100": list(range(76, 101))}

evaluator = Evaluator(name=METRIC)
group = admet_group(path="data/")
benchmark = group.get(ENDPOINT)
test = benchmark["test"]
y_test = test["Y"].to_numpy()
y_tv = benchmark["train_val"]["Y"].to_numpy()
floor = float((y_tv == 1).mean())

# sanity: TDC pr-auc == sklearn average_precision (so bootstrap matches the scorer)
_p = np.linspace(0, 1, len(y_test))
assert abs(float(evaluator(y_test, _p)) - average_precision_score(y_test, _p)) < 1e-9

splits = {s: group.get_train_valid_split(benchmark=ENDPOINT, split_type="default",
                                         seed=s) for s in SEEDS_ALL}

# ---- Phase A: fit every candidate on every seed; stash valid/test probs ----
valid_probs, test_probs, valid_scores = {}, {}, {}
for name, kind, make in CANDIDATES:
    X_test = featurize(test["Drug"].tolist(), kind)
    valid_probs[name], test_probs[name], valid_scores[name] = {}, {}, {}
    for s in SEEDS_ALL:
        tr, va = splits[s]
        Xtr = featurize(tr["Drug"].tolist(), kind); ytr = tr["Y"].to_numpy()
        Xva = featurize(va["Drug"].tolist(), kind); yva = va["Y"].to_numpy()
        m = make(s).fit(Xtr, ytr)
        vp = m.predict_proba(Xva)[:, 1]; tp = m.predict_proba(X_test)[:, 1]
        valid_probs[name][s] = vp; test_probs[name][s] = tp
        valid_scores[name][s] = float(evaluator(yva, vp))
n_fits = len(CANDIDATES) * len(SEEDS_ALL)


def select_on(seeds):
    sb = {}
    for name, _, _ in CANDIDATES:
        v = np.array([valid_scores[name][s] for s in seeds])
        m, hw = float(v.mean()), float(ci_halfwidth(v))
        sb[name] = (m, hw, (m - hw, m + hw))
    elig = [n for n in sb if sb[n][0] > floor]
    top = max(elig, key=lambda n: sb[n][0])
    tied = [n for n in elig if overlaps(sb[n][2], sb[top][2])]
    if len(tied) > CAP:
        tied = sorted(tied, key=lambda n: sb[n][0], reverse=True)[:CAP]
    return set(tied), sb, top


def ens_valid(selected, seeds):
    out = []
    for s in seeds:
        _, va = splits[s]; yva = va["Y"].to_numpy()
        out.append(float(evaluator(yva, np.mean([valid_probs[n][s] for n in selected], 0))))
    return np.array(out)


def ens_test(selected, seeds):
    return np.array([float(evaluator(y_test, np.mean([test_probs[n][s] for n in selected], 0)))
                     for s in seeds])


def summ(x):
    return f"{x.mean():.4f} +/- {ci_halfwidth(x):.4f}"


print("=" * 90)
print(f"GATE DIAGNOSIS  --  {ENDPOINT}  ({METRIC}); floor={floor:.4f}; test n={len(y_test)}, "
      f"positives={int(y_test.sum())}")
print("=" * 90)

# ---- (1)+(2) ensemble valid vs test on the main block (seeds 1-25) ----
block1 = BLOCKS["seeds 1-25"]
sel1, sb1, top1 = select_on(block1)
ev = ens_valid(sel1, block1)
et = ens_test(sel1, block1)
print(f"\n[1] ENSEMBLE valid vs test (seeds 1-25, {len(sel1)} members)")
print(f"    selected: {sorted(sel1)}")
print(f"    ensemble VALID {METRIC}: {summ(ev)}")
print(f"    ensemble TEST  {METRIC}: {summ(et)}")
best_single_valid = max(sb1[n][0] for n, _, _ in CANDIDATES)
print(f"    best SINGLE-candidate valid mean: {best_single_valid:.4f}")

print(f"\n[2] LEVEL vs STABILITY -- paired per-seed (test - valid), same seed")
diff = et - ev
print(f"    mean paired (test - valid): {diff.mean():+.4f} +/- {ci_halfwidth(diff):.4f}")
print(f"    seeds with test>valid: {(diff>0).sum()}/{len(diff)}")
print(f"    -> ensemble valid level {ev.mean():.3f} vs test level {et.mean():.3f}: "
      f"{'INVERSION persists (test systematically higher)' if diff.mean()>0.02 else 'consistent'}")

# ---- (3) bootstrap the test set ----
pred = np.mean([np.mean([test_probs[n][s] for n in sel1], 0) for s in block1], 0)
point = average_precision_score(y_test, pred)
rng = np.random.RandomState(0)
N = len(y_test); boots = []
for _ in range(5000):
    idx = rng.randint(0, N, N)
    if 0 < y_test[idx].sum() < len(idx):
        boots.append(average_precision_score(y_test[idx], pred[idx]))
boots = np.array(boots)
blo, bhi = np.percentile(boots, [2.5, 97.5])
print(f"\n[3] TEST-SET SAMPLING bootstrap (resample {N} test mols w/ replacement, 5000x)")
print(f"    point ensemble test {METRIC}: {point:.4f}")
print(f"    bootstrap 95% CI: [{blo:.4f}, {bhi:.4f}]  (half-width ~{(bhi-blo)/2:.4f})")
print(f"    seed-only CI half-width was ~0.007 -> sampling error is "
      f"{((bhi-blo)/2)/0.007:.1f}x larger")
print(f"    Is a leaderboard 0.70 inside this CI? {'YES' if blo <= 0.70 <= bhi else 'no'}")

# ---- (4) composition stability across disjoint seed blocks ----
print(f"\n[4] COMPOSITION STABILITY across disjoint 25-seed blocks")
print(f"    {'block':14s} {'n_sel':5s} {'ens test':16s} members")
memberships = []
for bname, seeds in BLOCKS.items():
    sel, sb, top = select_on(seeds)
    et_b = ens_test(sel, seeds)
    memberships.append(sel)
    print(f"    {bname:14s} {len(sel):<5d} {summ(et_b):16s} {sorted(sel)}")
# stability metrics
inter = set.intersection(*memberships); uni = set.union(*memberships)
always = [n for n, _, _ in CANDIDATES if all(n in m for m in memberships)]
never = [n for n, _, _ in CANDIDATES if all(n not in m for m in memberships)]
sometimes = [n for n, _, _ in CANDIDATES if n not in always and n not in never]
print(f"    ALWAYS selected ({len(always)}): {always}")
print(f"    SOMETIMES (flips) ({len(sometimes)}): {sometimes}")
print(f"    NEVER ({len(never)}): {never}")
print(f"    Jaccard(core/union) = {len(inter)}/{len(uni)} = {len(inter)/len(uni):.2f}")

print(f"\nCOMPUTE: {len(CANDIDATES)} candidates x {len(SEEDS_ALL)} seeds = {n_fits} fits")
