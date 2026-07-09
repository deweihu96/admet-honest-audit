"""Metric map across all 22 TDC ADMET endpoints.

Pulls metric + split from tdc.metadata (authoritative) and task type / sizes /
positive counts from the REAL loader objects. Prints a table to eyeball before
any scaled code depends on it.

WALL RULE: this session reads NO test labels. We read len(test) (a size, not a
score) and read Y ONLY from train_val (never test) for task type / positive
counts. Nothing is scored.

DIRECTION IS INFERRED, NOT READ: TDC stores the metric name per endpoint but
NOT its optimization direction. Direction below is derived from the metric by a
fixed, standard rule and is flagged as inferred.
"""
import numpy as np
from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics, admet_splits

# Fixed rule: only error metrics are lower-is-better. Correlation and AUC
# metrics are higher-is-better. This is the ONLY place direction is decided.
LOWER_IS_BETTER = {"mae", "rmse", "mse"}
HIGHER_IS_BETTER = {"spearman", "pearson", "pcc", "roc-auc", "pr-auc", "accuracy"}

# Which metrics imply which task family (used only as a consistency CROSS-CHECK
# against the actual labels, not as the source of truth for task type).
REGRESSION_METRICS = {"mae", "rmse", "mse", "spearman", "pearson", "pcc"}
CLASSIFICATION_METRICS = {"roc-auc", "pr-auc", "accuracy"}


def direction_of(metric):
    if metric in LOWER_IS_BETTER:
        return "lower-is-better"
    if metric in HIGHER_IS_BETTER:
        return "higher-is-better"
    return "AMBIGUOUS(flag)"


def main():
    group = admet_group(path="data/")
    names = group.dataset_names
    rows = []
    flags = []

    for name in names:
        key = name.lower()
        metric = admet_metrics.get(key, "MISSING")
        split = admet_splits.get(key, "MISSING")
        direction = direction_of(metric)

        benchmark = group.get(name)
        train_val = benchmark["train_val"]
        test = benchmark["test"]
        n_trainval = len(train_val)
        n_test = len(test)                      # size only; no labels read from test

        # Task type from REAL labels (train_val only). Binary => classification.
        y = train_val["Y"].to_numpy()
        uniq = np.unique(y)
        is_binary = set(uniq.tolist()) <= {0.0, 1.0}
        task = "classification" if is_binary else "regression"

        # Cross-check the label-derived task type against the metric family.
        metric_task = ("classification" if metric in CLASSIFICATION_METRICS
                       else "regression" if metric in REGRESSION_METRICS
                       else "unknown")
        if metric_task != "unknown" and metric_task != task:
            flags.append(f"{name}: label-type={task} but metric={metric} "
                         f"implies {metric_task}")

        # Positive count only meaningful for classification (train_val only).
        if is_binary:
            n_pos = int((y == 1).sum())
            pos_frac = n_pos / n_trainval
            pos_str = f"{n_pos} ({pos_frac:5.1%})"
        else:
            pos_str = "-"

        rows.append((name, task, metric, direction, n_trainval, n_test,
                     "yes" if is_binary else "no", pos_str))

    # ---- print table ----
    hdr = ("endpoint", "task", "metric", "direction", "train_val", "test",
           "binary?", "train_val pos (frac)")
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(hdr))
    print(line)
    print("-" * len(line))
    # Group regressions then classifications for readability.
    for task_group in ("regression", "classification"):
        for r in sorted(rows, key=lambda x: x[0]):
            if r[1] != task_group:
                continue
            print("  ".join(str(r[i]).ljust(w[i]) for i in range(len(r))))
        print()

    # ---- summary counts ----
    reg = [r for r in rows if r[1] == "regression"]
    clf = [r for r in rows if r[1] == "classification"]
    print(f"totals: {len(rows)} endpoints = {len(reg)} regression + {len(clf)} classification")
    from collections import Counter
    print("metrics:", dict(Counter(r[2] for r in rows)))
    print("directions:", dict(Counter(r[3] for r in rows)))
    print("splits:", dict(Counter(admet_splits.get(r[0].lower(), '?') for r in rows)))

    print("\nDIRECTION NOTE: every direction above is INFERRED from the metric "
          "name\n(MAE->lower; spearman/roc-auc/pr-auc->higher). TDC stores no "
          "per-endpoint direction.")
    if flags:
        print("\nCONSISTENCY FLAGS (label type vs metric family):")
        for f in flags:
            print("  !!", f)
    else:
        print("\nConsistency: label-derived task type agrees with metric family "
              "for all 22.")


if __name__ == "__main__":
    main()
