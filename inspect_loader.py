"""Task 2: Inspect exactly what the TDC ADMET_Group loader returns for Caco2_Wang.
No modeling here. Just print the real objects so we build on facts, not assumptions.
"""
import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# The group object downloads/caches the 22-endpoint ADMET benchmark under ./data
group = admet_group(path="data/")

print("=" * 70)
print("GROUP OBJECT")
print("=" * 70)
print("type:", type(group))
print("dataset_names (count = {}):".format(len(group.dataset_names)))
for n in group.dataset_names:
    print("   ", n)

BENCH = "Caco2_Wang"
benchmark = group.get(BENCH)
print("\n" + "=" * 70)
print(f"group.get('{BENCH}') RETURN STRUCTURE")
print("=" * 70)
print("type:", type(benchmark))
print("keys:", list(benchmark.keys()))
print("benchmark['name']:", benchmark["name"])

train_val = benchmark["train_val"]
test = benchmark["test"]
print("\ntrain_val type:", type(train_val), "shape:", train_val.shape)
print("test      type:", type(test), "shape:", test.shape)
print("\ntrain_val columns:", list(train_val.columns))
print("test      columns:", list(test.columns))

print("\n--- train_val.head() ---")
print(train_val.head())
print("\n--- test.head() ---")
print(test.head())

print("\n--- dtypes ---")
print(train_val.dtypes)

# Label column stats: is this regression? what is the range?
print("\n--- label (Y) summary on train_val ---")
print(train_val["Y"].describe())
print("n unique Y:", train_val["Y"].nunique())

# Now the official per-seed split: get_train_valid_split reshuffles train/valid,
# test stays fixed. Show that the split mechanism is seed-driven.
print("\n" + "=" * 70)
print("get_train_valid_split MECHANISM (seed reshuffles train/valid; test fixed)")
print("=" * 70)
for seed in [1, 2]:
    tr, va = group.get_train_valid_split(benchmark=BENCH, split_type="default", seed=seed)
    print(f"seed={seed}: train={len(tr):4d}  valid={len(va):4d}  "
          f"train_head_id={tr['Drug_ID'].iloc[0]!r}")

# Confirm the whole train_val = train+valid partition and test disjointness.
tr1, va1 = group.get_train_valid_split(benchmark=BENCH, split_type="default", seed=1)
print("\ntrain+valid sizes:", len(tr1), "+", len(va1), "=", len(tr1) + len(va1),
      "vs train_val:", len(train_val))
train_ids = set(tr1["Drug_ID"]) | set(va1["Drug_ID"])
test_ids = set(test["Drug_ID"])
print("Drug_ID overlap train_val vs test:", len(train_ids & test_ids))

# Official metric for this endpoint, straight from TDC metadata.
from tdc.metadata import admet_metrics
print("\n" + "=" * 70)
print("OFFICIAL METRIC (from tdc.metadata.admet_metrics)")
print("=" * 70)
key = BENCH.lower()
print(f"admet_metrics['{key}'] =", admet_metrics.get(key))

print("\nSizes summary:")
print(f"  train_val total : {len(train_val)}")
print(f"  test (fixed)    : {len(test)}")
print(f"  per-seed train  : {len(tr1)}")
print(f"  per-seed valid  : {len(va1)}")
