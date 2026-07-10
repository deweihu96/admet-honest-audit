"""TDC ADMET benchmark adapter -- reads cached CSVs, splits reproducibly.

Reads data/admet_group/<endpoint>/{train_val,test}.csv directly with pandas (the
same plain pd.read_csv TDC uses) and splits train_val with the vendored scaffold
algorithm. It does NOT import PyTDC at runtime, so the loop is reproducible from
committed data and needs no setuptools<81 pin. PyTDC is only a documented
fallback for downloading endpoints not yet cached (see download_endpoint).
"""
import os

import pandas as pd

from benchmarks.base import BenchmarkAdapter
from benchmarks._scaffold_split import create_scaffold_split

DATA_ROOT = "data/admet_group"
SPLIT_FRAC = [0.875, 0.125, 0.0]        # TDC ADMET train/valid frac (test frac 0)

# Verified metric map (vendored from tdc.metadata.admet_metrics via
# build_metric_map.py), keyed by EXACT endpoint name -- no family defaults, so
# cyp3a4_substrate_carbonmangels resolves to roc-auc, not the substrate-family
# pr-auc.
METRIC_MAP = {
    "ames": "roc-auc", "bbb_martins": "roc-auc", "bioavailability_ma": "roc-auc",
    "caco2_wang": "mae", "clearance_hepatocyte_az": "spearman",
    "clearance_microsome_az": "spearman",
    "cyp2c9_substrate_carbonmangels": "pr-auc", "cyp2c9_veith": "pr-auc",
    "cyp2d6_substrate_carbonmangels": "pr-auc", "cyp2d6_veith": "pr-auc",
    "cyp3a4_substrate_carbonmangels": "roc-auc", "cyp3a4_veith": "pr-auc",
    "dili": "roc-auc", "half_life_obach": "spearman", "herg": "roc-auc",
    "hia_hou": "roc-auc", "ld50_zhu": "mae", "lipophilicity_astrazeneca": "mae",
    "pgp_broccatelli": "roc-auc", "ppbr_az": "mae", "solubility_aqsoldb": "mae",
    "vdss_lombardo": "spearman",
}
# Only error metrics are lower-is-better (verified empirically in verify_direction.py).
LOWER_IS_BETTER = {"mae", "rmse", "mse"}
CLASSIFICATION_METRICS = {"roc-auc", "pr-auc", "accuracy"}
# Small endpoints get the 25-seed budget (variance analysis); large get 5.
# Threshold reproduces the validated budgets: cyp2d6/caco2 -> 25, solubility -> 5.
LARGE_SEED_BUDGET, SMALL_SEED_BUDGET, LARGE_THRESHOLD = 5, 25, 2000


class TDCAdmetAdapter(BenchmarkAdapter):
    def __init__(self, endpoint: str, data_root: str = DATA_ROOT):
        key = endpoint.lower()
        if key not in METRIC_MAP:
            raise KeyError(f"unknown ADMET endpoint: {endpoint}")
        self.endpoint = key
        self._dir = os.path.join(data_root, key)
        tv_path = os.path.join(self._dir, "train_val.csv")
        if not os.path.exists(tv_path):
            raise FileNotFoundError(
                f"{tv_path} not cached. Download once via "
                f"TDCAdmetAdapter.download_endpoint('{key}') (PyTDC fallback).")
        self._train_val = pd.read_csv(tv_path)
        self._test = pd.read_csv(os.path.join(self._dir, "test.csv"))

        self.metric_name = METRIC_MAP[key]
        self.higher_is_better = self.metric_name not in LOWER_IS_BETTER
        self.task_type = ("classification" if self.metric_name in CLASSIFICATION_METRICS
                          else "regression")
        self.seed_budget = (LARGE_SEED_BUDGET if len(self._train_val) >= LARGE_THRESHOLD
                            else SMALL_SEED_BUDGET)

    # ---- iteration-loop surface (train/valid only) ----
    def load_train_valid(self, seed: int):
        out = create_scaffold_split(self._train_val, seed, SPLIT_FRAC, entity="Drug")
        return out["train"], out["valid"]

    def load_train_reference(self) -> pd.DataFrame:
        return self._train_val.copy()

    def load_test_molecules(self) -> pd.DataFrame:
        return self._test.drop(columns=["Y"]).copy()   # SMILES/Drug_ID only, NO Y

    # ---- final-eval-only surface ----
    def load_test_labeled(self) -> pd.DataFrame:
        """ONLY the final-eval step may call this. See base.py docstring."""
        return self._test.copy()

    # ---- documented PyTDC fallback (NOT used at runtime) ----
    @staticmethod
    def download_endpoint(endpoint: str, data_root: str = "data"):
        """One-time cache population via PyTDC (needs the setuptools<81 pin).
        Writes data/admet_group/<endpoint>/{train_val,test}.csv. Not called by
        the loop; here only so uncached endpoints can be fetched."""
        from tdc.benchmark_group import admet_group
        g = admet_group(path=data_root)
        g.get(endpoint)      # materializes the cached CSVs under data_root
        return os.path.join(data_root, "admet_group", endpoint.lower())
