"""Benchmark adapter interface.

The test wall is built into the SHAPE of this interface: the three methods the
iteration loop is allowed to touch (load_train_valid, load_train_reference,
load_test_molecules) can NEVER return a test label. Test labels are reachable
only through load_test_labeled(), which exists solely for the final-eval step.
"""
from abc import ABC, abstractmethod

import pandas as pd


class BenchmarkAdapter(ABC):
    """One parameterized adapter per benchmark endpoint.

    Metadata attributes (set by the concrete adapter, resolved by EXACT endpoint
    name from the verified metric map):
      endpoint        : str   exact TDC endpoint name
      metric_name     : str   e.g. 'mae', 'pr-auc', 'roc-auc', 'spearman'
      task_type       : str   'regression' | 'classification'
      higher_is_better: bool  metric direction (verified empirically)
      seed_budget     : int   number of seeds for this endpoint's protocol
    """

    endpoint: str
    metric_name: str
    task_type: str
    higher_is_better: bool
    seed_budget: int

    # ---- iteration-loop surface (train/valid only; NEVER exposes test) ----
    @abstractmethod
    def load_train_valid(self, seed: int) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """Return (train_df, valid_df): the reshuffled train/valid split for a
        seed. NEVER returns test. This is what the executor trains/selects on."""

    @abstractmethod
    def load_train_reference(self) -> pd.DataFrame:
        """Return the full train_val molecule set (seed-independent). For the
        auditor's split-leakage reference. Contains no test rows."""

    @abstractmethod
    def load_test_molecules(self) -> pd.DataFrame:
        """Return the test molecules with the label column (Y) DROPPED -- SMILES
        (and Drug_ID) only. This is all the auditor may see of test, for
        similarity computation. Exposing no Y here is a structural guarantee."""

    # ---- final-eval-only surface (the ONLY path to test labels) ----
    @abstractmethod
    def load_test_labeled(self) -> pd.DataFrame:
        """Return test WITH labels (Y). This is the ONLY method that exposes test
        Y. It exists solely so the final-eval step can score a locked model
        exactly once. NOTHING IN THE ITERATION LOOP MAY CALL THIS -- calling it
        during iteration is a test-wall violation."""
