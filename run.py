"""Orchestrator: wires adapter + executor + arbiter + auditor into one loop.

TEST WALL (structural): the ITERATION path (run_loop, inject_idea, _run_candidate)
does NOT import final_eval, testwall, or load_test_labeled. Test is reached ONLY
through the explicit user-commanded trigger request_test(), which LAZILY imports
testwall.final_eval. grep run.py: 'final_eval' appears only inside request_test.
"""
import importlib

from benchmarks.tdc_admet import TDCAdmetAdapter
from config import Config
from roles import arbiter
from roles.executor import train_and_score
from roles.auditor import Auditor


def load_specs(version_names):
    """Proposer source for this stage: the existing models/*.py version files.
    Each is a model_spec (FEATURES + build)."""
    return [importlib.import_module(f"models.{v}") for v in version_names]


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.adapter = TDCAdmetAdapter(config.endpoint)
        allowed, msg = config.check_data_sufficiency(self.adapter)
        self.gate_message = msg
        if not allowed:
            raise ValueError(msg)          # high-tier data-sufficiency refusal
        self.history = []                  # ScoreResult per accepted candidate
        self.spec_names = []
        self.specs = []
        self.leakage = None                # split-level verdict (seed-independent)
        self.outcome = None
        self.read_log = None
        self._test_read = False

    # ---- iteration path (NO test access) ----
    def _run_candidate(self, spec):
        result = train_and_score(spec, self.adapter)     # validation only
        self.history.append(result)
        self.specs.append(spec)
        self.spec_names.append(spec.__name__.split(".")[-1])
        self.outcome = arbiter.run_selection(
            self.history, k=self.config.plateau_k,
            max_loops=self.config.max_loops, force_full=self.config.force_full)
        return result

    def run_loop(self, candidate_specs):
        # split-level leakage verdict once; rides alongside the whole run
        self.leakage = Auditor(self.adapter).audit()
        for spec in candidate_specs:
            if len(self.history) >= self.config.max_loops:
                break
            self._run_candidate(spec)
            # non-forced: stop when the plateau has fired
            if (not self.config.force_full and self.outcome.plateau_index != -1
                    and len(self.history) - 1 >= self.outcome.plateau_index):
                break
        return self.outcome

    def inject_idea(self, spec):
        """Human idea injection: a user spec enters the SAME executor->arbiter
        discipline with no privilege. If a test read already happened, the next
        test read is flagged adaptive."""
        if self._test_read and self.read_log is not None:
            self.read_log.mark_adaptive()
        return self._run_candidate(spec)

    @property
    def locked_spec(self):
        return self.specs[self.outcome.locked_index]

    @property
    def locked_name(self):
        return self.spec_names[self.outcome.locked_index]

    def decision_sequence(self):
        labels = ["baseline"] + list(self.outcome.compare_labels)
        locks = [self.spec_names[i] for i in self.outcome.lock_trajectory]
        return list(zip(self.spec_names, labels, locks))

    # ---- user-commanded test trigger (the ONLY path to test) ----
    def request_test(self):
        """Explicit, user-commanded. Test is NOT auto-fired. Lazily imports
        final_eval so the iteration path never touches test."""
        from testwall.final_eval import run_final_eval        # LAZY: wall boundary
        from testwall.read_counter import TestReadLog
        if self.read_log is None:
            self.read_log = TestReadLog()
        locked_result = self.history[self.outcome.locked_index]
        read = run_final_eval(self.adapter, self.locked_spec, locked_result,
                              self.read_log, leakage_verdict=self.leakage)
        self._test_read = True
        return read
