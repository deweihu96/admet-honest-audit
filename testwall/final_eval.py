"""final_eval: THE one test read. The only place besides the auditor that touches
test, and the ONLY caller of load_test_labeled().

Structurally separate from the loop: the orchestrator's ITERATION path does not
import this module. It is reached only via the explicit user-commanded test
trigger. Given a locked model spec + its validation ScoreResult, it scores test
once (per the seed budget), computes the bootstrap CI through the read_counter,
runs the second audit axis (valid_vs_test_gap), attaches the split leakage
verdict, and returns an honest, summary-only test read.
"""
from dataclasses import dataclass

import numpy as np

from features import build_matrix
from roles.auditor import Auditor, valid_vs_test_gap, GapVerdict, LeakageVerdict
from testwall.read_counter import ReadSummary


@dataclass(frozen=True)
class HonestTestRead:
    endpoint: str
    validation_headline: str          # the primary generalization estimate
    read: ReadSummary                  # test point + bootstrap/corrected CI + disclosure
    gap: GapVerdict                    # valid-vs-test second audit axis
    leakage: LeakageVerdict            # split-level leakage verdict


def run_final_eval(adapter, locked_spec, validation_result, read_log,
                   leakage_verdict=None) -> HonestTestRead:
    y_test = adapter.load_test_labeled()["Y"].to_numpy()      # <-- sole load_test_labeled call
    X_test = build_matrix(adapter.load_test_labeled()["Drug"].tolist(), locked_spec.FEATURES)
    is_clf = adapter.task_type == "classification"

    preds = []
    for s in range(1, adapter.seed_budget + 1):
        train, _ = adapter.load_train_valid(s)
        X_tr = build_matrix(train["Drug"].tolist(), locked_spec.FEATURES)
        y_tr = train["Y"].to_numpy()
        est = locked_spec.build(s).fit(X_tr, y_tr)
        preds.append(est.predict_proba(X_test)[:, 1] if is_clf else est.predict(X_test))
    pred = np.mean(preds, axis=0)      # seed-averaged prediction

    vh = f"{validation_result.metric_name} {validation_result.mean:.3f} +/- {validation_result.ci_halfwidth:.3f}"
    read = read_log.record_read(adapter.endpoint, y_test, pred,
                                validation_result.metric_name,
                                validation_result.higher_is_better, vh)
    gap = valid_vs_test_gap(validation_result.mean, validation_result.ci_halfwidth,
                            read.test_point, validation_result.higher_is_better)
    leakage = leakage_verdict if leakage_verdict is not None else Auditor(adapter).audit()

    return HonestTestRead(endpoint=adapter.endpoint, validation_headline=vh,
                          read=read, gap=gap, leakage=leakage)
