"""Stage-4 regression + wall check for the walled auditor.

    uv run python -m roles.verify_stage4

Confirms the walled auditor reproduces the validated leakage verdicts, the
synthetic-injection control still fires through the walled interface, the second
audit axis matches the known gaps, and the verdict boundary holds.
"""
import dataclasses

import pandas as pd

from benchmarks.tdc_admet import TDCAdmetAdapter
from roles.auditor import (Auditor, LeakageVerdict, valid_vs_test_gap,
                           GapVerdict)

# validated leakage findings (from the sessions): (n_exact, nn_max~, clean)
EXPECTED = {
    "solubility_aqsoldb": dict(n_exact=7, nn_max_min=0.999, clean=False),
    "caco2_wang": dict(n_exact=0, nn_max_lt=0.99, clean=True),
    "cyp2d6_substrate_carbonmangels": dict(n_exact=0, nn_max_lt=0.99, clean=True),
}


class _PlantedAdapter:
    """Wraps an adapter, planting N test molecules into the train reference so
    the auditor should detect them as exact duplicates. Only overrides the two
    test-touching methods the auditor uses."""
    def __init__(self, base, n_plant):
        self.base = base
        self.endpoint = base.endpoint + f"+planted{n_plant}"
        self._plant = base.load_test_molecules().head(n_plant).copy()

    def load_train_reference(self):
        ref = self.base.load_train_reference()
        planted = self._plant.copy()
        planted["Y"] = 0.0
        return pd.concat([ref, planted], ignore_index=True)

    def load_test_molecules(self):
        return self.base.load_test_molecules()


def main():
    ok = True
    print("=" * 90)
    print("STAGE 4 LEAKAGE VERDICTS  (walled auditor vs validated)")
    print("=" * 90)
    print(f"{'endpoint':32s} {'clean':6s} {'nn_max':7s} {'n_exact':7s} {'verdict vs validated'}")
    for ep, exp in EXPECTED.items():
        v = Auditor(TDCAdmetAdapter(ep)).audit()
        assert isinstance(v, LeakageVerdict)
        clean_ok = v.clean == exp["clean"]
        exact_ok = v.n_exact_identity == exp["n_exact"]
        nn_ok = (v.nn_similarity_max >= exp["nn_max_min"] if "nn_max_min" in exp
                 else v.nn_similarity_max < exp["nn_max_lt"])
        row_ok = clean_ok and exact_ok and nn_ok
        ok &= row_ok
        print(f"{ep:32s} {str(v.clean):6s} {v.nn_similarity_max:6.3f}  {v.n_exact_identity:^7d} "
              f"[{'MATCH' if row_ok else 'MISMATCH'}]  "
              f"(exp clean={exp['clean']}, n_exact={exp['n_exact']})")

    print("\n" + "=" * 90)
    print("SYNTHETIC INJECTION CONTROL  (plant 10 test molecules into reference)")
    print("=" * 90)
    base = TDCAdmetAdapter("cyp2d6_substrate_carbonmangels")   # verified-clean base
    v0 = Auditor(base).audit()
    v1 = Auditor(_PlantedAdapter(base, 10)).audit()
    fired = (v0.clean is True and v1.clean is False and v1.n_exact_identity >= 10
             and v1.nn_similarity_max >= 0.999)
    ok &= fired
    print(f"  baseline : clean={v0.clean}, n_exact={v0.n_exact_identity}, nn_max={v0.nn_similarity_max:.3f}")
    print(f"  planted  : clean={v1.clean}, n_exact={v1.n_exact_identity}, nn_max={v1.nn_similarity_max:.3f}")
    print(f"  detector fired: {fired}  [{'ok' if fired else 'FAIL'}]")

    print("\n" + "=" * 90)
    print("SECOND AXIS  valid_vs_test_gap  (unit check vs known gaps)")
    print("=" * 90)
    # caco2: valid 0.365 +/- 0.017, test 0.267, MAE lower-better -> flagged, suspicious
    caco = valid_vs_test_gap(0.365, 0.017, 0.267, higher_is_better=False)
    caco_ok = (abs(caco.gap - (-0.098)) < 1e-9 and caco.flagged and caco.suspicious)
    # solubility: valid 0.826 +/- 0.151, test 0.741, MAE -> gap -0.085, within CI, not flagged
    sol = valid_vs_test_gap(0.826, 0.151, 0.741, higher_is_better=False)
    sol_ok = (abs(sol.gap - (-0.085)) < 1e-9 and not sol.flagged and not sol.suspicious)
    ok &= caco_ok and sol_ok
    print(f"  caco2      gap={caco.gap:+.3f} flagged={caco.flagged} suspicious={caco.suspicious}  "
          f"[{'ok' if caco_ok else 'MISMATCH'}]  ({caco.note})")
    print(f"  solubility gap={sol.gap:+.3f} flagged={sol.flagged} suspicious={sol.suspicious}  "
          f"[{'ok' if sol_ok else 'MISMATCH'}]  ({sol.note})")

    print("\n" + "=" * 90)
    print("WALL / VERDICT-BOUNDARY CHECK")
    print("=" * 90)
    fields = {f.name for f in dataclasses.fields(LeakageVerdict)}
    # Whitelist of aggregate summary fields; anything else is a potential leak.
    allowed = {"endpoint", "clean", "nn_similarity_median", "nn_similarity_max",
               "nn_similarity_p95", "n_exact_identity", "n_test", "noise_floor_ok", "notes"}
    unexpected = fields - allowed
    fields_ok = not unexpected
    print(f"  LeakageVerdict fields: {sorted(fields)}")
    print(f"  unexpected (non-whitelisted) fields: {unexpected or 'NONE'}  "
          f"[{'ok' if fields_ok else 'LEAK'}]")
    # Value-level: every field is a scalar or a short tuple of strings -- no
    # per-molecule array, DataFrame, SMILES, or label can hide in the verdict.
    vv = Auditor(TDCAdmetAdapter("cyp2d6_substrate_carbonmangels")).audit()
    values_ok = True
    for f in fields:
        val = getattr(vv, f)
        if f == "notes":
            values_ok &= isinstance(val, tuple) and all(isinstance(x, str) for x in val)
        else:
            values_ok &= isinstance(val, (int, float, bool, str))
    print(f"  every verdict field is scalar / tuple-of-str (no array/df/SMILES/label): {values_ok}")
    # every audit() return is a LeakageVerdict
    returns_ok = all(isinstance(Auditor(TDCAdmetAdapter(e)).audit(), LeakageVerdict)
                     for e in EXPECTED)
    print(f"  Auditor.audit always returns LeakageVerdict: {returns_ok}")
    ok &= fields_ok and values_ok and returns_ok

    print("\n" + "=" * 90)
    print(f"OVERALL: {'verdicts reproduced + control fired + wall holds -> safe to commit' if ok else 'FAIL -> DO NOT COMMIT'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
