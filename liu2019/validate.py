"""Validation script: reproduce Liu 2019 Tables 1, 4 and Figure 12."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .aero import Liu2019AeroEvaluator
from .config import (
    PAPER_PARAMS,
    PAPER_REFERENCE_AERO,
    PAPER_REFERENCE_GEOMETRY,
    TOLERANCES,
)
from .geometry import build_liu2019_waverider


def _pass(value, reference, fractional_tol):
    if reference is None or reference == 0:
        return True, 0.0
    dev = (value - reference) / reference
    return abs(dev) <= fractional_tol, dev


def check_reference_consistency(reference=None, rel_tol: float = 0.05,
                                verbose: bool = True):
    """Assert the reference aero table obeys its own definitions.

    Coefficients normalised by a common reference area must satisfy two
    identities, independent of what that area is:

        CL / CD  == L/D
        Cmz / CL == Xcp

    A reference row that violates either one cannot be matched by any
    solver, so scoring against it says nothing about the solver. This
    check exists so that a future re-read of Fig. 12 is validated before
    it is trusted -- the original read had CD and Cmz mis-scaled by
    19-34x and 4.4-5.9x respectively (see :mod:`liu2019.config`).

    Entries whose value is ``None`` are skipped, not failed.

    Returns ``(ok, problems)`` where ``problems`` is a list of dicts.
    """
    if reference is None:
        reference = PAPER_REFERENCE_AERO
    problems = []
    for Ma in sorted(reference):
        row = reference[Ma]
        CL, CD  = row.get("CL"),  row.get("CD")
        L_D     = row.get("L_D")
        Cmz     = row.get("Cmz")
        Xcp     = row.get("Xcp")
        if None not in (CL, CD, L_D) and CD != 0 and L_D != 0:
            got = CL / CD
            if abs(got - L_D) / abs(L_D) > rel_tol:
                problems.append({"Ma": Ma, "identity": "CL/CD == L/D",
                                 "lhs": got, "rhs": L_D,
                                 "factor": L_D / got if got else float("inf")})
        if None not in (CL, Cmz, Xcp) and CL != 0 and Xcp != 0:
            got = Cmz / CL
            if abs(got - Xcp) / abs(Xcp) > rel_tol:
                problems.append({"Ma": Ma, "identity": "Cmz/CL == Xcp",
                                 "lhs": got, "rhs": Xcp,
                                 "factor": got / Xcp if Xcp else float("inf")})
    if verbose:
        if problems:
            print("Reference aero table is INTERNALLY INCONSISTENT:")
            for p in problems:
                print(f"  Ma {p['Ma']:>2}  {p['identity']:<16} "
                      f"got {p['lhs']:.3f} vs {p['rhs']:.3f} "
                      f"({p['factor']:.2f}x off)")
        else:
            print("Reference aero table: internally consistent.")
    return (not problems), problems


def run_paper_validation(params: Dict = None,
                         n_z: int = 200,
                         n_x: int = 100,
                         verbose: bool = True,
                         run_aero: bool = True) -> Dict:
    params = dict(params or PAPER_PARAMS)
    wr = build_liu2019_waverider(params, n_z=n_z, n_x=n_x)

    geom = {
        "Vol_m3":   wr.volume(),
        "S_wet_m2": wr.wetted_area(),
        "S_p_m2":   wr.planform_area(),
        "S_b_m2":   wr.base_area(),
        "eta":      wr.volumetric_efficiency(),
    }
    geom_checks = []
    for key, value in geom.items():
        ok, dev = _pass(value, PAPER_REFERENCE_GEOMETRY[key], TOLERANCES[key])
        geom_checks.append({
            "metric": key,
            "computed": value,
            "reference": PAPER_REFERENCE_GEOMETRY[key],
            "deviation": dev,
            "tolerance": TOLERANCES[key],
            "pass": ok,
        })

    aero_checks = []
    aero_rows = []
    if run_aero:
        evaluator = Liu2019AeroEvaluator(wr)
        aero_rows = evaluator.evaluate_paper_trajectory()
        for r in aero_rows:
            ref = PAPER_REFERENCE_AERO.get(int(r["Ma"]), {})
            for key in ("CL", "CD", "L_D", "Cmz", "Xcp"):
                ref_v = ref.get(key)
                ok, dev = _pass(r[key], ref_v, TOLERANCES[key])
                aero_checks.append({
                    "Ma":       r["Ma"],
                    "metric":   key,
                    "computed": r[key],
                    "reference": ref_v,
                    "deviation": dev,
                    "tolerance": TOLERANCES[key],
                    "pass":     ok,
                    # No usable reference -> skipped, not passed. Counting
                    # it as a pass would inflate the score with rows that
                    # were never actually checked.
                    "comparable": ref_v is not None,
                })

    if verbose:
        print("Liu 2019 validation — geometric metrics")
        print(f"  {'metric':>10} {'computed':>12} {'paper':>10} "
              f"{'dev':>8} {'tol':>6}  status")
        for c in geom_checks:
            status = "PASS" if c["pass"] else "FAIL"
            print(f"  {c['metric']:>10} {c['computed']:>12.4f} "
                  f"{c['reference']:>10.4f} "
                  f"{c['deviation']*100:>7.2f}% "
                  f"{c['tolerance']*100:>5.1f}%  {status}")
        if aero_checks:
            print("\nLiu 2019 validation — aerodynamic metrics")
            print(f"  {'Ma':>3} {'metric':>5} {'computed':>10} "
                  f"{'paper':>8} {'dev':>8} {'tol':>6}  status")
            for c in aero_checks:
                ref = c["reference"]
                if not c.get("comparable", True):
                    status = "SKIP"
                else:
                    status = "PASS" if c["pass"] else "FAIL"
                ref_s = f"{ref:>8.3f}" if ref is not None else "      --"
                print(f"  {int(c['Ma']):>3} {c['metric']:>5} "
                      f"{c['computed']:>10.3f} {ref_s} "
                      f"{c['deviation']*100:>7.2f}% "
                      f"{c['tolerance']*100:>5.1f}%  {status}")

    all_checks = geom_checks + aero_checks
    scored = [c for c in all_checks if c.get("comparable", True)]
    n_pass = sum(1 for c in scored if c["pass"])
    n_total = len(scored)
    n_skipped = len(all_checks) - len(scored)
    if verbose and n_skipped:
        print(f"\n{n_skipped} aero check(s) skipped: no usable paper "
              f"reference (see liu2019.config.PAPER_REFERENCE_AERO).")
    return {
        "skipped": n_skipped,
        "waverider": wr,
        "geometry":  geom,
        "aero":      aero_rows,
        "geometry_checks": geom_checks,
        "aero_checks": aero_checks,
        "pass_fraction": (n_pass, n_total),
    }


if __name__ == "__main__":
    run_paper_validation()
