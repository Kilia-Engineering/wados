"""Phase 2 numerical-equivalence test.

Builds the same waverider with both packages -- ``liu2019`` and ``mfof``
with the factory that mirrors ``liu2019``'s per-plane physics -- and asserts
the two agree on every geometric metric to within ``1e-6`` relative
deviation. This is the gate condition for Phase 2 acceptance: any drift
signals that the architectural refactor has introduced a numerical
regression.

The matching factory is *mixed*, not all-cone: ``liu2019`` uses 2D wedge
flow in the flat region (``|z| <= L_s``) and osculating-cone flow in the
curved region. See :func:`_build_mfof_equivalent`.

Run from the repo root:

    py -3.10 -m mfof.validate

Returns exit code 0 on success, 1 on any tolerance failure.
"""

from __future__ import annotations

import sys


def _build_liu(params, n_z=200, n_x=100):
    from liu2019.geometry import build_liu2019_waverider
    return build_liu2019_waverider(params, n_z=n_z, n_x=n_x)


def _build_mfof_equivalent(params, n_z=200, n_x=100):
    """Build the MFOF waverider that mirrors ``liu2019``'s physics exactly.

    ``liu2019.osculating.osculating_plane_geometry`` dispatches per plane:

    * ``|z| <= L_s`` (flat region, ``R_osc -> infinity``): 2D wedge flow,
      streamline at ``theta_w(beta, Ma_local)``.
    * ``|z| >  L_s`` (curved region): osculating-cone flow, streamline at
      the Taylor-Maccoll half-angle ``delta_c``.

    So the matching MFOF factory is *mixed* -- a ``WedgeFlowfield`` inboard
    of ``L_s`` and a ``ConeFlowfield`` outboard. This is the framework's
    first production use of a non-uniform factory, and it is the whole
    point of MFOF: the flowfield type is a per-plane decision.

    An all-cone factory does **not** reproduce ``liu2019``; it applies
    ``delta_c`` in the flat region too, which is the "paper's flat-region
    bug" that ``liu2019.osculating`` documents and deliberately avoids.
    """
    from mfof.cone_flowfield import ConeFlowfield
    from mfof.wedge_flowfield import WedgeFlowfield
    from mfof.geometry import build_mfof_waverider

    beta = float(params["beta_deg"])
    gamma = float(params.get("gamma", 1.4))
    L_s = float(params["L_s"])

    def liu_equivalent_factory(z, Ma_z):
        if abs(z) <= L_s:
            return WedgeFlowfield(Ma_z, beta, gamma)
        return ConeFlowfield(Ma_z, beta, gamma)

    return build_mfof_waverider(params, liu_equivalent_factory,
                                n_z=n_z, n_x=n_x)


def run_equivalence_test(params=None, n_z: int = 200, n_x: int = 100,
                          tol: float = 1e-6, verbose: bool = True) -> bool:
    """Run the Liu-2019 equivalence test.

    Parameters
    ----------
    params : dict, optional
        Liu-style design dict. Defaults to :data:`mfof.config.PAPER_PARAMS`.
    n_z, n_x : int
        Mesh resolution. Both packages use the same value.
    tol : float
        Relative-deviation acceptance threshold. Phase 2 spec is 1e-6.
        The mixed factory reproduces ``liu2019`` to ~1e-13 in practice.
    verbose : bool
        If True, print a per-metric table.

    Returns
    -------
    bool
        ``True`` iff every checked metric is within ``tol``.
    """
    if params is None:
        from liu2019.config import PAPER_PARAMS as params

    # ---- Warm-up the T-M solver -----------------------------------------
    # liu2019.shock.taylor_maccoll_cone_angle has a try/except that prefers
    # waverider_generator.flowfield.cone_angle but falls back to a local
    # solver if anything goes wrong. The very first call in a fresh Python
    # process returns the local-fallback value (~8.3852 deg at Ma=6, beta=13);
    # every subsequent call returns the waverider_generator value (~8.3834).
    # Whichever package builds first contaminates its delta_c grid with the
    # one-off first-call value. Warming up here ensures both packages see the
    # same post-warmup behavior.
    from liu2019.shock import taylor_maccoll_cone_angle as _tmca
    _ = _tmca(float(params["Ma_center"]), float(params["beta_deg"]),
              float(params.get("gamma", 1.4)))

    liu_wv  = _build_liu(params, n_z=n_z, n_x=n_x)
    mfof_wv = _build_mfof_equivalent(params, n_z=n_z, n_x=n_x)

    checks = [
        ("volume",       liu_wv.volume(),               mfof_wv.volume()),
        ("wetted_area",  liu_wv.wetted_area(),          mfof_wv.wetted_area()),
        ("planform",     liu_wv.planform_area(),        mfof_wv.planform_area()),
        ("base_area",    liu_wv.base_area(),            mfof_wv.base_area()),
        ("eta",          liu_wv.volumetric_efficiency(),
                         mfof_wv.volumetric_efficiency()),
    ]

    if verbose:
        print(f"{'Metric':<14} {'Liu 2019':>14} {'MFOF':>14} "
              f"{'Δ rel':>12}  status")
        print("-" * 64)

    all_pass = True
    for name, a, b in checks:
        denom = max(abs(a), 1e-30)
        delta = abs(a - b) / denom
        ok = delta < tol
        if not ok:
            all_pass = False
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"{name:<14} {a:>14.6f} {b:>14.6f} "
                  f"{delta:>12.2e}  {status}")

    if verbose:
        print()
        print(f"Phase 2 equivalence: "
              f"{'PASS' if all_pass else 'FAIL'} "
              f"(tol = {tol:.0e})")
    return all_pass


def main():
    ok = run_equivalence_test()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
