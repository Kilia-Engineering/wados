"""Phase 4 tests: published cases (R1) and robustness.

V6: R1 waverider case I, Eq. (23), M 6, FCT Z = 0.12 (Eq. 24), against R1 Table 2
    (LTOCs row) and the wall pressure of R1 Fig. 23.
V7: R1 waverider case II, Eq. (25), M 7, FCT Z = 0.25 (Eq. 26), against R1 Table 3
    and Fig. 27.
V8: R1 test case II shock generator, Eq. (18), M 7, against the wall pressure of
    R1 Fig. 16.
V9: flat, concave (``test_ltoc_core``), sub-Mach-angle, detached and
    under-determined inputs give the specified messages, never a silent bad
    geometry.

Wall-pressure references are R1's own LTOCs curves, digitised into
``ltoc/data/r1_wall_pressure_digitised.json`` (about 0.03 in p/p_inf and 0.003
in X or Y). On the steep parts of the curves the position accuracy alone is
worth 2-3 % in pressure, so the maxima are checked against 4 %.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ltoc.forces import (body_grid, crosswise_section, lower_surface_grid, panel_forces,
                         streamwise_section)
from ltoc.ltoc import find_leading_edge, r1_frame_to_gui, solve_stream_surface
from ltoc.shock_surface import BSplineShock, ConeShock, RuledShock
from ltoc.waverider import LTOCWaverider

DIGITISED = json.loads((Path(__file__).resolve().parents[1] / "data"
                        / "r1_wall_pressure_digitised.json").read_text())

R1_WAVERIDERS = {
    # shock (R1 Eqs. 23, 25), M, FCT Z (Eqs. 24, 26), R1 Tables 2/3 LTOCs row, figure
    "I": (dict(x_p=0.0, a_p=0.0, b_p=0.0, x_q=1.7, a_q=0.595, b_q=0.68), 6.0, 0.12,
          (0.2205, 0.0750, 2.9378), "fig23"),
    "II": (dict(x_p=0.1, a_p=0.3, b_p=0.2, x_q=1.5, a_q=0.65, b_q=0.8), 7.0, 0.25,
           (0.2034, 0.0648, 3.1395), "fig27"),
}


def r1_waverider(case, n_stations=25, n_points=40):
    shock, M, zf, _, _ = R1_WAVERIDERS[case]
    s = RuledShock.elliptic(**shock, m_range=(0.0, np.pi), n_range=(0.0, 1.0))
    y_tip = shock["a_q"] * np.sqrt(1.0 - (zf / shock["b_q"]) ** 2)
    u = np.linspace(0.0, 1.0, n_stations)
    ys = y_tip * (1.0 - np.cos(0.5 * np.pi * u))           # clustered at the symmetry plane
    return LTOCWaverider(s, np.column_stack([ys, np.full_like(ys, zf)]), shock["x_q"], M,
                         n_points=n_points, to_gui=r1_frame_to_gui, strict=True)


@pytest.fixture(scope="module", params=["I", "II"])
def r1_case(request):
    return request.param, r1_waverider(request.param)


def _section_deviation(points, p, fig):
    worst, rms = 0.0, []
    for key, kind in ((fig + "a", "stream"), (fig + "b", "cross")):
        for name, c in DIGITISED[key].items():
            val = float(name.split("=")[1])
            fn = streamwise_section if kind == "stream" else crosswise_section
            xs, ps = fn(points, p, val)
            xd, pd = np.asarray(c["x"]), np.asarray(c["p"])
            inside = (xd >= xs.min()) & (xd <= xs.max())
            assert inside.sum() >= 0.8 * xd.size, (key, name)       # same extent as R1's curve
            dev = np.interp(xd[inside], xs, ps) / pd[inside] - 1.0
            worst = max(worst, float(np.max(np.abs(dev))))
            rms.append(float(np.sqrt(np.mean(dev ** 2))))
    return worst, max(rms)


# ---------------------------------------------------------------------------
#  V6 and V7: R1 Tables 2 and 3, R1 Figs. 23 and 27
# ---------------------------------------------------------------------------

def test_v6_v7_lift_and_drag_match_r1_tables(r1_case):
    case, wr = r1_case
    CL, CD, LD = R1_WAVERIDERS[case][3]
    f = panel_forces(lower_surface_grid(wr, 101), wr.M_inf)
    assert f.reference == "planform"
    for ours, ref in ((f.CL, CL), (f.CD, CD), (f.L_over_D, LD)):
        assert abs(ours / ref - 1.0) < 0.01                     # spec target
        assert abs(ours / ref - 1.0) < 0.005                    # actual margin at 25 stations


def test_r1_coefficients_are_referenced_to_planform_not_wetted_area(r1_case):
    """R1 Eq. (22) says "total wetted area"; R1's tables need the planform area."""
    case, wr = r1_case
    CL, CD, _ = R1_WAVERIDERS[case][3]
    f = panel_forces(lower_surface_grid(wr, 101), wr.M_inf)
    assert f.CL_wetted / CL < 0.85 and f.CD_wetted / CD < 0.85
    w = panel_forces(lower_surface_grid(wr, 101), wr.M_inf, reference="wetted")
    assert w.CL == pytest.approx(f.CL_wetted) and w.L_over_D == pytest.approx(f.L_over_D)


def test_v6_v7_wall_pressure_matches_r1_figures(r1_case):
    case, wr = r1_case
    g = lower_surface_grid(wr, 201)
    worst, rms = _section_deviation(g.points_native, g.p, R1_WAVERIDERS[case][4])
    assert rms < 0.015 and worst < 0.04


# ---------------------------------------------------------------------------
#  V8: R1 test case II shock generator, R1 Fig. 16
# ---------------------------------------------------------------------------

def test_v8_test_case_2_generator_wall_pressure_matches_r1_fig16():
    s = RuledShock.elliptic(0.1, 0.2, 0.3, 1.5, 0.8, 0.65, m_range=(0.0, np.pi),
                            n_range=(0.0, 1.0))                  # R1 Eqs. (16)-(18)
    stations = [solve_stream_surface(s, (m, 0.0), 1.5, 7.0, n_points=40, station=k)
                for k, m in enumerate(np.linspace(0.5 * np.pi, 0.0, 25))]
    assert all(st.status == "ok" for st in stations)
    pts, p = body_grid(stations, 201)
    worst, rms = _section_deviation(pts, p, "fig16")
    assert rms < 0.02 and worst < 0.04


# ---------------------------------------------------------------------------
#  V9: robustness
# ---------------------------------------------------------------------------

def _plane_shock(beta):
    """Planar shock Z = X tan(beta) as a ruled surface (kappa_b = 0 everywhere)."""
    tb = np.tan(beta)
    vec = lambda f: (lambda m: np.stack(np.broadcast_arrays(*f(m)), -1))
    p, q = vec(lambda m: (0.0, m, 0.0)), vec(lambda m: (1.0, m, tb))
    d1, d2 = vec(lambda m: (0.0, 1.0, 0.0)), vec(lambda m: (0.0, 0.0, 0.0))
    return RuledShock(p, d1, d2, q, d1, d2, m_range=(-1.0, 1.0), n_range=(0.0, 1.0))


def test_v9_flat_shock_gives_the_wedge_derived_waverider():
    from gvwd.thermo.oblique_shock import rankine_hugoniot

    M, b = 6.0, np.radians(15.0)
    yy = np.linspace(0.0, 0.5, 6)
    fct = np.column_stack([yy, 0.1 + 0.05 * yy ** 2])           # curved FCT, flat shock
    wr = LTOCWaverider(_plane_shock(b), fct, 1.0, M, n_points=40, to_gui=r1_frame_to_gui,
                       strict=True)
    rh = rankine_hugoniot(M, b, 1.0, 1.0, 1.4)
    for st in wr.stations:
        P = st.body.points
        assert np.max(np.abs(np.diff(P[:, 2]) / np.diff(P[:, 0]) - np.tan(rh["theta"]))) < 1e-6
        assert np.max(np.abs(st.body.p - rh["p2"])) < 1e-5
        assert np.ptp(P[:, 1]) < 1e-12                          # planar stream surfaces
    # Panel forces on a wedge: C_L = p2 / q, L/D = 1 / tan(theta) (planform reference).
    f = panel_forces(lower_surface_grid(wr, 41), M, reference="planform")
    q = 0.7 * M ** 2
    assert f.CL == pytest.approx(rh["p2"] / q, rel=1e-5)
    assert f.L_over_D == pytest.approx(1.0 / np.tan(rh["theta"]), rel=1e-5)


@pytest.mark.parametrize("beta_deg, M, fct_z, status", [(8.0, 6.0, 0.1, "sub_mach"),
                                                        (75.0, 3.0, 2.0, "detached")])
def test_v9_sub_mach_and_detached_shocks_are_refused(beta_deg, M, fct_z, status):
    cone = ConeShock(np.radians(beta_deg), x_range=(0.0, 2.0))
    uv, _, _ = find_leading_edge(cone, np.array([[0.0, fct_z]]))
    s = solve_stream_surface(cone, uv[0], 1.0, M, n_points=20)
    assert s.status == status and "spec flag 7: refused" in s.message


def test_v9_under_determined_body_is_reported():
    cone = ConeShock(np.radians(20.0), x_range=(0.0, 2.0))
    uv, _, _ = find_leading_edge(cone, np.array([[0.0, 0.2]]))
    s = solve_stream_surface(cone, uv[0], 1.0, 6.0, n_points=20, max_extension=0.1)
    assert s.status == "undetermined" and "spec flag 4" in s.message
    wr = LTOCWaverider(cone, np.array([[0.0, 0.2]]), 1.0, 6.0, n_points=20)
    assert wr.ok                                                 # default extension suffices


def test_v9_data_shock_is_never_extrapolated():
    """A B-spline shock that ends at the base plane cannot determine the body up
    to the base: refused (flag 4). With data past the base it reproduces the
    analytic cone."""
    cone = ConeShock(np.radians(20.0), x_range=(0.0, 3.0))
    short = BSplineShock.from_surface(ConeShock(np.radians(20.0), x_range=(0.0, 1.0)),
                                      u_range=(0.5, 2.5), v_range=(0.0, 1.0))
    uv, _, ok = find_leading_edge(short, np.array([[0.0, 0.2]]))
    s = solve_stream_surface(short, uv[0], 1.0, 6.0, n_points=20)
    assert ok.all() and s.status == "undetermined" and "shock data end" in s.message
    long = BSplineShock.from_surface(cone, u_range=(0.5, 2.5), v_range=(0.0, 3.0), nv=80)
    uv, _, _ = find_leading_edge(long, np.array([[0.0, 0.2]]))
    uvc, _, _ = find_leading_edge(cone, np.array([[0.0, 0.2]]))
    a = solve_stream_surface(long, uv[0], 1.0, 6.0, n_points=20)
    b = solve_stream_surface(cone, uvc[0], 1.0, 6.0, n_points=20)
    assert a.status == "ok" and np.max(np.abs(a.body.points[-1] - b.body.points[-1])) < 1e-5
