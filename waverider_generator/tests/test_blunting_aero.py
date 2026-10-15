"""Tests for ``blunting_aero``: a rounded leading edge on a sharp mesh.

The modified Newtonian force of a half-cylinder normal to the flow is
(4/3) Cp_max R q per unit length, and sweep lowers it by cos^3(Lambda). The
leading edge must be found on a closed mesh (and the trailing or base edges
not), and the radius, bleed and heat flux must follow the models they cite.
Applied through ``aero_corrections``, the blunt LE adds drag in proportion to
R and leaves a sharp edge (R = 0) unchanged.
"""
import contextlib
import io
import math

import numpy as np
import pytest

from aero_corrections import AeroCorrections, corrected_coefficients
from waverider_generator import blunting_aero as B


@pytest.mark.parametrize("sweep_deg", [0.0, 30.0, 60.0, 75.0])
def test_half_cylinder_drag(sweep_deg):
    lam, eps, R, M = math.radians(sweep_deg), 1e-4, 0.01, 6.0
    t = [math.sin(lam), 0.0, math.cos(lam)]                 # edge swept back by lam
    F = B.newtonian_arc_force([[0, 0, 0]], [t], [[-eps, 1, 0]], [[-eps, -1, 0]], R, [1, 0, 0],
                              B.cp_max(M), n_arc=400)
    assert F[0, 0] == pytest.approx(4 / 3 * B.cp_max(M) * R * math.cos(lam) ** 3, rel=1e-4)
    assert abs(F[0, 1]) < 1e-12 * F[0, 0]                   # symmetric arc: no lift


@pytest.fixture(scope="module")
def caret():
    import method_optimization as M

    g = M.build("gvwd", dict(mode="caret", M_design=6.0, theta_d_deg=14.0, Lambda_deg=70.0,
                             L=10.0, gamma=1.4))
    return g.mesh


def test_leading_edge_found_on_the_caret(caret):
    """The caret's LE is the two apex-to-wingtip edges; its base edges are not."""
    v = np.asarray(caret.vertices)
    p0, p1, na, nb = B.sharp_leading_edges(v, caret.faces, [1, 0, 0])
    assert len(p0) == 2
    apex = v[np.argmin(v[:, 0])]
    tips = v[[np.argmin(v[:, 2]), np.argmax(v[:, 2])]]
    found = sorted(map(tuple, np.round(np.vstack([p0, p1]), 9)))
    assert found == sorted(map(tuple, np.round(np.vstack([apex, apex, *tips]), 9)))
    assert np.all((na + nb)[:, 0] < 0.0)                    # facing upstream
    # 70 deg in planform; the edge also drops (anhedral), so it is less swept to the flow
    edge = tips[0] - apex
    assert B.flow_sweep(p0, p1, [1, 0, 0]) == pytest.approx(
        [math.asin(edge[0] / np.linalg.norm(edge))] * 2, abs=1e-12)
    assert math.degrees(math.atan2(edge[0], abs(edge[2]))) == pytest.approx(70.0)
    assert len(B.sharp_leading_edges(v, caret.faces, [-1, 0, 0])[0]) == 0   # from behind: no LE


def test_models_follow_their_sources():
    from gvwd.heating.fay_riddell import swept_LE_heat_flux
    from planar_waverider_aero import pressure_bleed_correction
    from waverider_generator.leading_edge_blunting import compute_sweep_scaled_radius

    le = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [2.5, 0.05, 1.0], [4.0, 0.1, 1.2]])
    r = B.edge_radius(le[:-1], le[1:], 0.01, sweep_scaled=True)
    # same rule as the OC tab's blunting, per edge direction
    expected = [compute_sweep_scaled_radius(le[i:i + 2], 0.01)[0] for i in range(len(le) - 1)]
    assert r == pytest.approx(expected, rel=1e-12)
    assert np.all(B.edge_radius(le[:-1], le[1:], 0.01) == 0.01)
    assert B.pressure_bleed(6.0, 0.05, 1.2, 40.0) == pytest.approx(
        pressure_bleed_correction(6.0, 0.05, 1.2, 40.0))
    q = B.le_heat_flux(0.04, 1800.0, np.array([0.005, 0.0]), np.radians([60.0, 60.0]))
    assert q[0] == pytest.approx(swept_LE_heat_flux(0.04, 1800.0, 0.005, math.radians(60.0)))
    assert q[1] == np.inf


def _solve(mesh, mach):
    from pysagas.cfd import OPM
    from pysagas.flow import FlowState
    from pysagas.geometry import Cell, Vector

    v = np.asarray(mesh.vertices, float)
    cells = [Cell.from_points([Vector(*map(float, q)) for q in v[t]]) for t in np.asarray(mesh.faces)]
    solver = OPM(cells, FlowState(mach=mach, pressure=2549.0, temperature=221.55), verbosity=0)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(aoa=0.0)
    return solver


def test_blunt_le_in_the_corrections(caret):
    from geometry_export import mesh_planform_area

    solver, S = _solve(caret, 6.0), mesh_planform_area(caret)
    sharp = corrected_coefficients(solver, S, 10.0, AeroCorrections())
    assert sharp["CD_le"] == sharp["CD_bleed"] == sharp["CL_blunt"] == 0.0
    assert math.isnan(sharp["q_LE_max"])
    r1 = corrected_coefficients(solver, S, 10.0, AeroCorrections(le_radius=0.005, le_bleed=False))
    r2 = corrected_coefficients(solver, S, 10.0, AeroCorrections(le_radius=0.010, le_bleed=False))
    assert r1["CD_le"] > 0.0 and r2["CD_le"] == pytest.approx(2.0 * r1["CD_le"], rel=1e-12)
    assert r1["CD"] == pytest.approx(sharp["CD"] + r1["CD_le"], rel=1e-12)
    assert r1["CL"] / r1["CD"] < sharp["CL"] / sharp["CD"]
    v = np.asarray(caret.vertices)
    edge = v[np.argmax(v[:, 2])] - v[np.argmin(v[:, 0])]               # apex to a wingtip
    assert r1["LE_length"] == pytest.approx(2 * np.linalg.norm(edge), rel=1e-12)
    # the bleed adds pressure behind the LE on the compression (lower) side
    bleed = corrected_coefficients(solver, S, 10.0, AeroCorrections(le_radius=0.005))
    assert bleed["CD_le"] == pytest.approx(r1["CD_le"]) and bleed["CD_bleed"] != 0.0
    assert bleed["CD"] == pytest.approx(
        bleed["CD_inviscid"] + bleed["CD_base"] + bleed["CD_friction"] + bleed["CD_le"]
        + bleed["CD_bleed"], rel=1e-12)
    # heat flux: Tauber-Sutton at the LE's sweep to the flow, R = 5 mm
    from gvwd.heating.fay_riddell import swept_LE_heat_flux
    rho, V = 2549.0 / (287.0 * 221.55), 6.0 * math.sqrt(1.4 * 287.0 * 221.55)
    sweep = math.asin(edge[0] / np.linalg.norm(edge))
    assert r1["q_LE_max"] == pytest.approx(swept_LE_heat_flux(rho, V, 0.005, sweep), rel=1e-9)
    assert "blunt LE: R = 5 mm (uniform)" in r1["corrections"]
