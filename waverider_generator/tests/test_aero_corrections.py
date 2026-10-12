"""Tests for ``aero_corrections``: base pressure and skin friction on PySAGAS results.

The base-pressure correlation must follow Igra 2018 Eq. (6); the correction
must change only the faces PySAGAS leaves at vacuum, by exactly p_b times
their area. The friction of a flat plate must match the integral of the same
local skin-friction law, and its force must point downstream.
"""
import contextlib
import io
import math

import numpy as np
import pytest

from aero_corrections import (INVISCID, AeroCorrections, base_pressure_ratio,
                              corrected_coefficients, friction_forces)

P, T = 2549.0, 221.55            # about 25 km


def _solve(mesh, mach, aoa=0.0):
    from pysagas.cfd import OPM
    from pysagas.flow import FlowState
    from pysagas.geometry import Cell, Vector

    v = np.asarray(mesh.vertices, float)
    cells = [Cell.from_points([Vector(*map(float, q)) for q in v[t]]) for t in np.asarray(mesh.faces)]
    solver = OPM(cells, FlowState(mach=mach, pressure=P, temperature=T), verbosity=0)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(aoa=aoa)
    return solver


@pytest.fixture(scope="module")
def caret():
    import method_optimization as M
    from geometry_export import mesh_planform_area

    g = M.build("gvwd", dict(mode="caret", M_design=6.0, theta_d_deg=14.0, Lambda_deg=70.0,
                             L=10.0, gamma=1.4))
    return g.mesh, _solve(g.mesh, 6.0), mesh_planform_area(g.mesh)


def test_base_pressure_ratio():
    g, M = 1.4, 8.0
    cp = 2 / (g * M**2) * ((2 / (g + 1)) ** 1.4 * (1 / M) ** 2.8 * (2 * g * M**2 - (g - 1)) / (g + 1) - 1)
    assert base_pressure_ratio("igra", M) == pytest.approx(1 + 0.5 * g * M**2 * cp, rel=1e-12)
    assert base_pressure_ratio("igra", M) == pytest.approx(0.1709, abs=1e-4)
    ratios = [base_pressure_ratio("igra", m) for m in (2, 4, 6, 10, 14, 20)]
    assert all(0.0 < r < 1.0 for r in ratios) and ratios == sorted(ratios, reverse=True)
    assert base_pressure_ratio("vacuum", 6.0) == 0.0
    assert base_pressure_ratio("freestream", 6.0) == 1.0
    assert base_pressure_ratio("maxwell", 6.0) == pytest.approx(1 / 6)


def test_settings():
    c = AeroCorrections("maxwell", "transition", 800.0, 5e5)
    assert AeroCorrections.from_dict(c.to_dict()) == c
    assert not INVISCID.active and AeroCorrections().active
    assert "Igra" in AeroCorrections().describe() and "turbulent" in AeroCorrections().describe()
    with pytest.raises(ValueError):
        AeroCorrections(base_pressure="zero")
    with pytest.raises(ValueError):
        AeroCorrections(friction="fast")


def test_inviscid_is_pysagas_as_is(caret):
    _, solver, S = caret
    r = corrected_coefficients(solver, S, 10.0, INVISCID)
    CL, CD, Cm = solver.flow_result.coefficients(A_ref=S, c_ref=10.0)
    assert (r["CL"], r["CD"], r["Cm"]) == (pytest.approx(CL, abs=1e-14), pytest.approx(CD, abs=1e-14),
                                           pytest.approx(Cm, abs=1e-14))
    assert r["CD_base"] == 0.0 and r["CD_friction"] == 0.0


def test_base_pressure_acts_on_the_base_only(caret):
    """The caret's base is a flat triangle facing downstream: a base pressure
    p_b lowers the drag by p_b A_base / (q S) and leaves the lift alone."""
    mesh, solver, S = caret
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    n2 = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    A_base = 0.5 * float(np.sum(np.abs(n2[:, 0])[n2[:, 0] / np.linalg.norm(n2, axis=1) > 0.99]))
    q = 0.5 * 1.4 * P * 6.0**2
    for model in ("freestream", "igra", "maxwell"):
        r = corrected_coefficients(solver, S, 10.0, AeroCorrections(model, "off"))
        p_b = base_pressure_ratio(model, 6.0) * P
        assert r["CD_base"] == pytest.approx(-p_b * A_base / (q * S), rel=1e-9)
        assert r["CD"] == pytest.approx(r["CD_inviscid"] + r["CD_base"], rel=1e-12)
        assert r["CL"] == pytest.approx(r["CL_inviscid"], abs=1e-12)
    assert r["base_faces"] == 2


def _plate(L=2.0, b=0.5, n=2000):
    """Both sides of a flat plate along the flow: centroids, normals, areas."""
    x = np.linspace(0.0, L, n + 1)
    h = x[1] - x[0]
    xc = np.concatenate([x[:-1] + h / 3, x[:-1] + 2 * h / 3])      # two triangles per strip
    c = np.column_stack([xc, np.zeros_like(xc), np.full_like(xc, b / 2)])
    c = np.vstack([c, c])
    nrm = np.vstack([np.tile([0.0, 1.0, 0.0], (2 * n, 1)), np.tile([0.0, -1.0, 0.0], (2 * n, 1))])
    A = np.full(4 * n, 0.5 * h * b)
    return c, nrm, A


@pytest.mark.parametrize("model, k, power", [("laminar", 1.328, 0.5), ("turbulent", 0.074, 0.2)])
def test_flat_plate_friction(model, k, power):
    """Sum of face forces = 2 b (1/2 rho* u^2) k L / Re*_L^power, the integral
    of the local law (0.664 Re^-1/2 or 0.0592 Re^-1/5) over the plate."""
    from pswr.aero.viscous import eckert_reference_T, sutherland_viscosity

    L, b, M, Tw = 2.0, 0.5, 6.0, 1000.0
    c, nrm, A = _plate(L, b)
    one = np.ones(len(A))
    F, Re = friction_forces(nrm, A, c, M * one, P * one, T * one, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            wall_temperature=Tw, model=model, R=287.0)
    Ts = eckert_reference_T(T, M, Tw)
    rho_s, u = P / (287.0 * Ts), M * math.sqrt(1.4 * 287.0 * T)
    Re_L = rho_s * u * L / sutherland_viscosity(Ts)
    D = 2 * b * 0.5 * rho_s * u * u * k * L / Re_L**power
    assert F[:, 0].sum() == pytest.approx(D, rel=0.015)
    assert np.allclose(F[:, 1:], 0.0) and Re.max() == pytest.approx(Re_L, rel=1e-3)


def test_transition_lies_between_laminar_and_turbulent():
    c, nrm, A = _plate()
    one = np.ones(len(A))
    args = (nrm, A, c, 6.0 * one, P * one, T * one, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    D = {m: friction_forces(*args, wall_temperature=1000.0, model=m, transition_re=3e5)[0][:, 0].sum()
         for m in ("laminar", "transition", "turbulent")}
    assert D["laminar"] < D["transition"] < D["turbulent"]


def test_friction_on_a_waverider(caret):
    """Friction adds drag and so lowers L/D; the parts add up to the total."""
    _, solver, S = caret
    base_only = corrected_coefficients(solver, S, 10.0, AeroCorrections("igra", "off"))
    for model in ("laminar", "turbulent"):
        r = corrected_coefficients(solver, S, 10.0, AeroCorrections("igra", model))
        assert r["CD_friction"] > 0.0 and r["Re_max"] > 1e6
        assert r["CD"] == pytest.approx(r["CD_inviscid"] + r["CD_base"] + r["CD_friction"], rel=1e-12)
        assert r["CD_base"] == pytest.approx(base_only["CD_base"], rel=1e-12)
        assert r["CL"] / r["CD"] < base_only["CL"] / base_only["CD"]
    lam = corrected_coefficients(solver, S, 10.0, AeroCorrections("igra", "laminar"))
    turb = corrected_coefficients(solver, S, 10.0, AeroCorrections("igra", "turbulent"))
    assert lam["CD_friction"] < turb["CD_friction"]


def test_evaluate_design_applies_the_corrections():
    import method_optimization as M

    params = dict(mode="caret", M_design=6.0, theta_d_deg=14.0, Lambda_deg=70.0, L=10.0, gamma=1.4)
    flow = dict(mach=6.0, pressure=P, temperature=T, aoa=0.0)
    raw = M.evaluate_design("gvwd", params, flow)
    corr = M.evaluate_design("gvwd", params, dict(flow, corrections=AeroCorrections().to_dict()))
    assert raw["success"] and corr["success"]
    assert raw["CD_base"] == raw["CD_friction"] == 0.0 and raw["CD"] == raw["CD_inviscid"]
    assert corr["CD_inviscid"] == pytest.approx(raw["CD"], rel=1e-12)
    assert corr["CD_base"] < 0.0 < corr["CD_friction"]
    assert corr["CD"] == pytest.approx(corr["CD_inviscid"] + corr["CD_base"] + corr["CD_friction"])
