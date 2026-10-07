"""Verification tests for Mode B leading-edge fillet compensation."""

import math

import numpy as np
import pytest

from waverider_generator.le_fillet_compensation import (
    FilletCompensationConfig, compensate_le_mode_b, radius_schedule,
    smoothstep_weight, wedge_angle, write_station_csv,
    write_control_points_csv, plot_compensation, STATION_COLUMNS)


def _unit(v):
    return v / np.linalg.norm(v)


def _inscribed_center(apex, d_lower, d_upper, R):
    """Centre of the radius-R circle tangent to two rays from ``apex``."""
    d_lower, d_upper = _unit(d_lower), _unit(d_upper)
    half = 0.5 * math.atan2(np.linalg.norm(np.cross(d_lower, d_upper)),
                            np.dot(d_lower, d_upper))
    return apex + R / math.sin(half) * _unit(d_lower + d_upper)


# ---------------------------------------------------------------------------
# Synthetic geometries
# ---------------------------------------------------------------------------

def wedge_2d_extruded(theta_deg, chord=100.0, n_j=41, z=(0.0, 1.0, 2.0)):
    """Lower face y = 0, upper face y = x tan(theta), apex on the z axis."""
    th = math.radians(theta_deg)
    xi = np.linspace(0.0, chord, n_j)
    upper, lower = [], []
    for zi in z:
        lower.append(np.column_stack([xi, np.zeros(n_j), np.full(n_j, zi)]))
        upper.append(np.column_stack([xi * math.cos(th), xi * math.sin(th),
                                      np.full(n_j, zi)]))
    return upper, lower


def swept_wedge(theta_n_deg, sweep_deg, chord=200.0, n_j=61, n_i=5, dz=1.0):
    """
    Wedge with normal-plane angle theta_n, LE swept by `sweep_deg`, grid lines
    streamwise (+x).  Lower face y = 0; the streamwise section angle is
    atan(cos(sweep) tan(theta_n)).
    """
    th, lam = math.radians(theta_n_deg), math.radians(sweep_deg)
    x = np.linspace(0.0, chord, n_j)
    upper, lower = [], []
    for i in range(n_i):
        zi = i * dz
        P = np.array([zi * math.tan(lam), 0.0, zi])
        lower.append(P + np.column_stack([x, np.zeros(n_j), np.zeros(n_j)]))
        upper.append(P + np.column_stack([x, x * math.cos(lam) * math.tan(th),
                                          np.zeros(n_j)]))
    return upper, lower


def _unit_cfg(**kw):
    """Config in geometry units (mm_per_unit = 1)."""
    kw.setdefault("mm_per_unit", 1.0)
    return FilletCompensationConfig(blunting_enabled=True, **kw)


# ---------------------------------------------------------------------------
# Unit checks
# ---------------------------------------------------------------------------

def test_wedge_angle_matches_arccos_away_from_degeneracy():
    for th in np.radians([1.0, 5.0, 10.0, 45.0, 120.0]):
        n_l = np.array([0.0, -1.0, 0.0])
        n_u = np.array([-math.sin(th), math.cos(th), 0.0])
        got, s, c = wedge_angle(n_u, n_l)
        assert got == pytest.approx(th, abs=1e-14)
        assert s == pytest.approx(math.sin(th / 2), abs=1e-15)
        assert c == pytest.approx(math.cos(th / 2), abs=1e-15)


def test_smoothstep_weight():
    w = smoothstep_weight([0.0, 2.0, 2.5, 3.0, 4.0, 9.0], xi_p=2.0, L_b=2.0)
    assert w[0] == 1.0 and w[1] == 1.0
    assert w[2] == pytest.approx(1 - 3 * 0.25 ** 2 + 2 * 0.25 ** 3)
    assert w[3] == pytest.approx(0.5)
    assert w[4] == 0.0 and w[5] == 0.0


# ---------------------------------------------------------------------------
# 1. 2D wedge, analytical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("theta_deg", [5.0, 10.0, 20.0, 35.0])
def test_2d_wedge_analytical(theta_deg, refine):
    R = 1.0
    th = math.radians(theta_deg)
    upper, lower = wedge_2d_extruded(theta_deg, n_j=201)
    res = compensate_le_mode_b(upper, lower, _unit_cfg(R_mm=R, refine_upper=refine),
                               symmetry_station=None)
    t = res.table
    i = 1                                     # interior station
    z = upper[i][0, 2]

    assert t["theta_deg"][i] == pytest.approx(theta_deg, abs=1e-12)
    assert t["L_B_mm"][i] == pytest.approx(1 / math.tan(th / 2) - 1, rel=1e-12)
    assert t["h_u_mm"][i] == pytest.approx(1 + math.cos(th) - math.sin(th), rel=1e-12)
    assert t["t_u_mm"][i] == pytest.approx(1 / math.tan(th / 2), rel=1e-12)
    assert bool(t["feasible"][i])

    # New faces from the compensated grid: lower is still y = 0, upper is the
    # plateau segment starting at P'.
    up, lo = res.upper_streams[i], res.lower_streams[i]
    P_new = res.le_new[i]
    assert np.array_equal(up[0], P_new) and np.array_equal(lo[0], P_new)
    assert np.all(lo[:, 1] == 0.0)
    d_lower = lo[-1] - lo[0]
    d_upper = up[1] - up[0]
    C = _inscribed_center(P_new, d_lower, d_upper, R)
    assert C[0] == pytest.approx(R, abs=1e-12)
    assert C[1] == pytest.approx(R, abs=1e-12)
    assert C[2] == pytest.approx(z, abs=1e-12)
    assert C[0] - R == pytest.approx(0.0, abs=1e-12)   # foremost point at x = 0

    # Upper face over the fillet zone is an exact parallel offset.
    xi_up = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(up, axis=0), axis=1))])
    n_u = np.array([-math.sin(th), math.cos(th), 0.0])
    in_zone = xi_up <= t["t_u_mm"][i]
    assert in_zone.sum() >= 3
    offset = (up[in_zone] - upper[i][0]) @ n_u
    assert np.allclose(offset, t["h_u_mm"][i], atol=1e-12)


@pytest.mark.parametrize("theta_deg, L_B, t_u", [
    (10.0, 10.430052, None),
    (20.0, 4.671282, 5.671282),
])
def test_2d_wedge_reference_values(theta_deg, L_B, t_u):
    upper, lower = wedge_2d_extruded(theta_deg)
    res = compensate_le_mode_b(upper, lower, _unit_cfg(R_mm=1.0), symmetry_station=None)
    assert res.table["L_B_mm"][1] == pytest.approx(L_B, abs=5e-7)
    if t_u is not None:
        assert res.table["t_u_mm"][1] == pytest.approx(t_u, abs=5e-7)


# ---------------------------------------------------------------------------
# 2. Swept wedge in 3D: normal-plane angle, 2D results per station
# ---------------------------------------------------------------------------

def test_swept_wedge_recovers_normal_plane_theta():
    theta_n, sweep, R = 12.0, 70.0, 1.0
    th, lam = math.radians(theta_n), math.radians(sweep)
    theta_s = math.degrees(math.atan(math.cos(lam) * math.tan(th)))
    upper, lower = swept_wedge(theta_n, sweep)
    res = compensate_le_mode_b(upper, lower, _unit_cfg(R_mm=R), symmetry_station=None)
    t = res.table

    assert theta_s < 5.0                       # streamwise angle is very different
    assert np.allclose(t["theta_deg"], theta_n, atol=1e-10)
    assert np.allclose(t["L_B_mm"], 1 / math.tan(th / 2) - 1, rtol=1e-10)
    assert np.allclose(t["h_u_mm"], 1 + math.cos(th) - math.sin(th), rtol=1e-10)
    d_u = _unit(np.array([1.0, math.cos(lam) * math.tan(th), 0.0]))
    phi_u = math.degrees(math.acos(d_u[0] * math.sin(lam)))
    assert np.allclose(t["phi_u_deg"], phi_u, atol=1e-8)
    assert np.allclose(t["phi_l_deg"], 90.0 - sweep, atol=1e-8)
    assert t["feasible"].all()

    e = np.array([math.sin(lam), 0.0, math.cos(lam)])          # LE tangent
    m = np.array([math.cos(lam), 0.0, -math.sin(lam)])         # aft, in lower face
    y = np.array([0.0, 1.0, 0.0])
    for i in range(1, len(upper) - 1):
        P, P_new = upper[i][0], res.le_new[i]
        delta = P_new - P
        assert abs(np.dot(delta, e)) < 1e-12                    # Δ in the normal plane

        # The new upper face in the fillet zone is the old face moved by Δ.
        # The tangent point lies at normal distance t_u, i.e. t_u / sin(phi)
        # along the streamwise grid line.
        up = res.upper_streams[i]
        xi = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(up, axis=0), axis=1))])
        reach = t["t_u_mm"][i] / math.sin(math.radians(t["phi_u_deg"][i]))
        zone = xi <= reach * 1.0000001
        moved = up[zone] - _original_points(upper[i], up[zone] - delta)
        assert np.allclose(moved, delta, atol=1e-12)

        # Circle in the normal plane through P, tangent to both new faces.
        n_plane = np.cross(up[1] - up[0], res.upper_streams[i + 1][0] - up[0])
        d_upper = _unit(np.cross(n_plane, e))
        if np.dot(d_upper, m) < 0:
            d_upper = -d_upper
        C = _inscribed_center(P_new, m, d_upper, R)
        local = np.array([np.dot(C - P, m), np.dot(C - P, y), np.dot(C - P, e)])
        assert np.allclose(local, [R, R, 0.0], atol=1e-11)

        # Lower points are untouched and the edges meet at P'.
        lo = res.lower_streams[i]
        assert np.array_equal(lo[-len(lower[i]):], lower[i])
        assert np.array_equal(lo[0], P_new) and np.array_equal(up[0], P_new)


def _original_points(stream, approx):
    """Points on the original polyline closest to ``approx`` (straight line)."""
    a, b = stream[0], stream[-1]
    d = _unit(b - a)
    return a + np.outer((approx - a) @ d, d)


# ---------------------------------------------------------------------------
# 3. Real WADOS case (GUI-default OC design)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oc_default():
    from waverider_generator.generator import waverider
    return waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0,
                     dp=[0.11, 0.63, 0.0, 0.46], n_upper_surface=1000,
                     n_shockwave=1000, n_planes=40, n_streamwise=30,
                     delta_streamwise=0.1)


@pytest.mark.parametrize("cfg", [
    FilletCompensationConfig(blunting_enabled=True, R_mm=5.0),
    FilletCompensationConfig(blunting_enabled=True, R_mm=5.0, refine_upper=True),
    FilletCompensationConfig(blunting_enabled=True, fillet_mode="variable",
                             R_start_mm=20.0, R_end_mm=5.0, N_steps=5),
], ids=["constant", "constant-refined", "variable"])
def test_real_oc_waverider(oc_default, cfg):
    us = oc_default.upper_surface_streams
    ls = oc_default.lower_surface_streams
    us_before = [s.copy() for s in us]
    ls_before = [s.copy() for s in ls]
    res = compensate_le_mode_b(us, ls, cfg)
    t = res.table
    n = len(us)

    # inputs untouched
    for a, b in zip(us, us_before):
        assert np.array_equal(a, b)
    for a, b in zip(ls, ls_before):
        assert np.array_equal(a, b)

    for i in range(n):
        lo, up = res.lower_streams[i], res.upper_streams[i]
        # lower: original points bit-identical aft of the new segment
        assert np.array_equal(lo[-len(ls[i]):], ls[i])
        # upper and lower edges coincide at P'
        assert np.array_equal(lo[0], res.le_new[i])
        assert np.array_equal(up[0], res.le_new[i])
        # upper points beyond the blend zone are untouched
        xi = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(up, axis=0), axis=1))])
        far = xi > (t["xi_p_mm"][i] + t["L_b_mm"][i]) / 1000.0 + np.linalg.norm(res.le_new[i] - us[i][0])
        for p in up[far]:
            assert np.any(np.all(us[i] == p, axis=1))

    # nose stays on the symmetry plane
    assert res.le_new[0, 2] == 0.0
    assert np.all(res.upper_streams[0][:, 2] == 0.0)
    assert np.all(res.lower_streams[0][:, 2] == 0.0)

    # the zero-chord wingtip is reported, not clamped
    assert bool(t["degenerate"][-1]) and not bool(t["feasible"][-1])
    assert t["R_max_mm"][-1] == 0.0
    assert n - 1 in res.infeasible_stations
    assert "zero chord" in res.feasibility_report()

    # wedge angles are physical and finite
    assert np.all((t["theta_deg"] > 1.0) & (t["theta_deg"] < 30.0))


def test_real_oc_outputs(oc_default, tmp_path):
    cfg = FilletCompensationConfig(blunting_enabled=True, fillet_mode="variable",
                                   R_start_mm=20.0, R_end_mm=5.0, N_steps=5)
    res = compensate_le_mode_b(oc_default.upper_surface_streams,
                               oc_default.lower_surface_streams, cfg)
    write_station_csv(res, tmp_path / "st.csv")
    write_control_points_csv(res, tmp_path / "cp.csv")
    plot_compensation(res, tmp_path / "plot.png")

    lines = (tmp_path / "st.csv").read_text().splitlines()
    assert lines[0].split(",") == STATION_COLUMNS
    assert len(lines) == 1 + len(oc_default.upper_surface_streams)
    cp = (tmp_path / "cp.csv").read_text().splitlines()
    assert len(cp) == 1 + 6
    assert (tmp_path / "plot.png").stat().st_size > 10_000


# ---------------------------------------------------------------------------
# 4. Variable schedule
# ---------------------------------------------------------------------------

def test_variable_schedule_20_to_5_in_5_steps():
    cfg = FilletCompensationConfig(fillet_mode="variable", R_start_mm=20.0,
                                   R_end_mm=5.0, N_steps=5)
    s_total = 3000.0
    s = np.linspace(0.0, s_total, 31)
    R, cp = radius_schedule(s, cfg, s_total)

    assert np.allclose(cp["R_mm"], [20, 17, 14, 11, 8, 5], atol=1e-12)
    assert cp["s_mm"][0] == s_total and cp["s_mm"][-1] == 0.0  # wingtip -> nose
    assert np.allclose(np.diff(cp["s_mm"]), -s_total / 5, atol=1e-9)
    assert np.allclose(np.interp(cp["s_mm"][::-1], s, R), cp["R_mm"][::-1], atol=1e-9)
    assert np.all(np.diff(R) > 0)                    # monotone between ends


def test_variable_schedule_clamped_outside_range():
    cfg = FilletCompensationConfig(fillet_mode="variable", R_start_mm=20.0,
                                   R_end_mm=5.0, N_steps=5,
                                   s_start_mm=2000.0, s_end_mm=500.0)
    s = np.array([0.0, 400.0, 500.0, 1250.0, 2000.0, 2500.0, 3000.0])
    R, cp = radius_schedule(s, cfg, 3000.0)
    assert np.allclose(R, [5, 5, 5, 12.5, 20, 20, 20], atol=1e-12)
    assert np.allclose(cp["s_mm"], [2000, 1700, 1400, 1100, 800, 500])


def test_variable_schedule_on_geometry_control_points():
    upper, lower = swept_wedge(12.0, 40.0, chord=2000.0, n_i=11, dz=10.0)
    cfg = _unit_cfg(fillet_mode="variable", R_start_mm=20.0, R_end_mm=5.0, N_steps=5)
    res = compensate_le_mode_b(upper, lower, cfg, symmetry_station=None)
    cp = res.control_points
    assert np.allclose(cp["R_mm"], [20, 17, 14, 11, 8, 5], atol=1e-12)
    assert np.allclose(np.diff(cp["s_mm"]), np.diff(cp["s_mm"])[0], atol=1e-9)
    # control points sit on the new edge (straight here)
    d = _unit(res.le_new[-1] - res.le_new[0])
    rel = cp["xyz_mm"] - res.le_new[0]
    assert np.allclose(rel - np.outer(rel @ d, d), 0.0, atol=1e-9)
