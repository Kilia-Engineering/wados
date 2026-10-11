"""Phase 3 tests: LTOCs core (leading edge, Steps A-C, waverider assembly).

V4: LTOCs on a circular conical shock reproduces the cone-derived waverider.
    The 1e-4 L criterion is applied against the tight Taylor-Maccoll
    reference (Gate 0 section 3.7). The deviation from the existing
    ``ShadowWaverider`` is also bounded.
V5: LTOCs on the shock of a WADOS osculating-cone design reproduces it.
    The 1e-4 L criterion is applied against a tight per-plane reference:
    Taylor-Maccoll in each local cone, exact wedge flow in the flat region.
    The deviation from the OC generator is bounded by that generator's own
    Taylor-Maccoll accuracy (default solve_ivp tolerances).

Also tested: the leading-edge projection; Step C invariants (on a cone each
stream surface is a plane through the axis, and the 3-D distance to the axis
equals the meridian radius); refusals with a clear status; the stream
protocol and STL export.
"""
import contextlib
import io

import numpy as np
import pytest

from ltoc.ltoc import find_leading_edge, solve_stream_surface
from ltoc.reference import ConicalFlowReference, osculating_cone_streamline
from ltoc.shock_geometry import local_geometry
from ltoc.shock_surface import ConeShock, OsculatingConeShock, RuledShock
from ltoc.waverider import LTOCWaverider


# ---------------------------------------------------------------------------
#  Leading edge and Step C on a cone
# ---------------------------------------------------------------------------

def test_leading_edge_projection_lies_on_shock_and_fct():
    s = RuledShock.elliptic(0.1, 0.3, 0.2, 1.5, 0.65, 0.8)          # R1 Eq. (25)
    fct = np.column_stack([np.linspace(0.0, 0.6, 7), np.full(7, 0.25)])
    uv, pts, ok = find_leading_edge(s, fct)
    assert ok.all()
    assert np.max(np.abs(pts[:, 1:] - fct)) < 1e-12
    assert np.all(pts[:, 0] < 1.5)


@pytest.mark.parametrize("fct", [(0.0, 0.18), (0.12, 0.15)])
def test_step_c_on_cone_keeps_stream_surface_in_meridian_plane(fct):
    b = np.radians(20.0)
    cone = ConeShock(b, x_range=(0.0, 2.0))
    uv, pts, ok = find_leading_edge(cone, np.array([fct]))
    s = solve_stream_surface(cone, uv[0], 1.0, 6.0, n_points=40)
    assert s.status == "ok" and s.body.x[-1] == pytest.approx(1.0)
    v = s.moc.valid
    P = s.points3[v]
    phi = np.arctan2(P[:, 2], P[:, 1])
    assert np.ptp(phi) < 1e-12                                          # plane through the axis
    # 3-D distance to the axis equals the meridian radius y - y_axis (coaxial).
    r_m = (s.moc.y - (s.curve.y - s.curve.geometry.r)[0])[v]
    assert np.max(np.abs(np.hypot(P[:, 1], P[:, 2]) - r_m)) < 1e-12
    assert s.diagnostics["stepc_fallbacks"] == 0


# ---------------------------------------------------------------------------
#  V4: cone-derived waverider
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v4():
    from shadow_waverider import ShadowWaverider

    with contextlib.redirect_stdout(io.StringIO()):
        sh = ShadowWaverider(mach=6.0, shock_angle=12.0, poly_coeffs=[-1.0, 0.0, 0.5],
                             n_leading_edge=11, n_streamwise=40)
    le = sh.leading_edge                               # x stream, y vertical, z span
    half = le[:, 2] >= -1e-12
    order = np.argsort(le[half, 2])
    le_half = le[half][order]
    lower_half = sh.lower_surface[half][order]
    # Shock frame (X, Y_s, Z_s) = (x, z_span, y_vert): the symmetry plane is Y_s = 0.
    fct = np.column_stack([le_half[:, 2], le_half[:, 1]])
    to_gui = lambda P: np.stack([P[..., 0], P[..., 2], P[..., 1]], axis=-1)
    wr = LTOCWaverider(ConeShock(np.radians(12.0), x_range=(0.0, 1.2 * sh.x_end)), fct,
                       sh.x_end, 6.0, n_points=40, to_gui=to_gui, strict=True)
    return sh, wr, lower_half


def test_v4_cone_waverider_matches_taylor_maccoll(v4):
    sh, wr, _ = v4
    ref = ConicalFlowReference(6.0, np.radians(12.0))
    length = sh.x_end - sh.x_start
    worst = 0.0
    for s in wr.stations:
        if s.status == "tip":
            continue
        P = s.body.points
        rho = np.hypot(P[:, 1], P[:, 2])
        y_ref = ref.streamline(s.le_point[0], np.hypot(*s.le_point[1:]), sh.x_end)(P[:, 0])
        worst = max(worst, float(np.max(np.abs(rho - y_ref))))
    assert worst < 1e-4 * length
    assert worst < 1e-6 * length                         # actual margin at n_points = 40


def test_v4_cone_waverider_matches_shadow_waverider(v4):
    sh, wr, lower_half = v4
    length = sh.x_end - sh.x_start
    worst = 0.0
    for s, S in zip(wr.stations, lower_half):
        if s.status == "tip":
            continue
        P = s.body.points
        rho = np.hypot(P[:, 1], P[:, 2])
        worst = max(worst, float(np.max(np.abs(np.interp(S[:, 0], P[:, 0], rho)
                                                  - np.hypot(S[:, 1], S[:, 2])))))
    assert worst < 1e-4 * length


# ---------------------------------------------------------------------------
#  V5: osculating-cone waverider
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v5():
    from waverider_generator.generator import waverider

    M, bdeg = 5.0, 15.0
    oc_wr = waverider(M_inf=M, beta=bdeg, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                      n_upper_surface=10000, n_shockwave=10000, n_planes=10, n_streamwise=40,
                      delta_streamwise=0.1)
    shock = OsculatingConeShock.from_oc_waverider(oc_wr)
    fct = np.asarray(oc_wr.leading_edge)[:, [1, 2]]          # the OC design's own LE points
    lt = LTOCWaverider(shock, fct, oc_wr.length, M, n_points=40, strict=True)
    return oc_wr, shock, lt


def _oc_reference(shock, s, x):
    """Exact lower-surface streamline in one osculating plane."""
    return osculating_cone_streamline(shock, s.le_uv, s.le_point, x, 5.0)


def test_v5_oc_waverider_matches_tight_reference(v5):
    oc_wr, shock, lt = v5
    L = oc_wr.length
    worst = 0.0
    for s in lt.stations:
        if s.status == "tip":
            continue
        P = s.body.points
        worst = max(worst, float(np.max(np.linalg.norm(P - _oc_reference(shock, s, P[:, 0]),
                                                          axis=1))))
    assert worst < 1e-4 * L
    assert worst < 2e-6 * L                                  # actual margin at n_points = 40


def test_v5_oc_waverider_matches_oc_generator_within_its_accuracy(v5):
    oc_wr, shock, lt = v5
    L = oc_wr.length
    dev_ltoc, dev_oc = [], []
    for k, s in enumerate(lt.stations):
        if s.status == "tip":
            continue
        S = np.asarray(oc_wr.lower_surface_streams[k])
        P = s.body.points
        yi, zi = np.interp(S[:, 0], P[:, 0], P[:, 1]), np.interp(S[:, 0], P[:, 0], P[:, 2])
        dev_ltoc.append(float(np.max(np.hypot(yi - S[:, 1], zi - S[:, 2]))))
        dev_oc.append(float(np.max(np.linalg.norm(S - _oc_reference(shock, s, S[:, 0]),
                                                  axis=1))))
    # The gap to the OC generator is the generator's own error (default
    # solve_ivp tolerances in waverider_generator/flowfield.py).
    assert np.allclose(dev_ltoc, dev_oc, atol=1e-5 * L)
    assert max(dev_ltoc) < 2e-3 * L


# ---------------------------------------------------------------------------
#  Waverider assembly, protocol, export, refusals
# ---------------------------------------------------------------------------

def test_stream_protocol_and_closed_stl(v5, tmp_path):
    _, _, lt = v5
    us, ls = lt.upper_surface_streams, lt.lower_surface_streams
    assert len(us) == len(ls) == len(lt.stations)
    assert len({u.shape for u in us + ls}) == 1                         # equal length
    assert np.allclose(us[0][:, 2], 0.0) and np.allclose(ls[0][:, 2], 0.0)   # symmetry plane
    assert np.allclose([u[-1, 0] for u in us], lt.length)
    mesh = lt.mesh(full_span=True)
    f = mesh.faces
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert np.all(counts == 2)
    path = lt.export_stl(tmp_path / "ltoc.stl")
    assert path.stat().st_size == 84 + 50 * mesh.n_faces


def test_fct_outside_shock_and_concave_shock_are_refused():
    cone = ConeShock(np.radians(20.0), x_range=(0.0, 2.0))
    R = np.tan(np.radians(20.0))
    wr = LTOCWaverider(cone, np.array([[0.0, 0.5 * R], [0.0, 1.5 * R]]), 1.0, 6.0, n_points=20)
    assert not wr.ok and wr.stations[1].status == "outside_shock"
    with pytest.raises(Exception, match="incomplete"):
        _ = wr.lower_surface_streams
    # A shock whose cross-section curves away from the body (z = 0.3 x + 0.6 y^2 / x,
    # body below): the local axis would sit on the freestream side, kappa_b < 0.
    zero = lambda m: np.zeros(np.shape(m) + (3,))
    q = lambda m: np.stack(np.broadcast_arrays(1.0, m, 0.3 + 0.6 * m ** 2), -1)
    dq = lambda m: np.stack(np.broadcast_arrays(0.0, 1.0, 1.2 * m), -1)
    d2q = lambda m: np.stack(np.broadcast_arrays(0.0, 0.0, 1.2), -1)
    concave = RuledShock(zero, zero, zero, q, dq, d2q, m_range=(-0.4, 0.4))
    uv, _, ok = find_leading_edge(concave, np.array([[0.1, 0.25]]))
    assert ok.all() and local_geometry(concave, *uv[0], 1.0).kappa_b < 0
    s = solve_stream_surface(concave, uv[0], 1.0, 6.0, n_points=20)
    assert s.status == "concave" and "refused" in s.message


# ---------------------------------------------------------------------------
#  Twisted stream surfaces (R1 case II shock): Step C consistency and order
# ---------------------------------------------------------------------------

def test_step_c_on_twisted_stream_surface_is_consistent_and_first_order():
    """Off the symmetry plane of R1 Eq. (25) the shock curves turn, so the
    stream surface twists. The 3-D body velocity must stay tangent to the 3-D
    body streamline, and the body converges at first order (R1's linear
    interpolation in Step C; see the ``ltoc.ltoc`` module docstring)."""
    s = RuledShock.elliptic(0.1, 0.3, 0.2, 1.5, 0.65, 0.8)
    y_tip = 0.65 * np.sqrt(1.0 - (0.25 / 0.8) ** 2)
    uv, _, _ = find_leading_edge(s, np.array([[0.3 * y_tip, 0.25]]))
    out = {n: solve_stream_surface(s, uv[0], 1.5, 7.0, n_points=n) for n in (20, 40, 80)}
    assert all(o.status == "ok" for o in out.values())
    assert out[20].diagnostics["azimuth_drift"] > 0.2                  # the surface really turns
    tang = [out[n].diagnostics["stepc_tangency_deg"] for n in (20, 40, 80)]
    assert tang[2] < 0.03 and tang[0] / tang[1] > 1.8 and tang[1] / tang[2] > 1.8
    e1 = np.linalg.norm(out[20].body.points[-1] - out[40].body.points[-1])
    e2 = np.linalg.norm(out[40].body.points[-1] - out[80].body.points[-1])
    assert 1.8 < e1 / e2 < 2.2 and e2 < 2e-4 * 1.5
    d = out[40].diagnostics
    assert 0.0 < d["extension_needed_chords"] <= d["extension_chords"]
