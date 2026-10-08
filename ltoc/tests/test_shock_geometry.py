"""Phase 2 tests: shock surfaces and shock geometry (LTOCs spec V3).

V3 requirements:
* on a cone, the shock curve is the generator and r equals the cone radius;
* the flag 1 identity: for a surface Z(X, Y) the shock curve has
  dY/dX = -Z_X Z_Y / (1 + Z_Y^2), the opposite of R1 Eq. (1) as printed;
* the flag 3 identity: r = cos(beta) / kappa_b equals the radius of
  curvature of the cross-section X = const (R1's definition), checked on
  elliptic cones and on R1's variable elliptic shock.

Also: the osculating-cone shock against the WADOS OC generator, the B-spline
shock against its source surface, the published shocks (R1 Eqs. 18, 23, 25;
R2 Eq. 13), and the intrinsic meridian ordinate (Gate 0 decision).
"""
import numpy as np
import pytest

from ltoc.shock_geometry import local_geometry, trace_shock_curves
from ltoc.shock_surface import BSplineShock, ConeShock, OsculatingConeShock, RuledShock

R1_CASE_I = dict(x_p=0.0, a_p=0.0, b_p=0.0, x_q=1.7, a_q=0.595, b_q=0.68)            # Eq. (23)
R1_CASE_II = dict(x_p=0.1, a_p=0.3, b_p=0.2, x_q=1.5, a_q=0.65, b_q=0.8)             # Eq. (25)
R1_TEST_II = dict(x_p=0.1, a_p=0.2, b_p=0.3, x_q=1.5, a_q=0.8, b_q=0.65)             # Eq. (18)


def ellipse_radius(a, b, m):
    """Radius of curvature of (a cos m, b sin m) at parameter m."""
    return (a * a * np.sin(m) ** 2 + b * b * np.cos(m) ** 2) ** 1.5 / (a * b)


# ---------------------------------------------------------------------------
#  V3: cone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta_deg", [12.0, 20.0, 35.0])
def test_v3_cone_shock_curve_is_generator_and_r_is_cone_radius(beta_deg):
    b = np.radians(beta_deg)
    cone = ConeShock(b)
    phis = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    for cv in trace_shock_curves(cone, phis, 0.2, 1.0, n_steps=40):
        g = cv.geometry
        assert np.ptp(cv.uv[:, 0]) < 1e-14                               # stays on its generator
        P = cv.points
        assert np.max(np.abs(np.cross(P, P[-1]))) < 1e-14               # collinear with the apex
        assert np.max(np.abs(g.beta - b)) < 1e-14
        assert np.max(np.abs(g.r - cv.x * np.tan(b))) < 1e-14            # r = cone radius
        assert np.max(np.abs(g.axis_centre[:, 1:])) < 1e-14              # centre on the axis
        # Intrinsic meridian ordinate rises at tan(beta); the axis stays at y_axis = const.
        assert np.max(np.abs(cv.y - (cv.x - cv.x[0]) * np.tan(b))) < 1e-13
        assert np.ptp(cv.y_axis) < 1e-13


# ---------------------------------------------------------------------------
#  V3: flag 1 identity
# ---------------------------------------------------------------------------

def test_v3_flag1_shock_curve_slope_identity():
    """Elliptic cone, upper half as Z(X, Y) = b sqrt(X^2 - Y^2 / a^2)."""
    a, bb = 0.35, 0.40
    ell = RuledShock.elliptic(0.0, 0.0, 0.0, 1.0, a, bb, m_range=(0.0, np.pi))
    cv = trace_shock_curves(ell, [0.3 * np.pi], [0.2], 1.0, n_steps=200)[0]
    P = cv.points
    X, Y = P[:, 0], P[:, 1]
    root = np.sqrt(X ** 2 - Y ** 2 / a ** 2)
    ZX, ZY = bb * X / root, -bb * Y / (a ** 2 * root)
    slope_formula = -ZX * ZY / (1.0 + ZY ** 2)
    slope_curve = cv.geometry.t[:, 1] / cv.geometry.t[:, 0]
    assert np.max(np.abs(slope_curve - slope_formula)) < 1e-12
    # The printed R1 Eq. (1) has the opposite sign; the difference is not small.
    assert np.max(np.abs(slope_curve + slope_formula)) > 0.1


# ---------------------------------------------------------------------------
#  V3: flag 3 identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [(0.35, 0.35), (0.35, 0.40), (0.25, 0.60)])
def test_v3_flag3_radius_equals_cross_section_radius_elliptic_cone(a, b):
    ell = RuledShock.elliptic(0.0, 0.0, 0.0, 1.0, a, b)
    m = np.linspace(0.05, 2 * np.pi, 37)
    n = np.full_like(m, 0.7)
    g = local_geometry(ell, m, n)
    R = ellipse_radius(a * n, b * n, m)
    assert np.max(np.abs(g.r / R - 1.0)) < 1e-12
    assert np.max(np.abs(g.b[:, 0])) < 1e-14                            # b is crosswise
    # The axis centre is the centre of curvature of the cross-section.
    assert np.max(np.abs(np.linalg.norm(g.axis_centre - g.point, axis=1) - R)) < 1e-12


def test_v3_flag3_on_variable_elliptic_shock_by_finite_differences():
    """R1 test case II shock (Eq. 18): cross-section radius by finite differences."""
    s = RuledShock.elliptic(**R1_TEST_II)
    m0, n0, h = 1.1, 0.55, 1e-4
    m = np.array([m0 - h, m0, m0 + h])
    P = s.point(m, np.full(3, n0))                     # section X = const: n fixed
    d1 = (P[2] - P[0]) / (2 * h)
    d2 = (P[2] - 2 * P[1] + P[0]) / h ** 2
    k_cs = np.linalg.norm(np.cross(d1, d2)) / np.linalg.norm(d1) ** 3
    g = local_geometry(s, m0, n0)
    assert float(g.r) == pytest.approx(1.0 / k_cs, rel=1e-7)


# ---------------------------------------------------------------------------
#  Osculating-cone shock against the WADOS OC generator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oc_waverider():
    from waverider_generator.generator import waverider

    return waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                     n_upper_surface=10000, n_shockwave=10000, n_planes=20, n_streamwise=30,
                     delta_streamwise=0.1)


def test_osculating_cone_shock_matches_oc_generator(oc_waverider):
    wr = oc_waverider
    oc = OsculatingConeShock.from_oc_waverider(wr)
    L, b = wr.length, np.radians(wr.beta)
    # Base-plane shock against the OC generator's own SWPC samples.
    z = np.asarray(wr.z_local_shockwave)
    y_oc = np.asarray(wr.y_local_shockwave).ravel() - wr.height
    assert np.max(np.abs(oc.point(z, np.full_like(z, L))[:, 1] - y_oc)) < 1e-6 * wr.height
    # Local axis centres against the generator's cone centres (curved planes).
    g = local_geometry(oc, z, np.full_like(z, L), length_scale=L)
    curved = np.isfinite(g.r)
    centres = np.asarray(wr.cone_centers)[curved]
    assert curved.sum() > 5
    assert np.max(np.abs(g.axis_centre[curved, 1:] - centres[:, 1:])) < 1e-6 * wr.height
    # Shock curves are the osculating-plane generators (constant z_s), beta is
    # constant, the axis centre is fixed, and r = R - (L - x) tan(beta).
    for cv in trace_shock_curves(oc, [0.1, 1.0, 2.0, 2.8], 0.5 * L, L, n_steps=60):
        gg = cv.geometry
        assert np.ptp(cv.uv[:, 0]) < 1e-12
        assert np.max(np.abs(gg.beta - b)) < 1e-13
        if np.all(cv.flags["flat"]):
            continue
        assert np.max(np.ptp(gg.axis_centre[:, 1:], axis=0)) < 1e-10
        assert np.max(np.abs(np.diff(gg.r) / np.diff(cv.x) - np.tan(b))) < 1e-9   # r grows downstream


# ---------------------------------------------------------------------------
#  B-spline shock
# ---------------------------------------------------------------------------

def test_bspline_shock_reproduces_source_surface():
    src = RuledShock.elliptic(**R1_CASE_II, m_range=(0.3, 2.8), n_range=(0.0, 1.0))
    bs = BSplineShock.from_surface(src, nu=60, nv=30)
    m = np.linspace(0.6, 2.5, 9)
    n = np.linspace(0.2, 0.9, 9)
    g_src, g_bs = local_geometry(src, m, n), local_geometry(bs, m, n)
    assert np.max(np.abs(g_bs.point - g_src.point)) < 1e-6
    assert np.max(np.abs(g_bs.beta - g_src.beta)) < 1e-5
    assert np.max(np.abs(g_bs.r / g_src.r - 1.0)) < 1e-3


# ---------------------------------------------------------------------------
#  Published shocks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case,M", [(R1_CASE_I, 6.0), (R1_CASE_II, 7.0), (R1_TEST_II, 7.0)],
                         ids=["R1-Eq23", "R1-Eq25", "R1-Eq18"])
def test_published_r1_shocks_are_convex_and_attached(case, M):
    s = RuledShock.elliptic(**case)
    m = np.linspace(0.2, np.pi - 0.2, 7)
    curves = trace_shock_curves(s, m, 0.15, case["x_q"], n_steps=60, M_inf=M)
    for cv in curves:
        for k in ("concave", "sub_mach", "detached"):
            assert not cv.flags[k].any(), k
        assert np.all(cv.geometry.r > 0.0)


def test_r2_quartic_shock_curvature_is_positive_toward_the_body():
    """R2 Eq. (13): p = 0, q = (1.3, m, 5.28 m^4 + 0.16 m^2 - 0.37)."""
    zero = lambda m: np.zeros(np.shape(m) + (3,))
    q = lambda m: np.stack(np.broadcast_arrays(1.3, m, 5.28 * m ** 4 + 0.16 * m ** 2 - 0.37), -1)
    dq = lambda m: np.stack(np.broadcast_arrays(0.0, 1.0, 21.12 * m ** 3 + 0.32 * m), -1)
    d2q = lambda m: np.stack(np.broadcast_arrays(0.0, 0.0, 63.36 * m ** 2 + 0.32), -1)
    s = RuledShock(zero, zero, zero, q, dq, d2q, m_range=(-0.5, 0.5))
    g = local_geometry(s, np.linspace(-0.45, 0.45, 11), 0.8)
    assert np.all(g.kappa_b > 0.0)
    assert np.all(np.degrees(g.beta) > np.degrees(np.arcsin(1 / 6.0)))


def test_shock_curves_turn_on_elliptic_shocks():
    """The point of LTOCs: on a non-circular shock the shock curve leaves its
    starting meridian (local turning), so the stream surface is not planar."""
    s = RuledShock.elliptic(**R1_CASE_II)
    cv = trace_shock_curves(s, [0.25 * np.pi], [0.1], 1.5, n_steps=100)[0]
    assert np.ptp(cv.uv[:, 0]) > 0.02                     # azimuth drift [rad]
    assert np.ptp(cv.y_axis) > 1e-3                       # axis centre drifts (noncoaxial)


def test_start_downstream_of_end_is_refused():
    with pytest.raises(ValueError):
        trace_shock_curves(ConeShock(0.3), [0.0], [0.9], 0.5)
