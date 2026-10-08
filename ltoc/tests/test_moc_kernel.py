"""Phase 1 tests: two-family inverse MOC kernel (``ltoc.moc_noncoaxial``).

Spec V1: coaxial rule, circular conical shock at Mach 6 and 10. The kernel must
match the existing Taylor-Maccoll solver (``liu2019.shock.
taylor_maccoll_cone_field``) to 0.1 %, and converge with refinement. The order
is measured against the tight (rtol 1e-12) reference in ``ltoc.reference``.

Spec V2: planar rule, straight shock. The post-shock flow must be uniform to
round-off.

Also tested:
* the pluggable axis rule (translation invariance; the noncoaxial rule
  reduces to the coaxial one);
* the body streamline against the Taylor-Maccoll streamline;
* rotational flow behind a curved shock (the two schemes agree, and the
  convergence is second order);
* limit-surface detection (spec flag 6, R4 Sec. 2);
* input validation.
"""
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from ltoc.moc_noncoaxial import (
    SCHEMES, AxisRule, AxisCrossedError, InitialLine, LimitSurfaceError, MOCError,
    SubsonicError, CHARACTERISTIC_FOLD, shock_initial_line, solve_inverse,
)
from ltoc.reference import ConicalFlowReference

CONES = [(6.0, 20.0), (10.0, 14.0)]          # (M_inf, beta_deg)
X0, X1 = 0.25, 1.0                           # shock extent (apex at the origin)


def cone_solution(M, beta_deg, N, scheme, rule=None, y_shift=0.0):
    b = np.radians(beta_deg)
    x = np.linspace(X0, X1, N + 1)
    y = x * np.tan(b) + y_shift
    rule = AxisRule.coaxial() if rule is None else rule
    return solve_inverse(shock_initial_line(x, y, b, M), rule, scheme=scheme)


def rel_errors(sol, ref):
    """Max relative errors of p, M, theta over valid net points covered by ref."""
    _, _, Mn, _ = sol.derived()
    st = ref.state_at(sol.x, sol.y)
    ok = sol.valid & np.isfinite(st["p"])
    return (np.max(np.abs(sol.p[ok] / st["p"][ok] - 1.0)),
            np.max(np.abs(Mn[ok] / st["M"][ok] - 1.0)),
            np.max(np.abs(sol.theta[ok] / st["theta"][ok] - 1.0)),
            int(ok.sum()))


def curved_shock(N, b0, b1, ramp="smooth", x0=X0, x1=X1, y0=0.09):
    """Shock with beta going from b0 to b1 (deg); 'smooth' over the whole
    length, 'steep' over 0.45 < x < 0.55."""
    if ramp == "smooth":
        def beta(xx):
            s = (xx - x0) / (x1 - x0)
            return np.radians(b0 + (b1 - b0) * (3 * s ** 2 - 2 * s ** 3))
    else:
        def beta(xx):
            return np.radians(b0 + (b1 - b0) * 0.5 * (1 + np.tanh((xx - 0.5) / 0.025)))
    x = np.linspace(x0, x1, N + 1)
    xf = np.linspace(x0, x1, 40001)
    yf = y0 + cumulative_trapezoid(np.tan(beta(xf)), xf, initial=0.0)
    return x, np.interp(x, xf, yf), beta(x)


# ---------------------------------------------------------------------------
#  V2: planar shock -> uniform flow
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
def test_v2_planar_shock_gives_uniform_flow(scheme):
    M, b = 6.0, np.radians(20.0)
    x = np.linspace(X0, X1, 61)
    il = shock_initial_line(x, x * np.tan(b), b, M)
    sol = solve_inverse(il, AxisRule.planar(), scheme=scheme, strict=True)
    v = sol.valid
    assert v.sum() > 0.5 * sol.exists.sum()
    assert np.max(np.abs(sol.p[v] / il.p[0] - 1.0)) < 1e-13
    assert np.max(np.abs(sol.theta[v] - il.theta[0])) < 1e-13
    assert np.max(np.abs(sol.K[v] / sol.K[0, 0] - 1.0)) < 1e-13


@pytest.mark.parametrize("scheme", SCHEMES)
def test_v2_large_noncoaxial_radius_tends_to_planar(scheme):
    M, b = 6.0, np.radians(20.0)
    x = np.linspace(X0, X1, 41)
    il = shock_initial_line(x, x * np.tan(b), b, M)
    sol = solve_inverse(il, AxisRule.offset(-1e8), scheme=scheme, strict=True)
    v = sol.valid
    assert np.max(np.abs(sol.p[v] / il.p[0] - 1.0)) < 1e-7


# ---------------------------------------------------------------------------
#  V1: conical shock, coaxial rule
# ---------------------------------------------------------------------------

def existing_solver(M, beta_deg, g=1.4):
    """Flow-state evaluator built on the existing liu2019 Taylor-Maccoll splines.

    Returns ``(state(omega) -> dict, cone_angle, beta)`` in radians.
    """
    from gvwd.thermo.oblique_shock import stagnation_pressure_ratio
    from liu2019.shock import taylor_maccoll_cone_field

    Vr_s, Vt_s, dc, b = taylor_maccoll_cone_field(M, beta_deg, g)
    p02 = stagnation_pressure_ratio(M, b, g) * (1 + 0.5 * (g - 1) * M ** 2) ** (g / (g - 1))

    def state(omega):
        Vr, Vt = Vr_s(omega), Vt_s(omega)
        Vp2 = Vr ** 2 + Vt ** 2
        Mloc = np.sqrt(2.0 / (g - 1.0) * Vp2 / (1.0 - Vp2))
        p = p02 * (1 + 0.5 * (g - 1) * Mloc ** 2) ** (-g / (g - 1))
        return {"p": p, "M": Mloc, "theta": omega + np.arctan2(Vt, Vr)}

    return state, dc, b


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("M,beta_deg", CONES)
def test_v1_matches_existing_taylor_maccoll_solver(M, beta_deg, scheme):
    sol = cone_solution(M, beta_deg, 100, scheme)
    _, _, Mn, _ = sol.derived()
    state, dc, b = existing_solver(M, beta_deg)
    omega = np.arctan2(sol.y, sol.x)
    ok = sol.valid & (omega >= dc) & (omega <= b)
    assert ok.sum() > 1000
    st = state(omega[ok])
    assert np.max(np.abs(sol.p[ok] / st["p"] - 1.0)) < 1e-3
    assert np.max(np.abs(Mn[ok] / st["M"] - 1.0)) < 1e-3
    assert np.max(np.abs(sol.theta[ok] / st["theta"] - 1.0)) < 1e-3


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("M,beta_deg", CONES)
def test_v1_second_order_convergence(M, beta_deg, scheme):
    ref = ConicalFlowReference(M, np.radians(beta_deg))
    errs = np.array([rel_errors(cone_solution(M, beta_deg, N, scheme), ref)[:3]
                     for N in (50, 100, 200)])
    order = np.log2(errs[:-1] / errs[1:])
    assert np.all(order > 1.8), f"observed orders {order}"
    assert np.all(errs[-1] < 2e-5), f"errors at N=200: {errs[-1]}"


def test_tight_reference_agrees_with_existing_solver():
    for M, beta_deg in CONES:
        ref = ConicalFlowReference(M, np.radians(beta_deg))
        omega = np.linspace(ref.cone_angle, np.radians(beta_deg), 50)
        state, dc, _ = existing_solver(M, beta_deg)
        st, tight = state(omega), ref.state(omega)
        assert abs(dc - ref.cone_angle) < 1e-6
        # The public splines are rtol 1e-6 with sparse knots: measured
        # ~7e-5 in theta and ~3e-5 in p, well inside the 0.1 % V1 criterion.
        assert np.max(np.abs(tight["theta"] / st["theta"] - 1.0)) < 2e-4
        assert np.max(np.abs(tight["p"] / st["p"] - 1.0)) < 1e-4


@pytest.mark.parametrize("M,beta_deg", CONES)
def test_body_streamline_matches_taylor_maccoll(M, beta_deg):
    """Column 0 of the streamline scheme is the streamline from the first
    shock point; it must follow the Taylor-Maccoll streamline."""
    ref = ConicalFlowReference(M, np.radians(beta_deg))
    sol = cone_solution(M, beta_deg, 200, "streamline")
    sl = sol.streamline_column(0)
    assert sl.reached_end
    y_ref = ref.streamline(sl.x[0], sl.y[0], sl.x[-1])(sl.x)
    p_ref = ref.state_at(sl.x, y_ref)["p"]
    assert np.max(np.abs(sl.y - y_ref)) < 1e-6 * (X1 - X0)
    assert np.max(np.abs(sl.p / p_ref - 1.0)) < 5e-6
    # The characteristic scheme traces the same streamline by interpolation.
    tr = cone_solution(M, beta_deg, 200, "characteristic").trace_streamline(0, 0)
    y_ref_tr = ref.streamline(tr.x[0], tr.y[0], tr.x[-1])(tr.x)
    assert tr.x[-1] > 0.95 * sl.x[-1]
    assert np.max(np.abs(tr.y - y_ref_tr)) < 1e-6 * (X1 - X0)
    assert np.max(np.abs(tr.p / ref.state_at(tr.x, y_ref_tr)["p"] - 1.0)) < 5e-5


# ---------------------------------------------------------------------------
#  Axis rule plumbing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
def test_axis_rule_translation_invariance(scheme):
    base = cone_solution(6.0, 20.0, 60, scheme)
    c = 0.37
    shifted = cone_solution(6.0, 20.0, 60, scheme, rule=AxisRule.offset(c), y_shift=c)
    v = base.valid
    assert np.array_equal(v, shifted.valid)
    assert np.max(np.abs(shifted.y[v] - c - base.y[v])) < 1e-12
    assert np.max(np.abs(shifted.p[v] / base.p[v] - 1.0)) < 1e-11
    assert np.max(np.abs(shifted.theta[v] - base.theta[v])) < 1e-11


@pytest.mark.parametrize("scheme", SCHEMES)
def test_noncoaxial_rule_with_axis_centres_on_axis_is_coaxial(scheme):
    """For a circular cone every local axis centre lies on the cone axis, so
    the noncoaxial rule built from the shock's own radii must be coaxial."""
    M, b = 6.0, np.radians(20.0)
    x = np.linspace(X0, X1, 61)
    y = x * np.tan(b)
    r_shock = y.copy()                                    # crosswise radius of the cone
    rule = AxisRule.noncoaxial(x, y - r_shock)
    a = solve_inverse(shock_initial_line(x, y, b, M), rule, scheme=scheme)
    c = solve_inverse(shock_initial_line(x, y, b, M), AxisRule.coaxial(), scheme=scheme)
    v = c.valid
    assert np.array_equal(a.valid, v)
    assert np.max(np.abs(a.p[v] - c.p[v])) == 0.0


def test_axis_rule_validation():
    with pytest.raises(ValueError):
        AxisRule.noncoaxial([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        AxisRule.noncoaxial([0.0, 1.0], [0.0])


# ---------------------------------------------------------------------------
#  Rotational flow behind a curved shock
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", [AxisRule.planar(), AxisRule.coaxial()], ids=["planar", "coaxial"])
def test_rotational_flow_schemes_agree_and_converge(rule):
    """Curved shock (beta 22 -> 18 deg, smooth), so entropy varies across
    streamlines. The two schemes are independent discretisations and must
    agree in the interior. The streamline scheme's body pressure must
    converge at second order."""
    mids, bodies = {}, []
    for N in (50, 100, 200):
        x, y, beta = curved_shock(N, 22.0, 18.0)
        il = shock_initial_line(x, y, beta, 6.0)
        s_sol = solve_inverse(il, rule, scheme="streamline", strict=True)
        c_sol = solve_inverse(il, rule, scheme="characteristic", strict=True)
        body = s_sol.streamline_column(0)
        assert body.reached_end
        bodies.append(body)
        xs = np.linspace(0.3, 0.9 * body.x[-1], 25)
        yq = 0.5 * (np.interp(xs, body.x, body.y) + np.interp(xs, x, y))
        mids[N] = (s_sol.interpolate("p", xs, yq), c_sol.interpolate("p", xs, yq))
    s200, c200 = mids[200]
    assert np.max(np.abs(s200 / c200 - 1.0)) < 2e-5
    xg = np.linspace(0.26, 0.97 * min(b.x[-1] for b in bodies), 100)
    P = [np.interp(xg, b.x, b.p) for b in bodies]
    d1, d2 = np.max(np.abs(P[1] / P[0] - 1)), np.max(np.abs(P[2] / P[1] - 1))
    assert np.log2(d1 / d2) > 1.8


@pytest.mark.parametrize("scheme", SCHEMES)
def test_smooth_curved_shock_has_no_failures(scheme):
    x, y, beta = curved_shock(200, 22.0, 18.0)
    sol = solve_inverse(shock_initial_line(x, y, beta, 6.0), AxisRule.coaxial(),
                        scheme=scheme, strict=True)
    assert sol.first_failure() is None


# ---------------------------------------------------------------------------
#  Limit surfaces (spec flag 6)
# ---------------------------------------------------------------------------

def test_limit_surface_detected_by_both_schemes():
    """A shock that weakens 28 -> 14 deg over dx ~ 0.1 needs C+
    characteristics that cross a short distance below it (R4 Sec. 2)."""
    x, y, beta = curved_shock(200, 28.0, 14.0, ramp="steep")
    il = shock_initial_line(x, y, beta, 6.0)
    where = {}
    for scheme in SCHEMES:
        with pytest.raises(LimitSurfaceError):
            solve_inverse(il, AxisRule.planar(), scheme=scheme, strict=True)
        sol = solve_inverse(il, AxisRule.planar(), scheme=scheme)
        f = sol.first_failure()
        assert f is not None and f["code"] == CHARACTERISTIC_FOLD
        where[scheme] = f["x"]
    assert abs(where["streamline"] - where["characteristic"]) < 0.02
    assert 0.45 < where["streamline"] < 0.55


# ---------------------------------------------------------------------------
#  Input validation
# ---------------------------------------------------------------------------

def test_initial_line_must_be_space_like():
    M, b = 6.0, np.radians(20.0)
    x = np.linspace(X0, X1, 21)
    il = shock_initial_line(x, x * np.tan(b), b, M)
    # A line at 40 deg is steeper than the C+ direction theta + mu (25.8 deg).
    steep = InitialLine(il.x, 0.1 + il.x * np.tan(np.radians(40.0)), il.theta, il.p, il.rho, il.V)
    with pytest.raises(MOCError, match="space-like"):
        solve_inverse(steep, AxisRule.planar())
    # A line at -5 deg is shallower than the C- direction theta - mu (-1.0 deg).
    shallow = InitialLine(il.x, 0.5 - il.x * np.tan(np.radians(5.0)), il.theta, il.p, il.rho, il.V)
    with pytest.raises(MOCError, match="space-like"):
        solve_inverse(shallow, AxisRule.planar())


def test_shock_below_mach_angle_or_detached_is_refused():
    x = np.linspace(X0, X1, 11)
    with pytest.raises(MOCError, match="Mach angle"):
        shock_initial_line(x, x * np.tan(0.1), 0.1, 6.0)            # 5.7 deg < 9.6 deg
    with pytest.raises((SubsonicError, ValueError)):
        shock_initial_line(x, x * np.tan(1.2), 1.2, 6.0)            # 68.8 deg, subsonic behind


def test_initial_line_below_axis_is_refused():
    M, b = 6.0, np.radians(20.0)
    x = np.linspace(X0, X1, 11)
    il = shock_initial_line(x, x * np.tan(b), b, M)
    with pytest.raises(AxisCrossedError):
        solve_inverse(il, AxisRule.offset(1.0))
