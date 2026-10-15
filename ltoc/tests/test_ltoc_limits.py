"""Input limits of the elliptic ruled shock (ltoc.limits) and the advice after a build."""
import numpy as np
import pytest

from ltoc.design import PRESETS, EllipticShock, LTOCDesign
from ltoc.limits import (MAX_SHAPE_CHANGE_DEG, MIN_POST_SHOCK_MACH, _measures, _rates, _trace,
                         post_shock_mach, refusal_advice, shock_limits)

CASE_II = EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 0.8)          # R1 waverider case II, M 7, Z 0.25


def _params(shock, M, z):
    return dict(x_p=shock.x_p, a_p=shock.a_p, b_p=shock.b_p, x_q=shock.x_q, a_q=shock.a_q,
                b_q=shock.b_q, M_inf=M, fct_z=z)


def test_closed_form_rates_match_the_shock_geometry():
    from ltoc.shock_geometry import _rhs, local_geometry

    rng = np.random.default_rng(0)
    m, n = rng.uniform(0.05, 1.55, 40), rng.uniform(-0.2, 2.4, 40)
    surface = CASE_II.surface()
    sinb, dm, dn = _rates(tuple(np.float64(v) for v in (0.1, 0.3, 0.2, 1.5, 0.65, 0.8)), m, n)
    du, dv, _ = _rhs(surface, m, n)
    assert np.allclose(np.arcsin(sinb), local_geometry(surface, m, n).beta, atol=1e-12)
    assert np.allclose(dm, du, atol=1e-12) and np.allclose(dn, dv, atol=1e-12)


def test_traced_shock_angles_match_the_solver_shock_curves():
    """The check traces the same shock curves as Step A of the build."""
    from ltoc.ltoc import find_leading_edge
    from ltoc.shock_geometry import trace_shock_curves

    beta, m, valid = _trace(_params(CASE_II, 7.0, 0.25))
    assert bool(valid)
    d = LTOCDesign(shock=CASE_II, M_inf=7.0, fct_z=0.25, n_stations=9)
    surface = CASE_II.surface()
    uv, P, ok = find_leading_edge(surface, d.fct()[:-1])
    assert ok.all()
    # the first station of both is the symmetry plane: same leading edge, same angle there
    assert m[0, 0] == pytest.approx(uv[0, 0], abs=1e-9)
    curves = trace_shock_curves(surface, uv[:, 0], uv[:, 1], 1.5 + 1.0, n_steps=200, M_inf=7.0)
    b_solver = min(np.degrees(c.geometry.beta).min() for c in curves)
    assert np.degrees(beta.min()) == pytest.approx(b_solver, abs=1.0)
    assert np.degrees(beta.max()) == pytest.approx(
        max(np.degrees(c.geometry.beta).max() for c in curves), abs=0.05)


def test_post_shock_mach_matches_the_oblique_shock_relations():
    from gvwd.thermo.oblique_shock import rankine_hugoniot

    for M, beta in ((3.0, 30.0), (7.0, 20.0), (10.0, 55.0)):
        b = np.radians(beta)
        assert float(post_shock_mach(M, b)) == pytest.approx(rankine_hugoniot(M, b, 1.0, 1.0)["M2"],
                                                             rel=1e-10)
    assert np.isnan(post_shock_mach(5.0, np.arcsin(1.0 / 5.0) * 0.9))


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_within_the_limits(name):
    shock, M, z, _ = PRESETS[name]
    lim = shock_limits(shock, M, z)
    assert lim.ok, lim.text()
    assert "Within the recommended limits" in lim.text()
    for key, value in dict(a_q=shock.a_q, b_q=shock.b_q, x_q=shock.x_q, M_inf=M, fct_z=z).items():
        lo, hi = lim.ranges[key]
        assert lo <= value <= hi, key


def test_a_weak_shock_at_the_sides_gets_ranges_that_fix_it():
    shock = EllipticShock(0.1, 0.3, 0.2, 1.5, 0.55, 0.8)       # refused: limit surface, undetermined
    lim = shock_limits(shock, 7.0, 0.25)
    assert [i.severity for i in lim.issues] == ["warning"]
    assert "too weak" in lim.issues[0].text and "base half-width a_q" in lim.issues[0].ranges
    lo, hi = lim.ranges["a_q"]
    assert 0.55 < lo < 0.65 < hi                               # case II's 0.65 is inside
    assert shock_limits(EllipticShock(0.1, 0.3, 0.2, 1.5, lo * 1.01, 0.8), 7.0, 0.25).ok
    assert not shock_limits(EllipticShock(0.1, 0.3, 0.2, 1.5, lo * 0.98, 0.8), 7.0, 0.25).ok
    # a faster flight or a higher FCT (away from the weak sides) also works
    assert lim.ranges["M_inf"][0] > 7.0 and lim.ranges["M_inf"][1] == np.inf
    assert shock_limits(shock, lim.ranges["M_inf"][0] * 1.02, 0.25).ok
    assert shock_limits(shock, 7.0, 0.6).ok                     # builds (scan in ltoc/limits.py)


def test_a_strong_shock_is_flagged_by_the_post_shock_mach_number():
    r = 1.5 * np.tan(np.radians(59.0))                          # cone shock: refused at Mach 7
    lim = shock_limits(EllipticShock(0.0, 0.0, 0.0, 1.5, r, r), 7.0, 0.4 * r)
    assert len(lim.issues) == 1 and "too strong" in lim.issues[0].text
    assert float(post_shock_mach(7.0, np.radians(lim.beta_max))) < MIN_POST_SHOCK_MACH
    assert lim.beta_high == pytest.approx(55.7, abs=0.1)
    lo, hi = lim.ranges["x_q"]                                  # a longer shock is weaker
    assert lo > 1.5


def test_a_shock_much_steeper_at_the_centre_is_flagged():
    shock = EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 1.6)       # refused: limit surface
    lim = shock_limits(shock, 7.0, 0.25)
    assert any("weaker near the side" in i.text for i in lim.issues)
    assert lim.beta_sym - lim.beta_min > MAX_SHAPE_CHANGE_DEG
    assert lim.ranges["b_q"][1] < 1.6 and lim.ranges["fct_z"][0] > 0.25
    # b_q = 1.3 is refused with Z = 0.25 and builds with Z = 1.0 (scan in ltoc/limits.py)
    assert not shock_limits(EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 1.3), 7.0, 0.25).ok
    assert shock_limits(EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 1.3), 7.0, 1.0).ok


def test_geometry_errors_come_with_a_fix():
    lim = shock_limits(CASE_II, 7.0, 0.9)
    assert lim.has_errors and "Try Z = 0.24" in lim.issues[0].fix and not lim.issues[0].ranges
    assert lim.ranges["fct_z"] == (0.0, 0.8)
    lim = shock_limits(EllipticShock(1.6, 0.3, 0.2, 1.5, 0.65, 0.8), 7.0, 0.25)
    assert lim.has_errors and not lim.ranges and "downstream" in lim.issues[0].text
    assert np.isnan(lim.beta_min) and "Error:" in lim.text()


def test_the_check_is_vectorised_over_designs():
    p = _params(CASE_II, 7.0, 0.25)
    p["a_q"] = np.array([0.4, 0.65, 2.0])
    q = _measures(p)
    assert q["beta_min"].shape == (3,) and np.all(np.diff(q["beta_min"]) > 0)


def test_refusal_advice_names_the_refusal_and_what_to_change():
    d = LTOCDesign(shock=CASE_II, M_inf=3.0, fct_z=0.25, n_stations=5, n_points=20)
    res = d.build()
    assert not res.ok
    advice = refusal_advice(res)
    assert any("too weak (at or below the Mach angle)" in a for a in advice)
    k = next(i for i, a in enumerate(advice) if a.startswith("- The shock is too weak"))
    assert "fly faster" in advice[k + 1] and "Mach number" in advice[k + 1]
    ok = LTOCDesign.from_preset("Elliptic cone shock, Mach 6 (published case I)",
                                n_stations=5, n_points=20).build()
    assert ok.ok and refusal_advice(ok) == []


def test_a_shock_below_the_mach_angle_is_only_too_weak():
    tb = np.tan(np.radians(12.0))                               # 12 deg cone at Mach 4 (mu 14.5 deg)
    lim = shock_limits(EllipticShock(0.0, 0.0, 0.0, 1.0, tb, tb), 4.0, 0.1)
    assert len(lim.issues) == 1
    assert "too weak" in lim.issues[0].text and "at or below the Mach angle" in lim.issues[0].text
