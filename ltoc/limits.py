"""Input limits for an elliptic ruled shock (the LTOCs tab's custom shock).

An LTOCs station is refused when its shock curve gets too weak (at or below
the Mach angle) or too strong, or when the flow behind the shock folds (a
limit surface). Which part of the shock a waverider uses depends on the FCT:
the shock curves start on the leading edge and drift toward the sides as they
run downstream, past the base plane. This module traces those curves for an
``ltoc.design.EllipticShock`` (Step A of ``ltoc.ltoc``, in closed form for
the two ellipses, at ``N_CHECK_STATIONS`` stations) and checks the shock angle
beta along them.

The limits come from 54 builds: variants of R1's cases I and II, elliptic
cones and circular cone shocks at Mach 3 to 12 (gamma = 1.4; 9 stations,
30 points, some rechecked at 25 stations and 40 points), with beta along
curves traced CHECK_EXTENSION chords past the base:

* **Weak shock**: beta >= mu + 1.5 deg + 0.2 (beta_sym - beta), where
  beta_sym is the shock angle at the symmetry plane: a shock that weakens
  along its curves, as they drift toward a weaker part, needs more margin
  over the Mach angle mu. Refused designs had margins of 0.7 deg with no
  weakening, 1.8 deg with 9 deg, 3.1 deg with 12 deg and 5.8 deg with
  24 deg.
* **Strong shock**: post-shock Mach number M2 >= 1.5 (oblique-shock
  relations at the largest beta). Cone shocks were refused at M2 = 1.34 to
  1.43 and built at 1.49 and above: the flow keeps compressing toward the
  body and turns sonic there. This also keeps the shock attached.
* **Shape change**: beta no more than 25 deg below beta_sym. Shocks much
  steeper at the symmetry plane than where the curves drift fold the flow
  (refused at 26 to 31 deg).
* **Flow capture tube**: 0 < Z < b_q, so the FCT meets the base ellipse.

Every one of the 21 refused designs breaks a limit. Of the 33 that built,
5 are flagged, all close to a limit (within 1.1 deg of the weak limit, or
M2 = 1.49). They are rules of thumb: a new kind of shock can still be
refused inside them, so keep away from the ends of the ranges.

``shock_limits`` checks a design and gives, for each input, the range that
meets the limits with the other inputs fixed. ``refusal_advice`` turns the
refusals of a build into suggestions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["WEAK_MARGIN_DEG", "WEAK_MARGIN_PER_DROP", "MIN_POST_SHOCK_MACH",
           "MAX_SHAPE_CHANGE_DEG", "Issue", "ShockLimits", "post_shock_mach", "shock_limits",
           "refusal_advice"]

#: Smallest recommended margin of the shock angle over the Mach angle (deg),
#: plus WEAK_MARGIN_PER_DROP times the drop of the shock angle from its
#: symmetry-plane value.
WEAK_MARGIN_DEG = 1.5
WEAK_MARGIN_PER_DROP = 0.2
#: Smallest recommended post-shock Mach number at the largest shock angle.
MIN_POST_SHOCK_MACH = 1.5
#: Largest recommended drop of the shock angle from its symmetry-plane value (deg).
MAX_SHAPE_CHANGE_DEG = 25.0

#: Inputs whose recommended range ``shock_limits`` gives, with their labels.
RANGE_LABELS = {
    "a_q": "base half-width a_q",
    "b_q": "base half-height b_q",
    "x_q": "base X position x_q",
    "M_inf": "Mach number",
    "fct_z": "FCT height Z",
}


@dataclass
class Issue:
    """One input problem: ``severity`` "error" (cannot be built) or
    "warning" (stations will probably be refused), what is wrong, how to fix
    it, and the recommended ranges of the inputs that fix it (``ranges``,
    empty for errors)."""

    severity: str
    text: str
    fix: str
    ranges: str = ""


@dataclass
class ShockLimits:
    """Shock angles of an elliptic shock and its recommended input ranges.

    Angles are in degrees. ``beta_min`` and ``beta_max`` are over the traced
    shock curves, with where they occur in ``*_where``.
    ``ranges`` maps an input name (``RANGE_LABELS``) to the interval
    ``(lo, hi)`` that meets every limit with the other inputs fixed (``hi``
    may be inf; ``None`` if no value does).
    """

    M_inf: float
    mach_angle: float
    beta_low: float            # mu + the recommended margin (see _weak_margin)
    beta_high: float           # beta at which M2 = MIN_POST_SHOCK_MACH (nan if M is too low)
    beta_sym: float            # at the leading edge on the symmetry plane
    beta_min: float
    beta_max: float
    beta_min_where: str
    beta_max_where: str
    ranges: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def range_text(self, key: str, value: float, inline: bool = False) -> str:
        lo_hi = self.ranges.get(key)
        label = RANGE_LABELS[key]
        if lo_hi is None:
            return f"{label}: no value works with the other inputs"
        lo, hi = lo_hi
        if not np.isfinite(value) or lo <= value <= hi:
            mark = ""
        else:
            mark = f" (now {_fmt(value)})" if inline else f"  <- now {_fmt(value)}"
        return f"{label} {_fmt(lo)} to {_fmt(hi)}{mark}" if inline else \
            f"{label}: {_fmt(lo)} to {_fmt(hi)}{mark}"

    def text(self, values: Optional[dict] = None) -> str:
        """Plain-text summary for the GUI: shock angles, limits, ranges, issues."""
        lines = []
        if np.isfinite(self.beta_min):
            lines.append(f"Shock angle {self.beta_min:.1f}° ({self.beta_min_where}) to "
                         f"{self.beta_max:.1f}° ({self.beta_max_where}).")
        high = (f"{self.beta_high:.1f}° (post-shock Mach {MIN_POST_SHOCK_MACH:g})"
                if np.isfinite(self.beta_high) else "none: the Mach number is too low")
        margin = self.beta_low - self.mach_angle
        lines.append(f"At Mach {self.M_inf:g} keep it between {self.beta_low:.1f}° (Mach angle "
                     f"{self.mach_angle:.1f}° + {margin:.1f}°) and {high}, and no more than "
                     f"{MAX_SHAPE_CHANGE_DEG:g}° below its value at the symmetry plane.")
        if self.ranges:
            lines.append("Recommended, with the other inputs fixed:")
            vals = values or {}
            for key in RANGE_LABELS:
                if key in self.ranges:
                    lines.append("  " + self.range_text(key, vals.get(key, np.nan)))
        for i in self.issues:
            lines.append(("Error: " if i.severity == "error" else "Warning: ") + i.text
                         + " " + i.fix)
        if not self.issues:
            lines.append("Within the recommended limits.")
        return "\n".join(lines)


def _fmt(x: float) -> str:
    return "no limit" if not np.isfinite(x) else f"{x:.4g}"


# ---------------------------------------------------------------------------
#  Gas dynamics (oblique shock, vectorised)
# ---------------------------------------------------------------------------

def post_shock_mach(M, beta, gamma: float = 1.4):
    """Mach number behind an oblique shock of angle ``beta`` (rad); NaN at or
    below the Mach angle."""
    M, beta = np.broadcast_arrays(np.asarray(M, float), np.asarray(beta, float))
    mn1 = M * np.sin(beta)
    with np.errstate(invalid="ignore", divide="ignore"):
        mn2 = np.sqrt((1.0 + 0.5 * (gamma - 1.0) * mn1 ** 2) / (gamma * mn1 ** 2 - 0.5 * (gamma - 1.0)))
        tan_theta = (2.0 / np.tan(beta) * (mn1 ** 2 - 1.0)
                     / (M ** 2 * (gamma + np.cos(2.0 * beta)) + 2.0))
        m2 = mn2 / np.sin(beta - np.arctan(tan_theta))
    return np.where(mn1 > 1.0, m2, np.nan)


def _beta_high(M: float, gamma: float) -> float:
    """Shock angle (rad) at which the post-shock Mach number is MIN_POST_SHOCK_MACH."""
    if M <= MIN_POST_SHOCK_MACH:
        return np.nan
    from scipy.optimize import brentq

    mu = np.arcsin(1.0 / M)
    f = lambda b: float(post_shock_mach(M, b, gamma)) - MIN_POST_SHOCK_MACH  # noqa: E731
    return float(brentq(f, mu + 1e-9, 0.5 * np.pi - 1e-9, xtol=1e-12))


# ---------------------------------------------------------------------------
#  Shock curves of the elliptic ruled shock (vectorised over designs)
# ---------------------------------------------------------------------------

#: Stations and RK4 steps per shock curve of the check (the build uses more;
#: the shock angle along a curve is smooth).
N_CHECK_STATIONS = 8
N_CHECK_STEPS = 16
#: The check follows every shock curve this many chords past the base plane:
#: the most the build extends it (``solve_stream_surface(max_extension=6)``),
#: which a weak shock needs.
CHECK_EXTENSION = 6.0
_KEYS = ("x_p", "a_p", "b_p", "x_q", "a_q", "b_q")


def _rates(P, m, n):
    """sin(beta) and d(m, n)/dX along the shock curve at (m, n); ``P`` holds
    the six ellipse parameters (arrays broadcast with m and n). As
    ``shock_geometry._rhs`` for ``RuledShock.elliptic``."""
    x_p, a_p, b_p, x_q, a_q, b_q = P
    da, db, dx = a_q - a_p, b_q - b_p, x_q - x_p
    a, b = a_p + n * da, b_p + n * db
    c, s = np.cos(m), np.sin(m)
    # Su = (0, -a s, b c), Sv = (dx, da c, db s); N = Su x Sv
    nx = -(a * db * s * s + b * da * c * c)
    ny = b * c * dx
    nz = a * s * dx
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    sign = np.where(nx < 0.0, -1.0, 1.0) / norm               # toward the post-shock side
    nx, ny, nz = nx * sign, ny * sign, nz * sign
    cos2 = 1.0 - nx * nx
    wx, wy, wz = (1.0 - nx * nx) / cos2, -nx * ny / cos2, -nx * nz / cos2
    g11 = a * a * s * s + b * b * c * c
    g12 = s * c * (b * db - a * da)
    g22 = dx * dx + da * da * c * c + db * db * s * s
    r1 = -a * s * wy + b * c * wz
    r2 = dx * wx + da * c * wy + db * s * wz
    det = g11 * g22 - g12 * g12
    return nx, (g22 * r1 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det


def _leading_edge(P, y, z):
    """(m, n) of the shock point (Y, Z) = (y, z): bisection on n for
    (y/a)^2 + (z/b)^2 = 1 between the base plane and upstream."""
    x_p, a_p, b_p, x_q, a_q, b_q = P
    lo = np.full(np.broadcast(y, z, a_q).shape, -20.0)
    hi = np.ones_like(lo)

    def g(n):
        a, b = a_p + n * (a_q - a_p), b_p + n * (b_q - b_p)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (y / a) ** 2 + (z / b) ** 2 - 1.0
        return np.where((a > 0.0) & (b > 0.0), v, np.inf)

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        out = g(mid) > 0.0                    # outside the ellipse at n = mid: LE is downstream
        lo, hi = np.where(out, mid, lo), np.where(out, hi, mid)
    n = hi
    a, b = a_p + n * (a_q - a_p), b_p + n * (b_q - b_p)
    return np.arctan2(z / b, y / a), n


def _trace(p: dict):
    """Shock angle and m along the shock curves of ``N_CHECK_STATIONS`` stations.

    ``p`` maps the input names to arrays (broadcast together). Returns
    ``(beta, m, valid)``: beta and m of shape (..., stations, steps + 1) in
    radians, and whether the design is geometrically valid.
    """
    arrs = np.broadcast_arrays(*(np.asarray(p[k], float) for k in _KEYS + ("fct_z",)))
    P = tuple(v[..., None] for v in arrs[:6])
    z = arrs[6][..., None]
    x_p, a_p, b_p, x_q, a_q, b_q = P
    valid = ((x_q > x_p) & (a_q > 0.0) & (b_q > 0.0) & (z > 0.0) & (z < b_q))[..., 0]
    # LTOCDesign.fct clustering, up to the last station before the tip of a
    # 25-station design (the tip has no chord)
    s = np.linspace(0.0, 23.0 / 24.0, N_CHECK_STATIONS)
    frac = 1.0 - np.cos(0.5 * np.pi * s)
    with np.errstate(invalid="ignore", divide="ignore"):
        y_tip = a_q * np.sqrt(np.clip(1.0 - (z / b_q) ** 2, 0.0, None))
        m, n = _leading_edge(P, y_tip * frac, z)
        x0 = x_p + n * (x_q - x_p)
        chord = x_q - x0
        sinb, _, _ = _rates(P, m, n)
        h = (1.0 + CHECK_EXTENSION) * chord / N_CHECK_STEPS
        betas, ms = [np.arcsin(sinb)], [m]
        for _ in range(N_CHECK_STEPS):
            _, m1, n1 = _rates(P, m, n)
            _, m2, n2 = _rates(P, m + 0.5 * h * m1, n + 0.5 * h * n1)
            _, m3, n3 = _rates(P, m + 0.5 * h * m2, n + 0.5 * h * n2)
            _, m4, n4 = _rates(P, m + h * m3, n + h * n3)
            m = m + h / 6.0 * (m1 + 2.0 * m2 + 2.0 * m3 + m4)
            n = n + h / 6.0 * (n1 + 2.0 * n2 + 2.0 * n3 + n4)
            betas.append(np.arcsin(_rates(P, m, n)[0]))
            ms.append(m)
    return np.stack(betas, -1), np.stack(ms, -1), valid


def _measures(p: dict):
    """beta_min, beta_max (rad) over the shock curves, m at each, the
    symmetry-plane shock angle, and validity; arrays over the designs."""
    beta, m, valid = _trace(p)
    bad = ~np.isfinite(beta)
    flat_b = np.where(bad, np.inf, beta).reshape(beta.shape[:-2] + (-1,))
    flat_m = m.reshape(m.shape[:-2] + (-1,))
    i_min = np.argmin(flat_b, -1)
    i_max = np.argmax(np.where(bad, -np.inf, beta).reshape(flat_b.shape), -1)
    pick = lambda a, i: np.take_along_axis(a, i[..., None], -1)[..., 0]  # noqa: E731
    bmin = np.where(bad.reshape(flat_b.shape).any(-1), -np.inf, pick(flat_b, i_min))
    bmax = pick(beta.reshape(flat_b.shape), i_max)
    return dict(beta_min=bmin, beta_max=bmax, m_min=pick(flat_m, i_min),
                m_max=pick(flat_m, i_max), beta_sym=beta[..., 0, 0], valid=valid)


def _where(m: float) -> str:
    if m < np.radians(10.0):
        return "near the side"
    if m > np.radians(80.0):
        return "near the symmetry plane"
    return "between the symmetry plane and the side"


def _weak_margin(drop):
    """Recommended margin (rad) over the Mach angle for a shock angle that
    falls by ``drop`` (rad) from its symmetry-plane value."""
    return np.radians(WEAK_MARGIN_DEG) + WEAK_MARGIN_PER_DROP * np.maximum(drop, 0.0)


def _passes(p: dict, gamma: float):
    """Whether the inputs ``p`` (arrays broadcast together) meet every limit."""
    q = _measures(p)
    M = np.asarray(p["M_inf"], float)
    mu = np.arcsin(1.0 / np.maximum(M, 1.0 + 1e-12))
    with np.errstate(invalid="ignore"):
        ok = q["valid"] & (q["beta_min"] >= mu + _weak_margin(q["beta_sym"] - q["beta_min"]))
        ok &= post_shock_mach(M, q["beta_max"], gamma) >= MIN_POST_SHOCK_MACH
        ok &= np.degrees(q["beta_sym"] - q["beta_min"]) <= MAX_SHAPE_CHANGE_DEG
    return ok


def _interval(values: np.ndarray, ok: np.ndarray, current: float):
    """Indices (first, last) of the run of passing values that contains
    ``current``, else of the nearest run; None if no value passes."""
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    def dist(r):
        lo, hi = values[r[0]], values[r[-1]]
        return 0.0 if lo <= current <= hi else min(abs(current - lo), abs(current - hi))
    run = min(runs, key=dist)
    return int(run[0]), int(run[-1])


def _scan_values(key: str, p: dict) -> np.ndarray:
    """Candidate values for one input, around the current design."""
    size = max(p["a_q"], p["b_q"], 1e-6)
    length = max(p["x_q"] - p["x_p"], 1e-6)
    if key == "M_inf":
        return np.geomspace(1.05, 40.0, 120)
    if key == "fct_z":
        return np.linspace(0.0, p["b_q"], 122)[1:-1]
    if key == "x_q":
        return p["x_p"] + length * np.geomspace(0.02, 50.0, 120)
    lo = p["a_p"] if key == "a_q" else p["b_p"]
    return lo + size * np.geomspace(1e-3, 30.0, 120)


def _passes_batch(p: dict, gamma: float, blocks) -> list:
    """``_passes`` for several one-input scans in one call: ``blocks`` is a
    list of (key, values); returns one boolean array per block."""
    sizes = [len(v) for _, v in blocks]
    cols = {k: np.repeat(float(p[k]), sum(sizes)) for k in _KEYS + ("M_inf", "fct_z")}
    start = 0
    for (key, vals), size in zip(blocks, sizes):
        cols[key][start:start + size] = vals
        start += size
    return np.split(_passes(cols, gamma), np.cumsum(sizes)[:-1])


def _ranges(p: dict, gamma: float) -> dict:
    """Each input's passing interval around its current value, with the
    others fixed: a coarse scan, then two finer scans of each end."""
    keys = list(RANGE_LABELS)
    scans = [_scan_values(k, p) for k in keys]
    out, bounds = {}, []                      # bounds: [key, side, good, bad]
    for key, vals, ok in zip(keys, scans, _passes_batch(p, gamma, list(zip(keys, scans)))):
        run = _interval(vals, ok, p[key])
        if run is None:
            out[key] = None
            continue
        i_lo, i_hi = run
        out[key] = [float(vals[i_lo]), float(vals[i_hi])]
        if i_lo > 0:
            bounds.append([key, 0, float(vals[i_lo]), float(vals[i_lo - 1])])
        elif key == "fct_z":
            out[key][0] = 0.0                  # the open bound 0 < Z
        if i_hi + 1 < vals.size:
            bounds.append([key, 1, float(vals[i_hi]), float(vals[i_hi + 1])])
        elif key == "fct_z":
            out[key][1] = p["b_q"]             # the open bound Z < b_q
        elif key == "M_inf":
            out[key][1] = np.inf               # M2 tends to a limit as M grows
    for _ in range(2):
        if not bounds:
            break
        grids = [np.linspace(good, bad, 17) for _, _, good, bad in bounds]
        for b, vals, ok in zip(bounds, grids,
                               _passes_batch(p, gamma, [(b[0], g) for b, g in zip(bounds, grids)])):
            k = int(np.argmin(ok)) if not ok.all() else vals.size - 1    # first failing value
            b[2], b[3] = float(vals[max(k - 1, 0)]), float(vals[k])
    for key, side, good, _ in bounds:
        out[key][side] = good
    return {k: (tuple(v) if v is not None else None) for k, v in out.items()}


#: Inputs that move the shock angle where it is too weak or too strong.
_WEAK_KEYS = {"near the side": ("a_q", "fct_z"), "near the symmetry plane": ("b_q",)}
_STRONG_KEYS = {"near the side": ("a_q",), "near the symmetry plane": ("b_q",)}


def shock_limits(shock, M_inf: float, fct_z: float, gamma: float = 1.4) -> ShockLimits:
    """Check an ``EllipticShock`` at ``M_inf`` with the FCT at ``fct_z``."""
    p = dict(x_p=shock.x_p, a_p=shock.a_p, b_p=shock.b_p, x_q=shock.x_q, a_q=shock.a_q,
             b_q=shock.b_q, M_inf=float(M_inf), fct_z=float(fct_z))
    issues = []
    M = float(M_inf)
    mu = np.degrees(np.arcsin(1.0 / M)) if M > 1.0 else 90.0
    beta_low = mu + WEAK_MARGIN_DEG
    beta_high = np.degrees(_beta_high(M, gamma)) if M > 1.0 else np.nan
    geometry_ok = True
    if shock.x_q <= shock.x_p:
        geometry_ok = False
        issues.append(Issue("error", f"The base ellipse (X = {shock.x_q:g}) must lie downstream of "
                                     f"the upstream ellipse (X = {shock.x_p:g}).",
                            "Make the base X position larger than the upstream X position."))
    if min(shock.a_q, shock.b_q) <= 0.0:
        geometry_ok = False
        issues.append(Issue("error", "The base ellipse has a zero half-axis.",
                            "Give it a positive half-width and half-height."))
    if geometry_ok and not 0.0 < fct_z < shock.b_q:
        issues.append(Issue("error", f"The FCT height Z = {fct_z:g} must lie between 0 and the base "
                                     f"half-height b_q = {shock.b_q:g}: the FCT must meet the base "
                                     f"ellipse.",
                            f"Try Z = {0.3 * shock.b_q:.4g} (0.3 b_q)."))
    if not geometry_ok:
        return ShockLimits(M, mu, beta_low, beta_high, np.nan, np.nan, np.nan, "", "",
                           issues=issues)

    z_eval = fct_z if 0.0 < fct_z < shock.b_q else 0.3 * shock.b_q
    q = {k: float(v) for k, v in _measures(dict(p, fct_z=z_eval)).items()}
    if np.isfinite(q["beta_min"]):
        beta_low = mu + np.degrees(_weak_margin(q["beta_sym"] - q["beta_min"]))
    ranges = _ranges(dict(p, fct_z=z_eval), gamma)
    lim = ShockLimits(M, mu, beta_low, beta_high, np.degrees(q["beta_sym"]),
                      np.degrees(q["beta_min"]), np.degrees(q["beta_max"]),
                      _where(q["m_min"]), _where(q["m_max"]), ranges, issues)

    def fix(*keys) -> str:
        parts = [lim.range_text(k, p[k], inline=True) for k in dict.fromkeys(keys) if ranges.get(k)]
        if not parts:
            return "No single input fixes it: change several together."
        return "Recommended: " + "; ".join(parts) + "."

    if not np.isfinite(lim.beta_min):
        issues.append(Issue(
            "warning", "The shock is degenerate where the waverider uses it (an ellipse shrinks "
                       "to zero size, or the shock turns parallel to the flow).",
            "Make both base half-axes larger than the upstream ones, or raise the FCT.",
            fix("a_q", "b_q", "fct_z")))
        return lim
    if lim.beta_min < beta_low:
        what = ("at or below" if lim.beta_min <= mu
                else f"only {lim.beta_min - mu:.1f}° above")
        keys = _WEAK_KEYS.get(lim.beta_min_where, ("a_q", "b_q", "fct_z")) + ("x_q", "M_inf")
        issues.append(Issue(
            "warning", f"The shock is too weak {lim.beta_min_where}: {lim.beta_min:.1f}°, {what} "
                       f"the Mach angle ({mu:.1f}°); keep it at least {beta_low - mu:.1f}° above. "
                       f"Stations will probably be refused (sub-Mach, limit surface or "
                       f"undetermined).",
            "Make the shock steeper there (a larger base ellipse or a shorter shock), raise the "
            "FCT (the waverider then stays away from the sides), or fly faster.", fix(*keys)))
    m2 = float(post_shock_mach(M, np.radians(lim.beta_max), gamma))   # NaN if all below mu
    if m2 < MIN_POST_SHOCK_MACH:
        keys = _STRONG_KEYS.get(lim.beta_max_where, ("a_q", "b_q")) + ("x_q", "M_inf")
        issues.append(Issue(
            "warning", f"The shock is too strong {lim.beta_max_where}: {lim.beta_max:.1f}°, "
                       f"post-shock Mach {m2:.2f} (keep it at {MIN_POST_SHOCK_MACH:g} or more; the "
                       f"shock detaches at {_detach_deg(M, gamma):.1f}°). The flow turns sonic "
                       f"near the body and stations will probably be refused (limit surface).",
            "Make the shock less steep there (a smaller base ellipse or a longer shock), or fly "
            "slower.", fix(*keys)))
    drop = lim.beta_sym - lim.beta_min
    if drop > MAX_SHAPE_CHANGE_DEG:
        issues.append(Issue(
            "warning", f"The shock is {drop:.1f}° weaker {lim.beta_min_where} "
                       f"({lim.beta_min:.1f}°) than at the symmetry plane ({lim.beta_sym:.1f}°); "
                       f"beyond about {MAX_SHAPE_CHANGE_DEG:g}° the flow folds (limit surface).",
            "Make the two ellipses closer in shape (a wider or less tall base ellipse), or raise "
            "the FCT.", fix("a_q", "b_q", "fct_z")))
    return lim


def _detach_deg(M: float, gamma: float) -> float:
    from gvwd.thermo.oblique_shock import detachment_beta

    return float(np.degrees(detachment_beta(M, gamma)))


# ---------------------------------------------------------------------------
#  Advice after a build
# ---------------------------------------------------------------------------

_STATUS_TEXT = {
    "sub_mach": "the shock is too weak (at or below the Mach angle)",
    "detached": "the shock is too strong (beyond detachment)",
    "concave": "the shock curves away from the body across the span",
    "limit_surface": "the flow behind the shock folds (limit surface)",
    "moc_failure": "the flow solution failed",
    "moc_error": "the flow solution failed",
    "undetermined": "the body is not determined up to the base plane",
    "outside_shock": "the shock data end before the base plane",
}


def _span_part(fractions) -> str:
    parts = []
    f = np.asarray(fractions, float)
    if (f < 1.0 / 3.0).any():
        parts.append("near the symmetry plane")
    if ((f >= 1.0 / 3.0) & (f <= 2.0 / 3.0)).any():
        parts.append("at mid-span")
    if (f > 2.0 / 3.0).any():
        parts.append("near the tip")
    return ", ".join(parts)


def refusal_advice(result) -> list:
    """Suggestions (lines of text) for the refused stations of an ``LTOCResult``."""
    if result.ok or not result.refusals:
        return []
    design = result.design
    stations = result.waverider.stations
    y = np.array([abs(float(st.le_point[1])) for st in stations])
    y_tip = max(float(y.max()), 1e-300)
    by_status = {}
    for k, status, _ in result.refusals:
        by_status.setdefault(status, []).append(k)
    lines = []
    for status, ks in by_status.items():
        where = _span_part(y[ks] / y_tip)
        n = f"{len(ks)} station" + ("s" if len(ks) > 1 else "")
        lines.append(f"{n} {where}: {_STATUS_TEXT.get(status, status)}.")

    elliptic = (design.oc_waverider is None and design.shock_data is None
                and design.shock is not None)
    if elliptic:
        lim = shock_limits(design.shock, design.M_inf, design.fct_z, design.gamma)
        for i in lim.issues:
            lines.append("- " + i.text)
            lines.append("  " + i.fix + (" " + i.ranges if i.ranges else ""))
        if not lim.issues:
            lines.append("- The inputs are within the recommended limits, which are rules of "
                         "thumb. Move further inside the ranges in the 'Shock limits' box: "
                         "make the two ellipses closer in shape or raise the FCT.")
        return lines

    statuses = set(by_status)
    if statuses & {"outside_shock", "undetermined"} and design.shock_data is not None:
        lines.append("- Extend the shock data further past the base plane (typically 0.5 to "
                     "1.5 local chords): the body at the base depends on the shock "
                     "downstream of it.")
    if statuses & {"sub_mach", "limit_surface", "moc_failure", "moc_error", "undetermined"}:
        lines.append(f"- Keep the shock angle at least {WEAK_MARGIN_DEG:g}° above the Mach angle "
                     f"(more where it weakens across the span) and the post-shock Mach number "
                     f"at {MIN_POST_SHOCK_MACH:g} or more everywhere on the shock, and avoid "
                     f"strong changes of shape across the span.")
    if "concave" in statuses:
        lines.append("- The shock cross-section must be convex (curving toward the body) "
                     "across the span.")
    if "detached" in statuses:
        lines.append("- Make the shock less steep, or lower the Mach number.")
    return lines
