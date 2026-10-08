"""Two-family rotational inverse method of characteristics with a pluggable
axis rule, marching inward from a prescribed shock (LTOCs spec, Phase 1).

Meridian-plane frame
--------------------
``x`` runs along the freestream (+x downstream) and ``y`` is transverse. The
post-shock region lies on the right-hand side of the initial line walked in
+x: for a shock rising with x, that is below it. For axisymmetric flow the
source-term radius is ``r = y - y_axis(x)``:

* ``AxisRule.coaxial()`` -- ``y_axis = 0``, ordinary axisymmetric MOC;
* ``AxisRule.noncoaxial()`` -- ``y_axis(x)`` interpolated from the local axis
  centres of the shock points. An interior point takes the centre of the
  shock point at the same axial position (Zheng et al. 2020, R1 Sec. II.B;
  Zheng et al. 2020, R2 Sec. III.B and Fig. 15b);
* ``AxisRule.planar()`` -- ``delta = 0`` (R1 Eq. 9).

Governing relations
-------------------
Zucrow & Hoffman (1976, Vol. 2, pp. 187-192), restated in R1 Eqs. 4-8.

Along streamlines, ``dy/dx = tan(theta)`` (R1 Eq. 7)::

    rho V dV + dP = 0                                    (R1 Eq. 4)
    dP - c^2 d(rho) = 0                                  (R1 Eq. 5)

For a perfect gas these integrate exactly to the streamline invariants
``K = P / rho^gamma`` and ``H = gamma P / ((gamma - 1) rho) + V^2 / 2``. Both
are carried along streamlines, so the flow may be rotational.

Along Mach lines, ``dy/dx = tan(theta +/- mu)`` (R1 Eq. 8)::

    sqrt(M^2 - 1) / (rho V^2) dP +/- d(theta)
        + delta sin(theta) dx / (r M cos(theta +/- mu)) = 0     (R1 Eq. 6)

Two marching schemes (spec flag 2)
----------------------------------
R1 and R2 locate a new point with one streamline and one Mach line, which is
one relation short. Both schemes below use the C+ *and* the C- compatibility
relation. They differ in the mesh.

``scheme="characteristic"``
    The standard characteristic net (Zucrow & Hoffman; Qian & Sobieczky 2002,
    R4 Fig. 1). New point P = C- through A ∩ C+ through B, where A and B are
    adjacent points of the previous row. K and H are interpolated where the
    streamline through P, traced back, crosses the previous row. Behind a
    curved shock that crossing is usually upstream of A, so the row is
    searched. Points whose crossing lies upstream of the first point of the
    previous row, beyond ``extrapolation_limit`` segment lengths, have no
    determined entropy and are marked ``UNDETERMINED``.

``scheme="streamline"``
    R1's own mesh topology (R1 Fig. 3; R2 Fig. 15b), with the missing
    relation added. Point ``(d, j)`` is the d-th point on the streamline from
    shock point ``A_j``. It is the intersection of that streamline, continued
    from ``(d-1, j)``, and the C+ characteristic through ``(d-1, j+1)``. The
    C- relation comes from the foot of the backward C- characteristic on the
    known cell boundary. K and H are exact (the values at ``A_j``), and
    column 0 is the streamline from the first shock point. This is the
    waverider body in LTOCs.

``scheme="streamline"`` is the default (Gate 1 decision, 2026-10-08).

Both schemes use the modified-Euler average-property predictor-corrector of
Zucrow & Hoffman. Both store the solution in the same triangular layout:
row ``d`` has ``n - d`` points, where ``n`` is the number of initial-line
points.

Variables are nondimensional: ``p / p_inf``, ``rho / rho_inf`` and
``V / sqrt(p_inf / rho_inf)``. With these, ``c^2 = gamma p / rho``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = [
    "AxisRule",
    "InitialLine",
    "MOCSolution",
    "StreamlineResult",
    "MOCError",
    "LimitSurfaceError",
    "SubsonicError",
    "AxisCrossedError",
    "shock_initial_line",
    "solve_inverse",
    "SCHEMES",
    "VALID",
    "PARENT_INVALID",
    "SUBSONIC",
    "AXIS_CROSSED",
    "CHARACTERISTIC_FOLD",
    "NOT_CONVERGED",
    "UNDETERMINED",
    "STATUS_TEXT",
]

SCHEMES = ("characteristic", "streamline")

# Point status codes.
VALID = 0
PARENT_INVALID = 1
SUBSONIC = 2
AXIS_CROSSED = 3
CHARACTERISTIC_FOLD = 4
NOT_CONVERGED = 5
UNDETERMINED = 6

STATUS_TEXT = {
    VALID: "valid",
    PARENT_INVALID: "depends on an invalid point",
    SUBSONIC: "M <= 1 (MOC not applicable)",
    AXIS_CROSSED: "r <= 0 (point at or beyond the local axis)",
    CHARACTERISTIC_FOLD: "characteristics of the same family crossed (limit surface)",
    NOT_CONVERGED: "predictor-corrector did not converge",
    UNDETERMINED: "streamline originates upstream of the initial data (entropy undetermined)",
}


class MOCError(ValueError):
    """Base class for MOC kernel failures."""


class LimitSurfaceError(MOCError):
    """Characteristics of the same family crossed (R4 Sec. 2, spec flag 6)."""


class SubsonicError(MOCError):
    """The local Mach number dropped to 1 or below."""


class AxisCrossedError(MOCError):
    """A point reached the local axis (r <= 0) in axisymmetric mode."""


_STRICT_ERRORS = {
    SUBSONIC: SubsonicError,
    AXIS_CROSSED: AxisCrossedError,
    CHARACTERISTIC_FOLD: LimitSurfaceError,
    NOT_CONVERGED: MOCError,
}


# ---------------------------------------------------------------------------
#  Axis rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisRule:
    """Radius used in the axisymmetric source term of R1 Eq. (6).

    ``delta = 0`` is planar flow. ``delta = 1`` uses ``r = y - y_axis(x)``.
    """

    delta: int
    y_axis: Optional[Callable[[np.ndarray], np.ndarray]] = None
    name: str = ""

    @classmethod
    def planar(cls) -> "AxisRule":
        """Two-dimensional flow, ``delta = 0`` (R1 Eq. 9)."""
        return cls(0, None, "planar")

    @classmethod
    def coaxial(cls) -> "AxisRule":
        """Ordinary axisymmetric flow about ``y = 0`` (``r = y``)."""
        return cls(1, lambda x: np.zeros_like(np.asarray(x, dtype=float)), "coaxial")

    @classmethod
    def offset(cls, y0: float) -> "AxisRule":
        """Coaxial rule about the straight axis ``y = y0``."""
        return cls(1, lambda x: np.full_like(np.asarray(x, dtype=float), y0), f"offset({y0:g})")

    @classmethod
    def noncoaxial(cls, x_centre, y_centre) -> "AxisRule":
        """Noncoaxial rule of R1 Sec. II.B (R2 Fig. 15b).

        ``(x_centre, y_centre)`` are the meridian-plane axis centres of the
        shock points; a centre lies at the axial station of its shock point.
        A point at axial position ``x`` uses the centre interpolated linearly
        at ``x``, i.e. "the local axis centre of the shock point at the same
        axial position" (R1).
        """
        xc = np.asarray(x_centre, dtype=float)
        yc = np.asarray(y_centre, dtype=float)
        if xc.ndim != 1 or xc.shape != yc.shape or xc.size < 2:
            raise ValueError("x_centre and y_centre must be 1-D arrays of equal length >= 2")
        if np.any(np.diff(xc) <= 0.0):
            raise ValueError("x_centre must be strictly increasing")
        return cls(1, lambda x: np.interp(x, xc, yc), "noncoaxial")

    def radius(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.delta == 0:
            return np.full(np.broadcast(x, y).shape, np.inf)
        return y - self.y_axis(x)


# ---------------------------------------------------------------------------
#  Initial data
# ---------------------------------------------------------------------------

@dataclass
class InitialLine:
    """State on a non-characteristic initial-value line (normally the shock)."""

    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    p: np.ndarray
    rho: np.ndarray
    V: np.ndarray

    def __post_init__(self):
        for name in ("x", "y", "theta", "p", "rho", "V"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float).ravel())
        n = self.x.size
        if n < 2:
            raise ValueError("an initial line needs at least two points")
        if any(getattr(self, k).size != n for k in ("y", "theta", "p", "rho", "V")):
            raise ValueError("initial-line arrays must have equal length")
        if np.any(np.diff(self.x) <= 0.0):
            raise ValueError("initial-line x must be strictly increasing")
        if np.any(self.p <= 0.0) or np.any(self.rho <= 0.0) or np.any(self.V <= 0.0):
            raise ValueError("p, rho and V must be positive on the initial line")


def shock_initial_line(x, y, beta, M_inf: float, gamma: float = 1.4) -> InitialLine:
    """Post-shock state on a prescribed shock in uniform flow along +x.

    ``beta`` is the local shock angle (rad), one value per point; R1 Eq. (3)
    defines it from the shock normal. The post-shock state comes from the
    oblique-shock Rankine-Hugoniot relations, as R1 Sec. II.A prescribes
    without an equation number; they are in NACA Report 1135 (Ames Research
    Staff, 1953). The relations are implemented in
    ``gvwd.thermo.oblique_shock.rankine_hugoniot``. The flow is deflected by
    ``+theta(beta, M_inf)`` toward the shock side, which is the +y side for a
    shock rising with x.
    """
    from gvwd.thermo.oblique_shock import rankine_hugoniot

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    beta = np.broadcast_to(np.asarray(beta, dtype=float), x.shape)
    theta = np.empty_like(x)
    p = np.empty_like(x)
    rho = np.empty_like(x)
    V = np.empty_like(x)
    mu_inf = np.arcsin(1.0 / M_inf)
    for i, b in enumerate(beta):
        if not (mu_inf < b < 0.5 * np.pi):
            raise MOCError(
                f"shock angle {np.degrees(b):.3f} deg at x = {x[i]:.6g} is not above the "
                f"freestream Mach angle {np.degrees(mu_inf):.3f} deg")
        rh = rankine_hugoniot(M_inf, float(b), 1.0, 1.0, gamma)
        if rh["M2"] <= 1.0:
            raise SubsonicError(
                f"subsonic post-shock flow (M2 = {rh['M2']:.4f}) at x = {x[i]:.6g}; "
                "the shock angle is beyond the sonic point")
        theta[i] = rh["theta"]
        p[i] = rh["p_ratio"]
        rho[i] = rh["rho_ratio"]
        V[i] = rh["M2"] * np.sqrt(gamma * p[i] / rho[i])
    return InitialLine(x, y, theta, p, rho, V)


# ---------------------------------------------------------------------------
#  Gas relations and characteristic coefficients
# ---------------------------------------------------------------------------

def _invariants(p, rho, V, gamma):
    """Streamline invariants K = p / rho^gamma and H (R1 Eqs. 4-5 integrated)."""
    K = p / rho ** gamma
    H = gamma * p / ((gamma - 1.0) * rho) + 0.5 * V * V
    return K, H


def _state_from_pKH(p, K, H, gamma):
    """rho, V, M from pressure and the streamline invariants (NaN if invalid)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (p / K) ** (1.0 / gamma)
        c2 = gamma * p / rho
        V2 = 2.0 * (H - c2 / (gamma - 1.0))
        V = np.sqrt(np.where(V2 > 0.0, V2, np.nan))
        M = V / np.sqrt(c2)
    return rho, V, M


def _char_coeffs(x, y, theta, p, rho, V, sign, rule, gamma):
    """Slope, Q and source coefficient S of R1 Eq. (6), C+ (sign=+1) or C- (-1)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        M = V / np.sqrt(gamma * p / rho)
        mu = np.arcsin(1.0 / M)
        ang = theta + sign * mu
        lam = np.tan(ang)
        Q = np.sqrt(M * M - 1.0) / (rho * V * V)
        if rule.delta:
            S = np.sin(theta) / (rule.radius(x, y) * M * np.cos(ang))
        else:
            S = np.zeros_like(theta)
    return lam, Q, S


def _avg(s1, s2):
    return tuple(0.5 * (a + b) for a, b in zip(s1, s2))


def _segment_param(x0, y0, x1, y1, xP, yP, lam):
    """Parameter t of the intersection of segment (0 -> 1) with the line of
    slope ``lam`` through P; ``t = 0`` at end 0, ``t = 1`` at end 1."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return -((y0 - yP) - lam * (x0 - xP)) / ((y1 - y0) - lam * (x1 - x0))


def _solve_compat(Qp, pU, thU, Sp, xU, Qm, pF, thF, Sm, xF, xP):
    """p and theta at P from C+ (through U) and C- (through F), R1 Eq. (6)."""
    p = (Qp * pU + Qm * pF + thU - thF - Sm * (xP - xF) - Sp * (xP - xU)) / (Qp + Qm)
    th = thF + Qm * (p - pF) + Sm * (xP - xF)
    return p, th


def _lagrange(nodes, x):
    """Lagrange basis values and derivatives at ``x`` for the given nodes."""
    k = len(nodes)
    vals, ders = [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(k):
            den = np.ones_like(x)
            num = np.ones_like(x)
            dnum = np.zeros_like(x)
            for jj in range(k):
                if jj == i:
                    continue
                den = den * (nodes[i] - nodes[jj])
                dnum = dnum * (x - nodes[jj]) + num
                num = num * (x - nodes[jj])
            vals.append(num / den)
            ders.append(dnum / den)
    return vals, ders


def _row_foot(row, seg, xP, yP, lam):
    """Foot of the line through P (slope ``lam``) on the curve of a known row.

    ``row`` holds the row arrays (x, y, th, p, ...), NaN where invalid. Near
    segment ``seg`` the row is represented by a cubic through four
    neighbouring points (in ``x``). A quadratic leaning toward the foot is used
    at the row ends, and a straight line for two-point rows. The line is
    intersected with that curve (Newton from the chord intersection), and all
    fields are interpolated with the same stencil. Linear interpolation would
    add an O(h^2) error at every step, which accumulates to first order
    overall.

    Returns ``(foot_state, t)``. ``foot_state`` holds every field of ``row`` at
    the foot. ``t`` is the chord parameter on ``seg`` (0 at ``seg``, 1 at
    ``seg + 1``), used for the bracketing checks.
    """
    rx, ry = row[0], row[1]
    n = rx.size
    s = np.clip(seg, 0, n - 2)
    t = _segment_param(rx[s], ry[s], rx[s + 1], ry[s + 1], xP, yP, lam)
    fin = lambda i: (i >= 0) & (i <= n - 1) & np.isfinite(rx[np.clip(i, 0, n - 1)])
    has_l, has_r = fin(s - 1), fin(s + 2)
    cubic = has_l & has_r
    quad_l = ~cubic & has_l
    quad_r = ~cubic & ~has_l & has_r

    out = [rx[s] + t * (rx[s + 1] - rx[s])]
    out += [f[s] + t * (f[s + 1] - f[s]) for f in row[1:]]
    for mask, offs in ((cubic, (-1, 0, 1, 2)), (quad_l, (-1, 0, 1)), (quad_r, (0, 1, 2))):
        if not mask.any():
            continue
        ii = np.flatnonzero(mask)
        idx = [np.clip(s[ii] + o, 0, n - 1) for o in offs]
        nodes = [rx[i] for i in idx]
        yn = [ry[i] for i in idx]
        xF = out[0][ii]
        yPi, xPi, lami = yP[ii], xP[ii], lam[ii]
        with np.errstate(invalid="ignore", divide="ignore"):
            for _ in range(4):                        # Newton on the row curve
                b, db = _lagrange(nodes, xF)
                g = sum(bi * yi for bi, yi in zip(b, yn)) - yPi - lami * (xF - xPi)
                dg = sum(di * yi for di, yi in zip(db, yn)) - lami
                xF = xF - g / dg
        b, _ = _lagrange(nodes, xF)
        out[0][ii] = xF
        for fi, f in enumerate(row[1:], start=1):
            out[fi][ii] = sum(bi * f[i] for bi, i in zip(b, idx))
    return tuple(out), t


# ---------------------------------------------------------------------------
#  Row unit processes
# ---------------------------------------------------------------------------

def _row_characteristic(prev, rule, gamma, tol, max_iter, origin_min):
    """Characteristic-net row: P_k = C-(A = prev[k]) ∩ C+(B = prev[k+1]).

    ``prev`` holds the previous row (x, y, th, p, K, H, origin), NaN where
    invalid. K, H and the streamline origin are taken where the streamline
    through P, traced back, meets the previous row (:func:`_row_foot`). A
    point is determined only if its streamline originates on the initial
    line: ``origin >= x_first - ext_limit * dx_first``.

    Returns the new row, the convergence mask, the geometric-fold mask and the
    undetermined mask.
    """
    x = prev[0]
    m = x.size - 1
    A = tuple(v[:-1] for v in prev)
    B = tuple(v[1:] for v in prev)
    rho, V, _ = _state_from_pKH(prev[3], prev[4], prev[5], gamma)
    stA = (A[0], A[1], A[2], A[3], rho[:-1], V[:-1])
    stB = (B[0], B[1], B[2], B[3], rho[1:], V[1:])
    m_state, p_state = stA, stB
    xP = yP = pP = thP = None
    th_E = 0.5 * (A[2] + B[2])
    converged = np.zeros(m, dtype=bool)
    undetermined = np.zeros(m, dtype=bool)
    seg_found = None
    scale = np.maximum(np.abs(B[0] - A[0]), 1e-300)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for it in range(max_iter):
            lam_m, Q_m, S_m = _char_coeffs(*m_state, -1, rule, gamma)
            lam_p, Q_p, S_p = _char_coeffs(*p_state, +1, rule, gamma)
            xP_n = (B[1] - A[1] - lam_p * B[0] + lam_m * A[0]) / (lam_m - lam_p)
            yP_n = A[1] + lam_m * (xP_n - A[0])
            pP_n, thP_n = _solve_compat(Q_p, B[3], B[2], S_p, B[0], Q_m, A[3], A[2], S_m, A[0], xP_n)
            lam0 = np.tan(0.5 * (thP_n + th_E))

            if seg_found is None:
                # Predictor: find the previous-row segment crossed by the
                # streamline through P, walking upstream from segment k (AB).
                seg_found, undetermined = _walk_to_streamline_foot(x, prev[1], xP_n, yP_n, lam0)
            F, _ = _row_foot(prev, seg_found, xP_n, yP_n, lam0)
            undetermined = undetermined | ~(F[6] >= origin_min)
            undetermined &= np.isfinite(xP_n)
            th_E = F[2]
            KP = np.where(undetermined, np.nan, F[4])
            HP = np.where(undetermined, np.nan, F[5])
            OP = np.where(undetermined, np.nan, F[6])
            rhoP, VP, _ = _state_from_pKH(pP_n, KP, HP, gamma)

            if xP is not None:
                converged = ((np.abs(xP_n - xP) / scale < tol) & (np.abs(yP_n - yP) / scale < tol)
                             & (np.abs(pP_n - pP) / np.abs(pP_n) < tol) & (np.abs(thP_n - thP) < tol))
            xP, yP, pP, thP = xP_n, yP_n, pP_n, thP_n
            if it > 0 and np.all(converged | ~np.isfinite(xP) | undetermined):
                break
            stP = (xP, yP, thP, pP, rhoP, VP)
            m_new, p_new = _avg(stA, stP), _avg(stB, stP)
            # Undetermined points keep predictor coefficients (they are discarded).
            m_state = tuple(np.where(undetermined, a, b) for a, b in zip(stA, m_new))
            p_state = tuple(np.where(undetermined, a, b) for a, b in zip(stB, p_new))

        # Geometry: A < P < B in x, P on the post-shock side of AB.
        cross = (B[0] - A[0]) * (yP - A[1]) - (B[1] - A[1]) * (xP - A[0])
        span2 = (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2
        fold = ~((xP > A[0]) & (xP < B[0]) & (cross < -1e-12 * span2))
        fold &= np.isfinite(A[0]) & np.isfinite(B[0])
    return (xP, yP, thP, pP, KP, HP, OP), converged, fold, undetermined


def _walk_to_streamline_foot(x, y, xP, yP, lam0, max_extrapolation=2.0):
    """Previous-row segment crossed by the backward streamline from each P.

    Starts at segment k (between A and B) and walks upstream until the
    crossing parameter lies in [0, 1]. A crossing upstream of the first valid
    point is extrapolated on the first valid segment, by at most
    ``max_extrapolation`` segment lengths. The caller then decides
    determinacy from the extrapolated streamline origin.
    """
    m = xP.size
    seg = np.arange(m)
    found = np.zeros(m, dtype=bool)
    active = np.isfinite(xP)
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(m + 1):
            look = active & ~found & (seg >= 0)
            if not look.any():
                break
            li = np.flatnonzero(look)
            s = seg[li]
            ok_seg = np.isfinite(x[s]) & np.isfinite(x[s + 1])
            tt = _segment_param(x[s], y[s], x[s + 1], y[s + 1], xP[li], yP[li], lam0[li])
            ok = ok_seg & (tt >= -1e-12) & (tt <= 1.0 + 1e-12)
            found[li[ok]] = True
            # Walk upstream; stop at the start of the row or at an invalid point.
            nxt = s - 1
            dead = ~ok & ((nxt < 0) | ~ok_seg | ~np.isfinite(x[np.maximum(nxt, 0)]))
            seg[li[~ok]] = np.where(dead[~ok], -1, nxt[~ok])
        # Crossing upstream of the first valid segment: allow a short extrapolation.
        undetermined = active & ~found
        for k in np.flatnonzero(undetermined):
            s0 = _first_valid_segment(x, k)
            if s0 < 0:
                continue
            tt = _segment_param(x[s0], y[s0], x[s0 + 1], y[s0 + 1], xP[k], yP[k], lam0[k])
            if np.isfinite(tt) and -max_extrapolation <= tt <= 0.0:
                seg[k] = s0
                undetermined[k] = False
    seg = np.where(undetermined | ~active, np.clip(np.arange(m), 0, None), seg)
    return seg, undetermined


def _first_valid_segment(x, k):
    """Most upstream segment of the valid run that contains segment k."""
    s = int(k)
    if not (np.isfinite(x[s]) and np.isfinite(x[s + 1])):
        return -1
    while s - 1 >= 0 and np.isfinite(x[s - 1]):
        s -= 1
    return s


def _row_streamline(prev, rule, gamma, tol, max_iter):
    """Streamline-net row: P_j = streamline(L = prev[j]) ∩ C+(U = prev[j+1]).

    The C- relation is taken at the foot F of the backward C- through P. F
    always lies between L and U, because segment L-U is a diagonal of the
    cell, and its state comes from the quadratic row curve (:func:`_row_foot`).
    K and H are exact: they are constant along each streamline (column).
    """
    m = prev[0].size - 1
    L = tuple(v[:-1] for v in prev)
    U = tuple(v[1:] for v in prev)
    rho, V, _ = _state_from_pKH(prev[3], prev[4], prev[5], gamma)
    stL = (L[0], L[1], L[2], L[3], rho[:-1], V[:-1])
    stU = (U[0], U[1], U[2], U[3], rho[1:], V[1:])
    KP, HP, OP = L[4], L[5], L[6]
    th0 = L[2]
    p_state = stU
    m_state = _avg(stL, stU)
    xP = yP = pP = thP = None
    t_foot = np.full(m, 0.5)
    converged = np.zeros(m, dtype=bool)
    seg = np.arange(m)
    scale = np.maximum(np.abs(U[0] - L[0]), 1e-300)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for it in range(max_iter):
            lam0 = np.tan(th0)
            lam_p, Q_p, S_p = _char_coeffs(*p_state, +1, rule, gamma)
            lam_m, Q_m, S_m = _char_coeffs(*m_state, -1, rule, gamma)
            xP_n = (U[1] - L[1] - lam_p * U[0] + lam0 * L[0]) / (lam0 - lam_p)
            yP_n = L[1] + lam0 * (xP_n - L[0])
            F, t_foot = _row_foot(prev, seg, xP_n, yP_n, lam_m)
            pP_n, thP_n = _solve_compat(Q_p, U[3], U[2], S_p, U[0], Q_m, F[3], F[2], S_m, F[0], xP_n)
            rhoP, VP, _ = _state_from_pKH(pP_n, KP, HP, gamma)

            if xP is not None:
                converged = ((np.abs(xP_n - xP) / scale < tol) & (np.abs(yP_n - yP) / scale < tol)
                             & (np.abs(pP_n - pP) / np.abs(pP_n) < tol) & (np.abs(thP_n - thP) < tol))
            xP, yP, pP, thP = xP_n, yP_n, pP_n, thP_n
            if it > 0 and np.all(converged | ~np.isfinite(xP)):
                break
            stP = (xP, yP, thP, pP, rhoP, VP)
            rhoF, VF, _ = _state_from_pKH(F[3], F[4], F[5], gamma)
            th0 = 0.5 * (L[2] + thP)
            p_state = _avg(stU, stP)
            m_state = _avg((F[0], F[1], F[2], F[3], rhoF, VF), stP)

        # Geometry: L < P < U in x, P on the post-shock side of L-U, and the
        # C- foot between L and U.
        cross = (U[0] - L[0]) * (yP - L[1]) - (U[1] - L[1]) * (xP - L[0])
        span2 = (U[0] - L[0]) ** 2 + (U[1] - L[1]) ** 2
        fold = ~((xP > L[0]) & (xP < U[0]) & (cross < -1e-12 * span2)
                 & (t_foot >= -1e-9) & (t_foot <= 1.0 + 1e-9))
        fold &= np.isfinite(L[0]) & np.isfinite(U[0])
    return (xP, yP, thP, pP, KP, HP, OP), converged, fold, np.zeros(m, dtype=bool)


# ---------------------------------------------------------------------------
#  Solution container
# ---------------------------------------------------------------------------

@dataclass
class StreamlineResult:
    """A streamline in the solution."""

    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    p: np.ndarray
    rho: np.ndarray
    V: np.ndarray
    M: np.ndarray
    reached_end: bool
    stop_reason: str


@dataclass
class MOCSolution:
    """Solution on the triangular layout: row ``d`` holds columns ``k < n - d``.

    For ``scheme="characteristic"``, ``(d, k)`` is the net point between
    initial points ``k`` and ``k + d``. For ``scheme="streamline"``, column ``j``
    is the streamline from initial point ``j`` and ``(d, j)`` is its d-th point.
    ``origin`` is the x coordinate on the initial line where the streamline
    through each point starts.
    """

    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    p: np.ndarray
    K: np.ndarray
    H: np.ndarray
    origin: np.ndarray
    status: np.ndarray
    rule: AxisRule
    gamma: float
    scheme: str
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_rows(self) -> int:
        return self.x.shape[0]

    @property
    def n_cols(self) -> int:
        return self.x.shape[1]

    @property
    def exists(self) -> np.ndarray:
        """Mask of layout positions that belong to the computed rows."""
        d, k = np.indices(self.x.shape)
        return k < self.n_cols - d

    @property
    def valid(self) -> np.ndarray:
        return self.exists & (self.status == VALID)

    def derived(self):
        """(rho, V, M, mu) on the layout (NaN where invalid)."""
        rho, V, M = _state_from_pKH(self.p, self.K, self.H, self.gamma)
        with np.errstate(invalid="ignore"):
            mu = np.arcsin(1.0 / M)
        return rho, V, M, mu

    def status_counts(self) -> dict:
        ex = self.exists
        return {STATUS_TEXT[c]: int(np.sum(ex & (self.status == c)))
                for c in STATUS_TEXT if np.any(ex & (self.status == c))}

    def first_failure(self, include_undetermined: bool = False) -> Optional[dict]:
        """Location and reason of the first failure in marching order."""
        skip = {VALID, PARENT_INVALID} | (set() if include_undetermined else {UNDETERMINED})
        bad = self.exists & ~np.isin(self.status, list(skip))
        if not bad.any():
            return None
        d, k = np.argwhere(bad)[0]
        code = int(self.status[d, k])
        return {"row": int(d), "col": int(k), "x": float(self.x[d, k]), "y": float(self.y[d, k]),
                "code": code, "reason": STATUS_TEXT[code]}

    # -- interpolation -----------------------------------------------------
    def _triangulation(self):
        if "tri" in self._cache:
            return self._cache["tri"]
        import matplotlib.tri as mtri

        valid = self.valid
        idx = -np.ones(self.x.shape, dtype=np.int64)
        idx[valid] = np.arange(int(valid.sum()))
        tris = []
        for d in range(self.n_rows - 1):
            n = self.n_cols - d
            if n < 2:
                break
            k = np.arange(n - 1)
            tris.append(np.stack([idx[d, k], idx[d, k + 1], idx[d + 1, k]], axis=1))
            if n >= 3:
                k = np.arange(n - 2)
                tris.append(np.stack([idx[d, k + 1], idx[d + 1, k + 1], idx[d + 1, k]], axis=1))
        tris = np.concatenate(tris, axis=0) if tris else np.zeros((0, 3), dtype=np.int64)
        tris = tris[np.all(tris >= 0, axis=1)]
        if tris.shape[0] == 0:
            raise MOCError("the solution has no valid cells")
        tri = mtri.Triangulation(self.x[valid], self.y[valid], tris)
        self._cache["tri"] = tri
        self._cache["interp"] = {
            "theta": mtri.LinearTriInterpolator(tri, self.theta[valid]),
            "p": mtri.LinearTriInterpolator(tri, self.p[valid]),
        }
        return tri

    def interpolate(self, name: str, x, y) -> np.ndarray:
        """Piecewise-linear interpolation of ``theta`` or ``p`` (NaN outside)."""
        self._triangulation()
        f = self._cache["interp"][name]
        out = f(np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float)))
        return np.ma.filled(out.astype(float), np.nan)

    def streamline_column(self, j: int = 0) -> StreamlineResult:
        """Exact streamline ``j`` of a ``scheme="streamline"`` solution.

        The streamline runs from initial point ``j`` to the last valid point of
        column ``j``. ``reached_end`` is True when the column is valid to the
        end of the layout, i.e. up to the C+ characteristic through the last
        initial point. Beyond that the data does not determine it (spec flag 4).
        """
        if self.scheme != "streamline":
            raise MOCError("streamline_column needs scheme='streamline'; use trace_streamline")
        last = min(self.n_cols - j, self.n_rows)   # rows that exist in column j
        col = self.valid[:last, j]
        stop = last if col.all() else int(np.argmin(col))
        sl = slice(0, stop)
        rho, V, M = _state_from_pKH(self.p[sl, j], self.K[sl, j], self.H[sl, j], self.gamma)
        reached = stop == last
        reason = ("end of the determined region" if reached
                  else STATUS_TEXT[int(self.status[stop, j])])
        return StreamlineResult(self.x[sl, j], self.y[sl, j], self.theta[sl, j], self.p[sl, j],
                                rho, V, M, reached, reason)

    def trace_streamline(self, row: int = 0, col: int = 0, x_end: Optional[float] = None,
                         n_steps_per_cell: int = 4) -> StreamlineResult:
        """Trace the streamline that starts at layout point ``(row, col)``.

        ``tan(theta)`` and ``p`` are interpolated linearly on the triangulated
        solution, and the integration uses Heun's method. K and H keep their
        values at the start point.

        Tracing stops at ``x_end`` or where the streamline leaves the valid
        region. In the second case ``reached_end`` is False.
        """
        if not self.valid[row, col]:
            raise MOCError(f"start point ({row}, {col}) is not a valid point")
        x0, y0 = float(self.x[row, col]), float(self.y[row, col])
        th0 = float(self.theta[row, col])
        K0, H0 = float(self.K[row, col]), float(self.H[row, col])
        h = float(np.median(np.diff(self.x[0]))) / n_steps_per_cell
        if x_end is None:
            x_end = float(np.nanmax(self.x[self.valid]))
        xs, ys, ths = [x0], [y0], [th0]
        reached, reason = False, ""
        x, y, th = x0, y0, th0
        while x < x_end - 1e-14:
            step = min(h, x_end - x)
            y_pred = y + step * np.tan(th)
            th_pred = float(self.interpolate("theta", x + step, y_pred)[0])
            if not np.isfinite(th_pred):
                reason = "left the valid region"
                break
            y_new = y + 0.5 * step * (np.tan(th) + np.tan(th_pred))
            th_new = float(self.interpolate("theta", x + step, y_new)[0])
            if not np.isfinite(th_new):
                reason = "left the valid region"
                break
            x, y, th = x + step, y_new, th_new
            xs.append(x)
            ys.append(y)
            ths.append(th)
        else:
            reached, reason = True, "reached x_end"
        xs, ys, ths = np.array(xs), np.array(ys), np.array(ths)
        p = self.interpolate("p", xs, ys)
        p[0] = self.p[row, col]
        rho, V, M = _state_from_pKH(p, K0, H0, self.gamma)
        return StreamlineResult(xs, ys, ths, p, rho, V, M, reached, reason)


# ---------------------------------------------------------------------------
#  Marching
# ---------------------------------------------------------------------------

def solve_inverse(initial: InitialLine, rule: AxisRule, gamma: float = 1.4, *,
                  scheme: str = "streamline", n_rows: Optional[int] = None,
                  strict: bool = False, tol: float = 1e-10, max_iter: int = 60,
                  extrapolation_limit: float = 3.0) -> MOCSolution:
    """March inward from ``initial`` with the chosen scheme.

    Parameters
    ----------
    initial : InitialLine
        State on the initial line (usually from :func:`shock_initial_line`).
        The line must be space-like: at every point its direction lies
        strictly between the C- and C+ directions.
    rule : AxisRule
        Planar, coaxial or noncoaxial source-term radius.
    scheme : {"streamline", "characteristic"}
        Mesh topology; see the module docstring. ``"streamline"`` is the
        default for LTOCs (Gate 1 decision). ``"characteristic"`` is kept as
        an independent cross-check.
    n_rows : int, optional
        Rows to compute after the initial line. The default and the maximum is
        ``len(initial.x) - 1``.
    strict : bool
        If True, raise on the first failure (``LimitSurfaceError``,
        ``SubsonicError``, ``AxisCrossedError`` or ``MOCError``).
        ``UNDETERMINED`` is not a failure: it marks the edge of the domain of
        determinacy. If False, failures are recorded in ``status``.
    tol : float
        Predictor-corrector convergence tolerance. Positions are measured
        relative to the local cell size, pressure relatively, and the flow
        angle in radians.
    extrapolation_limit : float
        ``scheme="characteristic"`` only. A point counts as determined if its
        streamline starts no further upstream of the first initial point than
        this many first-segment lengths. This keeps a thin band of points just
        below the streamline from the first initial point, so the band can be
        interpolated across.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    n = initial.x.size
    max_rows = n - 1
    n_rows = max_rows if n_rows is None else int(min(max(n_rows, 0), max_rows))
    shape = (n_rows + 1, n)
    X, Y, TH, P, KK, HH, OR = (np.full(shape, np.nan) for _ in range(7))
    ST = np.zeros(shape, dtype=np.int8)

    # Row 0: the initial line. Check that it is supersonic and space-like.
    K0, H0 = _invariants(initial.p, initial.rho, initial.V, gamma)
    _, _, M0 = _state_from_pKH(initial.p, K0, H0, gamma)
    if np.any(~np.isfinite(M0)) or np.any(M0 <= 1.0):
        raise SubsonicError("the initial line has M <= 1")
    mu0 = np.arcsin(1.0 / M0)
    seg = np.arctan2(np.diff(initial.y), np.diff(initial.x))
    phi = np.empty(n)
    phi[0], phi[-1] = seg[0], seg[-1]
    phi[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    lo, hi = initial.theta - mu0, initial.theta + mu0
    if np.any(phi <= lo) or np.any(phi >= hi):
        i = int(np.argmax((phi <= lo) | (phi >= hi)))
        raise MOCError(
            f"initial line is not space-like at x = {initial.x[i]:.6g}: direction "
            f"{np.degrees(phi[i]):.3f} deg outside ({np.degrees(lo[i]):.3f}, "
            f"{np.degrees(hi[i]):.3f}) deg")
    if rule.delta and np.any(rule.radius(initial.x, initial.y) <= 0.0):
        raise AxisCrossedError("initial line has r <= 0 under the given axis rule")
    X[0], Y[0], TH[0], P[0], KK[0], HH[0], OR[0] = (initial.x, initial.y, initial.theta, initial.p,
                                                     K0, H0, initial.x)
    origin_min = initial.x[0] - extrapolation_limit * (initial.x[1] - initial.x[0])

    for d in range(1, n_rows + 1):
        m = n - d
        prev = tuple(A[d - 1, :m + 1] for A in (X, Y, TH, P, KK, HH, OR))
        # Invalid points of the previous row carry NaN state, but their x, y
        # are kept for diagnostics; hide them from the unit processes.
        bad_prev = ST[d - 1, :m + 1] != VALID
        prev = tuple(np.where(bad_prev, np.nan, v) for v in prev)
        parent_bad = bad_prev[:-1] | bad_prev[1:]
        if scheme == "characteristic":
            new, conv, fold, undet = _row_characteristic(prev, rule, gamma, tol, max_iter,
                                                         origin_min)
        else:
            new, conv, fold, undet = _row_streamline(prev, rule, gamma, tol, max_iter)
        xP, yP, thP, pP, KP, HP, OP = new

        with np.errstate(invalid="ignore"):
            _, _, MP = _state_from_pKH(pP, KP, HP, gamma)
            if m >= 2:
                # Cells (P_k, prev[k+1], P_{k+1}) must not invert: no
                # crossing of same-family characteristics between rows.
                xb, yb = prev[0][1:m], prev[1][1:m]
                cross_up = ((xP[1:] - xP[:-1]) * (yb - yP[:-1])
                            - (yP[1:] - yP[:-1]) * (xb - xP[:-1]))
                inverted = ~((xP[1:] > xP[:-1]) & (cross_up > 0.0))
                inverted &= np.isfinite(xb) & np.isfinite(xP[1:]) & np.isfinite(xP[:-1])
                inverted &= ~(undet[1:] | undet[:-1])
                fold = fold.copy()
                fold[:-1] |= inverted
                fold[1:] |= inverted
            subsonic = ~(MP > 1.0)
            axis = (rule.radius(xP, yP) <= 0.0) if rule.delta else np.zeros(m, dtype=bool)

        status = np.zeros(m, dtype=np.int8)
        status[~conv] = NOT_CONVERGED
        status[fold] = CHARACTERISTIC_FOLD
        status[axis] = AXIS_CROSSED
        status[subsonic] = SUBSONIC
        status[undet] = UNDETERMINED
        status[parent_bad] = PARENT_INVALID

        bad = status != VALID
        X[d, :m], Y[d, :m] = xP, yP
        TH[d, :m] = np.where(bad, np.nan, thP)
        P[d, :m] = np.where(bad, np.nan, pP)
        KK[d, :m] = np.where(bad, np.nan, KP)
        HH[d, :m] = np.where(bad, np.nan, HP)
        OR[d, :m] = np.where(bad, np.nan, OP)
        ST[d, :m] = status

        if strict:
            new_fail = ~np.isin(status, [VALID, PARENT_INVALID, UNDETERMINED])
            if new_fail.any():
                k = int(np.argmax(new_fail))
                code = int(status[k])
                raise _STRICT_ERRORS[code](
                    f"{STATUS_TEXT[code]} at row {d}, column {k} (x = {xP[k]:.6g}, y = {yP[k]:.6g})")
        if np.all(bad):
            for dd in range(d + 1, n_rows + 1):
                ST[dd, : n - dd] = PARENT_INVALID
            break

    return MOCSolution(X, Y, TH, P, KK, HH, OR, ST, rule, gamma, scheme)
