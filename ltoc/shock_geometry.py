"""Shock geometry for LTOCs Step A (spec section 6, Phase 2).

For a prescribed shock surface (``ltoc.shock_surface``) and a freestream
along +X, this module gives:

* the local shock geometry at any point: unit normal, shock angle, shock-curve
  direction, crosswise direction, crosswise normal curvature, local crosswise
  radius and local axis centre (:func:`local_geometry`);
* the shock curve of a stream surface: the curve on the shock whose direction
  at every point lies in the local osculating plane, the plane containing
  the freestream and the shock normal (:func:`trace_shock_curves`).

Conventions
-----------
The unit normal ``n`` is oriented toward the post-shock side, so ``n_x > 0``
for an attached shock.

* Shock angle, R1 Eq. (3)::

      beta = pi/2 - acos(V . n / (|V| |n|)),   so sin(beta) = n_x.

* Shock-curve direction, R1 Sec. II.A and Fig. 2: ``C = n x V`` and
  ``D = C x n``, the projection of the freestream onto the shock tangent
  plane::

      t = (X - sin(beta) n) / cos(beta).

  For a surface ``Z(X, Y)`` this gives ``dY/dX = -Z_X Z_Y / (1 + Z_Y^2)``. The
  sign printed in R1 Eqs. (1)-(2) is a typo (spec flag 1, confirmed at
  Gate 0).

* Crosswise direction ``b = n x t``. It is parallel to ``C = n x V`` and so
  tangent to the section X = const.

* ``kappa_b`` is the normal curvature of the shock along ``b``, from the
  second fundamental form. It is positive when the shock curves toward the
  post-shock side (convex cross-section, centre on the body side).

* Local crosswise radius (R1 Sec. II.B, "the local radius of the shock in
  its crosswise plane"). By Meusnier's theorem (spec flag 3, Gate 0)::

      r = cos(beta) / kappa_b.

* Local axis centre (R1 Sec. II.B; R2 Fig. 15b): ``O = P + r n_cs``, where
  ``n_cs = (n - sin(beta) X) / cos(beta)`` is the in-plane normal of the
  cross-section.

Intrinsic meridian mapping (Gate 0 decision, 2026-10-08): along each shock
curve ``x = X`` and ``dy/dx = tan(beta)``; the axis centre sits at
``y_axis = y - r``. The meridian ordinate is integrated together with the
curve.

The shock curve is integrated with fourth-order Runge-Kutta in X (R1
Sec. II.A), in the surface parameters, so the curve stays on the surface
exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .shock_surface import ShockSurface

__all__ = ["LocalGeometry", "ShockCurve", "local_geometry", "trace_shock_curves",
           "FLAT_KAPPA_REL"]

X_HAT = np.array([1.0, 0.0, 0.0])

#: |kappa_b| * length_scale below this counts as flat (r = inf, planar flow).
FLAT_KAPPA_REL = 1e-9


@dataclass
class LocalGeometry:
    """Local shock geometry; every field has the broadcast shape of (u, v)."""

    point: np.ndarray        # (..., 3)
    normal: np.ndarray       # (..., 3), unit, toward the post-shock side
    beta: np.ndarray         # (...,) shock angle [rad], R1 Eq. (3)
    t: np.ndarray            # (..., 3) unit shock-curve direction (R1 D)
    b: np.ndarray            # (..., 3) unit crosswise direction (n x t)
    kappa_b: np.ndarray      # (...,) normal curvature along b [1/m]
    r: np.ndarray            # (...,) local crosswise radius [m]; inf if flat
    n_cs: np.ndarray         # (..., 3) unit in-plane normal of the cross-section
    axis_centre: np.ndarray  # (..., 3) local axis centre O (NaN if flat)


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def local_geometry(surface: ShockSurface, u, v, length_scale: float = 1.0) -> LocalGeometry:
    """Local shock geometry at the parameters ``(u, v)`` (see module docstring).

    ``length_scale`` sets the flatness threshold: ``|kappa_b| * length_scale <
    FLAT_KAPPA_REL`` gives ``r = inf``.
    """
    d = surface.evaluate(u, v)
    n = np.cross(d.Su, d.Sv)
    nrm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / nrm
    n = np.where(n[..., :1] < 0.0, -n, n)                       # toward the post-shock side
    sinb = np.clip(n[..., 0], -1.0, 1.0)
    beta = np.arcsin(sinb)                                       # R1 Eq. (3)
    cosb = np.cos(beta)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (X_HAT - sinb[..., None] * n) / cosb[..., None]       # R1 Fig. 2, D = (n x V) x n
        b = np.cross(n, t)
        # b in the tangent basis (Su, Sv), then the second fundamental form.
        g11, g12, g22 = _dot(d.Su, d.Su), _dot(d.Su, d.Sv), _dot(d.Sv, d.Sv)
        r1, r2 = _dot(d.Su, b), _dot(d.Sv, b)
        det = g11 * g22 - g12 * g12
        alpha = (g22 * r1 - g12 * r2) / det
        gamma = (g11 * r2 - g12 * r1) / det
        L, M, N = _dot(d.Suu, n), _dot(d.Suv, n), _dot(d.Svv, n)
        kappa_b = alpha * alpha * L + 2.0 * alpha * gamma * M + gamma * gamma * N
        flat = np.abs(kappa_b) * length_scale < FLAT_KAPPA_REL
        r = np.where(flat, np.inf, cosb / np.where(flat, 1.0, kappa_b))   # spec flag 3
        n_cs = (n - sinb[..., None] * X_HAT) / cosb[..., None]
        centre = np.where(np.isfinite(r)[..., None], d.S + r[..., None] * n_cs, np.nan)
    return LocalGeometry(d.S, n, beta, t, b, kappa_b, r, n_cs, centre)


@dataclass
class ShockCurve:
    """One shock curve: the shock line of one stream surface (R1 Fig. 1)."""

    x: np.ndarray            # (k,) axial stations X
    uv: np.ndarray           # (k, 2) surface parameters
    geometry: LocalGeometry  # fields of shape (k, ...)
    y: np.ndarray            # (k,) intrinsic meridian ordinate, dy/dx = tan(beta)
    y_axis: np.ndarray       # (k,) meridian position of the local axis centre, y - r
    in_domain: np.ndarray    # (k,) True inside the nominal (u, v) range
    flags: dict = field(default_factory=dict)

    @property
    def points(self) -> np.ndarray:
        return self.geometry.point


def _rhs(surface, u, v):
    """d(u, v, y)/dX along the shock curve: dP/dX = t / t_x = (X - sin(beta) n) / cos^2(beta)."""
    d = surface.evaluate(u, v)
    n = np.cross(d.Su, d.Sv)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(n[..., :1] < 0.0, -n, n)
    sinb = n[..., 0]
    cos2 = 1.0 - sinb * sinb
    w = (X_HAT - sinb[..., None] * n) / cos2[..., None]
    g11, g12, g22 = _dot(d.Su, d.Su), _dot(d.Su, d.Sv), _dot(d.Sv, d.Sv)
    r1, r2 = _dot(d.Su, w), _dot(d.Sv, w)
    det = g11 * g22 - g12 * g12
    du = (g22 * r1 - g12 * r2) / det
    dv = (g11 * r2 - g12 * r1) / det
    dy = sinb / np.sqrt(cos2)                                  # tan(beta)
    return du, dv, dy


def trace_shock_curves(surface: ShockSurface, u0, v0, x_end: float, n_steps: int = 200,
                       M_inf: Optional[float] = None, gamma: float = 1.4,
                       length_scale: Optional[float] = None) -> List[ShockCurve]:
    """Integrate shock curves from the start points ``(u0, v0)`` to ``X = x_end``.

    Fourth-order Runge-Kutta in X with ``n_steps`` equal steps per curve
    (R1 Sec. II.A). Curves may run past the nominal parameter range (spec
    flag 4); ``in_domain`` records where they are inside it.

    If ``M_inf`` is given, ``flags["sub_mach"]`` marks beta at or below the
    freestream Mach angle and ``flags["detached"]`` marks beta beyond
    detachment. ``flags["concave"]`` (kappa_b < 0) and ``flags["flat"]``
    (r = inf) are always set. The LTOCs orchestration refuses sub-Mach,
    detached and concave curves (spec flag 7).
    """
    u0 = np.atleast_1d(np.asarray(u0, dtype=float))
    v0 = np.atleast_1d(np.asarray(v0, dtype=float))
    u0, v0 = np.broadcast_arrays(u0, v0)
    x0 = surface.point(u0, v0)[..., 0]
    if np.any(x0 > x_end):
        raise ValueError("every start point must lie upstream of x_end")
    if length_scale is None:
        length_scale = float(np.max(np.abs(x_end - x0))) or 1.0
    h = (x_end - x0) / n_steps

    U = np.empty((n_steps + 1,) + u0.shape)
    Vv = np.empty_like(U)
    Ym = np.empty_like(U)
    U[0], Vv[0], Ym[0] = u0, v0, 0.0
    u, v, y = u0.copy(), v0.copy(), np.zeros_like(u0)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(n_steps):
            k1 = _rhs(surface, u, v)
            k2 = _rhs(surface, u + 0.5 * h * k1[0], v + 0.5 * h * k1[1])
            k3 = _rhs(surface, u + 0.5 * h * k2[0], v + 0.5 * h * k2[1])
            k4 = _rhs(surface, u + h * k3[0], v + h * k3[1])
            u = u + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            v = v + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            y = y + h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            U[k + 1], Vv[k + 1], Ym[k + 1] = u, v, y

    mu_inf = np.arcsin(1.0 / M_inf) if M_inf else None
    beta_det = None
    if M_inf:
        from gvwd.thermo.oblique_shock import detachment_beta
        beta_det = detachment_beta(M_inf, gamma)

    curves = []
    for i in range(u0.size):
        idx = np.unravel_index(i, u0.shape)
        uu, vv = U[(slice(None),) + idx], Vv[(slice(None),) + idx]
        geo = local_geometry(surface, uu, vv, length_scale)
        (ua, ub), (va, vb) = surface.u_range, surface.v_range
        tol = 1e-12 * max(1.0, abs(ub - ua), abs(vb - va))
        in_dom = (uu >= ua - tol) & (uu <= ub + tol) & (vv >= va - tol) & (vv <= vb + tol)
        flags = {
            "flat": ~np.isfinite(geo.r),
            "concave": geo.kappa_b < 0.0,
        }
        if mu_inf is not None:
            flags["sub_mach"] = geo.beta <= mu_inf
            flags["detached"] = geo.beta >= beta_det
        y_m = Ym[(slice(None),) + idx]
        curves.append(ShockCurve(
            x=geo.point[:, 0], uv=np.column_stack([uu, vv]), geometry=geo,
            y=y_m, y_axis=y_m - geo.r, in_domain=in_dom, flags=flags))
    return curves
