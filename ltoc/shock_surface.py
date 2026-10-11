"""Prescribed shock surfaces for LTOCs (spec section 6, Phase 2).

A shock surface is a parametric surface ``S(u, v)`` in a Cartesian frame
whose X axis is the freestream direction. Y and Z are transverse; their
orientation is the caller's choice. R1, for example, draws Z downward in its
waverider cases. Every class gives exact first and second derivatives, so
the shock angle, the shock-curve direction and the crosswise curvature are
exact (R1 Sec. II.A: "the surface slopes may be exactly defined").

Classes
-------
``RuledShock``
    ``S(m, n) = (1 - n) p(m) + n q(m)``: R1 Eq. (12), and R2 Eq. (1). It covers
    every published case, e.g. R1 Eqs. (15), (18), (23), (25) via
    :meth:`RuledShock.elliptic` and R2 Eq. (13).
``ConeShock``
    A circular cone about the X axis, the ruled surface from the apex to a
    circle.
``OsculatingConeShock``
    The shock implied by an osculating-cone design. In every osculating plane
    the shock is the straight generator at angle ``beta`` through the
    base-plane shock-wave profile curve (Sobieczky et al. 1990; R2 Sec. III).
    :meth:`OsculatingConeShock.from_oc_waverider` reads it from a WADOS
    ``waverider_generator.generator.waverider`` (OC frame: x streamwise, y up,
    z span).
``BSplineShock``
    A bicubic interpolating spline through a grid of points, for shocks
    defined by data.

The parameters double as the nominal domain (``u_range``, ``v_range``).
Evaluation outside it is allowed when the formula extends naturally; spec
flag 4 extends shocks past the base plane this way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

__all__ = [
    "SurfaceDerivatives",
    "ShockSurface",
    "RuledShock",
    "ConeShock",
    "OsculatingConeShock",
    "BSplineShock",
]


@dataclass
class SurfaceDerivatives:
    """``S`` and its partial derivatives, each an array of shape (..., 3)."""

    S: np.ndarray
    Su: np.ndarray
    Sv: np.ndarray
    Suu: np.ndarray
    Suv: np.ndarray
    Svv: np.ndarray


class ShockSurface(ABC):
    """Parametric shock surface ``S(u, v)``; X is the freestream direction.

    ``analytic`` is True when ``evaluate`` is a closed form that stays valid
    outside the nominal ``(u_range, v_range)``. The LTOCs core may then
    extend the shock past the base plane as spec flag 4 requires (R3
    Sec. 1.4 does the same). A data surface (``BSplineShock``) is not
    analytic, so a stream surface that needs it beyond its data is refused.
    """

    u_range: Tuple[float, float]
    v_range: Tuple[float, float]
    analytic: bool = True

    @abstractmethod
    def evaluate(self, u, v) -> SurfaceDerivatives:
        """Point and first and second derivatives at ``(u, v)`` (broadcast)."""

    def point(self, u, v) -> np.ndarray:
        return self.evaluate(u, v).S


def _stack(*c):
    return np.stack(np.broadcast_arrays(*c), axis=-1)


# ---------------------------------------------------------------------------
#  Ruled surfaces (R1 Eq. 12)
# ---------------------------------------------------------------------------

class RuledShock(ShockSurface):
    """``S(m, n) = (1 - n) p(m) + n q(m)`` (R1 Eq. 12; R2 Eq. 1).

    ``m`` is the crosswise and ``n`` the streamwise control parameter.
    ``p``, ``q`` and their first and second derivatives are callables that
    map an array of ``m`` to points of shape (..., 3).
    """

    def __init__(self, p, dp, d2p, q, dq, d2q, m_range=(0.0, 2.0 * np.pi), n_range=(0.0, 1.0)):
        self.p, self.dp, self.d2p = p, dp, d2p
        self.q, self.dq, self.d2q = q, dq, d2q
        self.u_range = tuple(m_range)
        self.v_range = tuple(n_range)

    def evaluate(self, u, v) -> SurfaceDerivatives:
        m, n = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        nn = n[..., None]
        P, Q = self.p(m), self.q(m)
        dP, dQ = self.dp(m), self.dq(m)
        d2P, d2Q = self.d2p(m), self.d2q(m)
        return SurfaceDerivatives(
            S=(1.0 - nn) * P + nn * Q,
            Su=(1.0 - nn) * dP + nn * dQ,
            Sv=Q - P,
            Suu=(1.0 - nn) * d2P + nn * d2Q,
            Suv=dQ - dP,
            Svv=np.zeros_like(P),
        )

    @classmethod
    def elliptic(cls, x_p, a_p, b_p, x_q, a_q, b_q, centre_p=(0.0, 0.0), centre_q=(0.0, 0.0),
                 m_range=(0.0, 2.0 * np.pi), n_range=(0.0, 1.0)) -> "RuledShock":
        """Ruled shock between two ellipses in planes X = const.

        ``p(m) = (x_p, cy_p + a_p cos m, cz_p + b_p sin m)``, and likewise for
        ``q``. This covers R1 Eqs. (13)-(18), (23) and (25). A zero-size
        ``p`` is the apex of an elliptic cone.
        """
        def ell(x0, a, b, c):
            cy, cz = c

            def f(m):
                return _stack(np.full_like(m, x0), cy + a * np.cos(m), cz + b * np.sin(m))

            def df(m):
                return _stack(np.zeros_like(m), -a * np.sin(m), b * np.cos(m))

            def d2f(m):
                return _stack(np.zeros_like(m), -a * np.cos(m), -b * np.sin(m))

            return f, df, d2f

        p, dp, d2p = ell(x_p, a_p, b_p, centre_p)
        q, dq, d2q = ell(x_q, a_q, b_q, centre_q)
        return cls(p, dp, d2p, q, dq, d2q, m_range, n_range)


class ConeShock(RuledShock):
    """Circular conical shock of half-angle ``beta`` (rad) about the X axis.

    ``u`` is the azimuth (from +Y toward +Z) and ``v`` is the axial distance
    from the apex.
    """

    def __init__(self, beta: float, apex=(0.0, 0.0, 0.0), x_range=(0.0, 1.0)):
        self.beta = float(beta)
        ax, ay, az = (float(c) for c in apex)
        tb = np.tan(self.beta)
        super().__init__(
            p=lambda m: _stack(np.full_like(m, ax), np.full_like(m, ay), np.full_like(m, az)),
            dp=lambda m: np.zeros(np.shape(m) + (3,)),
            d2p=lambda m: np.zeros(np.shape(m) + (3,)),
            q=lambda m: _stack(np.full_like(m, ax + 1.0), ay + tb * np.cos(m), az + tb * np.sin(m)),
            dq=lambda m: _stack(np.zeros_like(m), -tb * np.sin(m), tb * np.cos(m)),
            d2q=lambda m: _stack(np.zeros_like(m), -tb * np.cos(m), -tb * np.sin(m)),
            m_range=(0.0, 2.0 * np.pi), n_range=tuple(x_range))


# ---------------------------------------------------------------------------
#  Osculating-cone shock
# ---------------------------------------------------------------------------

class OsculatingConeShock(ShockSurface):
    """Shock surface of an osculating-cone design, in the OC frame.

    The frame is x streamwise, y up and z span. The base-plane shock-wave
    profile curve (SWPC) is the graph ``y = y_s(z)`` in the plane
    ``x = length``. In the osculating plane through the SWPC point at ``z_s``
    the shock is the straight line at angle ``beta`` to the freestream, rising
    toward the local centre of curvature (Sobieczky et al. 1990)::

        S(z_s, x) = C(z_s) + (length - x) (tan(beta) N(z_s) - X),
        C = (length, y_s(z_s), z_s),   N = (0, 1, -y_s') / sqrt(1 + y_s'^2)

    Parameters: ``u = z_s`` and ``v = x``. ``swpc(z)`` must return ``y_s`` and
    its first three derivatives in z.
    """

    def __init__(self, swpc: Callable, length: float, beta: float, z_range: Tuple[float, float]):
        self.swpc = swpc
        self.length = float(length)
        self.beta = float(beta)
        self.u_range = tuple(z_range)
        self.v_range = (0.0, self.length)

    def evaluate(self, u, v) -> SurfaceDerivatives:
        z, x = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        y, y1, y2, y3 = self.swpc(z)
        tb = np.tan(self.beta)
        d = (self.length - x) * tb
        g = 1.0 / np.sqrt(1.0 + y1 * y1)
        g1 = -y1 * y2 * g ** 3
        g2 = -(y2 * y2 + y1 * y3) * g ** 3 + 3.0 * y1 * y1 * y2 * y2 * g ** 5
        zero = np.zeros_like(z)
        N = _stack(zero, g, -y1 * g)
        N1 = _stack(zero, g1, -(y1 * g1 + y2 * g))
        N2 = _stack(zero, g2, -(y1 * g2 + 2.0 * y2 * g1 + y3 * g))
        C = _stack(np.full_like(z, self.length), y, z)
        C1 = _stack(zero, y1, np.ones_like(z))
        C2 = _stack(zero, y2, zero)
        dd = d[..., None]
        S = C + dd * N
        S[..., 0] = x                                   # C_x - (length - x) = x
        Sv = -tb * N
        Sv[..., 0] += 1.0
        return SurfaceDerivatives(
            S=S,
            Su=C1 + dd * N1,
            Sv=Sv,
            Suu=C2 + dd * N2,
            Suv=-tb * N1,
            Svv=np.zeros_like(C),
        )

    @classmethod
    def from_oc_waverider(cls, wr) -> "OsculatingConeShock":
        """Shock of a WADOS osculating-cone waverider (``generator.waverider``).

        Its SWPC is flat at ``y = -height`` for ``z <= X1 width``. Outboard it
        is the quartic Bezier with control points evenly spaced in z, all at
        ``y = -height`` except the last, which is ``X2 height`` higher. That
        Bezier is exactly ``y_s = -height + X2 height xi^4`` with
        ``xi = (z - X1 width) / ((1 - X1) width)``.
        """
        H, W, X1, X2 = float(wr.height), float(wr.width), float(wr.X1), float(wr.X2)
        z0, span = X1 * W, (1.0 - X1) * W

        def swpc(z):
            xi = np.clip((np.asarray(z, dtype=float) - z0) / span, 0.0, None)
            a = X2 * H
            return (-H + a * xi ** 4, 4.0 * a * xi ** 3 / span,
                    12.0 * a * xi ** 2 / span ** 2, 24.0 * a * xi / span ** 3)

        return cls(swpc, float(wr.length), np.radians(float(wr.beta)), (0.0, W))


# ---------------------------------------------------------------------------
#  B-spline shock
# ---------------------------------------------------------------------------

class BSplineShock(ShockSurface):
    """Bicubic interpolating spline through a grid of shock points.

    ``points`` has shape (nu, nv, 3) and is sampled at the strictly increasing
    parameters ``u`` (nu,) and ``v`` (nv,). Each coordinate is a
    ``scipy.interpolate.RectBivariateSpline`` with ``s = 0``.
    """

    analytic = False                    # data: no extension beyond (u_range, v_range)

    def __init__(self, u, v, points):
        from scipy.interpolate import RectBivariateSpline

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        pts = np.asarray(points, dtype=float)
        if pts.shape != (u.size, v.size, 3):
            raise ValueError("points must have shape (len(u), len(v), 3)")
        self._f = [RectBivariateSpline(u, v, pts[..., k], kx=3, ky=3, s=0) for k in range(3)]
        self.u_range = (float(u[0]), float(u[-1]))
        self.v_range = (float(v[0]), float(v[-1]))

    def evaluate(self, u, v) -> SurfaceDerivatives:
        uu, vv = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        shape = uu.shape
        a, b = uu.ravel(), vv.ravel()

        def d(i, j):
            return np.stack([f.ev(a, b, dx=i, dy=j) for f in self._f], axis=-1).reshape(shape + (3,))

        return SurfaceDerivatives(d(0, 0), d(1, 0), d(0, 1), d(2, 0), d(1, 1), d(0, 2))

    @classmethod
    def from_surface(cls, surface: ShockSurface, nu: int = 40, nv: int = 40,
                     u_range=None, v_range=None) -> "BSplineShock":
        """Interpolating spline through ``surface`` sampled on an nu x nv grid."""
        u0, u1 = u_range or surface.u_range
        v0, v1 = v_range or surface.v_range
        u = np.linspace(u0, u1, nu)
        v = np.linspace(v0, v1, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        return cls(u, v, surface.point(U, V))
