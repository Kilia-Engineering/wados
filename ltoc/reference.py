"""Tight conical-flow reference for verifying the MOC kernel (spec V1, V4).

This is the flow behind a straight conical shock of angle ``beta`` in uniform
flow at ``M_inf``, from the Taylor-Maccoll equation (Taylor, G. I., Maccoll,
J. W., "The air pressure on a cone moving at high speeds," Proc. R. Soc. Lond.
A 139:278-311, 1933). The right-hand side is coded in the velocity-component
form of Anderson, *Modern Compressible Flow*, 3rd ed., Eq. 10.15. The pieces
are reused from the repo:

* ``liu2019.shock._taylor_maccoll_rhs`` -- the Taylor-Maccoll right-hand side,
  integrated here at rtol 1e-12. The public ``taylor_maccoll_cone_field``
  integrates at rtol 1e-6 and is accurate to about 6e-5 in flow angle, which
  is enough for the 0.1 % V1 criterion but not for a convergence-order study.
* ``gvwd.thermo.oblique_shock`` -- the post-shock state and total-pressure
  ratio at the shock.

The integration continues inward past the cone surface until the equation
becomes singular, i.e. the ray becomes a Mach line. Characteristic-net points
that fall below the cone, which the inverse MOC also computes, can then be
compared as well.

Nondimensionalisation matches ``ltoc.moc_noncoaxial``: ``p / p_inf``,
``rho / rho_inf`` and ``V / sqrt(p_inf / rho_inf)``.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["ConicalFlowReference", "osculating_cone_streamline"]


class ConicalFlowReference:
    """Taylor-Maccoll flow behind a conical shock with its apex at the origin.

    The meridian-plane frame matches the MOC kernel: x along the axis, y the
    radius, shock ``y = x tan(beta)``.
    """

    def __init__(self, M_inf: float, beta: float, gamma: float = 1.4,
                 rtol: float = 1e-12, atol: float = 1e-14):
        from gvwd.thermo.oblique_shock import rankine_hugoniot, stagnation_pressure_ratio
        from liu2019.shock import _taylor_maccoll_rhs

        self.M_inf, self.beta, self.gamma = float(M_inf), float(beta), float(gamma)
        g = self.gamma
        rh = rankine_hugoniot(self.M_inf, self.beta, 1.0, 1.0, g)
        self.delta_shock = rh["theta"]
        self.M2 = rh["M2"]
        self.p2 = rh["p_ratio"]
        self.rho2 = rh["rho_ratio"]
        # Total pressure behind the shock, p02 / p_inf (constant: the flow
        # behind a straight conical shock is isentropic).
        p01 = (1.0 + 0.5 * (g - 1.0) * self.M_inf ** 2) ** (g / (g - 1.0))
        self.p02 = stagnation_pressure_ratio(self.M_inf, self.beta, g) * p01

        V0 = 1.0 / np.sqrt(2.0 / ((g - 1.0) * self.M2 ** 2) + 1.0)
        # Third state: ln(r) along a streamline, d ln(r) / d omega = V_r / V_theta
        # (spherical coordinates: dr/dt = V_r, r d(omega)/dt = V_theta).
        y0 = [V0 * np.cos(self.beta - self.delta_shock), -V0 * np.sin(self.beta - self.delta_shock), 0.0]

        def cone(t, y):
            return y[1]

        def singular(t, y):
            A = 0.5 * (g - 1.0) * (1.0 - y[0] ** 2 - y[1] ** 2)
            return A - y[1] ** 2 - 1e-6

        singular.terminal = True
        # Flow field: inward from the shock, past the cone surface, to the
        # singular ray where the ray becomes a Mach line.
        sol = solve_ivp(lambda t, y: _taylor_maccoll_rhs(t, y, g), (self.beta, 0.15 * self.beta),
                        y0[:2], method="DOP853", rtol=rtol, atol=atol, dense_output=True,
                        events=(cone, singular))
        if not sol.t_events[0].size:
            raise ValueError("Taylor-Maccoll integration did not reach the cone surface")
        self.cone_angle = float(sol.t_events[0][0])
        self.omega_min = float(sol.t[-1])
        self._sol = sol

        # Streamline quadrature F(omega) = int V_r / V_theta d omega. It diverges
        # logarithmically at the cone (streamlines approach it asymptotically),
        # so it stops just short of the cone.
        def rhs3(t, y):
            d = _taylor_maccoll_rhs(t, y[:2], g)
            return [d[0], d[1], y[0] / y[1]]

        self._w_lnr_min = self.cone_angle + 1e-7 * self.beta
        self._lnr = solve_ivp(rhs3, (self.beta, self._w_lnr_min), y0, method="DOP853",
                              rtol=rtol, atol=atol, dense_output=True)

    # ------------------------------------------------------------------
    def covers(self, omega) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        tol = 1e-12 * self.beta
        return (omega >= self.omega_min - tol) & (omega <= self.beta + tol)

    def state(self, omega) -> dict:
        """Flow state on ray(s) ``omega`` (rad); NaN outside the covered range."""
        omega = np.asarray(omega, dtype=float)
        shape = omega.shape
        omega = np.atleast_1d(omega).ravel()
        g = self.gamma
        ok = self.covers(omega)
        w = np.where(ok, np.clip(omega, self.omega_min, self.beta), self.beta)
        Vr, Vt = self._sol.sol(w)
        Vp2 = Vr * Vr + Vt * Vt
        M = np.sqrt(2.0 / (g - 1.0) * Vp2 / (1.0 - Vp2))
        theta = w + np.arctan2(Vt, Vr)
        p = self.p02 * (1.0 + 0.5 * (g - 1.0) * M * M) ** (-g / (g - 1.0))
        rho = self.rho2 * (p / self.p2) ** (1.0 / g)
        V = M * np.sqrt(g * p / rho)
        out = {"theta": theta, "M": M, "p": p, "rho": rho, "V": V}
        for k in out:
            out[k] = np.where(ok, out[k], np.nan).reshape(shape) if shape else \
                np.where(ok, out[k], np.nan)
        return out

    def state_at(self, x, y) -> dict:
        return self.state(np.arctan2(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))

    def streamline(self, x0: float, y0: float, x_end: float, n_table: int = 4001):
        """Streamline through ``(x0, y0)`` (between the shock and the cone).

        Along a streamline ``r(omega) = r0 exp(F(omega) - F(omega0))`` with
        ``F = int V_r / V_theta d omega`` from the Taylor-Maccoll integration
        (rtol 1e-12). Returns a callable ``y(x)`` valid on ``[x0, x_end]``,
        interpolated with a cubic spline on a dense table.
        """
        from scipy.interpolate import CubicSpline

        w0 = float(np.arctan2(y0, x0))
        r0 = float(np.hypot(x0, y0))
        lo = self._w_lnr_min
        if not (lo <= w0 <= self.beta + 1e-12):
            raise ValueError("start point must lie between the cone and the shock")
        F0 = self._lnr.sol(min(w0, self.beta))[2]
        x_of = lambda w: r0 * np.exp(self._lnr.sol(w)[2] - F0) * np.cos(w)
        # Bracket omega at x_end (x grows monotonically as omega decreases).
        w_lo = lo
        if x_of(w_lo) < x_end:
            raise ValueError("x_end lies beyond the asymptotic approach to the cone")
        w = np.linspace(w0, w_lo, n_table)
        # Cluster the table near the cone, where r grows fastest.
        w = w0 + (w_lo - w0) * (1.0 - (1.0 - np.linspace(0.0, 1.0, n_table)) ** 2)
        r = r0 * np.exp(self._lnr.sol(w)[2] - F0)
        x, y = r * np.cos(w), r * np.sin(w)
        keep = x <= x_end * (1.0 + 1e-9) + 1e-12
        k = int(np.count_nonzero(keep))
        x, y = x[: min(k + 2, x.size)], y[: min(k + 2, y.size)]
        spl = CubicSpline(x, y)
        return lambda xx: spl(np.asarray(xx, dtype=float))


def osculating_cone_streamline(shock, le_uv, le_point, x, M_inf: float, gamma: float = 1.4):
    """Exact lower-surface streamline of an osculating-cone design (V5 reference).

    ``shock`` is an ``OsculatingConeShock``. In the osculating plane of the
    leading-edge point (``le_uv``, ``le_point``) the flow is Taylor-Maccoll
    flow about the local cone, whose apex lies ``r / tan(beta)`` upstream of
    the base plane on the local axis, or exact wedge flow where the shock is
    flat (r = inf). Returns the points (n, 3) at the stations ``x``, in the
    shock frame.
    """
    from gvwd.thermo.oblique_shock import theta_from_beta_M

    from .shock_geometry import local_geometry

    x = np.asarray(x, dtype=float)
    L, beta = shock.length, shock.beta
    lep = np.asarray(le_point, dtype=float)
    g = local_geometry(shock, le_uv[0], L, length_scale=L)
    if not np.isfinite(g.r):                                  # flat region: wedge flow
        th = theta_from_beta_M(beta, M_inf, gamma)
        return np.column_stack([x, lep[1] - (x - lep[0]) * np.tan(th), np.full_like(x, lep[2])])
    O, R = g.axis_centre, float(g.r)
    x_apex = L - R / np.tan(beta)
    e = lep[1:] - O[1:]
    rho_le = float(np.hypot(*e))
    rho = ConicalFlowReference(M_inf, beta, gamma).streamline(lep[0] - x_apex, rho_le,
                                                               L - x_apex)(x - x_apex)
    return np.column_stack([x, O[1] + rho * e[0] / rho_le, O[2] + rho * e[1] / rho_le])
