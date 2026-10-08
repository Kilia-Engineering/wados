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

__all__ = ["ConicalFlowReference"]


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
        y0 = [V0 * np.cos(self.beta - self.delta_shock), -V0 * np.sin(self.beta - self.delta_shock)]

        def cone(t, y):
            return y[1]

        def singular(t, y):
            A = 0.5 * (g - 1.0) * (1.0 - y[0] ** 2 - y[1] ** 2)
            return A - y[1] ** 2 - 1e-6

        singular.terminal = True
        sol = solve_ivp(lambda t, y: _taylor_maccoll_rhs(t, y, g), (self.beta, 0.15 * self.beta),
                        y0, method="DOP853", rtol=rtol, atol=atol, dense_output=True,
                        events=(cone, singular))
        if not sol.t_events[0].size:
            raise ValueError("Taylor-Maccoll integration did not reach the cone surface")
        self.cone_angle = float(sol.t_events[0][0])
        self.omega_min = float(sol.t[-1])
        self._sol = sol

    # ------------------------------------------------------------------
    def covers(self, omega) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        return (omega >= self.omega_min) & (omega <= self.beta)

    def state(self, omega) -> dict:
        """Flow state on ray(s) ``omega`` (rad); NaN outside the covered range."""
        omega = np.asarray(omega, dtype=float)
        shape = omega.shape
        omega = np.atleast_1d(omega).ravel()
        g = self.gamma
        ok = self.covers(omega)
        w = np.where(ok, omega, self.beta)
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

    def streamline(self, x0: float, y0: float, x_end: float):
        """Streamline through ``(x0, y0)`` from Taylor-Maccoll (rtol 1e-12).

        Returns a dense callable ``y(x)`` valid on ``[x0, x_end]``.
        """
        def rhs(x, y):
            return [np.tan(float(self.state(np.arctan2(y[0], x))["theta"][0]))]

        sol = solve_ivp(rhs, (x0, x_end), [y0], method="DOP853", rtol=1e-12, atol=1e-14,
                        dense_output=True)
        return lambda x: sol.sol(np.asarray(x, dtype=float))[0]
