"""Cone basic flowfield using Taylor-Maccoll.

The initial (and currently only) concrete :class:`BasicFlowfield` subclass.
It wraps the validated Taylor-Maccoll solver from :mod:`liu2019.shock`, so the
cone half-angle and resulting compression-surface streamline are byte-for-byte
identical to what the Liu 2019 package produces -- the foundation of the
Phase 2 numerical-equivalence guarantee.

The compression-surface streamline on the cone is approximated as a straight
line at angle ``delta_c`` from the freestream (Sobieczky / Liu legacy
prescription), matching the production state of ``liu2019.osculating``.
"""

import numpy as np

from liu2019.shock import (
    DetachedShockError,
    beta_detachment,
    mach_angle,
    taylor_maccoll_cone_angle,
)

from .basic_flowfield import BasicFlowfield, StreamlineResult


class ConeFlowfield(BasicFlowfield):
    """Cone (Taylor-Maccoll) basic flowfield.

    The cone half-angle ``delta_c`` is derived from ``(Ma_inf, beta_design)``
    by integrating Taylor-Maccoll inward from the shock; the value is
    lazy-cached on the first call to :meth:`deflection_angle_deg`. The
    compression-surface streamline is a straight line at angle ``delta_c``
    in the local osculating-plane ``(x, r)`` frame.
    """

    def __init__(self, Ma_inf: float, beta_design_deg: float,
                 gamma: float = 1.4, streamline_model: str = "straight"):
        if streamline_model not in ("straight", "tm"):
            raise ValueError(
                f"streamline_model must be 'straight' or 'tm', "
                f"got {streamline_model!r}")
        super().__init__(Ma_inf, beta_design_deg, gamma)
        self._delta_c_deg = None    # lazy
        self.streamline_model = str(streamline_model)
        self._cone_field = None     # lazy (tm mode): (Vr, Vt, dc_rad, beta_rad)

    # --------------------------------------------------------------
    def name(self) -> str:
        tag = "" if self.streamline_model == "straight" else ",tm"
        return f"cone(Ma={self.Ma_inf:.2f},beta={self.beta_design_deg:.2f}{tag})"

    # --------------------------------------------------------------
    def attached_shock_check(self) -> tuple:
        """Confirm beta is between the Mach angle and the detachment angle."""
        mu_deg = mach_angle(self.Ma_inf)
        if self.beta_design_deg <= mu_deg:
            return False, (f"beta = {self.beta_design_deg:.2f} deg <= Mach "
                           f"angle mu = {mu_deg:.2f} deg")
        b_det = beta_detachment(self.Ma_inf, self.gamma)
        if self.beta_design_deg >= b_det:
            return False, (f"beta = {self.beta_design_deg:.2f} deg >= "
                           f"detachment beta_det = {b_det:.2f} deg")
        return True, "ok"

    # --------------------------------------------------------------
    def deflection_angle_deg(self) -> float:
        """Cone half-angle ``delta_c`` (degrees) from Taylor-Maccoll.

        Lazily evaluated, then cached. Identical to
        ``liu2019.shock.taylor_maccoll_cone_angle(Ma, beta, gamma)``.
        """
        if self._delta_c_deg is None:
            self._delta_c_deg = taylor_maccoll_cone_angle(
                self.Ma_inf, self.beta_design_deg, self.gamma)
        return float(self._delta_c_deg)

    # --------------------------------------------------------------
    def trace_streamline(self, x_LE: float, r_LE: float, x_end: float,
                         n_points: int = 100) -> StreamlineResult:
        """Compression-surface streamline in the local ``(x, r)`` frame.

        ``streamline_model="straight"`` (default): straight line at angle
        ``delta_c``, reproducing the Liu-2019 production prescription
        bit-for-bit. This is a *construction rule*, not the flow solution
        -- a real streamline leaves the shock at ``theta_w < delta_c`` --
        but it is crease-free, gives the largest volume of the available
        models, and is what the paper itself builds.

        ``streamline_model="tm"``: the true Taylor-Maccoll streamline,
        integrated through this plane's conical velocity field. The LE
        slope is ``theta_w`` (shock jump conditions are locally 2D) and
        steepens toward ``delta_c`` with depth into the shock layer, so a
        nearly-flat plane (huge ``R_osc``, tiny angular travel) recovers
        the wedge solution continuously -- no crease at a flat/curved
        boundary. A degenerate plane (``r_LE <= 0``, e.g. the flat-region
        anchor the sweep passes) IS that wedge limit, and is traced as a
        straight line at ``theta_w``.
        """
        if self.streamline_model == "tm":
            return self._trace_tm(x_LE, r_LE, x_end, n_points)

        delta_c_deg = self.deflection_angle_deg()
        delta_c_rad = np.radians(delta_c_deg)

        x_arr = np.linspace(float(x_LE), float(x_end), int(n_points))
        r_arr = float(r_LE) - (x_arr - float(x_LE)) * np.tan(delta_c_rad)

        return StreamlineResult(
            x_arr=x_arr,
            r_arr=r_arr,
            delta_LE_deg=delta_c_deg,
            delta_TE_deg=delta_c_deg,         # constant on a cone
            Ma_TE=self.Ma_inf,                # placeholder; refine in future phases
        )

    # --------------------------------------------------------------
    def _field(self):
        """Lazy Taylor-Maccoll velocity field for this plane's (Ma, beta)."""
        if self._cone_field is None:
            from liu2019.shock import taylor_maccoll_cone_field
            self._cone_field = taylor_maccoll_cone_field(
                self.Ma_inf, self.beta_design_deg, self.gamma)
        return self._cone_field

    def _surface_mach(self, theta_rad: float) -> float:
        """Local Mach from the normalised T-M velocity at angle theta."""
        Vr_s, Vt_s, dc_rad, beta_rad = self._field()
        th = float(np.clip(theta_rad, dc_rad, beta_rad))
        v = float(np.hypot(float(Vr_s(th)), float(Vt_s(th))))
        denom = 0.5 * (self.gamma - 1.0) * (1.0 - v * v)
        if denom <= 0.0:
            return float("nan")
        return float(np.sqrt(v * v / denom))

    def _trace_tm(self, x_LE, r_LE, x_end, n_points) -> StreamlineResult:
        from liu2019.shock import theta_from_beta_Ma
        theta_w = float(theta_from_beta_Ma(
            self.beta_design_deg, self.Ma_inf, self.gamma))

        # Degenerate plane (flat region anchor): the 2D wedge limit.
        if not (np.isfinite(r_LE) and float(r_LE) > 0.0):
            x_arr = np.linspace(float(x_LE), float(x_end), int(n_points))
            r_arr = float(0.0 if not np.isfinite(r_LE) else r_LE) \
                - (x_arr - float(x_LE)) * np.tan(np.radians(theta_w))
            from liu2019.shock import oblique_shock_ratios
            Ma2 = float(oblique_shock_ratios(
                self.Ma_inf, self.beta_design_deg, self.gamma)["Ma2"])
            return StreamlineResult(x_arr=x_arr, r_arr=r_arr,
                                    delta_LE_deg=theta_w,
                                    delta_TE_deg=theta_w, Ma_TE=Ma2)

        from liu2019.osculating import trace_tm_streamline
        Vr_s, Vt_s, dc_rad, beta_rad = self._field()
        # The LE sits on the conical shock, so the axis apex is where the
        # shock ray through (x_LE, r_LE) meets r = 0. This equals the
        # L_w - R_osc/tan(beta) that liu2019 computes, because
        # x_LE = L_w - (R_osc - r_LE)/tan(beta) in the curved branch.
        x_a = float(x_LE) - float(r_LE) / np.tan(beta_rad)
        x_arr, r_cone = trace_tm_streamline(
            float(x_LE), float(r_LE), x_a, float(x_end),
            Vr_s, Vt_s, beta_rad, dc_rad, n_samples=int(n_points))
        r_cone = np.asarray(r_cone, dtype=float)

        # ---- Cone frame -> framework frame ---------------------------
        # In the cone frame, r is measured from the osculating axis and
        # GROWS along the descending streamline (the shock ray itself has
        # dr/dx = tan(beta); the streamline runs shallower at
        # tan(theta_w)..tan(delta_c), staying inside it). The osculating
        # sweep consumes ``descent = r_LE - r_arr``, i.e. it expects r to
        # SHRINK with descent, so mirror about the entry radius:
        #
        #     descent(x) = r_cone(x) - r_LE   >= 0
        #     r_arr(x)   = r_LE - descent(x)  = 2*r_LE - r_cone(x)
        #
        # This is the same frame flip the power-law wrapper documents in
        # mfof/power_law_flowfield.py ("body increase = framework
        # descent"); back-projected, y = y_LE - descent*n_y reproduces
        # liu2019's y = y_c - r_cone*n_y identically because
        # y_LE = y_c - r_LE*n_y.
        r_arr = 2.0 * float(r_LE) - r_cone

        # End-state from the mesh-free conical field at the TE angle.
        dx_apex = max(float(x_arr[-1]) - x_a, 1e-12)
        theta_TE = float(np.arctan(max(float(r_cone[-1]), 0.0) / dx_apex))
        if len(x_arr) >= 2 and x_arr[-1] > x_arr[-2]:
            slope = (r_cone[-1] - r_cone[-2]) / (x_arr[-1] - x_arr[-2])
            delta_TE = float(np.degrees(np.arctan(slope)))
        else:
            delta_TE = float(np.degrees(dc_rad))
        return StreamlineResult(
            x_arr=np.asarray(x_arr, dtype=float),
            r_arr=r_arr,
            delta_LE_deg=theta_w,
            delta_TE_deg=delta_TE,
            Ma_TE=self._surface_mach(theta_TE),
        )
