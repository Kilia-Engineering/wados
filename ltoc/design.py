"""LTOCs design inputs and results for the GUI tab (spec Phase 5).

This module has no Qt dependency, so the GUI logic can be tested in CI. A
design is one of two kinds:

* **Elliptic ruled shock** (R1 Eqs. 12 and 15):
  ``S(m, n) = (1 - n) p(m) + n q(m)`` with ``p(m) = (x_p, a_p cos m, b_p sin m)``
  and ``q(m) = (x_q, a_q cos m, b_q sin m)``. It is given in R1's frame
  (X freestream, Y span, Z down), with a flow capture tube (FCT) ``Z = fct_z``
  in the base plane ``X = x_q`` (R1 Eqs. 24 and 26). It covers R1's
  waverider cases I and II (Eqs. 23 and 25), test case II (Eq. 18) and
  circular cones.
* **Osculating-cone import**: the shock of a WADOS OC design
  (``OsculatingConeShock.from_oc_waverider``), with that design's leading
  edge as the FCT. LTOCs then reproduces the OC waverider (V5).

``LTOCDesign.build`` returns an ``LTOCResult``. It holds the waverider, its
forces (R1 Eqs. 19-22, planform and wetted reference areas), the lower-surface
grid with p/p_inf, and the stations that were refused, with their messages.
All plotting data are in the GUI frame (x streamwise, y up, z span).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .forces import (crosswise_section, lower_surface_grid, panel_forces,
                     streamwise_section)
from .ltoc import identity_frame, r1_frame_to_gui
from .shock_surface import OsculatingConeShock, RuledShock
from .waverider import LTOCWaverider

__all__ = ["EllipticShock", "PRESETS", "LTOCDesign", "LTOCResult"]


@dataclass
class EllipticShock:
    """Elliptic ruled shock of R1 Eq. (15) form, in R1's frame (Z down)."""

    x_p: float
    a_p: float
    b_p: float
    x_q: float
    a_q: float
    b_q: float

    def surface(self) -> RuledShock:
        return RuledShock.elliptic(self.x_p, self.a_p, self.b_p, self.x_q, self.a_q, self.b_q,
                                   m_range=(0.0, np.pi), n_range=(0.0, 1.0))

    def tip_y(self, fct_z: float) -> float:
        """Span at which the FCT ``Z = fct_z`` meets the shock in the base plane."""
        if not 0.0 < fct_z < self.b_q:
            raise ValueError(f"FCT Z = {fct_z:g} must lie between 0 and b_q = {self.b_q:g}")
        return self.a_q * np.sqrt(1.0 - (fct_z / self.b_q) ** 2)


def _cone(beta_deg: float, length: float = 1.0) -> EllipticShock:
    r = length * np.tan(np.radians(beta_deg))
    return EllipticShock(0.0, 0.0, 0.0, length, r, r)


#: name -> (shock, M_inf, FCT Z, reference)
PRESETS = {
    "R1 waverider case I (Eq. 23), M 6": (EllipticShock(0.0, 0.0, 0.0, 1.7, 0.595, 0.68), 6.0, 0.12,
                                          "R1 Sec. IV.B, Eqs. (23)-(24); R1 Table 2"),
    "R1 waverider case II (Eq. 25), M 7": (EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 0.8), 7.0, 0.25,
                                           "R1 Sec. IV.C, Eqs. (25)-(26); R1 Table 3"),
    "R1 test case II shock (Eq. 18), M 7": (EllipticShock(0.1, 0.2, 0.3, 1.5, 0.8, 0.65), 7.0, 0.35,
                                            "R1 Sec. III.B, Eqs. (16)-(18), with an FCT"),
    "Circular cone, beta 12 deg, M 6": (_cone(12.0), 6.0, 0.5 * np.tan(np.radians(12.0)),
                                        "cone-derived waverider (V4)"),
}


@dataclass
class LTOCResult:
    design: "LTOCDesign"
    waverider: LTOCWaverider
    elapsed: float
    forces: Optional[object] = None          # ltoc.forces.PanelForces (planform reference)
    grid: Optional[object] = None            # ltoc.forces.LowerSurfaceGrid
    refusals: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.waverider.ok

    # -- plotting data, GUI frame -------------------------------------
    def shock_curves(self, every: int = 1, to_base: bool = True) -> list:
        """Shock curves of the stream surfaces, GUI frame, LE to base."""
        out = []
        for st in self.waverider.stations[::every]:
            if st.curve is None:
                continue
            keep = st.curve.x <= self.waverider.x_base * (1 + 1e-12) if to_base else slice(None)
            out.append(self.waverider.to_gui(st.curve.points[keep]))
        return out

    def base_shock(self, n: int = 200) -> np.ndarray:
        """Shock trace in the base plane (half span), GUI frame."""
        s = self.waverider.shock
        u = np.linspace(*s.u_range, n)
        v = np.full_like(u, s.v_range[1])
        P = self.waverider.to_gui(s.point(u, v))
        return P[P[:, 2] >= -1e-12]

    def streamwise_sections(self, fractions=(0.0, 1.0 / 3.0, 2.0 / 3.0)) -> dict:
        """Wall pressure on planes z = const (GUI span), at fractions of the tip span."""
        z_tip = float(np.max(self.grid.points[..., 2]))
        out = {}
        for f in fractions:
            x, p = streamwise_section(self.grid.points, self.grid.p, f * z_tip, span_axis=2)
            out[f * z_tip] = (x, p)
        return out

    def crosswise_sections(self, fractions=(0.5, 0.75, 1.0)) -> dict:
        """Wall pressure on planes x = const, at fractions between the first LE and the base."""
        x0, x1 = float(np.min(self.grid.points[..., 0])), self.waverider.x_base
        out = {}
        for f in fractions:
            xc = x0 + f * (x1 - x0)
            z, p = crosswise_section(self.grid.points, self.grid.p, xc, span_axis=2)
            out[xc] = (z, p)
        return out

    def station_diagnostics(self) -> list:
        """Per-station diagnostics (spec section 6) for the diagnostics view."""
        wr = self.waverider
        t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wr.fct_yz, axis=0), axis=1))])
        rows = []
        for st, tk in zip(wr.stations, t / max(t[-1], 1e-300)):
            if not st.ok or st.status == "tip":
                continue
            cv, d = st.curve, st.diagnostics
            m_base = np.interp(wr.x_base, cv.x, cv.uv[:, 0])
            ya_base = np.interp(wr.x_base, cv.x, cv.y_axis)
            rows.append({
                "fct_fraction": float(tk),
                "azimuth_drift_deg": float(np.degrees(abs(m_base - cv.uv[0, 0]))),
                "axis_drift_over_L": float(abs(ya_base - cv.y_axis[0]) / wr.x_base),
                "extension_needed": d["extension_needed_chords"],
                "extension_used": d["extension_chords"],
                "body_step_ratio": d["body_step_ratio"],
                "beta_min_deg": d["beta_deg"][0], "beta_max_deg": d["beta_deg"][1],
                "r_min": d["r_min"],
            })
        return rows

    def summary(self) -> str:
        wr = self.waverider
        n = len(wr.stations)
        n_ok = sum(st.ok for st in wr.stations)
        lines = [f"LTOCs waverider: {n_ok}/{n} stations solved in {self.elapsed:.1f} s"]
        if self.forces is not None:
            f = self.forces
            lines += [f"Length {wr.length:.4g}, half span {wr.width:.4g}, height {wr.height:.4g}",
                      f"CL {f.CL:.4f}  CD {f.CD:.4f}  L/D {f.L_over_D:.3f}  (planform reference, "
                      f"A = {2 * f.area_planform:.4g} full span)",
                      f"CL {f.CL_wetted:.4f}  CD {f.CD_wetted:.4f}  (wetted reference, "
                      f"A = {2 * f.area_wetted:.4g} full span)",
                      f"Spanwise smoothness {wr.spanwise_smoothness():.2e} L"]
        for k, status, msg in self.refusals:
            lines.append(f"station {k}: {status}: {msg}")
        return "\n".join(lines)


@dataclass
class LTOCDesign:
    """Inputs of one LTOCs waverider design (see module docstring)."""

    shock: Optional[EllipticShock] = None
    M_inf: float = 6.0
    fct_z: float = 0.12
    n_stations: int = 25
    clustering: str = "symmetry"             # "symmetry" or "uniform"
    n_points: int = 40
    gamma: float = 1.4
    oc_waverider: Optional[object] = None    # an OC design to import instead of ``shock``

    @classmethod
    def from_preset(cls, name: str, **kw) -> "LTOCDesign":
        shock, M, z, _ = PRESETS[name]
        return cls(shock=shock, M_inf=M, fct_z=z, **kw)

    def fct(self) -> np.ndarray:
        """FCT points (Y, Z) in the shock frame, station 0 on the symmetry plane."""
        if self.oc_waverider is not None:
            return np.asarray(self.oc_waverider.leading_edge, dtype=float)[:, [1, 2]]
        y_tip = self.shock.tip_y(self.fct_z)
        u = np.linspace(0.0, 1.0, self.n_stations)
        if self.clustering == "symmetry":                # Gate 4 report section 2
            u = 1.0 - np.cos(0.5 * np.pi * u)
        elif self.clustering != "uniform":
            raise ValueError("clustering must be 'symmetry' or 'uniform'")
        return np.column_stack([y_tip * u, np.full_like(u, self.fct_z)])

    def validate(self) -> None:
        if self.oc_waverider is None and self.shock is None:
            raise ValueError("no shock surface given")
        if self.M_inf <= 1.0:
            raise ValueError("the design Mach number must be supersonic")
        if self.n_stations < 3 or self.n_points < 10:
            raise ValueError("need at least 3 stations and 10 points per stream surface")
        if self.oc_waverider is None:
            s = self.shock
            if min(s.a_q, s.b_q) <= 0.0 or s.x_q <= s.x_p:
                raise ValueError("the base ellipse must be non-degenerate and downstream of p(m)")
            self.shock.tip_y(self.fct_z)

    def build(self, progress: Optional[Callable] = None) -> LTOCResult:
        """Solve every stream surface; refusals are recorded, not raised."""
        self.validate()
        t0 = time.time()
        if self.oc_waverider is not None:
            oc = self.oc_waverider
            shock, x_base, M, to_gui = OsculatingConeShock.from_oc_waverider(oc), oc.length, \
                float(oc.M_inf), identity_frame
        else:
            shock, x_base, M, to_gui = self.shock.surface(), self.shock.x_q, self.M_inf, \
                r1_frame_to_gui
        wr = LTOCWaverider(shock, self.fct(), x_base, M, gamma=self.gamma, n_points=self.n_points,
                           to_gui=to_gui, strict=False, label="ltoc", progress=progress)
        res = LTOCResult(design=self, waverider=wr, elapsed=0.0, refusals=wr.failures())
        if wr.ok:
            res.grid = lower_surface_grid(wr, 101)
            res.forces = panel_forces(res.grid, M, self.gamma)
        res.elapsed = time.time() - t0
        return res
