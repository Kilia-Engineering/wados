"""LTOCs design inputs and results for the GUI tab (spec Phase 5).

This module has no Qt dependency, so the GUI logic can be tested in CI. A
design takes its shock from one of three sources:

* **Elliptic ruled shock** (R1 Eqs. 12 and 15):
  ``S(m, n) = (1 - n) p(m) + n q(m)`` with ``p(m) = (x_p, a_p cos m, b_p sin m)``
  and ``q(m) = (x_q, a_q cos m, b_q sin m)``. It is given in R1's frame
  (X freestream, Y span, Z down), with a flow capture tube (FCT) ``Z = fct_z``
  in the base plane ``X = x_q`` (R1 Eqs. 24 and 26). It covers R1's
  waverider cases I and II (Eqs. 23 and 25), test case II (Eq. 18) and
  circular cones.
* **Shock data** (``ShockData``): points on a structured (u, v) grid, read
  from a CSV file and interpolated by a bicubic spline (``BSplineShock``).
  The frame and FCT are as above; the base plane ``X = x_base`` is an
  input. Data are never extrapolated, so they must extend past the base
  plane (spec flag 4).
* **Osculating-cone import**: the shock of a WADOS OC design
  (``OsculatingConeShock.from_oc_waverider``), with that design's leading
  edge as the FCT. LTOCs then reproduces the OC waverider (V5).

``LTOCDesign.build`` returns an ``LTOCResult``. It holds the waverider, its
forces (R1 Eqs. 19-22, planform and wetted reference areas), the lower-surface
grid with p/p_inf, and the stations that were refused, with their messages.
All plotting data are in the GUI frame (x streamwise, y up, z span).
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Callable, ClassVar, Optional

import numpy as np

from .forces import (crosswise_section, lower_surface_grid, panel_forces,
                     streamwise_section)
from .ltoc import identity_frame, r1_frame_to_gui
from .shock_surface import BSplineShock, OsculatingConeShock, RuledShock, ShockSurface
from .waverider import LTOCWaverider

__all__ = ["EllipticShock", "ShockData", "PRESETS", "PRESET_ALIASES", "LTOCDesign", "LTOCResult",
           "base_trace", "fct_tip_y"]


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

    def to_data(self, n_max: float = 2.5, nu: int = 61, nv: int = 41, **meta) -> "ShockData":
        """Sample this shock for a data file: m in [0, pi], n in [0, n_max]. With
        ``n_max`` > 1 the data extend past the base plane (n = 1), as LTOCs
        needs (spec flag 4); 2.5 covers the extensions of R1's cases."""
        return ShockData.sample(self.surface(), (0.0, np.pi), (0.0, n_max), nu, nv,
                                meta=dict(x_base=self.x_q, **meta))

    def tip_y(self, fct_z: float) -> float:
        """Span at which the FCT ``Z = fct_z`` meets the shock in the base plane."""
        if not 0.0 < fct_z < self.b_q:
            raise ValueError(f"FCT Z = {fct_z:g} must lie between 0 and b_q = {self.b_q:g} (the base "
                             f"half-height), so that the FCT meets the base ellipse")
        return self.a_q * np.sqrt(1.0 - (fct_z / self.b_q) ** 2)


# ---------------------------------------------------------------------------
#  Shock data (B-spline input)
# ---------------------------------------------------------------------------

#: Metadata keys a shock data file may carry in "# key = value" lines.
SHOCK_FILE_META = ("x_base", "M_inf", "fct_z")


@dataclass(eq=False)
class ShockData:
    """Shock surface given as points on a structured (u, v) grid.

    ``points[i, j]`` is the shock point (X, Y, Z) at parameters ``u[i]``,
    ``v[j]``, both strictly increasing, at least 4 of each. The frame is
    R1's: X freestream, Y span (symmetry plane Y = 0), Z down, so the body
    lies at larger Z than the FCT. ``v`` should run downstream. The surface
    is the interpolating bicubic spline through the points (``BSplineShock``).
    It is never extrapolated, so the data must extend past the base plane
    far enough to determine the body there: typically 0.5 to 1.5 local
    chords (spec flag 4; the Gate 3 report gives the needed extensions).

    ``meta`` holds optional design inputs read from the file
    (``SHOCK_FILE_META``).
    """

    u: np.ndarray
    v: np.ndarray
    points: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.u = np.asarray(self.u, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        self.points = np.asarray(self.points, dtype=float)
        nu, nv = self.u.size, self.v.size
        if self.u.ndim != 1 or self.v.ndim != 1 or self.points.shape != (nu, nv, 3):
            raise ValueError(f"shock data: points must have shape (len(u), len(v), 3) = "
                             f"({nu}, {nv}, 3), got {self.points.shape}")
        if nu < 4 or nv < 4:
            raise ValueError(f"shock data: need at least 4 values of u and of v for a bicubic "
                             f"spline, got {nu} x {nv}")
        if np.any(np.diff(self.u) <= 0) or np.any(np.diff(self.v) <= 0):
            raise ValueError("shock data: u and v must be strictly increasing")
        if not np.all(np.isfinite(self.points)):
            raise ValueError("shock data: the points contain NaN or infinite values")
        self._surface = None

    def __eq__(self, other):
        return (isinstance(other, ShockData) and np.array_equal(self.u, other.u)
                and np.array_equal(self.v, other.v) and np.array_equal(self.points, other.points))

    @property
    def shape(self):
        return self.points.shape[:2]

    @property
    def x_range(self):
        return float(self.points[..., 0].min()), float(self.points[..., 0].max())

    def surface(self) -> BSplineShock:
        if self._surface is None:
            self._surface = BSplineShock(self.u, self.v, self.points)
        return self._surface

    @classmethod
    def sample(cls, surface: ShockSurface, u_range, v_range, nu: int = 61, nv: int = 41,
               name: str = "", meta: Optional[dict] = None) -> "ShockData":
        """Sample a shock surface on a uniform nu x nv parameter grid."""
        u = np.linspace(*u_range, nu)
        v = np.linspace(*v_range, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        return cls(u, v, surface.point(U, V), name=name, meta=dict(meta or {}))

    # -- files ----------------------------------------------------------
    @classmethod
    def read_csv(cls, path) -> "ShockData":
        """Read a shock data file.

        Format: comma-separated, one row per grid point, with a header row
        naming the columns ``u, v, X, Y, Z`` (any order, case-insensitive);
        without a header the columns are taken in that order. Lines starting
        with ``#`` are comments; ``# x_base = 1.5``, ``# M_inf = 7`` and
        ``# fct_z = 0.25`` are read as design inputs. The points must form a
        full u x v grid. Errors name the offending line.
        """
        from pathlib import Path

        path = Path(path)
        meta, header, rows = {}, None, []
        with open(path, newline="") as f:
            for lineno, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                if text.startswith("#"):
                    key, sep, val = text[1:].partition("=")
                    key = key.strip()
                    match = [k for k in SHOCK_FILE_META if k.lower() == key.lower()]
                    if sep and match:
                        try:
                            meta[match[0]] = float(val)
                        except ValueError:
                            raise ValueError(f"{path.name}, line {lineno}: {match[0]} must be a "
                                             f"number, got {val.strip()!r}") from None
                    continue
                cells = [c.strip() for c in next(csv.reader([text]))]
                if header is None and not rows and any(c[:1].isalpha() for c in cells):
                    header = [c.lower() for c in cells]
                    missing = [c for c in ("u", "v", "x", "y", "z") if c not in header]
                    if missing:
                        raise ValueError(f"{path.name}, line {lineno}: the header must name the "
                                         f"columns u, v, X, Y, Z; missing {', '.join(missing)}")
                    continue
                try:
                    values = [float(c) for c in cells]
                except ValueError:
                    raise ValueError(f"{path.name}, line {lineno}: expected numbers, got "
                                     f"{text!r}") from None
                width = len(header) if header else 5
                if len(values) != width:
                    raise ValueError(f"{path.name}, line {lineno}: expected {width} values, "
                                     f"got {len(values)}")
                rows.append(values)
        if not rows:
            raise ValueError(f"{path.name}: no data rows")
        data = np.asarray(rows)
        cols = [header.index(c) for c in ("u", "v", "x", "y", "z")] if header else range(5)
        uc, vc, X, Y, Z = (data[:, c] for c in cols)
        u, iu = np.unique(uc, return_inverse=True)
        v, iv = np.unique(vc, return_inverse=True)
        if u.size * v.size != len(data):
            raise ValueError(f"{path.name}: the points must form a full u x v grid: "
                             f"{u.size} values of u and {v.size} of v need {u.size * v.size} "
                             f"rows, the file has {len(data)}")
        pts = np.full((u.size, v.size, 3), np.nan)
        pts[iu, iv] = np.column_stack([X, Y, Z])
        if np.isnan(pts).any():
            i, j = np.argwhere(np.isnan(pts[..., 0]))[0]
            raise ValueError(f"{path.name}: the grid point u = {u[i]:g}, v = {v[j]:g} is "
                             f"missing (another is repeated)")
        return cls(u, v, pts, name=path.name, meta=meta)

    def write_csv(self, path) -> None:
        """Write the data in the format read by :meth:`read_csv`."""
        with open(path, "w", newline="") as f:
            f.write("# LTOCs shock surface data (WADOS)\n")
            f.write("# Frame: R1 (X freestream, Y span, Z down); one row per (u, v) grid point;\n")
            f.write("# v runs downstream. The data must extend past the base plane.\n")
            for key in SHOCK_FILE_META:
                if key in self.meta:
                    f.write(f"# {key} = {self.meta[key]:.10g}\n")
            w = csv.writer(f)
            w.writerow(["u", "v", "X", "Y", "Z"])
            for i, ui in enumerate(self.u):
                for j, vj in enumerate(self.v):
                    w.writerow([repr(float(ui)), repr(float(vj))]
                               + [repr(float(c)) for c in self.points[i, j]])

    # -- project files ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "u": self.u.tolist(), "v": self.v.tolist(),
                "points": self.points.tolist(), "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict) -> "ShockData":
        try:
            return cls(d["u"], d["v"], d["points"], name=str(d.get("name", "")),
                       meta=dict(d.get("meta", {})))
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed LTOCs shock data entry: {e!r}") from None


def base_trace(surface: ShockSurface, x_base: float, n: int = 401) -> np.ndarray:
    """Points (k, 3) where the shock meets the plane X = x_base, ordered by u.

    For every u the first crossing along v is found on a sample grid and
    refined by Newton's method. Values of u whose v-line does not reach the
    plane are left out.
    """
    uv = _trace_parameters(surface, x_base, np.linspace(*surface.u_range, n))
    uv = uv[np.isfinite(uv[:, 1])]
    if uv.shape[0] == 0:
        return np.zeros((0, 3))
    return surface.point(uv[:, 0], uv[:, 1])


def fct_tip_y(surface: ShockSurface, x_base: float, fct_z: float) -> float:
    """Span at which the FCT ``Z = fct_z`` meets the shock in the base plane
    (R1's frame, half span Y >= 0; the outermost crossing)."""
    P = base_trace(surface, x_base)
    P = P[P[:, 1] >= -1e-12]
    if P.shape[0] < 2:
        raise ValueError(f"the shock does not reach the base plane X = {x_base:g}")
    if not np.any(np.sign(P[:-1, 2] - fct_z) * np.sign(P[1:, 2] - fct_z) <= 0):
        raise ValueError(f"the FCT Z = {fct_z:g} does not meet the shock in the base plane "
                         f"X = {x_base:g} (the shock trace there spans Z = "
                         f"{P[:, 2].min():.4g} to {P[:, 2].max():.4g})")
    # Refine the crossing on the trace itself (u solved by Brent's method, v by
    # Newton in the base plane), so the tip station gets exactly zero chord.
    from scipy.optimize import brentq

    u_all = np.linspace(*surface.u_range, 401)
    trace_u = _trace_parameters(surface, x_base, u_all)
    keep = np.isfinite(trace_u[:, 1])
    U, Vb = trace_u[keep, 0], trace_u[keep, 1]
    pts = surface.point(U, Vb)
    half = pts[:, 1] >= -1e-12
    U, Vb, pts = U[half], Vb[half], pts[half]
    g = pts[:, 2] - fct_z
    k = np.flatnonzero(np.sign(g[:-1]) * np.sign(g[1:]) <= 0)

    def z_minus_fct(uu):
        vv = _trace_parameters(surface, x_base, np.array([uu]))[0, 1]
        return float(surface.point(uu, vv)[2] - fct_z)

    ys = []
    for i in k:
        if g[i] == 0.0:
            u_star = U[i]
        else:
            u_star = brentq(z_minus_fct, U[i], U[i + 1], xtol=1e-15, rtol=4 * np.finfo(float).eps)
        v_star = _trace_parameters(surface, x_base, np.array([u_star]))[0, 1]
        ys.append(float(surface.point(u_star, v_star)[1]))
    return float(max(ys))


def _trace_parameters(surface: ShockSurface, x_base: float, u: np.ndarray) -> np.ndarray:
    """(u, v) on the base plane X = x_base for the given u (v = NaN where the
    v-line does not reach the plane): first crossing on a sample grid, then
    Newton's method on X(u, v) = x_base."""
    vs = np.linspace(*surface.v_range, 201)
    U, V = np.meshgrid(u, vs, indexing="ij")
    F = surface.point(U, V)[..., 0] - x_base
    cross = (F[:, :-1] < 0.0) & (F[:, 1:] >= 0.0)
    has = cross.any(axis=1)
    out = np.column_stack([u, np.full(u.shape, np.nan)])
    if not has.any():
        return out
    rows = np.flatnonzero(has)
    j = np.argmax(cross[rows], axis=1)
    f0, f1 = F[rows, j], F[rows, j + 1]
    v = vs[j] + (vs[j + 1] - vs[j]) * f0 / (f0 - f1)
    uu = u[rows]
    for _ in range(12):
        d = surface.evaluate(uu, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(d.Sv[:, 0] != 0.0, (d.S[:, 0] - x_base) / d.Sv[:, 0], 0.0)
        v = np.clip(v - step, *surface.v_range)
    out[rows, 1] = v
    return out


def _cone(beta_deg: float, length: float = 1.0) -> EllipticShock:
    r = length * np.tan(np.radians(beta_deg))
    return EllipticShock(0.0, 0.0, 0.0, length, r, r)


#: name -> (shock, M_inf, FCT Z, description shown under the list)
PRESETS = {
    "Elliptic cone shock, Mach 6 (published case I)": (
        EllipticShock(0.0, 0.0, 0.0, 1.7, 0.595, 0.68), 6.0, 0.12,
        "Waverider case I of Zheng et al. (2020): an elliptic cone shock with its apex at "
        "X = 0 and a base ellipse 0.595 wide x 0.68 high (half-axes) at X = 1.7. "
        "Published: CL 0.2205, CD 0.0750, L/D 2.94."),
    "Shape-changing elliptic shock, Mach 7 (published case II)": (
        EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 0.8), 7.0, 0.25,
        "Waverider case II of Zheng et al. (2020): the shock cross-section changes from a wide "
        "ellipse (0.3 x 0.2 at X = 0.1) to a tall one (0.65 x 0.8 at the base, X = 1.5). "
        "Published: CL 0.2034, CD 0.0648, L/D 3.14."),
    "Shape-changing elliptic shock, Mach 7 (published test shock)": (
        EllipticShock(0.1, 0.2, 0.3, 1.5, 0.8, 0.65), 7.0, 0.35,
        "Test shock of Zheng et al. (2020): the cross-section changes from a tall ellipse "
        "(0.2 x 0.3 at X = 0.1) to a wide one (0.8 x 0.65 at X = 1.5). Used here with the "
        "FCT at Z = 0.35."),
    "Circular cone shock 12°, Mach 6": (
        _cone(12.0), 6.0, 0.5 * np.tan(np.radians(12.0)),
        "A circular cone shock with a 12° half-angle. LTOCs then gives the classic "
        "cone-derived waverider."),
}

#: Earlier preset names, still accepted in project files and ``from_preset``.
PRESET_ALIASES = {
    "R1 waverider case I (Eq. 23), M 6": "Elliptic cone shock, Mach 6 (published case I)",
    "R1 waverider case II (Eq. 25), M 7": "Shape-changing elliptic shock, Mach 7 (published case II)",
    "R1 test case II shock (Eq. 18), M 7":
        "Shape-changing elliptic shock, Mach 7 (published test shock)",
    "Circular cone, beta 12 deg, M 6": "Circular cone shock 12°, Mach 6",
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
        P = self.waverider.to_gui(base_trace(self.waverider.shock, self.waverider.x_base, n))
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

    def info(self, title: str = "LTOCs") -> dict:
        """Characteristics for the information panel over the 3-D view (full span,
        same layout as the GVWD and PSWR-1 tabs). Volume from the closed mesh;
        volumetric efficiency = V^(2/3) / planform area."""
        wr, f = self.waverider, self.forces
        mesh = wr.mesh(full_span=True)
        tri = mesh.vertices[mesh.faces]
        volume = abs(float(np.einsum("ij,ij->i", tri[:, 0],
                                     np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)
        planform = 2.0 * f.area_planform
        return {
            "__title__": title,
            "M_design": f"{wr.M_inf:.2f}",
            "Length": f"{wr.length:.4f} m",
            "Span": f"{2.0 * wr.width:.4f} m",
            "Height": f"{wr.height:.4f} m",
            "Volume": f"{volume:.4g} m^3",
            "Planform": f"{planform:.4g} m^2",
            "Volumetric Efficiency": f"{volume ** (2.0 / 3.0) / planform:.4f}",
            "CL": f"{f.CL:.4f}",
            "CD": f"{f.CD:.4f}",
            "L/D": f"{f.L_over_D:.3f}",
            "Stations": f"{len(wr.stations)} x {wr.n_points} pts",
        }

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
    shock_data: Optional[ShockData] = None   # shock data (B-spline) instead of ``shock``
    x_base: Optional[float] = None           # base plane for ``shock_data``

    @classmethod
    def from_preset(cls, name: str, **kw) -> "LTOCDesign":
        shock, M, z, _ = PRESETS[PRESET_ALIASES.get(name, name)]
        return cls(shock=shock, M_inf=M, fct_z=z, **kw)

    def fct(self) -> np.ndarray:
        """FCT points (Y, Z) in the shock frame, station 0 on the symmetry plane."""
        if self.oc_waverider is not None:
            return np.asarray(self.oc_waverider.leading_edge, dtype=float)[:, [1, 2]]
        if self.shock_data is not None:
            y_tip = fct_tip_y(self.shock_data.surface(), self.x_base, self.fct_z)
        else:
            y_tip = self.shock.tip_y(self.fct_z)
        u = np.linspace(0.0, 1.0, self.n_stations)
        if self.clustering == "symmetry":                # Gate 4 report section 2
            u = 1.0 - np.cos(0.5 * np.pi * u)
        elif self.clustering != "uniform":
            raise ValueError("clustering must be 'symmetry' or 'uniform'")
        return np.column_stack([y_tip * u, np.full_like(u, self.fct_z)])

    #: Version of the dictionary written by :meth:`to_dict`.
    DICT_VERSION: ClassVar[int] = 1

    def to_dict(self, source: Optional[str] = None) -> dict:
        """JSON-serialisable inputs, for WADOS project files (section "ltoc_waverider").

        ``source`` records where the shock came from in the GUI: a preset name,
        "custom", "bspline" or "oc_import". The shock numbers are always stored,
        so a design still loads if a preset is later renamed. Shock data are
        embedded, so the project does not depend on the data file. An imported
        OC design is not stored here; its parameters are in the project's own
        "oc_waverider" section, and the import is redone after loading.
        """
        if source is None:
            source = ("oc_import" if self.oc_waverider is not None
                      else "bspline" if self.shock_data is not None else "custom")
        return {
            "version": self.DICT_VERSION,
            "source": source,
            "shock": asdict(self.shock) if self.shock is not None else None,
            "M_inf": float(self.M_inf),
            "fct_z": float(self.fct_z),
            "n_stations": int(self.n_stations),
            "clustering": self.clustering,
            "n_points": int(self.n_points),
            "gamma": float(self.gamma),
            "x_base": None if self.x_base is None else float(self.x_base),
            "shock_data": self.shock_data.to_dict() if self.shock_data is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict, oc_waverider: Optional[object] = None) -> "LTOCDesign":
        """Inverse of :meth:`to_dict`.

        Missing entries take the defaults and unknown entries are ignored. A
        design saved with source "oc_import" gets ``oc_waverider`` if one is
        given; otherwise only its numbers are restored. Raises ``ValueError``
        for a newer format version or a malformed shock entry.
        """
        version = int(d.get("version", 1))
        if version > cls.DICT_VERSION:
            raise ValueError(f"LTOCs design format version {version} is newer than this "
                             f"WADOS supports ({cls.DICT_VERSION})")
        shock = d.get("shock")
        if shock is not None:
            try:
                shock = EllipticShock(**{f.name: float(shock[f.name]) for f in fields(EllipticShock)})
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"malformed LTOCs shock entry: {e!r}") from None
        base = cls()
        design = cls(shock=shock,
                     M_inf=float(d.get("M_inf", base.M_inf)),
                     fct_z=float(d.get("fct_z", base.fct_z)),
                     n_stations=int(d.get("n_stations", base.n_stations)),
                     clustering=str(d.get("clustering", base.clustering)),
                     n_points=int(d.get("n_points", base.n_points)),
                     gamma=float(d.get("gamma", base.gamma)))
        if d.get("x_base") is not None:
            design.x_base = float(d["x_base"])
        if d.get("shock_data") is not None:
            design.shock_data = ShockData.from_dict(d["shock_data"])
        if d.get("source") == "oc_import":
            design.oc_waverider = oc_waverider
        return design

    def validate(self) -> None:
        if self.oc_waverider is None and self.shock is None and self.shock_data is None:
            raise ValueError("no shock surface given")
        if self.M_inf <= 1.0:
            raise ValueError("the design Mach number must be supersonic")
        if self.n_stations < 3 or self.n_points < 10:
            raise ValueError("need at least 3 stations and 10 points per stream surface")
        if self.oc_waverider is not None:
            return
        if self.shock_data is not None:
            x0, x1 = self.shock_data.x_range
            if self.x_base is None:
                raise ValueError("shock data: the base plane X is not set")
            if not x0 < self.x_base < x1:
                raise ValueError(f"shock data: the base plane X = {self.x_base:g} must lie inside "
                                 f"the data, X = {x0:.4g} to {x1:.4g}, and the data must extend "
                                 f"past it")
            fct_tip_y(self.shock_data.surface(), self.x_base, self.fct_z)
        else:
            s = self.shock
            if min(s.a_q, s.b_q) <= 0.0 or s.x_q <= s.x_p:
                raise ValueError("the base ellipse must be non-degenerate (a positive half-width "
                                 "and half-height) and downstream of the upstream ellipse "
                                 f"(base X = {s.x_q:g}, upstream X = {s.x_p:g})")
            self.shock.tip_y(self.fct_z)

    def base_x(self) -> float:
        """Base-plane position of this design."""
        if self.oc_waverider is not None:
            return float(self.oc_waverider.length)
        return float(self.x_base) if self.shock_data is not None else float(self.shock.x_q)

    def build(self, progress: Optional[Callable] = None) -> LTOCResult:
        """Solve every stream surface; refusals are recorded, not raised."""
        self.validate()
        t0 = time.time()
        if self.oc_waverider is not None:
            oc = self.oc_waverider
            shock, x_base, M, to_gui = OsculatingConeShock.from_oc_waverider(oc), oc.length, \
                float(oc.M_inf), identity_frame
        else:
            surface = self.shock_data.surface() if self.shock_data is not None else self.shock.surface()
            shock, x_base, M, to_gui = surface, self.base_x(), self.M_inf, r1_frame_to_gui
        wr = LTOCWaverider(shock, self.fct(), x_base, M, gamma=self.gamma, n_points=self.n_points,
                           to_gui=to_gui, strict=False, label="ltoc", progress=progress)
        res = LTOCResult(design=self, waverider=wr, elapsed=0.0, refusals=wr.failures())
        if wr.ok:
            res.grid = lower_surface_grid(wr, 101)
            res.forces = panel_forces(res.grid, M, self.gamma)
        res.elapsed = time.time() - t0
        return res
