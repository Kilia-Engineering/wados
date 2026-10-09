"""LTOCs waverider assembly (R1 Sec. IV.A; spec Phase 3).

* Lower surface: the body streamlines of the stream surfaces, one per
  leading-edge point, from the leading edge to the base plane.
* Upper surface: the freestream surface through the leading edge, i.e. the
  FCT projected forward (R1 Sec. IV.A, Fig. 18).

The geometry is built in the shock frame (X = freestream) and exposed in the
WADOS "OC stream protocol" in the GUI frame (x streamwise, y up, z span):
``upper_surface_streams``, ``lower_surface_streams`` (equal-length lists, half
model, station 0 on the symmetry plane), ``leading_edge``, ``length``, ``width``,
``height``. This is what export, volume and reference-area code consume
(Gate 0 report, section 1.3). ``to_gui`` maps the shock frame to the GUI
frame, e.g. ``ltoc.ltoc.r1_frame_to_gui`` for R1's Z-down frame.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .ltoc import LTOCError, StreamSurface, find_leading_edge, identity_frame, solve_stream_surface
from .shock_surface import ShockSurface

__all__ = ["LTOCWaverider"]


class LTOCWaverider:
    """Waverider from a prescribed shock and a flow capture tube (R1 Sec. IV).

    Parameters
    ----------
    shock : ShockSurface
        Prescribed shock in a frame whose X axis is the freestream.
    fct_yz : (n, 2) array
        FCT points (Y, Z) in the base plane ``X = x_base``, ordered from the
        symmetry plane (station 0) outward. ``to_gui`` must map the symmetry
        plane to the GUI plane ``z = 0`` (e.g. ``Y = 0`` for R1's frame with
        ``r1_frame_to_gui``; ``Z = 0`` when the shock is given in the GUI
        frame). The last point may lie on the shock (zero-chord tip).
    x_base : float
        Base-plane position.
    M_inf, gamma : float
        Design Mach number and ratio of specific heats.
    n_points : int
        Shock points per stream surface between the leading edge and the base.
    n_streamwise : int, optional
        Points per stream in the OC stream protocol (default ``n_points + 1``).
    to_gui : callable
        Maps shock-frame points (..., 3) to the GUI frame.
    strict : bool
        Raise ``LTOCError`` as soon as a stream surface fails. Otherwise the
        failure is recorded in ``stations`` and ``ok`` is False; the
        stream-protocol attributes then raise.
    """

    def __init__(self, shock: ShockSurface, fct_yz, x_base: float, M_inf: float, *,
                 gamma: float = 1.4, n_points: int = 80, n_streamwise: Optional[int] = None,
                 to_gui: Callable = identity_frame, strict: bool = False, label: str = "ltoc"):
        self.shock = shock
        self.fct_yz = np.atleast_2d(np.asarray(fct_yz, dtype=float))
        self.x_base = float(x_base)
        self.M_inf = float(M_inf)
        self.gamma = float(gamma)
        self.n_points = int(n_points)
        self.n_streamwise = int(n_streamwise or n_points + 1)
        self.to_gui = to_gui
        self.label = label
        self.stations: list[StreamSurface] = []
        self._build(strict)

    # ------------------------------------------------------------------
    def _build(self, strict: bool) -> None:
        uv, le_pts, converged = find_leading_edge(self.shock, self.fct_yz)
        self.le_uv, self.le_points_native = uv, le_pts
        for k in range(self.fct_yz.shape[0]):
            if not converged[k]:
                st = StreamSurface(station=k, status="no_leading_edge",
                                   message="FCT point could not be projected onto the shock")
            elif le_pts[k, 0] > self.x_base * (1 + 1e-9) + 1e-12:
                st = StreamSurface(station=k, status="outside_shock",
                                   message="FCT point lies outside the shock at the base plane")
            else:
                st = solve_stream_surface(self.shock, uv[k], self.x_base, self.M_inf,
                                          gamma=self.gamma, n_points=self.n_points, station=k)
            if strict and not st.ok:
                raise LTOCError(f"station {k}: {st.status}: {st.message}")
            self.stations.append(st)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stations)

    def failures(self) -> list:
        return [(s.station, s.status, s.message) for s in self.stations if not s.ok]

    def _require_ok(self):
        if not self.ok:
            lines = "; ".join(f"station {k}: {st} ({msg})" for k, st, msg in self.failures())
            raise LTOCError(f"waverider incomplete: {lines}")

    # ------------------------------------------------------------------
    #  Native (shock-frame) geometry
    # ------------------------------------------------------------------
    def lower_streams_native(self) -> list:
        self._require_ok()
        out = []
        for s in self.stations:
            if s.status == "tip":
                out.append(np.repeat(s.le_point[None], 2, axis=0))
            else:
                out.append(s.body.points)
        return out

    def upper_streams_native(self) -> list:
        self._require_ok()
        out = []
        for s, fct in zip(self.stations, self.fct_yz):
            te = np.array([self.x_base, fct[0], fct[1]])
            out.append(np.vstack([s.le_point, te]))
        return out

    # ------------------------------------------------------------------
    #  OC stream protocol (GUI frame)
    # ------------------------------------------------------------------
    def _streams_gui(self, streams):
        from geometry_export import resample_stream

        return [resample_stream(self.to_gui(s), self.n_streamwise) for s in streams]

    @property
    def lower_surface_streams(self) -> list:
        return self._streams_gui(self.lower_streams_native())

    @property
    def upper_surface_streams(self) -> list:
        return self._streams_gui(self.upper_streams_native())

    @property
    def leading_edge(self) -> np.ndarray:
        self._require_ok()
        return self.to_gui(np.vstack([s.le_point for s in self.stations]))

    @property
    def length(self) -> float:
        return self.x_base

    @property
    def width(self) -> float:
        return float(np.max(np.abs(self.leading_edge[:, 2])))

    @property
    def height(self) -> float:
        ys = np.concatenate([s[:, 1] for s in self.lower_surface_streams + self.upper_surface_streams])
        return float(np.ptp(ys))

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------
    def mesh(self, full_span: bool = True):
        """Closed triangle mesh (GUI frame) through ``geometry_export``."""
        from geometry_export import streams_to_mesh

        return streams_to_mesh(self.upper_surface_streams, self.lower_surface_streams,
                               full_span=full_span, kind=f"{self.label}_waverider")

    def export_stl(self, path, full_span: bool = True):
        from geometry_export import write_stl

        return write_stl(self.mesh(full_span), path)

    def export_step(self, path, full_span: bool = True):
        from geometry_export import write_step

        return write_step(self.mesh(full_span), path)

    # ------------------------------------------------------------------
    #  Inviscid forces (R1 Eqs. 19-22)
    # ------------------------------------------------------------------
    def aerodynamics(self, n_streamwise: int = 101, reference: str = "planform"):
        """Lower-surface lift and drag coefficients (``ltoc.forces.panel_forces``).

        The default planform reference area reproduces R1 Tables 2 and 3; see
        ``ltoc.forces`` for why it differs from the "wetted area" of R1 Eq. (22).
        """
        from .forces import lower_surface_grid, panel_forces

        return panel_forces(lower_surface_grid(self, n_streamwise), self.M_inf, self.gamma,
                            reference=reference)

    # ------------------------------------------------------------------
    #  Diagnostics (spec section 6)
    # ------------------------------------------------------------------
    def spanwise_residuals(self, fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0)) -> dict:
        """Spec flag 8: spanwise smoothness of the lower surface.

        The stream surfaces are solved independently and nothing is smoothed.
        Cross-flow cuts are taken at ``X = x_min + f (x_base - x_min)``, where
        ``x_min`` is the most upstream leading-edge point. At each cut, the
        lower-surface point of every station with two neighbours on each side
        is compared with the cubic through those four neighbours. The cubic
        is parameterised by arc length along the FCT. On a smooth surface the
        residual is the cubic interpolation error, O(h^4) in the station
        spacing. Noise or a kink between neighbouring stream surfaces shows up
        directly.

        Returns ``{f: (stations, residuals / length)}``.
        """
        lower = self.lower_streams_native()
        fct = self.fct_yz
        t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fct, axis=0), axis=1))])
        x_min = min(float(st[0, 0]) for st in lower)
        out = {}
        for f in fractions:
            xc = x_min + f * (self.x_base - x_min)
            idx, pts = [], []
            for k, st in enumerate(lower):
                x = st[:, 0]
                if x[0] <= xc + 1e-12 * self.x_base and xc <= x[-1] + 1e-12 * self.x_base:
                    idx.append(k)
                    pts.append([np.interp(xc, x, st[:, c]) for c in (1, 2)])
            idx, pts = np.asarray(idx), np.asarray(pts)
            ks, res = [], []
            for a in range(2, len(idx) - 2):
                nb = np.array([a - 2, a - 1, a + 1, a + 2])
                if np.any(np.diff(idx[np.r_[nb[:2], a, nb[2:]]]) != 1):
                    continue
                tn, tk = t[idx[nb]], t[idx[a]]
                w = np.array([np.prod([(tk - tn[q]) / (tn[p_] - tn[q]) for q in range(4) if q != p_])
                              for p_ in range(4)])
                res.append(float(np.linalg.norm(pts[a] - w @ pts[nb])))
                ks.append(int(idx[a]))
            out[float(f)] = (np.asarray(ks, dtype=int), np.asarray(res) / self.length)
        return out

    def spanwise_smoothness(self, fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0)) -> float:
        """Largest residual of :meth:`spanwise_residuals`, divided by the length."""
        vals = [r.max() for _, r in self.spanwise_residuals(fractions).values() if r.size]
        return float(max(vals)) if vals else 0.0

    def diagnostics(self) -> list:
        rows = []
        for s in self.stations:
            row = {"station": s.station, "status": s.status, "message": s.message}
            if s.le_point is not None:
                row["x_le"] = float(s.le_point[0])
            row.update(s.diagnostics)
            rows.append(row)
        return rows
