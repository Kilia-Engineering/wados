"""Inviscid lower-surface forces of an LTOCs waverider (R1 Sec. IV.A, Eqs. 19-22).

R1 integrates the pressure over quadrilateral elements ABCD of the lower
surface (R1 Fig. 19):

* area from the cross product of the diagonals, ``A_w = |AC x DB|`` (Eq. 19);
* pressure as the mean of the four corners (Eq. 20);
* ``L = P_avg A_w n_z / |n|`` and ``D = P_avg A_w n_x / |n|`` (Eq. 21);
* ``C_L = L / (0.5 rho_inf V_inf^2 A)`` and ``C_D = D / (0.5 rho_inf V_inf^2 A)``
  (Eq. 22).

Two points differ from the printed text, and both were settled by reproducing
R1 Tables 2 and 3 (Gate 4 report):

* **Reference area.** R1 Eq. (22) calls A "the total wetted area of the
  waverider". R1's tabulated coefficients are matched by the *planform*
  (projected) area of the lower surface instead, to 0.3 % in C_L for both
  waverider cases. With the wetted area, C_L and C_D are 17-23 % lower. The
  default here is ``reference="planform"``; the wetted-area coefficients are
  also returned.
* **Factor 1/2.** R1 Eq. (19) leaves out the factor 1/2 of the quadrilateral
  area. It cancels in the coefficients, but the true area ``|AC x DB| / 2`` is
  used here so that the reported areas are physical.

The pressure is the absolute lower-surface pressure, and the upper
(freestream) surface and the base contribute nothing (R1 Sec. IV.A treats
the lower surface only). The spec inferred this from R1's numbers, and the
reproduction confirms it.

The lower-surface grid is built from the body streamlines: one row per FCT
station, resampled at the same fractions of each local chord, with p/p_inf
interpolated along the streamline. Forces are evaluated in the GUI frame
(x streamwise, y up), so lift is the y component and drag the x component.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LowerSurfaceGrid", "PanelForces", "lower_surface_grid", "body_grid", "panel_forces",
           "streamwise_section", "crosswise_section"]


@dataclass
class LowerSurfaceGrid:
    """Structured lower-surface grid (half model), station i x streamwise k."""

    points: np.ndarray       # (n_stations, n_streamwise, 3), GUI frame
    p: np.ndarray            # (n_stations, n_streamwise), p / p_inf
    points_native: np.ndarray  # same points in the shock frame


@dataclass
class PanelForces:
    """Lower-surface forces (half model; the coefficients equal the full model's)."""

    CL: float                # referenced to ``reference`` area
    CD: float
    L_over_D: float
    reference: str           # "planform" or "wetted"
    area_wetted: float       # half model, geometry units^2
    area_planform: float     # projection on the horizontal plane (GUI x-z plane)
    CL_wetted: float
    CD_wetted: float
    lift: float              # in p_inf * length^2 (half model)
    drag: float
    n_panels: int


def _tip_pressure(wr, st) -> float:
    """Post-shock pressure at a zero-chord station (exact Rankine-Hugoniot)."""
    from gvwd.thermo.oblique_shock import rankine_hugoniot

    from .shock_geometry import local_geometry

    g = local_geometry(wr.shock, st.le_uv[0], st.le_uv[1], wr.x_base)
    return float(rankine_hugoniot(wr.M_inf, float(g.beta), 1.0, 1.0, wr.gamma)["p2"])


def body_grid(stations, n_streamwise: int = 101, spacing: str = "uniform", tip_pressure=None):
    """Body streamlines of ``stations`` resampled at the same chord fractions.

    Returns ``(points, p)`` in the shock frame, shapes (n, n_streamwise, 3) and
    (n, n_streamwise). ``spacing`` is "uniform" or "cosine" (clustered at the
    leading edge and the base). Zero-chord ("tip") stations repeat their
    leading-edge point with ``tip_pressure(station)``.
    """
    s = np.linspace(0.0, 1.0, n_streamwise)
    if spacing == "cosine":
        s = 0.5 * (1.0 - np.cos(np.pi * s))
    pts, pres = [], []
    for st in stations:
        if st.status == "tip":
            pts.append(np.repeat(st.le_point[None], n_streamwise, axis=0))
            pres.append(np.full(n_streamwise, tip_pressure(st)))
            continue
        b = st.body
        xs = b.x[0] + s * (b.x[-1] - b.x[0])
        pts.append(np.column_stack([np.interp(xs, b.x, b.points[:, c]) for c in range(3)]))
        pres.append(np.interp(xs, b.x, b.p))
    return np.asarray(pts), np.asarray(pres)


def lower_surface_grid(wr, n_streamwise: int = 101, spacing: str = "uniform") -> LowerSurfaceGrid:
    """Lower-surface grid of an ``LTOCWaverider`` (see :func:`body_grid`)."""
    wr._require_ok()
    native, p = body_grid(wr.stations, n_streamwise, spacing, lambda st: _tip_pressure(wr, st))
    return LowerSurfaceGrid(points=wr.to_gui(native), p=p, points_native=native)


def streamwise_section(points, p, value: float, span_axis: int = 1):
    """Wall pressure in the streamwise plane ``points[..., span_axis] = value``.

    ``points`` and ``p`` come from :func:`body_grid` (shock frame). The plane
    is crossed between neighbouring stations along every chord fraction.
    Returns ``(X, p)`` sorted by X (R1 Figs. 16a, 23a, 27a).
    """
    yv = points[..., span_axis] - value
    eps = 1e-12 * max(float(np.ptp(points[..., 0])), 1.0)
    yv = np.where(np.abs(yv) <= eps, 0.0, yv)                # a station lying on the plane
    X, P = [], []
    for k in range(points.shape[1]):
        a, b = yv[:-1, k], yv[1:, k]
        hit = (a == 0.0) | (a * b < 0.0)
        hit[-1] |= b[-1] == 0.0 and a[-1] != 0.0
        for i in np.flatnonzero(hit):
            t = 0.0 if a[i] == b[i] else a[i] / (a[i] - b[i])
            X.append(points[i, k, 0] + t * (points[i + 1, k, 0] - points[i, k, 0]))
            P.append(p[i, k] + t * (p[i + 1, k] - p[i, k]))
    X, P = np.asarray(X), np.asarray(P)
    order = np.argsort(X)
    return X[order], P[order]


def crosswise_section(points, p, x: float, span_axis: int = 1):
    """Wall pressure in the crosswise plane ``X = x``: ``(span coordinate, p)``
    over the stations whose chord contains x, sorted by the span coordinate
    (R1 Figs. 16b, 23b, 27b)."""
    Y, P = [], []
    for row, prow in zip(points, p):
        xs = row[:, 0]
        if xs[0] <= x <= xs[-1] and xs[-1] > xs[0]:
            Y.append(np.interp(x, xs, row[:, span_axis]))
            P.append(np.interp(x, xs, prow))
    Y, P = np.asarray(Y), np.asarray(P)
    order = np.argsort(Y)
    return Y[order], P[order]


def panel_forces(grid: LowerSurfaceGrid, M_inf: float, gamma: float = 1.4,
                 reference: str = "planform") -> PanelForces:
    """Lift and drag coefficients of the lower surface (R1 Eqs. 19-22).

    ``reference`` selects the area A of Eq. (22): "planform" (default; it
    reproduces R1 Tables 2 and 3) or "wetted" (the printed text).
    """
    if reference not in ("planform", "wetted"):
        raise ValueError("reference must be 'planform' or 'wetted'")
    P, p = grid.points, grid.p
    A, B = P[:-1, :-1], P[1:, :-1]
    C, D = P[1:, 1:], P[:-1, 1:]
    n = np.cross(C - A, B - D)                                   # R1 Eq. (19), AC x DB
    norm = np.linalg.norm(n, axis=-1)
    # Orient every normal out of the body, i.e. downward for the lower surface.
    n = np.where((n[..., 1] > 0.0)[..., None], -n, n)
    area = 0.5 * norm
    p_avg = 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[1:, 1:] + p[:-1, 1:])   # R1 Eq. (20)
    ok = norm > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        nhat = np.where(ok[..., None], n / norm[..., None], 0.0)
    force = -(p_avg * area)[..., None] * nhat                   # pressure acts against n_out
    lift = float(np.sum(force[..., 1]))                         # R1 Eq. (21)
    drag = float(np.sum(force[..., 0]))
    A_w = float(np.sum(area))
    A_p = float(np.sum(area * np.abs(nhat[..., 1])))
    if A_p <= 1e-6 * A_w:
        raise ValueError("the lower surface has no planform area in the GUI frame (y up); "
                         "check the waverider's to_gui mapping")
    q = 0.5 * gamma * M_inf ** 2                                # q_inf / p_inf
    A_ref = A_p if reference == "planform" else A_w
    CL, CD = lift / (q * A_ref), drag / (q * A_ref)             # R1 Eq. (22)
    return PanelForces(CL=CL, CD=CD, L_over_D=CL / CD, reference=reference, area_wetted=A_w,
                       area_planform=A_p, CL_wetted=lift / (q * A_w), CD_wetted=drag / (q * A_w),
                       lift=lift, drag=drag, n_panels=int(np.sum(ok)))
