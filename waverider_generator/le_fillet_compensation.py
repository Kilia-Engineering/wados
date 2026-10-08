"""
Leading-edge fillet compensation, Mode B
========================================

Pre-compensates a sharp-LE waverider so that a rolling-ball fillet of
radius R applied later in CAD lands back on the original sharp leading
edge.  The lower (compression) surface is preserved: only its first R of
length is consumed by the fillet.  All thickening goes onto the upper
(freestream) surface, which is offset outward by a rigid translation that
decays to zero downstream.

Per LE station i, in the plane normal to the LE:

    s = sin(θ/2) = |n_u + n_l| / 2,   c = cos(θ/2) = |n_u − n_l| / 2
    θ   = 2·atan2(s, c)                   interior wedge angle
    L_B = R (c/s − 1)                     lower-face extension
    P'  = P + L_B a_l,  Δ = P' − P        new sharp edge, upper translation
    h_u = Δ·n_u = R (1 + cos θ − sin θ)   outward upper-face offset
    t_u = R cot(θ/2)  (from P'),  t_l = R  (from P)   fillet setbacks

Upper grid:  X' = X + w(d) Δ(s*).  A point at arc length ξ along the
upper grid line of station i lies in the normal plane of the LE point at
s* = s_i + ξ cos φ_u (φ_u = angle between the grid line and the LE) at
normal-plane distance d = ξ sin φ_u from the edge.  It is translated by
the Δ of *that* station (linear interpolation of Δ along s), so every
normal plane sees its own parallel offset h_u even where R, θ or the tip
taper change along the LE; with Δ constant along the LE this reduces to
X' = X + w(ξ) Δ_i.  w = 1 for d ≤ k_p t_u(s*), a smoothstep
1 − 3u² + 2u³ over the blend length L_b(s*), and 0 beyond.  At ξ = 0 the
LE point maps exactly to P'_i.  The lower-face setback t_l is converted to
grid-line distance with sin φ_l in the feasibility check.
Lower grid:  original points untouched; points on the straight segment
P' → P are prepended.

Tip rule:    the wingtip has zero chord, so any R > 0 there is infeasible.
Over the last ``tip_taper_mm`` of the LE the radius is capped by a straight
ramp from the schedule value at the taper start down to 0 at the tip:
R_applied = min(R_schedule, ramp).  Both are piecewise linear, so the
applied R(s) is piecewise linear and the control-point table reproduces
it exactly with linear transitions in CAD.  ``tip_taper_mm=None`` picks
the shortest taper (starting at an LE station) that makes every station
feasible; 0 disables the taper and infeasible stations are only reported.

Convention (OC generator): x streamwise, y vertical, z spanwise, half
model with the symmetry plane at z = 0; streams are lists of (n_j, 3)
arrays with point 0 on the LE.  Geometry is in metres; all user inputs
and reported lengths are in mm.
"""

import csv
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEGENERATE_CHORD = 1e-12   # relative to LE arc length


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FilletCompensationConfig:
    """User inputs for Mode B compensation (lengths in mm)."""
    blunting_enabled: bool = False
    fillet_mode: str = "constant"          # "constant" | "variable"
    R_mm: float = 5.0
    R_start_mm: float = 20.0
    R_end_mm: float = 5.0
    N_steps: int = 5
    s_start_mm: Optional[float] = None     # None -> wingtip end of the LE
    s_end_mm: Optional[float] = None       # None -> nose
    blend_length_mm: Optional[float] = None  # None/0 -> 0.30 x local upper chord
    plateau_factor: float = 1.2
    tip_taper_mm: Optional[float] = None   # None -> auto, 0 -> off, >0 fixed length
    # In-plane landing correction: iterate L_B per station until the ball of
    # radius R tangent to both compensated grid sections (in the normal plane)
    # has its foremost point on the original LE.  Off: closed-form L_B only.
    exact_landing: bool = False
    landing_tol_mm: float = 1e-2
    landing_max_iter: int = 15
    section_points: int = 200              # resampled points per stream for sections
    mm_per_unit: float = 1000.0            # geometry is in metres
    # Insert collinear points along the (straight) upper grid lines in the
    # plateau/blend zone before applying the weights.  Off by default: on the
    # OC grids the exported interpPlate surface fits the unrefined grid better.
    refine_upper: bool = False

    def validate(self):
        if self.fillet_mode not in ("constant", "variable"):
            raise ValueError(f"fillet_mode must be 'constant' or 'variable', "
                             f"got {self.fillet_mode!r}")
        if self.fillet_mode == "constant":
            if self.R_mm < 0:
                raise ValueError("R must be >= 0 mm")
        else:
            if self.R_start_mm < 0 or self.R_end_mm < 0:
                raise ValueError("R_start and R_end must be >= 0 mm")
            if int(self.N_steps) != self.N_steps or self.N_steps < 1:
                raise ValueError("N_steps must be an integer >= 1")
        if self.plateau_factor < 1.0:
            raise ValueError("plateau_factor k_p must be >= 1")
        if self.blend_length_mm is not None and self.blend_length_mm < 0:
            raise ValueError("blend_length must be >= 0 mm (0 = automatic)")
        if self.landing_tol_mm <= 0 or self.landing_max_iter < 1 or self.section_points < 20:
            raise ValueError("landing_tol_mm > 0, landing_max_iter >= 1, section_points >= 20")
        if self.tip_taper_mm is not None and self.tip_taper_mm < 0:
            raise ValueError("tip_taper must be >= 0 mm (0 = off) or None for auto")
        if self.mm_per_unit <= 0:
            raise ValueError("mm_per_unit must be > 0")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def le_arc_length(le):
    """Cumulative arc length along the LE polyline, starting at 0."""
    seg = np.linalg.norm(np.diff(le, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def le_tangents(le, symmetry_station=None, span_axis=2):
    """
    Unit LE tangents: central differences, one-sided at the ends.

    At a symmetry station that is an end point, the missing neighbour is
    the mirror image of the existing one, so the tangent is exactly
    spanwise and the compensation translation stays in the symmetry plane
    by construction.
    """
    n = len(le)
    e = np.empty_like(le, dtype=float)
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
        if i == symmetry_station and (i == 0 or i == n - 1):
            nb = le[b] if i == 0 else le[a]
            mirrored = nb.copy()
            mirrored[span_axis] = -mirrored[span_axis]
            e[i] = _unit(nb - mirrored) if i == 0 else _unit(mirrored - nb)
        else:
            e[i] = _unit(le[b] - le[a])
    return e


def polyline_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def streamwise_tangent(stream):
    """
    Unit tangent of a grid line at its first point (pointing aft).

    Second-order one-sided difference in arc length when three distinct
    points are available (exact for straight lines), first-order otherwise.
    """
    h = np.linalg.norm(np.diff(stream[:3], axis=0), axis=1)
    if len(h) >= 2 and h[0] > 0 and h[1] > 0:
        h1, h2 = h
        d = (-(2 * h1 + h2) / (h1 * (h1 + h2)) * stream[0]
             + (h1 + h2) / (h1 * h2) * stream[1]
             - h1 / (h2 * (h1 + h2)) * stream[2])
        return _unit(d)
    return _unit(stream[1] - stream[0])


def wedge_angle(n_u, n_l):
    """Interior wedge angle from outward unit normals (no arccos)."""
    s = np.linalg.norm(n_u + n_l) / 2.0
    c = np.linalg.norm(n_u - n_l) / 2.0
    return 2.0 * math.atan2(s, c), s, c


def smoothstep_weight(xi, xi_p, L_b):
    """w = 1 on [0, ξ_p], 1 − 3u² + 2u³ on (ξ_p, ξ_p + L_b), 0 beyond."""
    xi = np.asarray(xi, dtype=float)
    if L_b <= 0:
        return np.where(xi <= xi_p, 1.0, 0.0)
    u = np.clip((xi - xi_p) / L_b, 0.0, 1.0)
    return 1.0 - 3.0 * u ** 2 + 2.0 * u ** 3


def smoothstep_weight_vec(xi, xi_p, L_b):
    """smoothstep_weight with per-point plateau and blend length."""
    xi, xi_p, L_b = (np.asarray(a, dtype=float) for a in (xi, xi_p, L_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(L_b > 0, np.clip((xi - xi_p) / np.where(L_b > 0, L_b, 1.0), 0.0, 1.0),
                     np.where(xi <= xi_p, 0.0, 1.0))
    return 1.0 - 3.0 * u ** 2 + 2.0 * u ** 3


def radius_schedule(s_mm, cfg, s_total_mm):
    """
    Fillet radius at each LE arc-length position.

    Returns (R_mm array, control points dict or None).  Variable mode uses
    N+1 equally spaced control points from s_start to s_end with linearly
    stepped radii, linear interpolation between them and clamping outside.
    """
    s_mm = np.asarray(s_mm, dtype=float)
    if cfg.fillet_mode == "constant":
        return np.full_like(s_mm, float(cfg.R_mm)), None

    s_a = s_total_mm if cfg.s_start_mm is None else float(cfg.s_start_mm)
    s_b = 0.0 if cfg.s_end_mm is None else float(cfg.s_end_mm)
    tol = 1e-9 * max(1.0, s_total_mm)
    for name, val in (("start", s_a), ("end", s_b)):
        if val < -tol or val > s_total_mm + tol:
            raise ValueError(f"variable-fillet {name} point s={val:.3f} mm is "
                             f"outside the LE (0 .. {s_total_mm:.3f} mm)")
    if abs(s_a - s_b) <= tol:
        raise ValueError("variable-fillet start and end points coincide")

    N = int(cfg.N_steps)
    k = np.arange(N + 1)
    s_k = s_a + k * (s_b - s_a) / N
    R_k = cfg.R_start_mm - k * (cfg.R_start_mm - cfg.R_end_mm) / N
    order = np.argsort(s_k)
    R = np.interp(s_mm, s_k[order], R_k[order])
    return R, {"k": k, "s_mm": s_k, "R_mm": R_k}


def _schedule_breakpoints(cfg, control, s_total_mm):
    """Knots (s, R) of the requested piecewise-linear schedule, s ascending."""
    if control is None:
        return np.array([0.0, s_total_mm]), np.array([cfg.R_mm, cfg.R_mm], dtype=float)
    order = np.argsort(control["s_mm"])
    s_k, R_k = control["s_mm"][order], control["R_mm"][order]
    # clamped outside [s_a, s_b]: extend flat to the LE ends
    if s_k[0] > 0.0:
        s_k, R_k = np.insert(s_k, 0, 0.0), np.insert(R_k, 0, R_k[0])
    if s_k[-1] < s_total_mm:
        s_k, R_k = np.append(s_k, s_total_mm), np.append(R_k, R_k[-1])
    return s_k, R_k


def apply_tip_taper(s_mm, cfg, control, taper_mm):
    """
    Cap the schedule near the wingtip: R = min(R_schedule, ramp) over the
    last ``taper_mm`` of the LE, where the ramp falls linearly from the
    schedule value at the taper start to 0 at the tip.

    Returns (R_applied at s_mm, knots_s, knots_R) with the knots describing
    the applied piecewise-linear function exactly.
    """
    s_mm = np.asarray(s_mm, dtype=float)
    s_tip = float(s_mm[-1])
    ks, kR = _schedule_breakpoints(cfg, control, s_tip)
    sched = lambda x: np.interp(x, ks, kR)
    if not taper_mm or taper_mm <= 0.0:
        return sched(s_mm), ks, kR
    taper_mm = min(float(taper_mm), s_tip)
    s_c = s_tip - taper_mm
    R_c = float(sched(s_c))
    ramp = lambda x: R_c * (s_tip - np.asarray(x, dtype=float)) / taper_mm
    applied = lambda x: np.where(np.asarray(x) < s_c, sched(x), np.minimum(sched(x), ramp(x)))

    knots = set(float(v) for v in ks[ks <= s_c]) | {s_c, s_tip}
    inside = [float(v) for v in ks if s_c < v < s_tip]
    knots |= set(inside)
    # crossings of the ramp with each schedule segment inside the taper zone
    seg = sorted(set([s_c] + inside + [s_tip]))
    for s1, s2 in zip(seg[:-1], seg[1:]):
        f1, f2 = float(sched(s1) - ramp(s1)), float(sched(s2) - ramp(s2))
        if f1 * f2 < 0.0:
            knots.add(s1 + (s2 - s1) * f1 / (f1 - f2))
    knots_s = np.array(sorted(knots))
    return applied(s_mm), knots_s, applied(knots_s)


def resample_stream(stream, n, cluster=True):
    """
    Resample a polyline to n points by arc length.  With ``cluster`` the
    parameter is cosine-stretched so points concentrate at the LE end.
    Resampled points lie on the original polyline; point 0 is kept exactly.
    """
    stream = np.asarray(stream, dtype=float)
    if stream.shape[0] < 2 or np.ptp(stream, axis=0).max() < 1e-14:
        return np.tile(stream[0], (n, 1))
    xi = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(stream, axis=0), axis=1))])
    t = np.linspace(0.0, 1.0, n)
    frac = 1.0 - np.cos(0.5 * np.pi * t) if cluster else t
    out = np.column_stack([np.interp(frac * xi[-1], xi, stream[:, c]) for c in range(3)])
    out[0] = stream[0]
    return out


def _plane_section(grid, u, i, P, e, eps):
    """
    Intersection of a structured grid (n_i, n_j, 3) with the plane through P
    normal to e: one point per j, ordered from the LE aft, stopping where
    the plane leaves the grid.  Across stations the grid is interpolated
    with a C2 cubic spline in the parameter ``u`` (LE arc length), the same
    model the CAD builder uses for its surfaces; the crossing nearest
    station i is taken.
    """
    from scipy.interpolate import CubicSpline
    n_i, n_j, _ = grid.shape
    d = np.einsum("ijk,k->ij", grid - P, e)
    on_plane = np.abs(d) <= eps
    sgn = np.sign(d)
    crossing = (sgn[:-1] * sgn[1:] < 0) & ~on_plane[:-1] & ~on_plane[1:]

    pts = np.zeros((n_j, 3))
    kind = np.zeros(n_j, dtype=int)          # 0 none, 1 on-plane point, 2 bracket
    k_sel = np.zeros(n_j, dtype=int)
    for j in range(n_j):
        on = np.flatnonzero(on_plane[:, j])
        if on.size:
            k = on[np.argmin(np.abs(on - i))]
            pts[j], kind[j] = grid[k, j], 1
            continue
        cross = np.flatnonzero(crossing[:, j])
        if cross.size == 0:
            break
        k_sel[j], kind[j] = cross[np.argmin(np.abs(cross - i))], 2
    n_valid = int(np.argmax(kind == 0)) if (kind == 0).any() else n_j
    br = np.flatnonzero(kind[:n_valid] == 2)
    if br.size:
        spl = CubicSpline(u, grid[:, br], axis=0)
        k = k_sel[br]
        lo, hi = u[k].astype(float), u[k + 1].astype(float)
        f_lo = d[k, br]
        Pe = float(np.dot(P, e))
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            vals = spl(mid)                                   # (m, m, 3)
            f_mid = np.einsum("jjk,k->j", vals, e) - Pe
            go_lo = f_mid * f_lo > 0
            lo = np.where(go_lo, mid, lo)
            f_lo = np.where(go_lo, f_mid, f_lo)
            hi = np.where(go_lo, hi, mid)
            if np.max(hi - lo) <= 1e-12 * max(float(np.max(np.abs(hi))), 1.0):
                break
        mid = 0.5 * (lo + hi)
        pts[br] = np.einsum("jjk->jk", spl(mid))
    return pts[:n_valid]


def _signed_dist_to_polyline(C, poly, e, inward_ref):
    """Distance from C to a polyline; positive when C lies on the body side."""
    A, B = poly[:-1], poly[1:]
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300)
    tt = np.clip(np.einsum("ij,ij->i", C - A, AB) / L2, 0.0, 1.0)
    Q = A + tt[:, None] * AB
    dist = np.linalg.norm(Q - C, axis=1)
    k = int(np.argmin(dist))
    n_in = np.cross(e, AB[k])
    if np.dot(n_in, inward_ref) < 0:
        n_in = -n_in
    sign = 1.0 if np.dot(C - Q[k], n_in) >= 0 else -1.0
    return sign * dist[k]


def _ball_landing(sec_u, sec_l, P, a_l, n_l, n_u, e, R):
    """
    Ball of radius R tangent to both section polylines in the normal plane.
    Returns (landing, centre): landing > 0 means the ball's foremost point
    (along a_l) lies ahead of the original LE point P.  NaN if no ball fits.
    """
    seg = np.diff(sec_l, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    if cum[-1] <= 0 or len(sec_u) < 2:
        return float("nan"), None

    def centre(a):
        k = min(max(int(np.searchsorted(cum, a, side="right")) - 1, 0), len(seg) - 1)
        t = 0.0 if seg_len[k] == 0 else (a - cum[k]) / seg_len[k]
        p = sec_l[k] + t * seg[k]
        n_in = np.cross(e, seg[k])
        if np.dot(n_in, -n_l) < 0:
            n_in = -n_in
        return p + R * n_in / np.linalg.norm(n_in)

    def f(a):
        return _signed_dist_to_polyline(centre(a), sec_u, e, -n_u) - R

    lo, hi = 0.0, cum[-1]
    if f(lo) > 0 or f(hi) < 0:
        return float("nan"), None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(cum[-1], 1.0):
            break
    C = centre(0.5 * (lo + hi))
    return float(np.dot(C - P, a_l) + R), C


def _refine_polyline(stream, targets, min_gap):
    """
    Insert points at arc lengths ``targets`` by linear interpolation along
    the polyline.  Original points are kept bit-for-bit; targets within
    ``min_gap`` of an existing node are dropped.
    """
    xi = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(stream, axis=0), axis=1))])
    keep = [t for t in np.unique(targets)
            if 0.0 < t < xi[-1] and np.min(np.abs(xi - t)) > min_gap]
    if not keep:
        return stream
    keep = np.asarray(keep)
    new_pts = np.column_stack([np.interp(keep, xi, stream[:, c]) for c in range(3)])
    all_xi = np.concatenate([xi, keep])
    all_pts = np.vstack([stream, new_pts])
    order = np.argsort(all_xi, kind="stable")
    return all_pts[order]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CompensationResult:
    config: FilletCompensationConfig
    upper_streams: List[np.ndarray]
    lower_streams: List[np.ndarray]
    le_original: np.ndarray            # (n, 3) geometry units
    le_new: np.ndarray                 # (n, 3) geometry units, P'
    table: dict                        # per-station arrays (mm / deg / bool)
    control_points: Optional[dict]     # knots of the applied R(s); None if constant
    messages: List[str] = field(default_factory=list)
    tip_taper: dict = field(default_factory=dict)   # mode, length_mm, start_s_mm

    @property
    def infeasible_stations(self):
        return [int(i) for i in np.flatnonzero(~self.table["feasible"])]

    @property
    def all_feasible(self):
        return bool(np.all(self.table["feasible"]))

    def feasibility_report(self):
        """Human-readable list of failing stations with their max feasible R."""
        bad = self.infeasible_stations
        if not bad:
            return "All stations feasible."
        t = self.table
        lines = [f"{len(bad)} of {len(t['station'])} LE stations are infeasible "
                 f"(no clamping applied):",
                 "  station   s [mm]   R [mm]  R_max [mm]  reason"]
        for i in bad:
            lines.append(f"  {i:7d} {t['s_mm'][i]:8.1f} {t['R_mm'][i]:8.3f} "
                         f"{t['R_max_mm'][i]:11.3f}  {t['reason'][i]}")
        lines.append("R_max = chord_u sin(phi_u) / (k_p cot(theta/2)), ignoring the blend "
                     "length L_b (phi_u = angle between upper grid line and LE).")
        tt = self.tip_taper
        if tt.get("mode") == "off":
            lines.append("Tip taper is off; set tip_taper to auto or a length to cap R at the tip.")
        elif tt.get("mode") == "auto-failed":
            lines.append(tt.get("note", ""))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compensate_le_mode_b(upper_streams, lower_streams, cfg,
                         symmetry_station=0, span_axis=2):
    """
    Apply Mode B fillet compensation to sharp-LE surface grids.

    Parameters
    ----------
    upper_streams, lower_streams : list of (n_j, 3) ndarray
        Grid lines per LE station, point 0 on the shared LE.  Not modified.
    cfg : FilletCompensationConfig
    symmetry_station : int or None
        Station lying on the symmetry plane; the spanwise component of its
        translation is removed so the nose stays on the plane.
    span_axis : int
        Index of the spanwise coordinate (z = 2 in the OC convention).

    Returns
    -------
    CompensationResult
    """
    cfg.validate()
    n = len(upper_streams)
    if len(lower_streams) != n:
        raise ValueError("upper and lower stream counts differ")
    if n < 2:
        raise ValueError("need at least two LE stations")

    us = [np.asarray(s, dtype=float) for s in upper_streams]
    ls = [np.asarray(s, dtype=float) for s in lower_streams]
    mm = float(cfg.mm_per_unit)
    messages = []

    le = np.array([s[0] for s in us])
    s_le = le_arc_length(le)
    e_all = le_tangents(le, symmetry_station, span_axis)
    scale = max(s_le[-1], 1e-30)

    le_gap = max(np.linalg.norm(us[i][0] - ls[i][0]) for i in range(n))
    if le_gap > 1e-9 * scale:
        messages.append(f"upper/lower LE points differ by up to {le_gap * mm:.3e} mm; "
                        f"the upper LE is used as P_i")

    chord_u = np.array([polyline_length(s) for s in us])
    chord_l = np.array([polyline_length(s) for s in ls])
    degenerate = (chord_u <= _DEGENERATE_CHORD * scale) | (chord_l <= _DEGENERATE_CHORD * scale)
    if np.all(degenerate):
        raise ValueError("all LE stations have zero chord")

    # --- local frames -----------------------------------------------------
    n_u = np.zeros((n, 3))
    n_l = np.zeros((n, 3))
    sin_phi_u = np.ones(n)     # sine / cosine of angle between grid line and LE
    sin_phi_l = np.ones(n)
    cos_phi_u = np.zeros(n)
    for i in np.flatnonzero(~degenerate):
        P, e = le[i], e_all[i]
        interior = 0.5 * (us[i][1] + ls[i][1]) - P
        for out, sphi, stream in ((n_u, sin_phi_u, us[i]), (n_l, sin_phi_l, ls[i])):
            t = streamwise_tangent(stream)
            if out is n_u:
                cos_phi_u[i] = float(np.dot(t, e))
            sphi[i] = math.sqrt(max(0.0, 1.0 - float(np.dot(t, e)) ** 2))
            m = np.cross(t, e)
            m = _unit(m - np.dot(m, e) * e)
            if np.dot(m, interior) > 0:
                m = -m
            out[i] = m

    # degenerate (zero-chord) stations borrow the nearest valid frame
    valid = np.flatnonzero(~degenerate)
    for i in np.flatnonzero(degenerate):
        j = valid[np.argmin(np.abs(valid - i))]
        e = e_all[i]
        for arr in (n_u, n_l):
            m = arr[j] - np.dot(arr[j], e) * e
            arr[i] = _unit(m)
        messages.append(f"station {i}: zero chord, normals taken from station {j}")

    theta = np.zeros(n)
    s_half = np.zeros(n)
    c_half = np.zeros(n)
    a_l = np.zeros((n, 3))
    for i in range(n):
        theta[i], s_half[i], c_half[i] = wedge_angle(n_u[i], n_l[i])
        a = _unit(np.cross(e_all[i], n_l[i]))
        if degenerate[i]:
            j = valid[np.argmin(np.abs(valid - i))]
            ref = ls[j][1] - ls[j][0]
        else:
            ref = ls[i][1] - le[i]
        if np.dot(a, ref) > 0:
            a = -a
        a_l[i] = a

    # --- radius schedule --------------------------------------------------
    s_mm = s_le * mm
    R_req_mm, control = radius_schedule(s_mm, cfg, s_mm[-1])
    cot_half = c_half / s_half
    if cfg.blend_length_mm:
        L_b = np.full(n, cfg.blend_length_mm / mm)
    else:
        L_b = 0.30 * chord_u

    def feasibility(R):
        """Per-station feasibility for radii R (geometry units)."""
        ok = np.ones(n, dtype=bool)
        why = [""] * n
        xi_p_ = cfg.plateau_factor * R * cot_half / sin_phi_u
        t_l_ = R / sin_phi_l
        for i in range(n):
            if R[i] == 0.0:
                continue
            r = []
            if degenerate[i]:
                r.append("zero chord")
            else:
                if xi_p_[i] + L_b[i] > chord_u[i]:
                    r.append("xi_p + L_b > upper chord")
                if t_l_[i] > chord_l[i]:
                    r.append("t_l = R > lower chord")
            if r:
                ok[i] = False
                why[i] = "; ".join(r)
        return ok, why

    # --- tip taper ----------------------------------------------------------
    s_tip = float(s_mm[-1])
    tip_taper = {"mode": "off", "length_mm": 0.0, "start_s_mm": s_tip}
    if cfg.tip_taper_mm is None:
        # shortest taper starting at an LE station that makes every station feasible
        chosen = None
        for j in range(n - 2, -1, -1):
            length = s_tip - float(s_mm[j])
            if length > 0.5 * s_tip:
                break
            R_try, _, _ = apply_tip_taper(s_mm, cfg, control, length)
            if feasibility(R_try / mm)[0].all():
                chosen = (j, length)
                break
        if chosen is None:
            tip_taper = {"mode": "auto-failed", "length_mm": 0.0, "start_s_mm": s_tip,
                         "note": "auto tip taper: no taper of up to 50% of the LE makes "
                                 "all stations feasible; no taper applied"}
            messages.append(tip_taper["note"])
        else:
            tip_taper = {"mode": "auto", "length_mm": chosen[1],
                         "start_s_mm": s_tip - chosen[1], "start_station": chosen[0]}
            messages.append(f"auto tip taper: R capped over the last {chosen[1]:.1f} mm "
                            f"of the LE (from station {chosen[0]}, s = {s_tip - chosen[1]:.1f} mm)")
    elif cfg.tip_taper_mm > 0:
        length = min(float(cfg.tip_taper_mm), s_tip)
        tip_taper = {"mode": "fixed", "length_mm": length, "start_s_mm": s_tip - length}
        messages.append(f"tip taper: R capped over the last {length:.1f} mm of the LE")
    R_mm, knots_s, knots_R = apply_tip_taper(s_mm, cfg, control, tip_taper["length_mm"])
    R = R_mm / mm
    if tip_taper["length_mm"] > 0 or control is not None:
        sched_s, sched_R = _schedule_breakpoints(cfg, control, s_tip)
        control = {"k": np.arange(len(knots_s)), "s_mm": knots_s, "R_mm": knots_R,
                   "R_schedule_mm": np.interp(knots_s, sched_s, sched_R)}

    # --- Mode B per station (planar wedge) ---------------------------------
    L_B_planar = R * (cot_half - 1.0)
    t_u = R * cot_half
    t_l = R.copy()
    plateau = cfg.plateau_factor * t_u                # in the normal plane
    xi_p = plateau / sin_phi_u                        # along the upper grid line

    # --- feasibility ------------------------------------------------------
    feasible, reason = feasibility(R)
    R_max = np.where(degenerate, 0.0,
                     chord_u * sin_phi_u / (cfg.plateau_factor * cot_half))

    # --- apply to grids ---------------------------------------------------
    def apply_grids(L_B):
        delta = L_B[:, None] * a_l
        sym_mismatch = np.zeros(n)
        if symmetry_station is not None:
            k = int(symmetry_station)
            ideal = delta[k].copy()
            delta[k, span_axis] = 0.0
            sym_mismatch[k] = np.linalg.norm(ideal - delta[k])
        P_new = le + delta
        new_us, new_ls = [], []
        for i in range(n):
            P = le[i]
            if R[i] == 0.0:
                new_us.append(us[i].copy())
                new_ls.append(ls[i].copy())
                continue

            up = us[i]
            if cfg.refine_upper and not degenerate[i]:
                zone_p = np.linspace(0.0, xi_p[i], 9)[1:]
                zone_b = np.linspace(xi_p[i], xi_p[i] + L_b[i], 13)[1:]
                min_gap = 0.25 * min(xi_p[i] / 8, L_b[i] / 12)
                up = _refine_polyline(up, np.concatenate([zone_p, zone_b]), min_gap)
            xi = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(up, axis=0), axis=1))])
            # station whose normal plane contains each point, and its normal-plane distance
            s_star = np.clip(s_le[i] + xi * cos_phi_u[i], s_le[0], s_le[-1])
            d = xi * sin_phi_u[i]
            delta_star = np.column_stack([np.interp(s_star, s_le, delta[:, c]) for c in range(3)])
            delta_star[0] = delta[i]                       # LE maps exactly to P'_i
            w = smoothstep_weight_vec(d, np.interp(s_star, s_le, plateau), np.interp(s_star, s_le, L_b))
            up_new = up + w[:, None] * delta_star
            new_us.append(up_new)

            lo = ls[i]
            length = np.linalg.norm(delta[i])
            if length > 0.0:
                ds0 = np.linalg.norm(lo[1] - lo[0]) if not degenerate[i] else length
                n_pre = max(1, int(math.ceil(length / ds0 - 1e-9)))
                frac = np.arange(n_pre)[:, None] / n_pre
                pre = P_new[i] + frac * (P - P_new[i])
                lo_new = np.vstack([pre, lo])
            else:
                lo_new = lo.copy()
            new_ls.append(lo_new)

            gap = max(np.linalg.norm(up_new[0] - P_new[i]), np.linalg.norm(lo_new[0] - P_new[i]))
            if gap > 1e-12 * scale:
                raise AssertionError(f"station {i}: upper/lower edges do not meet at P' "
                                     f"(gap {gap * mm:.3e} mm)")
        return new_us, new_ls, delta, P_new, sym_mismatch

    L_B = L_B_planar.copy()
    new_us, new_ls, delta, P_new, sym_mismatch = apply_grids(L_B)

    # closed-form check: h_u = Delta . n_u = R (1 + cos theta - sin theta)
    h_u_formula = R * (1.0 + np.cos(theta) - np.sin(theta))
    h_u_planar = np.einsum("ij,ij->i", delta, n_u)
    for i in range(n):
        if abs(h_u_planar[i] - h_u_formula[i]) > 1e-9 * max(R[i], 1e-30) + 1e-15 * scale:
            raise AssertionError(f"station {i}: h_u check failed "
                                 f"({h_u_planar[i]} vs {h_u_formula[i]})")
    if symmetry_station is not None and R[int(symmetry_station)] > 0:
        k = int(symmetry_station)
        messages.append(
            f"symmetry station {k}: removed spanwise component of Delta "
            f"({sym_mismatch[k] * mm:.4e} mm, {sym_mismatch[k] / max(L_B[k], 1e-300):.3e} of L_B)")

    # --- in-plane landing correction (optional) ---------------------------
    landing = np.full(n, np.nan)
    landing_iters = 0

    def measure_landing(new_us, new_ls):
        gu = np.array([resample_stream(x, cfg.section_points) for x in new_us])
        gl = np.array([resample_stream(x, cfg.section_points) for x in new_ls])
        u = le_arc_length(gu[:, 0])              # station parameter, as in the CAD builder
        out = np.full(n, np.nan)
        for i in range(n):
            if R[i] == 0.0 or degenerate[i]:
                continue
            sec_u = _plane_section(gu, u, i, le[i], e_all[i], 1e-12 * scale)
            sec_l = _plane_section(gl, u, i, le[i], e_all[i], 1e-12 * scale)
            if len(sec_u) < 2 or len(sec_l) < 2:
                continue
            # the plane must stay inside the grid past the ball's tangent points
            reach_u = np.max((sec_u - le[i]) @ (-a_l[i]))
            reach_l = np.max((sec_l - le[i]) @ (-a_l[i]))
            if reach_u < t_u[i] + R[i] or reach_l < 3.0 * R[i]:
                continue
            out[i], _ = _ball_landing(sec_u, sec_l, le[i], a_l[i], n_l[i], n_u[i], e_all[i], R[i])
        return out

    if cfg.exact_landing:
        tol = cfg.landing_tol_mm / mm
        for landing_iters in range(1, cfg.landing_max_iter + 1):
            landing = measure_landing(new_us, new_ls)
            ok = np.isfinite(landing)
            if not ok.any() or np.max(np.abs(landing[ok])) < tol:
                break
            L_B = np.where(ok, L_B - landing, L_B)
            new_us, new_ls, delta, P_new, sym_mismatch = apply_grids(L_B)
        else:
            landing = measure_landing(new_us, new_ls)
        ok = np.isfinite(landing)
        bad_land = [int(i) for i in np.flatnonzero(~ok & (R > 0) & ~degenerate)]
        messages.append(f"exact landing: {landing_iters} iteration(s), max |residual| "
                        f"{(np.max(np.abs(landing[ok])) * mm if ok.any() else float('nan')):.4f} mm, "
                        f"L_B changed by {np.max(np.abs(L_B - L_B_planar)) * mm:.3f} mm at most"
                        + (f"; no ball fit at stations {bad_land}" if bad_land else ""))
    else:
        landing = measure_landing(new_us, new_ls)
        ok = np.isfinite(landing)
        if ok.any():
            messages.append(f"predicted landing error of the closed-form L_B on the grid sections: "
                            f"{np.min(landing[ok]) * mm:+.3f} .. {np.max(landing[ok]) * mm:+.3f} mm "
                            f"(exact_landing=True corrects it)")

    h_u = np.einsum("ij,ij->i", delta, n_u)

    table = {
        "station": np.arange(n),
        "s_mm": s_mm,
        "x": P_new[:, 0] * mm, "y": P_new[:, 1] * mm, "z": P_new[:, 2] * mm,
        "theta_deg": np.degrees(theta),
        "R_mm": R_mm,
        "R_requested_mm": R_req_mm,
        "L_B_mm": L_B * mm,
        "L_B_planar_mm": L_B_planar * mm,
        "landing_mm": landing * mm,
        "h_u_mm": h_u * mm,
        "t_u_mm": t_u * mm,
        "t_l_mm": t_l * mm,
        "xi_p_mm": xi_p * mm,
        "L_b_mm": L_b * mm,
        "chord_u_mm": chord_u * mm,
        "chord_l_mm": chord_l * mm,
        "R_max_mm": R_max * mm,
        "phi_u_deg": np.degrees(np.arcsin(np.clip(sin_phi_u, 0, 1))),
        "phi_l_deg": np.degrees(np.arcsin(np.clip(sin_phi_l, 0, 1))),
        "feasible": feasible,
        "degenerate": degenerate,
        "reason": reason,
        "sym_mismatch_mm": sym_mismatch * mm,
        # local frame (unit vectors, geometry axes)
        "e": e_all, "n_u": n_u, "n_l": n_l, "a_l": a_l,
    }

    if control is not None:
        s_new = le_arc_length(P_new) * mm
        pts = np.column_stack([np.interp(control["s_mm"], s_mm, P_new[:, c] * mm)
                               for c in range(3)])
        control = dict(control,
                       s_new_mm=np.interp(control["s_mm"], s_mm, s_new),
                       xyz_mm=pts)

    table["landing_iterations"] = landing_iters
    bad = int(np.count_nonzero(~feasible))
    if bad:
        messages.append(f"{bad} infeasible station(s): {[int(i) for i in np.flatnonzero(~feasible)]}")
    for msg in messages:
        logger.info("[LE comp] %s", msg)

    return CompensationResult(cfg, new_us, new_ls, le, P_new, table, control,
                              messages, tip_taper)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

STATION_COLUMNS = ["station", "s_mm", "x", "y", "z", "theta_deg", "R_mm",
                   "L_B_mm", "h_u_mm", "t_u_mm", "feasible", "R_requested_mm",
                   "L_B_planar_mm", "landing_mm", "side"]
CONTROL_COLUMNS = ["k", "s_mm", "s_new_edge_mm", "x", "y", "z", "R_mm",
                   "R_schedule_mm", "side"]


def _side_sign(side):
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' (z >= 0) or 'right' (z <= 0)")
    return -1.0 if side == "right" else 1.0


def write_station_csv(result, path, side="left"):
    """
    Per-station table for CAD entry; x, y, z are P' in mm.  ``side`` selects
    the half the coordinates describe: 'left' keeps z >= 0, 'right' mirrors
    z to match the right-side STEP export.
    """
    t = result.table
    sz = _side_sign(side)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(STATION_COLUMNS)
        for i in t["station"]:
            vals = {c: t[c][i] for c in STATION_COLUMNS if c in t}
            vals["z"] = sz * vals["z"]
            w.writerow([int(i)]
                       + [f"{vals[c]:.6f}" for c in STATION_COLUMNS[1:10]]
                       + [bool(t["feasible"][i]), f"{t['R_requested_mm'][i]:.6f}",
                          f"{t['L_B_planar_mm'][i]:.6f}", f"{t['landing_mm'][i]:.6f}", side])


def write_control_points_csv(result, path, side="left"):
    """
    Knots of the applied piecewise-linear R(s): s_k on the original LE, arc
    length of the same point along the new edge, its 3D position on the new
    edge (mm), R_k to enter in CAD, and the uncapped schedule value.
    """
    cp = result.control_points
    if cp is None:
        raise ValueError("no control points: constant radius without tip taper")
    sz = _side_sign(side)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CONTROL_COLUMNS)
        for k in range(len(cp["k"])):
            x, y, z = cp["xyz_mm"][k]
            w.writerow([int(cp["k"][k])] + [f"{v:.6f}" for v in
                       (cp["s_mm"][k], cp["s_new_mm"][k], x, y, sz * z,
                        cp["R_mm"][k], cp["R_schedule_mm"][k])] + [side])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"
_SERIES = "#2a78d6"
_CRITICAL = "#d03b3b"


def plot_compensation(result, path=None, title=None):
    """
    R(s), L_B(s), h_u(s) and θ(s) along the LE as four stacked panels.
    Infeasible stations are marked with red crosses; in variable mode the
    control points are shown on the R panel.  Returns the Figure.
    """
    from matplotlib.figure import Figure

    t = result.table
    s = t["s_mm"]
    bad = ~t["feasible"]
    panels = [("R_mm", "R [mm]"), ("L_B_mm", r"$L_B$ [mm]"),
              ("h_u_mm", r"$h_u$ [mm]"), ("theta_deg", r"$\theta$ [deg]")]

    fig = Figure(figsize=(11, 13), facecolor=_SURFACE)
    left, right = 0.12, 0.96
    fig.subplots_adjust(left=left, right=right, top=0.91, bottom=0.06, hspace=0.12)
    axes = fig.subplots(len(panels), 1, sharex=True)

    tapered = result.tip_taper.get("length_mm", 0.0) > 0
    for ax, (key, label) in zip(axes, panels):
        ax.set_facecolor(_SURFACE)
        if key == "L_B_mm" and result.config.exact_landing:
            ax.plot(s, t["L_B_planar_mm"], color=_INK_2, lw=2, ls="--", zorder=2,
                    label="closed-form $L_B$")
            ax.plot(s, t[key], color=_SERIES, lw=2, marker="o", ms=4, zorder=3,
                    label="$L_B$ after landing correction")
            ax.legend(fontsize=14, frameon=False, loc="best")
        elif key == "R_mm" and tapered:
            ax.plot(s, t["R_requested_mm"], color=_INK_2, lw=2, ls="--", zorder=2,
                    label="requested R(s)")
            ax.plot(s, t[key], color=_SERIES, lw=2, marker="o", ms=4, zorder=3,
                    label="applied R(s), tip taper")
        else:
            ax.plot(s, t[key], color=_SERIES, lw=2, marker="o", ms=4, zorder=3)
        if bad.any():
            ax.plot(s[bad], t[key][bad], ls="none", marker="x", ms=10, mew=2.5,
                    color=_CRITICAL, zorder=4, label="infeasible station")
        ax.set_ylabel(label, fontsize=18, color=_INK)
        ax.tick_params(labelsize=15, colors=_INK_2)
        ax.grid(True, color=_GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_INK_2)

    cp = result.control_points
    if cp is not None:
        axes[0].plot(cp["s_mm"], cp["R_mm"], ls="none", marker="s", ms=10,
                     mfc="none", mew=2, color=_INK, zorder=5,
                     label="control points $(s_k, R_k)$")
    if bad.any() or cp is not None or tapered:
        axes[0].legend(fontsize=14, frameon=False, loc="best")

    axes[-1].set_xlabel("LE arc length s from nose [mm]", fontsize=18, color=_INK)

    cfg = result.config
    if title is None:
        if cfg.fillet_mode == "constant":
            sched = f"R = {cfg.R_mm:g} mm"
        else:
            sched = (f"R {cfg.R_start_mm:g} → {cfg.R_end_mm:g} mm "
                     f"in {cfg.N_steps} steps")
        title = f"LE fillet compensation (Mode B), {sched}, $k_p$ = {cfg.plateau_factor:g}"
    fig.suptitle(title, fontsize=20, color=_INK, x=0.5 * (left + right),
                 ha="center", y=0.965)

    if path is not None:
        fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return fig
