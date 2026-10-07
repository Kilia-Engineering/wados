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

Upper grid:  X' = X + w(ξ) Δ_i, with ξ the arc length along the upper
grid line, w = 1 up to ξ_p, a smoothstep 1 − 3u² + 2u³ over the blend
length L_b, and 0 beyond.  The plateau must cover k_p t_u measured in the
normal plane; a grid line meeting the LE at angle φ reaches normal
distance d at ξ = d / sin φ, so ξ_p = k_p t_u / sin φ_u (= k_p t_u when
the grid line is normal to the LE).  The same conversion is used for the
lower-face setback t_l in the feasibility check.
Lower grid:  original points untouched; points on the straight segment
P' → P are prepended.

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


def le_tangents(le):
    """Unit LE tangents: central differences, one-sided at the ends."""
    n = len(le)
    e = np.empty_like(le, dtype=float)
    for i in range(n):
        a, b = max(i - 1, 0), min(i + 1, n - 1)
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
    control_points: Optional[dict]     # variable mode only
    messages: List[str] = field(default_factory=list)

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
    e_all = le_tangents(le)
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
    sin_phi_u = np.ones(n)     # sine of angle between grid line and LE
    sin_phi_l = np.ones(n)
    for i in np.flatnonzero(~degenerate):
        P, e = le[i], e_all[i]
        interior = 0.5 * (us[i][1] + ls[i][1]) - P
        for out, sphi, stream in ((n_u, sin_phi_u, us[i]), (n_l, sin_phi_l, ls[i])):
            t = streamwise_tangent(stream)
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
    R_mm, control = radius_schedule(s_mm, cfg, s_mm[-1])
    R = R_mm / mm

    # --- Mode B per station ----------------------------------------------
    cot_half = c_half / s_half
    L_B = R * (cot_half - 1.0)
    delta = L_B[:, None] * a_l
    sym_mismatch = np.zeros(n)
    if symmetry_station is not None:
        k = int(symmetry_station)
        ideal = delta[k].copy()
        delta[k, span_axis] = 0.0
        sym_mismatch[k] = np.linalg.norm(ideal - delta[k])
        if R[k] > 0:
            messages.append(
                f"symmetry station {k}: removed spanwise component of Delta "
                f"({sym_mismatch[k] * mm:.4e} mm, {sym_mismatch[k] / max(L_B[k], 1e-300):.3e} of L_B)")
    P_new = le + delta

    h_u = np.einsum("ij,ij->i", delta, n_u)
    h_u_formula = R * (1.0 + np.cos(theta) - np.sin(theta))
    for i in range(n):
        if i == symmetry_station:
            continue
        if abs(h_u[i] - h_u_formula[i]) > 1e-9 * max(R[i], 1e-30) + 1e-15 * scale:
            raise AssertionError(f"station {i}: h_u check failed "
                                 f"({h_u[i]} vs {h_u_formula[i]})")
    t_u = R * cot_half
    t_l = R.copy()
    xi_p = cfg.plateau_factor * t_u / sin_phi_u      # along the upper grid line
    t_l_grid = t_l / sin_phi_l                         # along the lower grid line
    if cfg.blend_length_mm:
        L_b = np.full(n, cfg.blend_length_mm / mm)
    else:
        L_b = 0.30 * chord_u

    # --- feasibility ------------------------------------------------------
    feasible = np.ones(n, dtype=bool)
    reason = [""] * n
    for i in range(n):
        if R[i] == 0.0:
            continue
        why = []
        if degenerate[i]:
            why.append("zero chord")
        else:
            if xi_p[i] + L_b[i] > chord_u[i]:
                why.append("xi_p + L_b > upper chord")
            if t_l_grid[i] > chord_l[i]:
                why.append("t_l = R > lower chord")
        if why:
            feasible[i] = False
            reason[i] = "; ".join(why)
    R_max = np.where(degenerate, 0.0,
                     chord_u * sin_phi_u / (cfg.plateau_factor * cot_half))

    # --- apply to grids ---------------------------------------------------
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
        w = smoothstep_weight(xi, xi_p[i], L_b[i])
        up_new = up + w[:, None] * delta[i]
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

    table = {
        "station": np.arange(n),
        "s_mm": s_mm,
        "x": P_new[:, 0] * mm, "y": P_new[:, 1] * mm, "z": P_new[:, 2] * mm,
        "theta_deg": np.degrees(theta),
        "R_mm": R_mm,
        "L_B_mm": L_B * mm,
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

    bad = int(np.count_nonzero(~feasible))
    if bad:
        messages.append(f"{bad} infeasible station(s): {[int(i) for i in np.flatnonzero(~feasible)]}")
    for msg in messages:
        logger.info("[LE comp] %s", msg)

    return CompensationResult(cfg, new_us, new_ls, le, P_new, table, control, messages)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

STATION_COLUMNS = ["station", "s_mm", "x", "y", "z", "theta_deg", "R_mm",
                   "L_B_mm", "h_u_mm", "t_u_mm", "feasible"]


def write_station_csv(result, path):
    """Per-station table for CAD entry; x, y, z are P' in mm (half model)."""
    t = result.table
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(STATION_COLUMNS)
        for i in t["station"]:
            w.writerow([int(i)] + [f"{t[c][i]:.6f}" for c in STATION_COLUMNS[1:-1]]
                       + [bool(t["feasible"][i])])


def write_control_points_csv(result, path):
    """
    Variable-fillet control points: s_k on the original LE, arc length of the
    same point along the new edge, its 3D position on the new edge (mm), R_k.
    """
    cp = result.control_points
    if cp is None:
        raise ValueError("control points exist only in variable fillet mode")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "s_mm", "s_new_edge_mm", "x", "y", "z", "R_mm"])
        for k in range(len(cp["k"])):
            x, y, z = cp["xyz_mm"][k]
            w.writerow([int(cp["k"][k])] + [f"{v:.6f}" for v in
                       (cp["s_mm"][k], cp["s_new_mm"][k], x, y, z, cp["R_mm"][k])])


def compensated_copy(waverider, result):
    """Shallow copy of a waverider object carrying the compensated streams."""
    import copy
    wr = copy.copy(waverider)
    wr.upper_surface_streams = result.upper_streams
    wr.lower_surface_streams = result.lower_streams
    wr.leading_edge = result.le_new
    return wr


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

    for ax, (key, label) in zip(axes, panels):
        ax.set_facecolor(_SURFACE)
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
    if bad.any() or cp is not None:
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
