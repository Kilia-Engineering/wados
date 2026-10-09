"""LTOCs core: one stream surface per leading-edge point (spec Phase 3).

Method: Zheng et al. (2020), AIAA J 58(8):3499-3513 (R1), Sec. II.

* **Leading edge** (R1 Sec. IV.A): the flow capture tube (FCT), a curve in
  the base plane, is projected upstream along the freestream onto the shock.
* **Step A** (R1 Sec. II.A): the shock curve of the stream surface starts at
  the leading-edge point and is traced with ``trace_shock_curves``. It runs
  past the base plane until the body streamline is determined up to the
  base (spec flag 4; R3 Sec. 1.4 extends the shock the same way).
* **Step B** (R1 Sec. II.B): the shock curve is developed into the meridian
  plane with the intrinsic mapping (``x = X``, ``dy/dx = tan(beta)``; Gate 0
  decision). The flow is then solved with the noncoaxial inverse MOC
  (``solve_inverse``, streamline mesh; Gate 1 decision). The local axis sits
  at ``y_axis = y - r``. Where the shock is flat (``r = inf``), r is capped,
  which is the planar limit (spec flag 7). Column 0 of the mesh is the
  streamline from the leading-edge point, i.e. the body.
* **Step C** (R1 Sec. II.C, Eqs. 10-11, vector form; spec section 4.3): each
  mesh point is placed in 3-D from its two parents L and U. L and U are the
  triangle's upper vertices, R1's A1, A2 for C2 and D1, C2 for D2 (Fig. 4):
  ``P = F' + j``. ``F'`` is the foot of the perpendicular from P onto L-U,
  interpolated linearly. ``j`` is parallel to ``V - (V.i) i/|i|^2`` with
  ``i = U - L``, and is scaled so its X component equals the meridian
  ``x_P - x_F'``. The 3-D velocity at P is "decomposed ... according to the
  geometric relationship of interior points" (R1 Sec. II.C). Here that means
  the image of the meridian flow direction under the triangle's
  meridian-to-3-D map. Its components along L-U and normal to it are scaled
  by the 3-D/meridian length ratios ``|i| / |U - L|_m`` and ``|j| / |P - F'|_m``,
  so the velocity is tangent to the 3-D streamline.

Accuracy: when the stream surface is planar (cone, osculating cone, symmetry
plane), the map is an isometry and Step C is exact, so the mesh keeps the
kernel's second order. When the stream surface twists, the chord through L
and U sags off the surface differently in 3-D than in the meridian plane.
That gives an O(h^2) error per row, so the body converges at first order
(Gate 3 report). Interpolating the 3-D rows with cubics removes the sag
mismatch, but the cubic row direction feeds position errors back into ``j``
and the march goes unstable, so R1's linear interpolation is kept.

Shock points get the exact 3-D post-shock velocity direction:
``V2 = V_inf - (1 - rho_inf/rho_2) (V_inf . n) n``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .moc_noncoaxial import (
    VALID, AxisRule, MOCError, MOCSolution, shock_initial_line, solve_inverse,
)
from .shock_geometry import ShockCurve, trace_shock_curves
from .shock_surface import ShockSurface

__all__ = [
    "LTOCError",
    "BodyLine",
    "StreamSurface",
    "find_leading_edge",
    "solve_stream_surface",
    "step_c",
    "r1_frame_to_gui",
    "identity_frame",
]

X_HAT = np.array([1.0, 0.0, 0.0])


class LTOCError(ValueError):
    """A stream surface or waverider could not be built."""


def identity_frame(P):
    return np.asarray(P, dtype=float)


def r1_frame_to_gui(P):
    """R1's waverider frame (X, Y, Z down; R1 Figs. 20-21, 24) to the WADOS GUI
    frame (x streamwise, y up, z span): ``(x, y, z) = (X, -Z, Y)``. This is a
    proper rotation, so handedness and normals are preserved."""
    P = np.asarray(P, dtype=float)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], axis=-1)


# ---------------------------------------------------------------------------
#  Leading edge
# ---------------------------------------------------------------------------

def find_leading_edge(surface: ShockSurface, fct_yz, n_grid: int = 61, tol: float = 1e-13,
                      max_iter: int = 50):
    """Project base-plane FCT points upstream onto the shock (R1 Sec. IV.A).

    ``fct_yz`` is an (n, 2) array of transverse coordinates (Y, Z). For each
    point, the parameters (u, v) solve ``S_y = Y, S_z = Z``. The start is the
    nearest node of a coarse grid on the nominal domain, refined by Newton.

    Returns ``(uv, points, converged)``.
    """
    fct = np.atleast_2d(np.asarray(fct_yz, dtype=float))
    (u0, u1), (v0, v1) = surface.u_range, surface.v_range
    ug = np.linspace(u0, u1, n_grid)
    vg = np.linspace(v0, v1, n_grid)
    U, V = np.meshgrid(ug, vg, indexing="ij")
    G = surface.point(U, V)[..., 1:].reshape(-1, 2)
    uv = np.empty((fct.shape[0], 2))
    ok = np.zeros(fct.shape[0], dtype=bool)
    scale = max(1.0, float(np.max(np.abs(G))))
    for k, target in enumerate(fct):
        i = int(np.argmin(np.sum((G - target) ** 2, axis=1)))
        u, v = U.ravel()[i], V.ravel()[i]
        for _ in range(max_iter):
            d = surface.evaluate(u, v)
            res = d.S[1:] - target
            if np.max(np.abs(res)) < tol * scale:
                ok[k] = True
                break
            J = np.array([[d.Su[1], d.Sv[1]], [d.Su[2], d.Sv[2]]])
            try:
                du, dv = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError:
                break
            u, v = u + du, v + dv
        uv[k] = (u, v)
    return uv, surface.point(uv[:, 0], uv[:, 1]), ok


# ---------------------------------------------------------------------------
#  Stream surface
# ---------------------------------------------------------------------------

@dataclass
class BodyLine:
    """Body streamline of one stream surface (shock frame), truncated at the base plane."""

    x: np.ndarray            # (k,)
    points: np.ndarray       # (k, 3)
    velocity: np.ndarray     # (k, 3) unit velocity direction
    p: np.ndarray            # (k,) p / p_inf
    rho: np.ndarray          # (k,) rho / rho_inf
    V: np.ndarray            # (k,) V / sqrt(p_inf / rho_inf)
    M: np.ndarray            # (k,)


@dataclass
class StreamSurface:
    """Result for one leading-edge station."""

    station: int
    status: str                          # "ok", "tip" or a failure keyword
    message: str = ""
    le_uv: Optional[np.ndarray] = None
    le_point: Optional[np.ndarray] = None
    curve: Optional[ShockCurve] = None
    moc: Optional[MOCSolution] = None
    points3: Optional[np.ndarray] = None     # (rows, cols, 3) Step C positions
    velocity3: Optional[np.ndarray] = None   # (rows, cols, 3) unit velocity
    body: Optional[BodyLine] = None
    x_end: float = np.nan                    # end of the (extended) shock curve
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "tip")


def _post_shock_direction(curve: ShockCurve, rho_ratio: np.ndarray) -> np.ndarray:
    """Exact 3-D post-shock velocity direction at the shock points."""
    g = curve.geometry
    vn = np.sin(g.beta)                                       # V_inf . n with V_inf = X
    v = X_HAT - ((1.0 - 1.0 / rho_ratio) * vn)[:, None] * g.normal
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def step_c(curve: ShockCurve, sol: MOCSolution, rho_ratio: np.ndarray,
           j_x_min: float = 0.05):
    """Map the meridian-plane mesh to 3-D (R1 Sec. II.C, Eqs. 10-11).

    Returns ``(points3, velocity3, n_fallback)``. ``n_fallback`` counts points
    where the X-component scaling of R1 was ill-conditioned
    (``|j_x| < j_x_min``), so ``j`` was scaled to the meridian distance
    instead. Under the intrinsic mapping the two scalings agree.
    """
    n_rows, n_cols = sol.x.shape
    P3 = np.full((n_rows, n_cols, 3), np.nan)
    V3 = np.full((n_rows, n_cols, 3), np.nan)
    P3[0] = curve.points
    V3[0] = _post_shock_direction(curve, rho_ratio)
    n_fallback = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for d in range(1, n_rows):
            m = n_cols - d
            ok = sol.valid[d, :m]
            Lm = np.stack([sol.x[d - 1, :m], sol.y[d - 1, :m]], axis=1)
            Um = np.stack([sol.x[d - 1, 1:m + 1], sol.y[d - 1, 1:m + 1]], axis=1)
            Pm = np.stack([sol.x[d, :m], sol.y[d, :m]], axis=1)
            seg = Um - Lm
            s = np.sum((Pm - Lm) * seg, axis=1) / np.sum(seg * seg, axis=1)
            Fm = Lm + s[:, None] * seg
            Xj = Pm[:, 0] - Fm[:, 0]
            dist = np.linalg.norm(Pm - Fm, axis=1)
            # Meridian angles: segment phi, flow theta. P lies on the right of L -> U.
            phi = np.arctan2(seg[:, 1], seg[:, 0])
            th = sol.theta[d, :m]

            L3, U3 = P3[d - 1, :m], P3[d - 1, 1:m + 1]
            i3 = U3 - L3
            ihat = i3 / np.linalg.norm(i3, axis=1, keepdims=True)
            F3 = L3 + s[:, None] * i3
            VF = (1.0 - s)[:, None] * V3[d - 1, :m] + s[:, None] * V3[d - 1, 1:m + 1]
            jdir = VF - np.sum(VF * ihat, axis=1, keepdims=True) * ihat     # R1 Eqs. 10-11
            jhat = jdir / np.linalg.norm(jdir, axis=1, keepdims=True)
            # Orient j toward P's side; the flow and P are both on the right of L -> U
            # in the meridian plane when theta < phi.
            side_v = np.sign(np.sin(phi - th))
            jhat = jhat * np.where(side_v < 0, -1.0, 1.0)[:, None]
            scale_x = Xj / jhat[:, 0]
            use_x = (np.abs(jhat[:, 0]) >= j_x_min) & (scale_x > 0)
            jlen = np.where(use_x, scale_x, dist)
            n_fallback += int(np.sum(ok & ~use_x))
            P3[d, :m] = np.where(ok[:, None], F3 + jlen[:, None] * jhat, np.nan)
            # Meridian flow direction mapped to 3-D with the triangle's stretch
            # ratios along L-U and along j (both 1 when the map is an isometry).
            lam_r = np.linalg.norm(i3, axis=1) / np.linalg.norm(seg, axis=1)
            lam_n = np.where(dist > 0.0, np.abs(jlen) / np.where(dist > 0.0, dist, 1.0), 1.0)
            vel = (np.cos(th - phi)[:, None] * lam_r[:, None] * ihat
                   + np.sin(phi - th)[:, None] * lam_n[:, None] * jhat)
            vel = vel / np.linalg.norm(vel, axis=1, keepdims=True)
            V3[d, :m] = np.where(ok[:, None], vel, np.nan)
    return P3, V3, n_fallback


def _truncate_body(sol: MOCSolution, P3, V3, x_base: float) -> BodyLine:
    col = sol.streamline_column(0)
    k = col.x.size
    pts, vel = P3[:k, 0], V3[:k, 0]
    idx = int(np.searchsorted(col.x, x_base))
    if idx >= k:
        raise LTOCError("body streamline does not reach the base plane")

    def cut(a):
        a = np.asarray(a)
        if idx == 0:
            return a[:1]
        t = (x_base - col.x[idx - 1]) / (col.x[idx] - col.x[idx - 1])
        last = a[idx - 1] + t * (a[idx] - a[idx - 1])
        return np.concatenate([a[:idx], last[None] if a.ndim > 1 else np.array([last])])

    vel_cut = cut(vel)
    vel_cut = vel_cut / np.linalg.norm(vel_cut, axis=1, keepdims=True)
    return BodyLine(x=cut(col.x), points=cut(pts), velocity=vel_cut, p=cut(col.p),
                    rho=cut(col.rho), V=cut(col.V), M=cut(col.M))


def solve_stream_surface(surface: ShockSurface, le_uv, x_base: float, M_inf: float, *,
                         gamma: float = 1.4, n_points: int = 80, extension: Optional[float] = None,
                         max_extension: float = 6.0, r_cap: Optional[float] = None,
                         station: int = 0, tip_tol: float = 1e-9) -> StreamSurface:
    """Steps A-C for the stream surface that starts at ``le_uv``.

    ``n_points`` is the number of shock points between the leading edge and
    the base plane; the spacing is kept the same on the extension. The shock
    curve is first extended past the base by ``extension`` times the local
    chord. The default is 1.2 times the planar shock-layer estimate at the
    leading edge, ``(tan b - tan th) / (tan(th + mu) - tan b)``: the forward
    C+ characteristic from the body at the base reaches the shock that far
    downstream. While the body streamline is not determined up to the base, the
    extension is re-estimated from how far the body got (the body end scales
    almost linearly with the shock end), with 10 % margin and at least a 1.3
    times larger extension, up to ``max_extension``. The march stops once the
    body passes the base plane.
    """
    le_uv = np.asarray(le_uv, dtype=float)
    le_point = surface.point(le_uv[0], le_uv[1])
    chord = x_base - float(le_point[0])
    res = StreamSurface(station=station, status="ok", le_uv=le_uv, le_point=le_point)
    L_ref = max(abs(x_base), abs(chord), 1e-12)
    if chord <= tip_tol * L_ref:
        res.status, res.message = "tip", "leading edge on the base plane (zero chord)"
        return res
    r_cap = 1e6 * L_ref if r_cap is None else r_cap

    if extension is None:
        from .shock_geometry import local_geometry
        from gvwd.thermo.oblique_shock import rankine_hugoniot

        b0 = float(local_geometry(surface, le_uv[0], le_uv[1], L_ref).beta)
        rh = rankine_hugoniot(M_inf, b0, 1.0, 1.0, gamma)
        th, mu = rh["theta"], np.arcsin(1.0 / rh["M2"])
        est = (np.tan(b0) - np.tan(th)) / max(np.tan(th + mu) - np.tan(b0), 1e-6)
        extension = float(np.clip(1.2 * est, 0.3, 0.8 * max_extension))
    ext = extension
    n_solves = 0
    while True:
        x_end = x_base + ext * chord
        n_steps = int(np.ceil(n_points * (x_end - le_point[0]) / chord))
        curve = trace_shock_curves(surface, le_uv[0], le_uv[1], x_end, n_steps=n_steps,
                                   M_inf=M_inf, gamma=gamma, length_scale=L_ref)[0]
        for flag, text in (("sub_mach", "shock angle at or below the Mach angle"),
                           ("detached", "shock angle beyond detachment"),
                           ("concave", "concave shock cross-section (kappa_b < 0)")):
            if curve.flags[flag].any():
                k = int(np.argmax(curve.flags[flag]))
                res.status = flag
                res.message = f"{text} at X = {curve.x[k]:.6g} (spec flag 7: refused)"
                res.curve = curve
                return res
        g = curve.geometry
        r = np.where(np.isfinite(g.r), np.minimum(g.r, r_cap), r_cap)
        try:
            il = shock_initial_line(curve.x, curve.y, g.beta, M_inf, gamma)
            sol = solve_inverse(il, AxisRule.noncoaxial(curve.x, curve.y - r), gamma,
                                scheme="streamline", x_stop=x_base)
        except MOCError as e:
            res.status, res.message, res.curve = "moc_error", str(e), curve
            return res
        n_solves += 1
        col = sol.streamline_column(0)
        if col.x[-1] >= x_base:
            break
        if not col.reached_end:
            f = sol.first_failure()
            res.status = "limit_surface" if f and f["code"] == 4 else "moc_failure"
            res.message = (f"body streamline stops at X = {col.x[-1]:.6g} before the base: "
                           f"{col.stop_reason}" + (f" (first failure at X = {f['x']:.6g})" if f else ""))
            res.curve, res.moc = curve, sol
            return res
        x_le = float(le_point[0])
        reach = (col.x[-1] - x_le) / (x_end - x_le)          # body end / shock end
        ext = max(1.3 * ext, 1.1 * chord / max(reach, 1e-3) / chord - 1.0)
        if ext > max_extension:
            res.status = "undetermined"
            res.message = (f"body not determined up to the base even with the shock extended "
                           f"{max_extension:g} chords past it (spec flag 4)")
            res.curve, res.moc = curve, sol
            return res

    P3, V3, n_fb = step_c(curve, sol, il.rho)
    body = _truncate_body(sol, P3, V3, x_base)
    steps = np.linalg.norm(np.diff(body.points[:-1], axis=0), axis=1)   # last step is the cut
    steps = steps[steps > 0]
    # Determinacy (spec flag 4): the body point of mesh row d depends on shock
    # points 0..d, so the shock was needed up to the row where the body passes
    # the base plane.
    x_needed = float(curve.x[min(sol.x.shape[0] - 1, curve.x.size - 1)])
    # Step C consistency: angle between the body velocity and the tangent of
    # the 3-D body streamline (central differences, interior points).
    Pb, Vb = body.points[:-1], body.velocity[:-1]
    if Pb.shape[0] >= 4:
        T = np.gradient(Pb, Pb[:, 0], axis=0, edge_order=2)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        tang = float(np.degrees(np.max(np.arccos(np.clip(np.sum(T[1:-1] * Vb[1:-1], axis=1),
                                                            -1.0, 1.0)))))
    else:
        tang = 0.0
    res.curve, res.moc, res.points3, res.velocity3, res.body, res.x_end = (
        curve, sol, P3, V3, body, x_end)
    res.diagnostics = {
        "extension_chords": float(ext),
        "extension_needed_chords": (x_needed - x_base) / chord,
        "moc_solves": n_solves,
        "beta_deg": (float(np.degrees(g.beta.min())), float(np.degrees(g.beta.max()))),
        "r_min": float(np.min(r)),
        "axis_drift": float(np.ptp(curve.y_axis[np.isfinite(curve.y_axis)])) if np.isfinite(curve.y_axis).any() else 0.0,
        "azimuth_drift": float(np.ptp(curve.uv[:, 0])),
        "body_points": int(body.x.size),
        "body_step_ratio": float(steps.max() / steps.min()) if steps.size > 1 else 1.0,
        "stepc_fallbacks": n_fb,
        "stepc_tangency_deg": tang,
        "moc_status": sol.status_counts(),
    }
    return res
