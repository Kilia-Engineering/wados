"""Aerodynamic evaluation for the Liu 2019 waverider.

Solvers
-------
All three are local-inclination (panel) methods: each triangle is assigned
a pressure from its own angle ``delta`` to the freestream, then forces are
summed. They differ only in the ``Cp(delta)`` law.

``"cone"`` (default)
    Tangent-cone. Windward panels take the Taylor-Maccoll cone-surface
    pressure; leeward panels expand through Prandtl-Meyer. Exact for a
    cone, which is what this geometry is derived from, so it is the right
    default here.
``"oblique"``
    Tangent-wedge. Windward panels take the 2D oblique-shock pressure.
    Correct for the flat region, but it ignores the three-dimensional
    relief behind a conical shock and so runs 28-55% high on the curved
    span.
``"newtonian"``
    Modified Newtonian, ``Cp = Cp_max(Ma) * sin^2(delta)``. The original
    implementation; kept for comparison and for shapes where impact
    theory is the appropriate idealisation.

Measured surface Cp at this vehicle's centreline body angle (8.385 deg),
with the cone column exact:

    Ma      cone      wedge            Newtonian
     6   0.05256   0.08149 (1.55x)   0.03866 (0.74x)
    13   0.04665   0.05968 (1.28x)   0.03902 (0.84x)

Normalisation
-------------
``_evaluate_surface`` returns per-panel force already divided by the
free-stream dynamic pressure (``dF/q_inf = -Cp * A * n_hat``), so the
summed force carries units of area. The coefficients are therefore

    CL, CD = (F . dir) / S_ref                  [m^2 / m^2 -> dimensionless]
    Cmz    = Mz / (S_ref * L_ref)               [m^3 / m^3 -> dimensionless]
    Xcp    = (Mz / Fy) / L_ref                  [m   / m   -> fraction of L]

which is the standard convention and is internally consistent: at
``alpha = 0`` the definitions force ``Cmz / CL == Xcp`` identically, and
the evaluator reproduces that to machine precision. ``S_ref = 1.0 m^2``
comes from paper Section 4.2; ``L_ref = 6.0 m`` is the vehicle length.

``L/D`` and ``Xcp`` are independent of ``S_ref``, so they are the only
quantities here that can be compared against the paper without trusting
the reference area. ``Xcp`` agrees to within 2.8% on the default solver.

What the shock-based solvers fixed, and what they did not
---------------------------------------------------------
They fixed the **CL Mach trend**. Newtonian's ``Cp_max(Ma)`` moves only
0.9% from Ma 6 to 13, so it predicts an essentially Mach-invariant CL.
Shock-based Cp falls properly (normalised to Ma 6):

    Ma    Newtonian   oblique    cone    paper Fig. 12
     6       1.000      1.000   1.000       1.000
    13       1.009      0.727   0.885       0.514

They did **not** fix the L/D trend, and no pressure model can. At fixed
``alpha`` an inviscid local-inclination method gives L/D close to
``cot(delta_effective)``, which is a property of the shape: all three
solvers sit at 6.25-6.45 across the whole trajectory while Fig. 12 rises
4.4 -> 5.8. That rise is a viscous effect -- skin-friction drag falling
faster than pressure drag along the constant-q trajectory -- so closing
it needs a solver that carries friction, e.g. the vendored ``pysagas``
panel method. Base drag is likewise absent (the triangulation has no base
panels).

Absolute CL remains far above Fig. 12 on every solver (+152% on the exact
cone model at Ma 6), and improving the pressure physics moves it further
out, not closer. Since the Fig. 12 CD and Cmz columns are already proven
mis-scaled, and ``S_ref`` is unverified, that gap is more likely a
reference-data or reference-area problem than a solver one: matching the
paper's CL with exact cone pressures would require ``S_ref`` near
2.5 m^2 rather than 1.0.

See :data:`liu2019.config.PAPER_REFERENCE_AERO` for why the Fig. 12 CD and
Cmz columns are not usable as references.
"""

from typing import Dict, Optional

import numpy as np

from .config import (
    MOMENT_REF,
    PAPER_REFERENCE_AERO,
    PAPER_TRAJECTORY,
    REF_AREA_M2,
    REF_LENGTH_M,
)
from .geometry import Liu2019Waverider
from .shock import (
    DetachedShockError,
    beta_detachment,
    cp_vacuum,
    isentropic_pressure_ratio,
    mach_angle,
    oblique_shock_ratios,
    prandtl_meyer,
    taylor_maccoll_cone_field,
    theta_from_beta_Ma,
)

SOLVERS = ("cone", "oblique", "newtonian")


# ---------------------------------------------------------------------------
# Newtonian Cp
# ---------------------------------------------------------------------------

def _cp_max_newtonian(Ma, gamma=1.4):
    """Modified-Newtonian Cp_max (stagnation Cp behind a normal shock)."""
    if Ma <= 1.0:
        return 2.0
    M2 = Ma * Ma
    p02_over_p1 = (
        ((gamma + 1.0) ** 2 * M2) /
        (4.0 * gamma * M2 - 2.0 * (gamma - 1.0))
    ) ** (gamma / (gamma - 1.0)) * (
        (1.0 - gamma + 2.0 * gamma * M2) / (gamma + 1.0)
    )
    return (2.0 / (gamma * M2)) * (p02_over_p1 - 1.0)


# ---------------------------------------------------------------------------
# Tangent-wedge / Prandtl-Meyer Cp  ("oblique" solver)
# ---------------------------------------------------------------------------
#
# Local-inclination method. Each panel is treated as a 2D wedge at its own
# angle to the freestream:
#
#   delta > 0 (windward)  -> attached oblique shock, Cp from p2/p1
#   delta < 0 (leeward)   -> Prandtl-Meyer expansion, floored at the vacuum
#                            limit Cp = -2/(gamma*Ma^2)
#   delta > theta_max     -> shock would detach; continued with a Newtonian
#                            sin^2 law scaled to match at theta_max, so Cp
#                            stays continuous. The Liu geometry never reaches
#                            this branch (panels sit at 4.8-8.4 deg against a
#                            theta_max of ~42 deg at Ma 6), but a user-edited
#                            design could.
#
# Both branches are built as FORWARD maps -- sweep the shock angle beta, and
# sweep the post-expansion Mach -- so no root-finding is needed and the whole
# curve comes out monotonic in delta. The result is cached per (Ma, gamma)
# and applied to the ~80k panels with a single np.interp.
#
# Why this and not modified Newtonian: Cp_max(Ma) moves only 0.9% from Ma 6
# to Ma 13, so Newtonian predicts an essentially Mach-invariant CL and an
# exactly Mach-invariant L/D. Shock-based Cp falls ~27-37% over the same
# range, which is the trend paper Fig. 12 reports.

_CP_TABLE_CACHE = {}


def _cp_table(Ma, gamma=1.4, n_compression=1200, n_expansion=1200):
    """Return ``(delta_deg, Cp)`` sampled monotonically in local inclination.

    ``delta_deg`` runs from the deepest expansion (most negative) up to
    +90 deg, so :func:`numpy.interp` clamps sensibly at both ends: the
    leeward end holds the vacuum Cp, the windward end the detached-shock
    continuation.
    """
    key = (round(float(Ma), 9), round(float(gamma), 9))
    cached = _CP_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    Ma = float(Ma)
    q_scale = 2.0 / (gamma * Ma * Ma)

    # ---- windward: sweep beta from the Mach angle to detachment ----------
    mu = mach_angle(Ma)
    b_det = beta_detachment(Ma, gamma)
    betas = np.linspace(mu, b_det, int(n_compression))
    thetas = np.array([theta_from_beta_Ma(b, Ma, gamma) for b in betas])
    M1n2 = (Ma * np.sin(np.radians(betas))) ** 2
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * np.clip(M1n2 - 1.0, 0.0, None)
    cp_comp = q_scale * (p2_p1 - 1.0)

    # theta(beta) is monotonic on the weak branch; enforce it against
    # round-off so np.interp gets a strictly increasing x.
    thetas[0] = 0.0
    cp_comp[0] = 0.0
    keep = np.concatenate([[True], np.diff(thetas) > 1e-12])
    thetas, cp_comp = thetas[keep], cp_comp[keep]

    # ---- detached continuation: Newtonian sin^2, scaled to match --------
    th_max = float(thetas[-1])
    cp_at_max = float(cp_comp[-1])
    sin2_max = np.sin(np.radians(th_max)) ** 2
    th_ext = np.linspace(th_max, 90.0, 40)[1:]
    cp_ext = cp_at_max * (np.sin(np.radians(th_ext)) ** 2) / max(sin2_max, 1e-30)

    # ---- leeward: sweep the post-expansion Mach -------------------------
    nu1 = prandtl_meyer(Ma, gamma)
    M2 = np.geomspace(Ma, 1.0e5, int(n_expansion))
    turn = prandtl_meyer(M2, gamma) - nu1            # 0 -> (nu_max - nu1)
    cp_exp = q_scale * (isentropic_pressure_ratio(Ma, M2, gamma) - 1.0)
    turn[0] = 0.0
    cp_exp[0] = 0.0
    keep = np.concatenate([[True], np.diff(turn) > 1e-12])
    turn, cp_exp = turn[keep], cp_exp[keep]
    # Past the maximum turn the flow has expanded to vacuum; hold it there
    # out to -90 deg so interp clamping is physical rather than accidental.
    cp_exp = np.maximum(cp_exp, cp_vacuum(Ma, gamma))
    if turn[-1] < 90.0:
        turn = np.append(turn, 90.0)
        cp_exp = np.append(cp_exp, cp_vacuum(Ma, gamma))

    delta = np.concatenate([-turn[::-1], thetas[1:], th_ext])
    cp = np.concatenate([cp_exp[::-1], cp_comp[1:], cp_ext])
    table = (delta, cp)
    _CP_TABLE_CACHE[key] = table
    return table


def cp_tangent_wedge(Ma, delta_deg, gamma=1.4):
    """Cp at local inclination ``delta_deg`` (scalar or array), degrees."""
    delta, cp = _cp_table(Ma, gamma)
    out = np.interp(np.asarray(delta_deg, dtype=float), delta, cp)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# Tangent-cone Cp  ("cone" solver, the default)
# ---------------------------------------------------------------------------
#
# Same local-inclination idea, but each windward panel is treated as a CONE
# at its own angle rather than a wedge, with the surface pressure taken from
# the Taylor-Maccoll solution. For a cone-derived shape this is the right
# model: 2D wedge theory ignores the three-dimensional relief behind a
# conical shock and so over-predicts surface pressure.
#
# Measured on this geometry at the centreline body angle (8.385 deg), Cp is
#
#     Ma      tangent-cone   tangent-wedge   modified Newtonian
#      6         0.05256        0.08149  (1.55x)   0.03866  (0.74x)
#      8         0.04956        0.07054  (1.42x)   0.03886  (0.78x)
#     10         0.04795        0.06458  (1.35x)   0.03895  (0.81x)
#     13         0.04665        0.05968  (1.28x)   0.03902  (0.84x)
#
# The cone column is exact for a cone, so the other two bracket it: Newtonian
# ~25% low with no Mach trend at all, tangent-wedge 28-55% high.
#
# The leeward branch is Prandtl-Meyer, identical to the wedge solver -- there
# is no conical relief to model in an expansion.

_CONE_TABLE_CACHE = {}


def _cp_table_cone(Ma, gamma=1.4, n_beta=90, n_expansion=1200):
    """``(delta_deg, Cp)`` using Taylor-Maccoll cone flow on the windward side.

    Built as a forward sweep over the shock angle: one T-M integration per
    beta yields both the cone half-angle it corresponds to and the surface
    velocity, hence Cp. delta_c(beta) is monotonic, so the result is already
    ordered for :func:`numpy.interp`. Cached per ``(Ma, gamma)``.
    """
    key = (round(float(Ma), 9), round(float(gamma), 9))
    cached = _CONE_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    Ma = float(Ma)
    q_scale = 2.0 / (gamma * Ma * Ma)
    mu = mach_angle(Ma)
    b_det = beta_detachment(Ma, gamma)

    deltas = [0.0]
    cps = [0.0]
    # The T-M integration is the expensive part, so keep n_beta modest; the
    # curve is smooth and linear interpolation between ~90 points is well
    # inside the accuracy of the surrounding model.
    for b in np.linspace(mu * 1.01, b_det * 0.995, int(n_beta)):
        try:
            Vr, Vt, dc_rad, _ = taylor_maccoll_cone_field(Ma, float(b), gamma)
            ratios = oblique_shock_ratios(Ma, float(b), gamma)
        except (DetachedShockError, ValueError, RuntimeError):
            continue
        v = float(np.hypot(float(Vr(dc_rad)), float(Vt(dc_rad))))
        denom = 0.5 * (gamma - 1.0) * (1.0 - v * v)
        if denom <= 0.0:
            continue
        Ma_s = np.sqrt(v * v / denom)
        # Total pressure is conserved along streamlines behind the shock, so
        # the surface pressure follows isentropically from the post-shock state.
        ps_p1 = isentropic_pressure_ratio(
            ratios["Ma2"], Ma_s, gamma) * ratios["p2_p1"]
        d = float(np.degrees(dc_rad))
        if d > deltas[-1] + 1e-9:
            deltas.append(d)
            cps.append(q_scale * (ps_p1 - 1.0))

    deltas = np.array(deltas)
    cps = np.array(cps)
    if deltas.size < 3:                       # pathological Ma; fall back
        return _cp_table(Ma, gamma)

    # Detached continuation, continuous at the last solved point.
    d_max, cp_at_max = float(deltas[-1]), float(cps[-1])
    sin2_max = np.sin(np.radians(d_max)) ** 2
    d_ext = np.linspace(d_max, 90.0, 40)[1:]
    cp_ext = cp_at_max * (np.sin(np.radians(d_ext)) ** 2) / max(sin2_max, 1e-30)

    # Leeward side: Prandtl-Meyer, reused from the wedge table.
    w_delta, w_cp = _cp_table(Ma, gamma, n_expansion=n_expansion)
    lee = w_delta < 0.0

    delta = np.concatenate([w_delta[lee], deltas, d_ext])
    cp = np.concatenate([w_cp[lee], cps, cp_ext])
    table = (delta, cp)
    _CONE_TABLE_CACHE[key] = table
    return table


def cp_tangent_cone(Ma, delta_deg, gamma=1.4):
    """Cp at local inclination ``delta_deg`` using tangent-cone theory."""
    delta, cp = _cp_table_cone(Ma, gamma)
    out = np.interp(np.asarray(delta_deg, dtype=float), delta, cp)
    return float(out) if out.ndim == 0 else out


def _panel_triangles(X, Y, Z):
    """Return (N, 3, 3) array of triangle vertices for a structured mesh."""
    P = np.stack([X, Y, Z], axis=-1)
    p0 = P[:-1, :-1]
    p1 = P[1:,  :-1]
    p2 = P[1:,  1:]
    p3 = P[:-1, 1:]
    tri_a = np.stack([p0, p1, p2], axis=-2).reshape(-1, 3, 3)
    tri_b = np.stack([p0, p2, p3], axis=-2).reshape(-1, 3, 3)
    return np.concatenate([tri_a, tri_b], axis=0)


def _tri_areas_centroids_normals(tris, outward_sign):
    """Compute per-triangle area, centroid, outward unit normal.

    ``outward_sign`` = +1 for upper surface (normal should point +y, away
    from the vehicle interior), -1 for lower surface (normal should point
    -y, into the freestream).
    """
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    centroids = (v0 + v1 + v2) / 3.0
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / np.where(areas[:, None] > 0, 2.0 * areas[:, None], 1.0)
    if outward_sign > 0:
        flip = normals[:, 1] < 0
    else:
        flip = normals[:, 1] > 0
    normals[flip] *= -1.0
    return areas, centroids, normals


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Liu2019AeroEvaluator:
    """Local-inclination aerodynamic evaluator for a Liu 2019 waverider.

    Reference length = 6.0 m, reference area = 1.0 m2 (paper Sec. 4.2).
    Pitching moment is taken about the nose (0, 0, 0).

    ``solver`` selects the pressure law -- ``"cone"`` (default, exact for
    this cone-derived geometry), ``"oblique"`` or ``"newtonian"``; see the
    module docstring. It can be set per evaluator or overridden per call:

    >>> ev = Liu2019AeroEvaluator(wr)                 # tangent-cone
    >>> ev.evaluate(6.0)["CL"]                        # doctest: +SKIP
    >>> ev.evaluate(6.0, solver="newtonian")["CL"]    # doctest: +SKIP
    """

    def __init__(self, waverider: Liu2019Waverider,
                 gamma: float = 1.4,
                 ref_length: float = REF_LENGTH_M,
                 ref_area: float = REF_AREA_M2,
                 moment_ref=MOMENT_REF,
                 solver: str = "cone"):
        if solver not in SOLVERS:
            raise ValueError(
                f"solver must be one of {SOLVERS}, got {solver!r}")
        self.wr = waverider
        self.gamma = float(gamma)
        self.ref_length = float(ref_length)
        self.ref_area = float(ref_area)
        self.moment_ref = np.asarray(moment_ref, dtype=float)
        self.solver = str(solver)

        X_u, Y_u, Z_u = waverider.upper_surface(mirror=True)
        X_l, Y_l, Z_l = waverider.lower_surface(mirror=True)
        self._upper_tris = _panel_triangles(X_u, Y_u, Z_u)
        self._lower_tris = _panel_triangles(X_l, Y_l, Z_l)

    # ------------------------------------------------------------------
    def _evaluate_surface(self, tris, outward_sign, flow_dir, cp_model):
        areas, centroids, normals = _tri_areas_centroids_normals(
            tris, outward_sign)
        # Local inclination to the freestream:
        #   sin(delta) = -n . flow_dir, positive windward.
        # NOT clipped here -- the oblique solver needs the negative
        # (leeward) values to drive the Prandtl-Meyer branch. The
        # Newtonian model does its own clipping in its cp_model.
        sin_delta = -np.einsum("ij,j->i", normals, flow_dir)
        Cp = cp_model(sin_delta)
        # Non-dim force per panel: dF/q_inf = -Cp * A * n_hat
        forces = -(Cp * areas)[:, None] * normals
        return forces, centroids, Cp, areas

    def _cp_model(self, Ma, solver):
        """Return a callable mapping ``sin(delta)`` per panel to ``Cp``."""
        if solver == "newtonian":
            Cp_max = _cp_max_newtonian(Ma, self.gamma)
            def model(sin_delta):
                return Cp_max * np.clip(sin_delta, 0.0, 1.0) ** 2
            return model
        if solver == "cone":
            delta_grid, cp_grid = _cp_table_cone(Ma, self.gamma)
        else:
            delta_grid, cp_grid = _cp_table(Ma, self.gamma)
        def model(sin_delta):
            delta_deg = np.degrees(np.arcsin(np.clip(sin_delta, -1.0, 1.0)))
            return np.interp(delta_deg, delta_grid, cp_grid)
        return model

    def evaluate(self, Ma, alpha_deg=0.0,
                 atm_conditions: Optional[Dict] = None,
                 solver: Optional[str] = None) -> Dict[str, float]:
        solver = self.solver if solver is None else str(solver)
        if solver not in SOLVERS:
            raise ValueError(
                f"solver must be one of {SOLVERS}, got {solver!r}")
        alpha = np.radians(alpha_deg)
        flow_dir = np.array([np.cos(alpha), -np.sin(alpha), 0.0])
        lift_dir = np.array([np.sin(alpha),  np.cos(alpha), 0.0])

        Cp_max = _cp_max_newtonian(Ma, self.gamma)
        cp_model = self._cp_model(Ma, solver)

        F_u, C_u, _, _ = self._evaluate_surface(
            self._upper_tris, +1, flow_dir, cp_model)
        F_l, C_l, _, _ = self._evaluate_surface(
            self._lower_tris, -1, flow_dir, cp_model)

        forces = np.concatenate([F_u, F_l], axis=0)
        centroids = np.concatenate([C_u, C_l], axis=0)

        F_total = forces.sum(axis=0)
        CD = float(np.dot(F_total, flow_dir)) / self.ref_area
        CL = float(np.dot(F_total, lift_dir)) / self.ref_area

        arms = centroids - self.moment_ref
        moments = np.cross(arms, forces).sum(axis=0)
        Cmz = float(moments[2]) / (self.ref_area * self.ref_length)

        L_D = CL / CD if abs(CD) > 1e-12 else 0.0

        if abs(F_total[1]) > 1e-12:
            Xcp = (moments[2] / F_total[1]) / self.ref_length
        else:
            Xcp = 0.0

        return {
            "Ma":  float(Ma),
            "alpha_deg": float(alpha_deg),
            "CL":  CL,
            "CD":  CD,
            "L_D": L_D,
            "Cmz": Cmz,
            "Xcp": Xcp,
            "Cp_max": float(Cp_max),
            "solver": solver,
        }

    # ------------------------------------------------------------------
    def evaluate_paper_trajectory(self, progress_callback=None, solver=None):
        results = []
        for i, row in enumerate(PAPER_TRAJECTORY):
            out = self.evaluate(row["Ma"], row["alpha"], solver=solver)
            out["H_km"] = row["H_km"]
            results.append(out)
            if progress_callback is not None:
                try:
                    progress_callback(i + 1)
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------
    def compare_with_paper(self, solver=None):
        """Return list of dicts comparing computed to PAPER_REFERENCE_AERO."""
        rows = []
        for r in self.evaluate_paper_trajectory(solver=solver):
            ref = PAPER_REFERENCE_AERO.get(int(r["Ma"]), {})
            rows.append({
                "Ma":  r["Ma"],
                "CL":  (r["CL"],  ref.get("CL")),
                "CD":  (r["CD"],  ref.get("CD")),
                "L_D": (r["L_D"], ref.get("L_D")),
                "Cmz": (r["Cmz"], ref.get("Cmz")),
                "Xcp": (r["Xcp"], ref.get("Xcp")),
            })
        return rows
