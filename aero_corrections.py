"""Base-pressure and skin-friction corrections to PySAGAS results.

PySAGAS (oblique shock / Prandtl-Meyer) is inviscid, and it sets every face
turned more than 20 deg away from the flow to vacuum (p = 0). On a waverider
that is the base. Both matter for L/D: on the He & Emanuel waverider, L/D
drops from 7.0 (laminar friction, freestream base pressure) to 5.82
(turbulent) and 4.74 (with a base-pressure correlation); CFD gives 4.33
(Igra 2019, Table 1, docs/refs R7).

``corrected_coefficients`` takes a solved PySAGAS ``OPM`` solver and

* gives the vacuum faces a base pressure p_b (``BASE_MODELS``):

  - ``"igra"``: Igra 2018 Eq. (6) [R6], the correlation of Guidi et al. 2003,
    Cp_b = 2/(g M^2) [(2/(g+1))^1.4 (1/M)^2.8 (2 g M^2 - (g-1))/(g+1) - 1];
  - ``"vacuum"``: p_b = 0 (PySAGAS as is);
  - ``"maxwell"``: p_b = p_inf / M_inf (the Planar tab's model);
  - ``"freestream"``: p_b = p_inf;

* adds the skin friction of every other face (``FRICTION_MODELS``): Eckert
  reference temperature T* = T_e (0.5 + 0.039 M_e^2 + 0.5 T_w/T_e) at the
  face's PySAGAS edge state, Re*_x = rho* u_e x / mu* with x the distance
  from the nose along the flow, local Cf = 0.664/sqrt(Re*_x) (laminar) or
  0.0592/Re*_x^0.2 (turbulent), shear Cf (1/2) rho* u_e^2 along the surface
  in the freestream direction. The relations are those of
  ``pswr.aero.viscous``.

Forces and moments use PySAGAS's own conventions (its cells carry inward
normals and a face's pressure force is n p A; moments about ``cog``), so the
corrections add to PySAGAS's net force and moment and go through its
``FlowResults.coefficients``. Lengths are in metres (``length_unit`` converts
other mesh units for the Reynolds number).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BASE_MODELS = {
    "igra": "Igra 2018 correlation",
    "vacuum": "vacuum (p_b = 0)",
    "maxwell": "p_inf / M_inf",
    "freestream": "freestream (p_b = p_inf)",
}
FRICTION_MODELS = {
    "turbulent": "turbulent",
    "laminar": "laminar",
    "transition": "transition",
    "off": "off (inviscid)",
}
VACUUM_METHOD = -1                 # PySAGAS cell attribute "method" for unsolved faces


@dataclass
class AeroCorrections:
    """Which corrections to apply. The defaults are the recommended ones."""

    base_pressure: str = "igra"
    friction: str = "turbulent"
    wall_temperature: float = 1000.0       # K
    transition_re: float = 1.0e6           # Re*_x at transition ("transition" only)
    le_radius: float = 0.0                 # m; rounded leading edge, 0 = sharp
    le_radius_sweep_scaled: bool = False   # R cos^2.2(planform sweep) instead of uniform
    le_bleed: bool = True                  # pressure bleed behind the rounded LE

    def __post_init__(self):
        if self.base_pressure not in BASE_MODELS:
            raise ValueError(f"unknown base-pressure model {self.base_pressure!r}")
        if self.friction not in FRICTION_MODELS:
            raise ValueError(f"unknown friction model {self.friction!r}")

    @property
    def active(self) -> bool:
        return self.base_pressure != "vacuum" or self.friction != "off" or self.le_radius > 0

    def describe(self) -> str:
        friction = FRICTION_MODELS[self.friction]
        if self.friction == "transition":
            friction += f" at Re_x = {self.transition_re:.3g}"
        if self.friction != "off":
            friction += f", T_w = {self.wall_temperature:.0f} K"
        text = f"Base pressure: {BASE_MODELS[self.base_pressure]}; skin friction: {friction}"
        if self.le_radius > 0:
            text += (f"; blunt LE: R = {self.le_radius * 1000:.3g} mm "
                     f"({'sweep-scaled' if self.le_radius_sweep_scaled else 'uniform'}), "
                     f"pressure bleed {'on' if self.le_bleed else 'off'}")
        return text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AeroCorrections":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


INVISCID = AeroCorrections(base_pressure="vacuum", friction="off")


def base_pressure_ratio(model: str, mach: float, gamma: float = 1.4) -> float:
    """p_b / p_inf for a base-pressure model."""
    if model == "vacuum":
        return 0.0
    if model == "freestream":
        return 1.0
    if model == "maxwell":
        return 1.0 / mach
    if model == "igra":
        g, M = gamma, mach
        cp = 2.0 / (g * M * M) * ((2.0 / (g + 1.0)) ** 1.4 * (1.0 / M) ** 2.8
                                  * (2.0 * g * M * M - (g - 1.0)) / (g + 1.0) - 1.0)
        return 1.0 + 0.5 * g * M * M * cp
    raise ValueError(f"unknown base-pressure model {model!r}")


def friction_forces(n, A, c, mach_e, p_e, T_e, flow_dir, nose, *, wall_temperature, model,
                    transition_re=1.0e6, gamma=1.4, R=287.0, length_unit=1.0):
    """Friction force on each face (N, along the surface in the flow direction).

    ``n``, ``c``: (N, 3) face normals (either sense) and centroids; ``A``: areas;
    ``mach_e``, ``p_e``, ``T_e``: edge state per face; ``flow_dir``: freestream
    direction; ``nose``: the most upstream point of the body. Returns
    ``(F, Re_x)``, the (N, 3) forces and Re*_x per face.
    """
    from pswr.aero.viscous import cf_laminar, cf_turbulent, eckert_reference_T, sutherland_viscosity

    n, c = np.asarray(n, float), np.asarray(c, float)
    A, M, p, T = (np.asarray(a, float) for a in (A, mach_e, p_e, T_e))
    v = np.asarray(flow_dir, float) / np.linalg.norm(flow_dir)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    t = v - (n @ v)[:, None] * n                       # flow direction on the face
    t_norm = np.linalg.norm(t, axis=1)
    t = np.divide(t, t_norm[:, None], out=np.zeros_like(t), where=t_norm[:, None] > 1e-9)

    x = np.maximum((c - np.asarray(nose, float)) @ v, 0.0) * length_unit   # from the nose, along the flow
    T_star = eckert_reference_T(T, M, wall_temperature)
    rho_star = p / (R * T_star)
    u_e = M * np.sqrt(gamma * R * T)
    Re = rho_star * u_e * x / sutherland_viscosity(T_star)
    Re_safe = np.maximum(Re, 1.0)
    cf_lam, cf_turb = cf_laminar(Re_safe), cf_turbulent(Re_safe)
    if model == "laminar":
        cf = cf_lam
    elif model == "turbulent":
        cf = cf_turb
    elif model == "transition":
        cf = np.where(Re < transition_re, cf_lam, cf_turb)
    else:
        raise ValueError(f"unknown friction model {model!r}")
    tau = 0.5 * rho_star * u_e * u_e * cf
    return (tau * A * length_unit ** 2)[:, None] * t, Re


def corrected_coefficients(solver, A_ref: float, c_ref: float = 1.0,
                           corrections: AeroCorrections | None = None, *,
                           cog=(0.0, 0.0, 0.0), length_unit: float = 1.0) -> dict:
    """CL, CD, Cm of a solved PySAGAS OPM solver, with the corrections applied.

    Returns ``CL``, ``CD``, ``Cm`` (corrected), ``CL_inviscid``, ``CD_inviscid``,
    ``Cm_inviscid`` (PySAGAS as is), ``CD_base`` and ``CD_friction`` (the drag
    each correction adds; ``CD_base`` is negative when the base pressure is
    above vacuum), ``CD_le`` and ``CD_bleed`` (the rounded leading edge's
    pressure force and pressure bleed, ``blunt_le_corrections``), ``CL_blunt``
    (the lift they add), ``q_LE_max`` (peak stagnation-line heat flux, W/m^2;
    NaN for a sharp edge), ``LE_length``, ``base_faces``, ``Re_max`` and
    ``corrections`` (a description).
    """
    from pysagas.cfd.solver import FlowResults
    from pysagas.geometry import Vector

    corr = corrections or AeroCorrections()
    result = solver.flow_result
    flow = result.freestream
    cells = solver.cells
    n = np.array([cell.n.vec for cell in cells], float)
    c = np.array([cell.c.vec for cell in cells], float)
    A = np.array([cell.A for cell in cells], float)
    attr = [cell.attributes for cell in cells]
    points = np.array([[q.vec for q in (cell.p0, cell.p1, cell.p2)] for cell in cells], float).reshape(-1, 3)
    method = np.array([a.get("method", 0) for a in attr], int)
    vacuum = method == VACUUM_METHOD
    r = c - np.asarray(cog, float)

    def coefficients(force, moment):
        return FlowResults(flow, Vector.from_coordinates(force),
                           Vector.from_coordinates(moment)).coefficients(A_ref=A_ref, c_ref=c_ref)

    CL0, CD0, Cm0 = result.coefficients(A_ref=A_ref, c_ref=c_ref)
    F, Mom = np.array(result.net_force.vec, float), np.array(result.net_moment.vec, float)

    p_b = base_pressure_ratio(corr.base_pressure, flow.M, flow.gamma) * flow.P
    F_base = n[vacuum] * (p_b * A[vacuum])[:, None]            # PySAGAS: F = n p A
    dF_base, dM_base = F_base.sum(axis=0), np.cross(r[vacuum], F_base).sum(axis=0)

    dF_fric, dM_fric, Re_max = np.zeros(3), np.zeros(3), 0.0
    wet = ~vacuum
    if corr.friction != "off" and wet.any():
        pick = lambda key: np.array([attr[i][key] for i in np.flatnonzero(wet)], float)
        F_fric, Re = friction_forces(
            n[wet], A[wet], c[wet], pick("Mach"), pick("pressure"), pick("temperature"),
            flow.direction.vec, points[np.argmin(points @ np.asarray(flow.direction.vec, float))],
            wall_temperature=corr.wall_temperature, model=corr.friction,
            transition_re=corr.transition_re, gamma=flow.gamma, R=flow.R, length_unit=length_unit)
        dF_fric, dM_fric = F_fric.sum(axis=0), np.cross(r[wet], F_fric).sum(axis=0)
        Re_max = float(Re.max())

    blunt = blunt_le_corrections(corr, flow, points, n, c, A, method, r, length_unit)
    dF = dF_base + dF_fric + blunt["F_le"] + blunt["F_bleed"]
    dM = dM_base + dM_fric + blunt["M_le"] + blunt["M_bleed"]
    CL, CD, Cm = coefficients(F + dF, Mom + dM)
    _, CD_base, _ = coefficients(dF_base, dM_base)
    _, CD_fric, _ = coefficients(dF_fric, dM_fric)
    CL_le, CD_le, _ = coefficients(blunt["F_le"], blunt["M_le"])
    CL_bleed, CD_bleed, _ = coefficients(blunt["F_bleed"], blunt["M_bleed"])
    return {"CL": float(CL), "CD": float(CD), "Cm": float(Cm),
            "CL_inviscid": float(CL0), "CD_inviscid": float(CD0), "Cm_inviscid": float(Cm0),
            "CD_base": float(CD_base), "CD_friction": float(CD_fric),
            "CD_le": float(CD_le), "CD_bleed": float(CD_bleed),
            "CL_blunt": float(CL_le + CL_bleed), "q_LE_max": blunt["q_LE_max"],
            "LE_length": blunt["LE_length"], "le_radius": corr.le_radius,
            "base_faces": int(vacuum.sum()), "Re_max": Re_max,
            "corrections": corr.describe()}


def blunt_le_corrections(corr: AeroCorrections, flow, points, n, c, A, method, r,
                         length_unit: float = 1.0) -> dict:
    """Force and moment of a leading edge rounded to ``corr.le_radius``.

    The leading edge is found on the mesh (``points``: the cells' corners, 3
    per cell, outward winding). Its rounded arc carries modified Newtonian
    pressure; the compression faces (PySAGAS oblique or normal shock) get the
    pressure bleed of the nearest LE edge, at their distance x behind it
    along the flow (``waverider_generator.blunting_aero``). ``n`` are
    PySAGAS's inward normals, ``r`` the centroids relative to the moment
    reference.
    """
    zero = np.zeros(3)
    out = {"F_le": zero, "M_le": zero, "F_bleed": zero, "M_bleed": zero,
           "q_LE_max": float("nan"), "LE_length": 0.0}
    if corr.le_radius <= 0.0:
        return out
    from waverider_generator import blunting_aero as B

    u = np.asarray(flow.direction.vec, float)
    u /= np.linalg.norm(u)
    # weld the cells' corners into a mesh, to find the edges shared by two faces
    size = float(np.ptp(points, axis=0).max())
    _, inverse = np.unique(np.round(points / (1e-9 * size)), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    vertices = np.zeros((inverse.max() + 1, 3))
    vertices[inverse] = points
    p0, p1, na, nb = B.sharp_leading_edges(vertices, inverse.reshape(-1, 3), u)
    if len(p0) == 0:
        return out
    # Forces in PySAGAS's convention (mesh units taken as metres); the radius
    # is converted to mesh units
    radius = B.edge_radius(p0, p1, corr.le_radius / length_unit, corr.le_radius_sweep_scaled)
    mid = 0.5 * (p0 + p1)
    reference = (c - r)[0]                               # the moment reference
    F_le = B.newtonian_arc_force(p0, p1, na, nb, radius, u, B.cp_max(flow.M, flow.gamma)) * flow.q
    out["F_le"] = F_le.sum(axis=0)
    out["M_le"] = np.cross(mid - reference, F_le).sum(axis=0)
    sweep = B.flow_sweep(p0, p1, u)
    q_le = B.le_heat_flux(flow.rho, flow.v, radius * length_unit, sweep)
    out["q_LE_max"] = float(q_le[np.isfinite(q_le)].max()) if np.isfinite(q_le).any() else float("nan")
    out["LE_length"] = float(np.linalg.norm(p1 - p0, axis=1).sum() * length_unit)

    compression = np.isin(method, (2, 3)) if corr.le_bleed else np.zeros(len(c), bool)
    if compression.any():
        cc = c[compression]
        lateral = lambda p: p - (p @ u)[:, None] * u
        lat_c, lat_m = lateral(cc), lateral(mid)
        nearest = np.empty(len(cc), int)
        for k in range(0, len(cc), 2048):                # nearest LE edge across the flow
            d = np.linalg.norm(lat_c[k:k + 2048, None, :] - lat_m[None, :, :], axis=2)
            nearest[k:k + 2048] = d.argmin(axis=1)
        x = (cc - mid[nearest]) @ u
        behind = (x > 0.0) & (radius[nearest] > 0.0)
        alpha = float(np.arctan2(u[1], u[0]))
        dp = np.zeros(len(cc))
        dp[behind] = B.pressure_bleed(flow.M, alpha, sweep[nearest][behind],
                                      x[behind] / radius[nearest][behind]) * flow.P
        F_bleed = n[compression] * (dp * A[compression])[:, None]          # F = n p A
        out["F_bleed"] = F_bleed.sum(axis=0)
        out["M_bleed"] = np.cross(r[compression], F_bleed).sum(axis=0)
    return out
