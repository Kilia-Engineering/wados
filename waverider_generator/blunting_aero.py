"""
Aerodynamics of a blunted (rounded) leading edge, for analysis of a sharp mesh.

The Aero Analysis tab analyses the sharp mesh with PySAGAS; a leading edge
rounded to radius R adds (``aero_corrections`` applies these):

1. **Leading-edge pressure force.** Modified Newtonian flow on a circular
   cylinder of radius R along each sharp LE edge, over the arc between the
   two surfaces' tangent points (the arc a fillet or G2 rounding puts in
   place of the edge). Cp = Cp_max (n . -v)^2 on the windward part, with
   the Rayleigh-pitot Cp_max [1]. Normal to the flow, a full half-cylinder
   has a drag of (4/3) Cp_max R q per unit length; sweep Lambda lowers it by
   cos^3(Lambda).
2. **Pressure bleed.** Overpressure behind the rounded LE on the
   compression side, Delta P / P_inf from Jessen et al. [2], Eq. (12), as the
   Planar tab uses it. It was fitted to planar waveriders; elsewhere it is an
   estimate.
3. **Stagnation-line heat flux.** Tauber-Sutton with the Beckwith-Cohen
   sweep factor sqrt(cos Lambda) [3], ``gvwd.heating.fay_riddell``.

Not modelled: the strip of sharp-wedge surface the rounding removes (about
R / tan(wedge angle) wide), and the bow shock's effect on the rest of the
flow beyond the bleed term.

The mesh frame is the analysis frame: x streamwise, y up, z span.

References:
  [1] Anderson (2006), Hypersonic and High-Temperature Gas Dynamics, 2nd ed., Sec. 3.3
  [2] Jessen, Larsson, Brehm (2026), Aerospace Science and Technology 172, 111703
  [3] Tauber & Sutton (1991), JSR 28(1); Beckwith & Cohen (1961), NASA TN D-625
"""

import numpy as np

SWEEP_EXPONENT = 2.2       # sweep-scaled radius R cos^2.2(Lambda), as leading_edge_blunting


def cp_max(M_inf, gamma=1.4):
    """Rayleigh-pitot stagnation Cp (modified Newtonian)."""
    from gvwd.thermo.newtonian import cp_max_modified_newtonian

    return cp_max_modified_newtonian(M_inf, gamma)


def sharp_leading_edges(vertices, faces, flow_dir, min_turn_deg=120.0):
    """Leading-edge segments of a closed triangle mesh with outward normals.

    An edge is on the leading edge when its two faces' normals differ by more
    than ``min_turn_deg`` (a wedge angle under 180 - min_turn_deg) and their
    bisector faces upstream; a sharp trailing edge faces downstream.

    Returns ``(p0, p1, n_a, n_b)``, each (m, 3): edge end points and the two
    faces' outward unit normals.
    """
    v = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    u = np.asarray(flow_dir, float) / np.linalg.norm(flow_dir)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    i = np.flatnonzero(same)                    # edges shared by faces owner[i], owner[i+1]
    na, nb = n[owner[i]], n[owner[i + 1]]
    turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", na, nb), -1.0, 1.0)))
    leading = (turn > min_turn_deg) & ((na + nb) @ u < 0.0)
    e = edges[i[leading]]
    return v[e[:, 0]], v[e[:, 1]], na[leading], nb[leading]


def edge_radius(p0, p1, radius, sweep_scaled=False):
    """LE radius per edge: uniform, or R cos^2.2 of the planform sweep
    (``leading_edge_blunting.compute_sweep_scaled_radius``)."""
    if not sweep_scaled:
        return np.full(len(p0), float(radius))
    t = np.asarray(p1, float) - np.asarray(p0, float)
    cos_sweep = np.abs(t[:, 2]) / np.maximum(np.hypot(t[:, 0], t[:, 2]), 1e-300)
    return radius * cos_sweep ** SWEEP_EXPONENT


def flow_sweep(p0, p1, flow_dir):
    """Sweep of each edge relative to the flow: sin(Lambda) = |t . v|."""
    t = np.asarray(p1, float) - np.asarray(p0, float)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    u = np.asarray(flow_dir, float) / np.linalg.norm(flow_dir)
    return np.arcsin(np.clip(np.abs(t @ u), 0.0, 1.0))


def newtonian_arc_force(p0, p1, n_a, n_b, radius, flow_dir, cp_max_value, n_arc=48):
    """Modified Newtonian force / q_inf (m^2) on the rounded edges.

    Each edge becomes a circular arc of radius ``radius`` (per edge) whose
    outward normal turns from ``n_a`` to ``n_b`` through the front. The
    windward part carries Cp = Cp_max (n . -v)^2; the force on the body is
    -Cp n per unit area. Returns the (m, 3) forces.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    na, nb = np.asarray(n_a, float), np.asarray(n_b, float)
    u = np.asarray(flow_dir, float) / np.linalg.norm(flow_dir)
    t = p1 - p0
    length = np.linalg.norm(t, axis=1)
    t /= np.maximum(length[:, None], 1e-300)
    # the arc lies in the plane normal to the edge
    na = na - np.einsum("ij,ij->i", na, t)[:, None] * t
    nb = nb - np.einsum("ij,ij->i", nb, t)[:, None] * t
    na /= np.maximum(np.linalg.norm(na, axis=1, keepdims=True), 1e-300)
    nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-300)
    angle = np.arccos(np.clip(np.einsum("ij,ij->i", na, nb), -1.0, 1.0))
    s = (np.arange(n_arc) + 0.5) / n_arc                       # midpoint rule
    sin_angle = np.maximum(np.sin(angle), 1e-12)[:, None, None]
    w_a = np.sin((1.0 - s)[None, :, None] * angle[:, None, None]) / sin_angle
    w_b = np.sin(s[None, :, None] * angle[:, None, None]) / sin_angle
    n = w_a * na[:, None, :] + w_b * nb[:, None, :]            # (m, n_arc, 3), unit
    windward = np.clip(-(n @ u), 0.0, None)
    cp = cp_max_value * windward ** 2
    d_phi = (angle / n_arc)[:, None]
    r = np.broadcast_to(np.asarray(radius, float), length.shape)
    return -np.einsum("ij,ijk->ik", cp * d_phi, n) * (r * length)[:, None]


def pressure_bleed(M_inf, alpha_rad, sweep_rad, x_over_R):
    """Delta P / P_inf behind a rounded LE (Jessen et al. 2026, Eq. 12)."""
    from planar_waverider_aero import pressure_bleed_correction

    return pressure_bleed_correction(M_inf, alpha_rad, sweep_rad, x_over_R)


def le_heat_flux(rho_inf, V_inf, radius, sweep_rad):
    """Stagnation-line heat flux (W/m^2) per edge (Tauber-Sutton, swept)."""
    from gvwd.heating.fay_riddell import swept_LE_heat_flux

    r = np.broadcast_to(np.asarray(radius, float), np.shape(sweep_rad))
    return np.array([swept_LE_heat_flux(rho_inf, V_inf, ri, si) if ri > 0 else np.inf
                     for ri, si in zip(r, np.asarray(sweep_rad, float))])
