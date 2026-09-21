"""Axisymmetric Method of Characteristics (MOC) for VMPLO.

Spec reference: VMPLO_implementation_prompt.md ``vmplo/moc.py``.

Used by :func:`powerlaw.solve_osculating_plane` when the per-plane
generating body is non-conical (|n - 1| >= CONE_TOL).  Builds a
characteristic mesh in the (x, r) plane from a post-shock initial data
line to the exit plane x = L, then integrates the leading-edge
streamline through the mesh via RK4.

Governing equations along the C+/C- characteristics (Shapiro Vol. II,
Zucrow & Hoffman Vol. II):

    Along C+:   dnu + dalpha = -S_Cp * dr
    Along C-:   dnu - dalpha = +S_Cm * dr

where nu(Ma) is the Prandtl-Meyer function, alpha is the local flow
angle, and the axisymmetric source terms are

    S_Cm = sin(alpha) * sin(mu) / (r * cos(alpha + mu))
    S_Cp = sin(alpha) * sin(mu) / (r * cos(alpha - mu))

with mu = arcsin(1/Ma).  The source terms vanish for 2D planar flow
(r -> inf) and for purely conical flow, but are nonzero for power-law
bodies with n != 1 — the reason T-M cannot be used there.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from waverider_generator.vmplo.shock import oblique_shock_ratios

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
#  Prandtl-Meyer function and inversion                                   #
# ---------------------------------------------------------------------- #

def prandtl_meyer(Ma: float, gamma: float = 1.4) -> float:
    """Prandtl-Meyer function nu(Ma) in radians (Ma > 1)."""
    if Ma <= 1.0:
        return 0.0
    g = np.sqrt((gamma + 1.0) / (gamma - 1.0))
    return float(
        g * np.arctan(np.sqrt((gamma - 1.0) / (gamma + 1.0) * (Ma**2 - 1.0)))
        - np.arctan(np.sqrt(Ma**2 - 1.0))
    )


def Ma_from_prandtl_meyer(nu_rad: float, gamma: float = 1.4,
                          tol: float = 1e-10, max_iter: int = 50) -> float:
    """Invert nu(Ma) via Newton's method.  Returns Ma >= 1."""
    if nu_rad <= 0.0:
        return 1.0001
    Ma = max(1.001, 1.0 + nu_rad)
    for _ in range(max_iter):
        f = prandtl_meyer(Ma, gamma) - nu_rad
        # d nu / d Ma
        fp = np.sqrt(max(Ma**2 - 1.0, 0.0)) / (
            Ma * (1.0 + (gamma - 1.0) / 2.0 * Ma**2))
        if abs(fp) < 1e-15:
            break
        step = f / fp
        Ma -= step
        if Ma < 1.0001:
            Ma = 1.0001
        if abs(f) < tol:
            break
    return float(max(1.0001, Ma))


# ---------------------------------------------------------------------- #
#  Source terms and point-dict helpers                                    #
# ---------------------------------------------------------------------- #

def _source_Cm(pt: dict) -> float:
    """Axisymmetric source term for the C- compatibility equation."""
    mu = np.arcsin(1.0 / pt["Ma"])
    denom = pt["r"] * np.cos(pt["alpha"] + mu)
    if abs(denom) < 1e-30:
        return 0.0
    return float(np.sin(pt["alpha"]) * np.sin(mu) / denom)


def _source_Cp(pt: dict) -> float:
    """Axisymmetric source term for the C+ compatibility equation."""
    mu = np.arcsin(1.0 / pt["Ma"])
    denom = pt["r"] * np.cos(pt["alpha"] - mu)
    if abs(denom) < 1e-30:
        return 0.0
    return float(np.sin(pt["alpha"]) * np.sin(mu) / denom)


def _make_point(x: float, r: float, alpha: float, Ma: float,
                gamma: float = 1.4) -> dict:
    """Build a MOC point dict with pre-computed nu and mu."""
    Ma = max(Ma, 1.0001)
    return {
        "x":     float(x),
        "r":     float(r),
        "alpha": float(alpha),
        "Ma":    float(Ma),
        "nu":    prandtl_meyer(Ma, gamma),
        "mu":    float(np.arcsin(1.0 / Ma)),
    }


# ---------------------------------------------------------------------- #
#  Initial data line                                                      #
# ---------------------------------------------------------------------- #

def initial_data_line(x_LE: float, r_LE: float, beta_design_deg: float,
                      Ma_inf: float, gamma: float = 1.4,
                      N: int = 12, n_body: float = 1.0) -> list[dict]:
    """N points along the shock at x = x_LE, from r=r_LE outward.

    All points share the same post-shock state because beta is constant
    across the osculating plane (VMPLO assumption).  Point 0 is the
    leading edge; subsequent points march along the shock toward the
    axis (r = 0).

    Shock direction in (x, r): ``(+cos beta, -sin beta)``.

    Resolution scales continuously with ``1/n_body``: a low-n body is
    concave-up and has steeper alpha gradients near the LE, so it needs
    more points along the initial line. The constant is set by convergence
    at the hardest case (beta = 16 deg, L_s = 0.05, Ma 8-14, n = 0.4),
    where spanwise trailing-edge curvature falls

        points    50      60      80     100     140
        max d2y  122.0    27.0    19.7    19.4    13.0
        volume   2.886   2.902   2.934   2.945   2.948

    i.e. the old ``20/n`` (50 points at n = 0.4) sat on the wrong side of
    a resolution cliff. ``32/n`` gives 80 points there, which is past the
    cliff and within 0.5% of the volume at 140.

    An earlier revision ("Phase 4 Fix 2") applied that boost only below
    ``n_body < 0.7`` and, in the same branch, switched the spacing from
    uniform to Chebyshev ``0.5 (1 - cos(pi t))``. Two problems:

    * It made the whole model discontinuous in ``n``. At beta = 16 deg,
      L_s = 0.05, Ma 8-14, crossing n = 0.699 -> 0.700 moved the vehicle
      volume 3.898 -> 3.442 m^3, an 11.7% step with no physics behind it.
    * Chebyshev clusters points at *both* ends of the arc. The far end is
      the shock-axis intersection, where the axisymmetric source terms
      (both carry 1/r) are singular. Packing points into that region
      feeds bad data into the mesh while thinning out the middle, where
      the traced streamline actually runs -- so the branch intended to
      help low-n bodies was making them rougher, not smoother.

    Spacing is therefore uniform for every ``n``, and the point count is a
    continuous function of it. ``n_body`` now only sets resolution.
    """
    beta = np.radians(beta_design_deg)
    post = oblique_shock_ratios(Ma_inf, beta_design_deg, gamma)
    alpha0 = np.radians(post["theta_deg"])
    Ma2 = post["Ma2"]

    # arc-length along the shock to the axis; stop short to avoid r=0
    xi_max = r_LE / max(np.sin(beta), 1e-6)

    # Continuous resolution law, uniform spacing at every n.
    n_actual = int(np.ceil(max(float(N), 32.0 / max(float(n_body), 0.2))))
    xis_norm = np.linspace(0.0, 1.0, n_actual)
    xis = xis_norm * (0.95 * xi_max)

    pts: list[dict] = []
    for xi in xis:
        x_s = x_LE + xi * np.cos(beta)
        r_s = r_LE - xi * np.sin(beta)
        if r_s <= 1e-9:
            break
        pts.append(_make_point(x_s, r_s, alpha0, Ma2, gamma))
    return pts


# ---------------------------------------------------------------------- #
#  Main MOC mesh                                                          #
# ---------------------------------------------------------------------- #

class MOCGrid:
    """Axisymmetric MOC mesh on the (x, r) domain of one osculating plane.

    Parameters
    ----------
    initial_points : list of point dicts
        Output of :func:`initial_data_line`.  Defines the upstream
        boundary (column 0).
    body : PowerLawBody
        Generating body (used by the wall-point solver for flow tangency).
    gamma : float, optional

    Attributes
    ----------
    cols : list[list[dict]]
        Columns of the characteristic mesh (column 0 = shock initial
        data, column k > 0 built by ``march``).
    """

    def __init__(self, initial_points: list[dict], body, gamma: float = 1.4):
        self.body = body
        self.gamma = float(gamma)
        self.cols: list[list[dict]] = [list(initial_points)]
        self.x_max = float(body.L)

    # ------------------------------------------------------------------ #
    #  Interior point (predictor-corrector)                              #
    # ------------------------------------------------------------------ #

    def _intersect_and_solve(self, A: dict, B: dict,
                             SCm: float, SCp: float) -> dict | None:
        """Core linear algebra for interior point from (A on C-, B on C+)."""
        gamma = self.gamma
        sl_Cm = np.tan(A["alpha"] - A["mu"])
        sl_Cp = np.tan(B["alpha"] + B["mu"])
        denom = sl_Cm - sl_Cp
        if abs(denom) < 1e-12:
            return None

        x_P = ((B["r"] - A["r"] - sl_Cp * B["x"] + sl_Cm * A["x"]) / denom)
        r_P = A["r"] + sl_Cm * (x_P - A["x"])

        if r_P < 1e-9 or x_P > self.x_max + 1e-6:
            return None
        # also reject retrograde x
        if x_P < min(A["x"], B["x"]) - 1e-9:
            return None

        dr_Cm = r_P - A["r"]
        dr_Cp = r_P - B["r"]

        # Compatibility:
        #   nu_P - nu_A - (alpha_P - alpha_A) = SCm * dr_Cm   (C-)
        #   nu_P - nu_B + (alpha_P - alpha_B) = -SCp * dr_Cp  (C+)
        # Rearranging:
        RHS_Cm = A["nu"] + A["alpha"] + SCm * dr_Cm
        RHS_Cp = B["nu"] - B["alpha"] - SCp * dr_Cp
        nu_P = 0.5 * (RHS_Cm + RHS_Cp)
        alpha_P = 0.5 * (RHS_Cm - RHS_Cp)
        if nu_P <= 0.0 or not np.isfinite(nu_P) or not np.isfinite(alpha_P):
            return None
        Ma_P = Ma_from_prandtl_meyer(nu_P, gamma)
        return _make_point(x_P, r_P, alpha_P, Ma_P, gamma)

    def _interior_point(self, A: dict, B: dict) -> dict | None:
        """Predictor-corrector interior point."""
        gamma = self.gamma
        P_pred = self._intersect_and_solve(A, B, _source_Cm(A), _source_Cp(B))
        if P_pred is None:
            return None

        def avg_key(p1: dict, p2: dict, key: str) -> float:
            return 0.5 * (p1[key] + p2[key])

        A_avg = _make_point(A["x"], A["r"],
                            avg_key(A, P_pred, "alpha"),
                            avg_key(A, P_pred, "Ma"), gamma)
        B_avg = _make_point(B["x"], B["r"],
                            avg_key(B, P_pred, "alpha"),
                            avg_key(B, P_pred, "Ma"), gamma)
        SCm_avg = 0.5 * (_source_Cm(A) + _source_Cm(P_pred))
        SCp_avg = 0.5 * (_source_Cp(B) + _source_Cp(P_pred))
        P_corr = self._intersect_and_solve(A_avg, B_avg, SCm_avg, SCp_avg)
        return P_corr if P_corr is not None else P_pred

    # ------------------------------------------------------------------ #
    #  Wall point                                                         #
    # ------------------------------------------------------------------ #

    def _wall_point(self, A: dict) -> dict | None:
        """C- characteristic from A intersects the body surface at W."""
        body = self.body
        gamma = self.gamma
        sl_Cm = np.tan(A["alpha"] - A["mu"])

        # Newton iteration for x_W: r_A + sl_Cm*(x_W - x_A) == body.radius(x_W)
        slope_at_A = body.slope(A["x"])
        denom = sl_Cm - slope_at_A
        if abs(denom) < 1e-15:
            return None
        x_W = A["x"] + (body.radius(A["x"]) - A["r"]) / denom
        x_W = float(np.clip(x_W, A["x"] + 1e-8, self.x_max))

        for _ in range(30):
            r_line = A["r"] + sl_Cm * (x_W - A["x"])
            r_bdy = float(body.radius(x_W))
            f = r_line - r_bdy
            fp = sl_Cm - float(body.slope(x_W))
            if abs(fp) < 1e-15:
                break
            x_W -= f / fp
            x_W = float(np.clip(x_W, A["x"] + 1e-8, self.x_max))
            # Phase 4 Fix 1: tightened from 1e-10 to 1e-12. Two orders
            # tighter than the previous tolerance; eight orders looser
            # than float64 epsilon (~2.2e-16) so convergence-stall risk
            # is negligible. Existing fallback (return None on
            # non-convergence after 30 iter) preserved.
            if abs(f) < 1e-12:
                break

        if x_W > self.x_max + 1e-6:
            return None

        r_W = float(body.radius(x_W))
        alpha_W = float(body.slope_angle_rad(x_W))   # flow tangency BC
        SCm_A = _source_Cm(A)

        # Predictor via C- compatibility with A source only
        nu_W_pred = A["nu"] - (alpha_W - A["alpha"]) + SCm_A * (r_W - A["r"])
        if nu_W_pred <= 0.0 or not np.isfinite(nu_W_pred):
            nu_W_pred = 1e-6
        Ma_W_pred = Ma_from_prandtl_meyer(nu_W_pred, gamma)
        W_pred = _make_point(x_W, r_W, alpha_W, Ma_W_pred, gamma)

        # Corrector with averaged sources
        SCm_avg = 0.5 * (SCm_A + _source_Cm(W_pred))
        nu_W = A["nu"] - (alpha_W - A["alpha"]) + SCm_avg * (r_W - A["r"])
        if nu_W <= 0.0 or not np.isfinite(nu_W):
            nu_W = 1e-6
        Ma_W = Ma_from_prandtl_meyer(nu_W, gamma)
        return _make_point(x_W, r_W, alpha_W, Ma_W, gamma)

    # ------------------------------------------------------------------ #
    #  Column-by-column march                                             #
    # ------------------------------------------------------------------ #

    def march(self, n_columns: int = 30) -> None:
        """Extend the mesh forward in x until the exit plane is reached
        or ``n_columns`` columns have been added."""
        for _ in range(n_columns):
            prev = self.cols[-1]
            if len(prev) < 2:
                break
            if all(p["x"] >= self.x_max - 1e-6 for p in prev):
                break

            new_col: list[dict] = []
            for i in range(len(prev) - 1):
                P = self._interior_point(prev[i], prev[i + 1])
                if P is not None:
                    new_col.append(P)
            if new_col:
                W = self._wall_point(new_col[-1])
                if W is not None:
                    new_col.append(W)
            if not new_col:
                break
            self.cols.append(new_col)
            self._alpha_cache = None      # mesh changed; drop the interpolator

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    def all_points(self) -> list[dict]:
        return [p for col in self.cols for p in col]

    def _alpha_interpolator(self):
        """Build (once) and cache the alpha interpolator over the mesh.

        ``LinearNDInterpolator`` Delaunay-triangulates its input, so
        constructing one costs far more than evaluating it. This used to be
        rebuilt on *every* ``interpolate_alpha`` call -- four per RK4 step
        in :func:`extract_streamline` -- which is why the step count had to
        be kept tiny to stay affordable, and a tiny step count is what made
        traced streamlines noisy from plane to plane.

        The mesh is built once by :meth:`march` and read-only afterwards,
        so the cache is keyed on the point count and cleared by ``march``.
        """
        pts = self.all_points()
        n = len(pts)
        cached = getattr(self, "_alpha_cache", None)
        if cached is not None and cached[0] == n:
            return cached[1], cached[2]

        xs = np.array([p["x"] for p in pts])
        rs = np.array([p["r"] for p in pts])
        als = np.array([p["alpha"] for p in pts])
        interp = None
        if n >= 3:
            try:
                interp = LinearNDInterpolator(
                    np.column_stack([xs, rs]), als)
            except Exception:
                interp = None
        self._alpha_cache = (n, interp, (xs, rs, als))
        return interp, (xs, rs, als)

    def interpolate_alpha(self, x: float, r: float) -> float:
        """Interpolate flow angle ``alpha(x, r)`` from the mesh."""
        pts = self.all_points()
        if len(pts) < 3:
            return pts[0]["alpha"] if pts else 0.0

        interp, (xs, rs, als) = self._alpha_interpolator()
        if interp is not None:
            try:
                val = interp([[x, r]])[0]
                if np.isfinite(val):
                    return float(val)
            except Exception:
                pass

        # Nearest-neighbour fallback (query outside the convex hull).
        dist = (xs - x) ** 2 + (rs - r) ** 2
        return float(als[int(np.argmin(dist))])


# ---------------------------------------------------------------------- #
#  Streamline extractor                                                   #
# ---------------------------------------------------------------------- #

def extract_streamline(grid: MOCGrid, x0: float, r0: float,
                       alpha0: float, n_steps: int = 300
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Integrate dr/dx = tan(alpha(x, r)) from (x0, r0) to grid.x_max via RK4.

    ``alpha0`` (radians) seeds the initial flow angle at (x0, r0) — used
    only for the initial guess; subsequent steps interpolate from the
    MOC mesh.

    Returns arrays of length ``n_steps + 1``.
    """
    xs = np.linspace(x0, grid.x_max, n_steps + 1)
    if xs.size < 2:
        return xs, np.full_like(xs, r0)
    dx = xs[1] - xs[0]
    r = float(r0)
    r_arr = np.empty(n_steps + 1)
    r_arr[0] = r0

    # Seed interpolator; if it fails (e.g. degenerate grid), fall back
    # to the seed angle alpha0.
    for i, x in enumerate(xs[:-1]):
        try:
            a1 = grid.interpolate_alpha(x, r)
        except Exception:
            a1 = alpha0
        k1 = np.tan(a1)

        try:
            a2 = grid.interpolate_alpha(x + dx / 2.0, r + dx / 2.0 * k1)
        except Exception:
            a2 = a1
        k2 = np.tan(a2)

        try:
            a3 = grid.interpolate_alpha(x + dx / 2.0, r + dx / 2.0 * k2)
        except Exception:
            a3 = a2
        k3 = np.tan(a3)

        try:
            a4 = grid.interpolate_alpha(x + dx, r + dx * k3)
        except Exception:
            a4 = a3
        k4 = np.tan(a4)

        r += dx / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        r_arr[i + 1] = max(r, 0.0)

    return xs, r_arr
