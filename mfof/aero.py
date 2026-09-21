"""Aerodynamic evaluator for the MFOF framework.

Thin alias around :class:`liu2019.aero.Liu2019AeroEvaluator`. The Liu
evaluator is fully duck-typed on the waverider object (it only calls
``wr.upper_surface(mirror=True)`` and ``wr.lower_surface(mirror=True)``), so
:class:`MFOFWaverider` works in place of :class:`Liu2019Waverider` without
modification.

The underlying evaluator offers three local-inclination pressure laws --
``"cone"`` (tangent-cone, the default), ``"oblique"`` (tangent-wedge) and
``"newtonian"`` -- selectable per evaluator or per call. See the
:mod:`liu2019.aero` module docstring for how they compare.

Note the default suits a cone-derived waverider. An MFOF waverider built
with the wedge or power-law factory has a different generating body, so
``solver="oblique"`` is the better match for an all-wedge build.
"""

import numpy as np

from liu2019.aero import Liu2019AeroEvaluator, panel_column_index

from .basic_flowfield import BasicFlowfield
from .cone_flowfield import ConeFlowfield
from .power_law_flowfield import PowerLawFlowfield
from .wedge_flowfield import WedgeFlowfield


# Which pressure law matches which generating body.
#
#   ConeFlowfield     -> "cone"     tangent-cone is exact for a cone
#   PowerLawFlowfield -> "cone"     axisymmetric body; tangent-cone captures
#                                   the three-dimensional relief that a
#                                   wedge law would miss
#   WedgeFlowfield    -> "oblique"  tangent-wedge is exact for a 2D wedge
#
# Getting this wrong is not cosmetic: at this vehicle's centreline body
# angle, wedge theory reads 1.55x the tangent-cone Cp at Ma 6.
SOLVER_FOR_FLOWFIELD = {
    ConeFlowfield:     "cone",
    PowerLawFlowfield: "cone",
    WedgeFlowfield:    "oblique",
}

# Same mapping keyed by the GUI's flowfield strings.
SOLVER_FOR_FLOWFIELD_KEY = {
    "cone":      "cone",
    "power-law": "cone",
    "wedge":     "oblique",
}

_DEFAULT_SOLVER = "cone"


def solver_for_flowfield(flowfield) -> str:
    """Pressure law matching ``flowfield``.

    Accepts a :class:`~mfof.basic_flowfield.BasicFlowfield` instance, a
    flowfield class, or one of the GUI keys ``"cone" / "wedge" /
    "power-law"``. Anything unrecognised falls back to tangent-cone, which
    suits the cone-derived geometry this framework started from.
    """
    if isinstance(flowfield, str):
        return SOLVER_FOR_FLOWFIELD_KEY.get(flowfield, _DEFAULT_SOLVER)
    cls = flowfield if isinstance(flowfield, type) else type(flowfield)
    for base, name in SOLVER_FOR_FLOWFIELD.items():
        if issubclass(cls, base):
            return name
    return _DEFAULT_SOLVER


class MFOFAeroEvaluator(Liu2019AeroEvaluator):
    """Local-inclination aero evaluator for an MFOF waverider.

    Constructor and public methods are inherited verbatim:

    * ``__init__(waverider, gamma=1.4, ref_length=6.0, ref_area=1.0,
                  moment_ref=(0,0,0), solver="cone")``
    * ``evaluate(Ma, alpha_deg=0.0, atm_conditions=None, solver=None)``
    * ``evaluate_paper_trajectory(progress_callback=None, solver=None)``
    * ``compare_with_paper(solver=None)``

    What this class adds is the wiring from geometry to pressure law: the
    solver is taken from the flowfields the waverider was actually built
    with, per osculating plane, instead of defaulting to tangent-cone
    regardless of shape.
    """

    # ------------------------------------------------------------------
    def _default_solver(self, waverider):
        """Solver implied by the waverider's flowfields.

        Uniform builds give one answer. For a mixed build this is only the
        headline value -- the per-plane assignment in
        :meth:`_assign_panel_solvers` is what actually gets used -- so
        report the law covering the most planes.
        """
        names = self._plane_solvers(waverider)
        if not names:
            # No plane records (e.g. a duck-typed waverider). Fall back to
            # the GUI tag if the geometry worker left one, else tangent-cone.
            return solver_for_flowfield(
                getattr(waverider, "_mfof_flowfield_type", None))
        return max(set(names), key=names.count)

    # ------------------------------------------------------------------
    @staticmethod
    def _plane_solvers(waverider):
        """Per-plane solver names, or ``[]`` if the planes are unreadable."""
        planes = getattr(waverider, "planes", None)
        if planes is None:
            return []
        names = []
        for p in planes:
            ff = getattr(p, "flowfield", None)
            if ff is None:
                return []
            names.append(solver_for_flowfield(ff))
        return names

    # ------------------------------------------------------------------
    def _assign_panel_solvers(self):
        """Map every panel to the pressure law of its osculating plane.

        The surface grids are ``(n_x, n_cols)`` with the spanwise direction
        along axis 1, and they arrive mirrored: columns
        ``[n_planes-1 ... 1]`` for the port half, then ``[0 ... n_planes-1]``
        for starboard. Translating a panel's grid column through that gives
        its plane, and the plane gives its flowfield.

        Returns ``None`` when every plane agrees, so uniform builds keep the
        single-solver fast path.
        """
        names = self._plane_solvers(self.wr)
        if not names or len(set(names)) == 1:
            return None

        order = sorted(set(names))
        per_plane = np.array([order.index(n) for n in names])

        def ids_for(shape):
            cols = panel_column_index(shape)
            plane_of_col = self._column_to_plane(len(names), int(shape[1]))
            return per_plane[plane_of_col[cols]]

        return ids_for(self._upper_shape), ids_for(self._lower_shape), order

    # ------------------------------------------------------------------
    @staticmethod
    def _column_to_plane(n_planes, n_cols):
        """Plane index owning each spanwise column of a surface grid.

        Mirrored grids are built as ``concat(X[:, ::-1][:, :-1], X)``, so the
        first ``n_planes - 1`` columns run from the tip back in to (but not
        including) the centreline, then the starboard half runs out again.
        """
        if n_cols == n_planes:                 # unmirrored
            return np.arange(n_planes)
        idx = np.empty(n_cols, dtype=int)
        idx[:n_planes - 1] = np.arange(n_planes - 1, 0, -1)
        idx[n_planes - 1:] = np.arange(n_planes)
        return idx
