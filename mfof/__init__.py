"""Multi-Flowfield Osculating Framework (MFOF).

Architectural refactor of the Liu 2019 osculating-cone waverider into a
plug-in framework. Each spanwise osculating plane carries its own
:class:`BasicFlowfield` instance, supplied by a factory at sweep time.
Three concrete flowfields ship today -- :class:`ConeFlowfield`,
:class:`WedgeFlowfield` and :class:`PowerLawFlowfield` -- and the factory
signature ``(z, Ma_z) -> BasicFlowfield`` already allows different types to
coexist in a single waverider; exposing that mixing in the GUI is still to
come.

With the factory that mirrors Liu's per-plane physics -- ``WedgeFlowfield``
in the flat region, ``ConeFlowfield`` in the curved region -- ``MFOF``
reproduces ``liu2019`` numerically to within ``1e-6`` -- see
:func:`mfof.validate.run_equivalence_test`.
"""

from .basic_flowfield import BasicFlowfield, StreamlineResult
from .cone_flowfield import ConeFlowfield
from .wedge_flowfield import WedgeFlowfield
from .power_law_flowfield import PowerLawFlowfield

# Lazy imports: osculating, geometry, aero pull in numpy / scipy / liu2019.
# Re-exported here for the canonical public surface.
from .osculating import (
    OsculatingPlaneData,
    OsculatingPlaneSet,
    build_all_osculating_planes,
)
from .geometry import MFOFWaverider, build_mfof_waverider
from .aero import MFOFAeroEvaluator

from .config import (
    DEFAULT_PARAMS,
    PAPER_PARAMS,
    PAPER_TRAJECTORY,
    PAPER_REFERENCE_GEOMETRY,
    PAPER_REFERENCE_AERO,
    TOLERANCES,
    MOMENT_REF,
    REF_AREA_M2,
    REF_LENGTH_M,
)
from .distributions import (
    Ma_distribution,
    shock_curve,
    shock_curve_coefficient,
    upper_surface_coefficients,
    upper_surface_trailing_edge,
)

__all__ = [
    # Flowfield interface
    "BasicFlowfield",
    "StreamlineResult",
    "ConeFlowfield",
    "WedgeFlowfield",
    "PowerLawFlowfield",
    # Osculating sweep
    "OsculatingPlaneData",
    "OsculatingPlaneSet",
    "build_all_osculating_planes",
    # 3D geometry + aero
    "MFOFWaverider",
    "build_mfof_waverider",
    "MFOFAeroEvaluator",
    # Config
    "DEFAULT_PARAMS",
    "PAPER_PARAMS",
    "PAPER_TRAJECTORY",
    "PAPER_REFERENCE_GEOMETRY",
    "PAPER_REFERENCE_AERO",
    "TOLERANCES",
    "MOMENT_REF",
    "REF_AREA_M2",
    "REF_LENGTH_M",
    # Distributions
    "Ma_distribution",
    "shock_curve",
    "shock_curve_coefficient",
    "upper_surface_coefficients",
    "upper_surface_trailing_edge",
]
