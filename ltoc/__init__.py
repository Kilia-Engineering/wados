"""LTOCs -- local-turning osculating cones waverider generator (in development).

Method: Zheng, X., Hu, Z., Li, Y., Zhu, C., You, Y., Song, W., "Local-Turning
Osculating Cones Method for Waverider Design," AIAA Journal 58(8):3499-3513,
2020, doi:10.2514/1.J059139 (R1). Spec: ``docs/ltoc/LTOC_implementation_spec.md``.

Status (Phase 3):
* the two-family rotational inverse MOC kernel with a pluggable axis rule
  (``ltoc.moc_noncoaxial``) and its conical-flow reference
  (``ltoc.reference``);
* prescribed shock surfaces (``ltoc.shock_surface``) and their shock
  geometry and shock curves (``ltoc.shock_geometry``);
* the LTOCs core: leading-edge projection and Steps A-C per stream surface
  (``ltoc.ltoc``), and the waverider assembly with the WADOS stream
  protocol and STL/STEP export (``ltoc.waverider``).

Published cases (Phase 4) and the GUI tab (Phase 5) follow.

Meridian-plane frame used by the kernel: x along the freestream, y
transverse; the intrinsic mapping (Gate 0 decision) keeps ``x = X`` and
develops each shock curve with ``dy/dx = tan(beta)``.
"""
from .moc_noncoaxial import (  # noqa: F401
    SCHEMES,
    AxisRule,
    InitialLine,
    MOCSolution,
    MOCError,
    LimitSurfaceError,
    SubsonicError,
    AxisCrossedError,
    shock_initial_line,
    solve_inverse,
)
from .shock_surface import (  # noqa: F401
    ShockSurface,
    RuledShock,
    ConeShock,
    OsculatingConeShock,
    BSplineShock,
)
from .shock_geometry import LocalGeometry, ShockCurve, local_geometry, trace_shock_curves  # noqa: F401
from .ltoc import (  # noqa: F401
    LTOCError,
    BodyLine,
    StreamSurface,
    find_leading_edge,
    solve_stream_surface,
    r1_frame_to_gui,
)
from .waverider import LTOCWaverider  # noqa: F401

__all__ = [
    "SCHEMES",
    "AxisRule",
    "InitialLine",
    "MOCSolution",
    "MOCError",
    "LimitSurfaceError",
    "SubsonicError",
    "AxisCrossedError",
    "shock_initial_line",
    "solve_inverse",
    "ShockSurface",
    "RuledShock",
    "ConeShock",
    "OsculatingConeShock",
    "BSplineShock",
    "LocalGeometry",
    "ShockCurve",
    "local_geometry",
    "trace_shock_curves",
    "LTOCError",
    "BodyLine",
    "StreamSurface",
    "find_leading_edge",
    "solve_stream_surface",
    "r1_frame_to_gui",
    "LTOCWaverider",
]
