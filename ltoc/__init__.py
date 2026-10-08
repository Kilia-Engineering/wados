"""LTOCs -- local-turning osculating cones waverider generator (in development).

Method: Zheng, X., Hu, Z., Li, Y., Zhu, C., You, Y., Song, W., "Local-Turning
Osculating Cones Method for Waverider Design," AIAA Journal 58(8):3499-3513,
2020, doi:10.2514/1.J059139 (R1). Spec: ``docs/ltoc/LTOC_implementation_spec.md``.

Status (Phase 1): the two-family rotational inverse MOC kernel with a
pluggable axis rule (``ltoc.moc_noncoaxial``) and its conical-flow reference
(``ltoc.reference``). Shock geometry, the LTOCs orchestration and the
waverider output follow in later phases.

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
]
