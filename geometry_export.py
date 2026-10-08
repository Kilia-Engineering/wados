"""Shared STL / STEP export for stream-based waverider geometries.

GUI-agnostic helpers used by the method tabs that had no export of their own
(OC Waverider STL, PSWR-1 STL and STEP).

Input geometry follows the OC stream convention:

* ``upper_streams`` and ``lower_streams`` are lists of ``(n, 3)`` arrays,
  one per spanwise station, ordered so the span coordinate increases with
  the station index;
* point 0 of every stream is on the leading edge (shared by the upper and
  lower stream) and the last point is on the base plane;
* frame: x streamwise, y up, z span (the GUI / PySAGAS frame).

Streams are resampled to a common point count, by arc length, before
meshing. Half models (station 0 on the symmetry plane z = 0) can be mirrored
to the full span. The mesh is closed by
``waverider_generator.stream_mesh.build_stream_mesh`` and zero-area triangles
(e.g. along a shared sharp leading edge) are dropped.

Writers are the GVWD ones: binary STL (``gvwd.export.stl.write_stl``) and a
faceted STEP solid (``gvwd.export.step.write_step``, which needs cadquery).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "resample_stream",
    "mirror_to_full_span",
    "grid_to_streams",
    "select_stations",
    "streams_to_mesh",
    "span_frame_to_gui",
    "write_stl",
    "write_step",
    "StepUnavailableError",
]


class StepUnavailableError(ImportError):
    """STEP export needs cadquery, which is not importable here."""


def resample_stream(stream: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points, uniform in arc length.

    End points are preserved exactly. A degenerate stream (zero length, e.g.
    the wing tip) becomes ``n`` copies of its first point.
    """
    s = np.asarray(stream, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3 or s.shape[0] < 1:
        raise ValueError("a stream must be an (n, 3) array")
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] <= 0.0:
        return np.repeat(s[:1], n, axis=0)
    keep = np.concatenate([[True], seg > 0.0])          # drop repeated points
    arc, s = arc[keep], s[keep]
    t = np.linspace(0.0, arc[-1], n)
    out = np.column_stack([np.interp(t, arc, s[:, k]) for k in range(3)])
    out[0], out[-1] = s[0], s[-1]
    return out


def grid_to_streams(X, Y, Z) -> list:
    """Streams from (n_stream, n_station) coordinate grids (one column per station)."""
    X, Y, Z = (np.asarray(a, dtype=float) for a in (X, Y, Z))
    return [np.column_stack([X[:, j], Y[:, j], Z[:, j]]) for j in range(X.shape[1])]


def select_stations(streams: Sequence[np.ndarray], n: int) -> list:
    """Keep ``n`` evenly spaced stations, always including the first and last."""
    k = len(streams)
    if n >= k:
        return list(streams)
    idx = np.unique(np.round(np.linspace(0, k - 1, max(n, 2))).astype(int))
    return [streams[i] for i in idx]


def span_frame_to_gui(mesh):
    """Mesh in the (x streamwise, y span, z up) frame of GVWD / PSWR-1, converted
    to the GUI frame (x streamwise, y up, z span).

    The axis swap is a reflection, so the triangle winding is reversed to keep
    the normals pointing outward.
    """
    from gvwd.geometry.mesh import Mesh

    v = np.asarray(mesh.vertices, dtype=float)[:, [0, 2, 1]]
    f = np.asarray(mesh.faces, dtype=int)[:, [0, 2, 1]]
    meta = dict(getattr(mesh, "metadata", {}) or {})
    meta["frame"] = "gui (x streamwise, y up, z span)"
    return Mesh(v, f, labels=getattr(mesh, "labels", None), metadata=meta)


def mirror_to_full_span(streams: Sequence[np.ndarray], span_axis: int = 2,
                        tol: float = 1e-9) -> list:
    """Mirror a half model about the symmetry plane ``span = 0``.

    Station 0 must lie on the symmetry plane. Returns the full-span list
    ordered from the negative tip to the positive tip.
    """
    streams = [np.asarray(s, dtype=float) for s in streams]
    scale = max(float(np.max(np.abs(s))) for s in streams) or 1.0
    if np.max(np.abs(streams[0][:, span_axis])) > tol * scale:
        raise ValueError("station 0 is not on the symmetry plane; cannot mirror")
    mirrored = []
    for s in reversed(streams[1:]):
        m = s.copy()
        m[:, span_axis] *= -1.0
        mirrored.append(m)
    return mirrored + streams


def streams_to_mesh(upper_streams: Sequence[np.ndarray], lower_streams: Sequence[np.ndarray], *,
                    full_span: bool = False, n_points: Optional[int] = None,
                    span_axis: int = 2, kind: str = "waverider"):
    """Closed triangle mesh (``gvwd.geometry.mesh.Mesh``) from upper/lower streams.

    Parameters
    ----------
    full_span : bool
        Mirror a half model (station 0 on the symmetry plane) to the full span.
        Leave False for geometry that is already full span; station 0 is
        then the negative tip.
    n_points : int, optional
        Points per stream after resampling. Default: the longest stream.
    kind : str
        Stored in ``Mesh.metadata["kind"]``; used as the STL header.
    """
    from gvwd.geometry.mesh import Mesh
    from waverider_generator.stream_mesh import build_stream_mesh

    if len(upper_streams) != len(lower_streams) or len(upper_streams) < 2:
        raise ValueError("need matching upper/lower stream lists with at least two stations")
    if n_points is None:
        n_points = max(max(len(s) for s in upper_streams), max(len(s) for s in lower_streams))
    n_points = max(int(n_points), 2)
    us = [resample_stream(s, n_points) for s in upper_streams]
    ls = [resample_stream(s, n_points) for s in lower_streams]
    if full_span:
        us = mirror_to_full_span(us, span_axis)
        ls = mirror_to_full_span(ls, span_axis)

    vertices, triangles = build_stream_mesh(us, ls)
    v, f = _weld(np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=int))
    return Mesh(v, f, metadata={"kind": kind})


def _weld(v: np.ndarray, f: np.ndarray, rel_tol: float = 1e-9):
    """Merge coincident vertices and drop degenerate triangles.

    ``build_stream_mesh`` keeps separate upper and lower copies of the shared
    leading-edge and tip points, joined by zero-area cap triangles. Welding
    makes the surface topologically closed along those seams.
    """
    size = float(np.max(np.ptp(v, axis=0))) or 1.0
    key = np.round(v / (rel_tol * size)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    f = inverse[f]
    distinct = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[distinct]
    v = v[first]
    area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    f = f[area2 > 1e-12 * size * size]
    used, compact = np.unique(f.ravel(), return_inverse=True)
    return v[used], compact.reshape(-1, 3)


def write_stl(mesh, path, *, scale: float = 1.0) -> Path:
    """Binary STL; coordinates multiplied by ``scale`` (default: metres)."""
    from gvwd.export.stl import write_stl as _write_stl

    return Path(_write_stl(mesh, path, scale=scale))


def write_step(mesh, path, *, scale: float = 1000.0) -> Path:
    """Faceted STEP solid; ``scale=1000`` writes millimetres, as the other tabs do."""
    try:
        from gvwd.export.step import CadqueryUnavailableError, write_step as _write_step
    except ImportError as e:                                  # pragma: no cover
        raise StepUnavailableError(str(e)) from e
    try:
        return Path(_write_step(mesh, path, scale=scale, close_to_solid=True))
    except CadqueryUnavailableError as e:
        raise StepUnavailableError(str(e)) from e
