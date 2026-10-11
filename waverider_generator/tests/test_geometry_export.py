"""Tests for ``geometry_export``: stream geometry to a closed mesh, STL and STEP.

Covers the exports added to the OC Waverider tab (STL) and the PSWR-1 tab
(STL and STEP). The meshes must be closed two-manifolds with outward normals.
Mirroring a half model must double the volume. The binary STL must have the
expected size. The STEP must re-import as one valid solid of the same volume;
that test is skipped without cadquery.
"""
import math

import numpy as np
import pytest

from geometry_export import mirror_to_full_span, resample_stream, streams_to_mesh, write_stl


def _closed_and_volume(mesh):
    f, v = mesh.faces, mesh.vertices
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    volume = np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0
    return bool(np.all(counts == 2)), float(volume)


@pytest.fixture(scope="module")
def oc_waverider():
    from waverider_generator.generator import waverider

    return waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                     n_upper_surface=10000, n_shockwave=10000, n_planes=20, n_streamwise=30,
                     delta_streamwise=0.1)


@pytest.fixture(scope="module")
def pswr_streams():
    from pswr.geometry.variable_wedge import VariableWedgeWaverider, to_gui_frame

    wr = VariableWedgeWaverider(M_inf=8.0, beta_knots=tuple(math.radians(b) for b in (14, 13, 12)),
                                Lambda=math.radians(70), body_length=1.0, flat_fraction=0.3,
                                n_span=41, n_chord=30)
    return ([to_gui_frame(s) for s in wr.upper_surface],
            [to_gui_frame(s) for s in wr.lower_surface_streams])


def test_resample_keeps_end_points_and_handles_degenerate_streams():
    s = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [0.5, 0.1, 0.0], [1.0, 0.0, 0.0]])
    r = resample_stream(s, 7)
    assert r.shape == (7, 3)
    assert np.array_equal(r[0], s[0]) and np.array_equal(r[-1], s[-1])
    tip = resample_stream(np.array([[2.0, 1.0, 3.0], [2.0, 1.0, 3.0]]), 5)
    assert np.all(tip == [2.0, 1.0, 3.0])


def test_mirror_needs_symmetry_station():
    s = [np.array([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1]]), np.array([[0.5, 0.0, 1.0], [1.0, 0.0, 1.0]])]
    with pytest.raises(ValueError):
        mirror_to_full_span(s)
    s[0][:, 2] = 0.0
    full = mirror_to_full_span(s)
    assert [st[0, 2] for st in full] == [-1.0, 0.0, 1.0]


def test_oc_waverider_mesh_is_closed_and_mirrors(oc_waverider):
    wr = oc_waverider
    # OC lower streams have uneven point counts; the helper resamples them.
    assert len({len(s) for s in wr.lower_surface_streams}) > 1
    half = streams_to_mesh(wr.upper_surface_streams, wr.lower_surface_streams)
    full = streams_to_mesh(wr.upper_surface_streams, wr.lower_surface_streams, full_span=True)
    closed_h, vol_h = _closed_and_volume(half)
    closed_f, vol_f = _closed_and_volume(full)
    assert closed_h and closed_f
    assert vol_h > 0.0                                   # outward normals
    # Re-indexing the mirrored half flips the quad diagonals, so the left half
    # is triangulated slightly differently: equal to second order only.
    assert vol_f == pytest.approx(2.0 * vol_h, rel=1e-3)
    assert np.min(full.vertices[:, 2]) == pytest.approx(-wr.width)
    assert np.max(full.vertices[:, 0]) == pytest.approx(wr.length, rel=1e-6)


def test_pswr_mesh_is_closed_and_two_points_suffice(pswr_streams):
    up, lo = pswr_streams
    fine = streams_to_mesh(up, lo)
    coarse = streams_to_mesh(up, lo, n_points=2)
    closed_f, vol_f = _closed_and_volume(fine)
    closed_c, vol_c = _closed_and_volume(coarse)
    assert closed_f and closed_c and vol_f > 0.0
    # Flat upper surface and straight lower streamlines: two points are exact.
    assert vol_c == pytest.approx(vol_f, rel=1e-4)
    assert coarse.n_faces < fine.n_faces / 10


def test_binary_stl_written(tmp_path, pswr_streams):
    mesh = streams_to_mesh(*pswr_streams)
    path = write_stl(mesh, tmp_path / "pswr.stl")
    assert path.stat().st_size == 84 + 50 * mesh.n_faces


def test_step_reimports_as_one_valid_solid(tmp_path, pswr_streams):
    cq = pytest.importorskip("cadquery")
    from geometry_export import write_step

    mesh = streams_to_mesh(*pswr_streams, n_points=2)
    _, vol = _closed_and_volume(mesh)
    path = write_step(mesh, tmp_path / "pswr.step")
    solids = cq.importers.importStep(str(path)).solids().vals()
    assert len(solids) == 1 and solids[0].isValid()
    assert solids[0].Volume() / 1e9 == pytest.approx(vol, rel=1e-5)     # mm^3 -> m^3


# ---------------------------------------------------------------------------
#  GVWD frame conversion, Liu 2019 / MFOF closed export, NumPy 2 compatibility
# ---------------------------------------------------------------------------

def test_gvwd_mesh_converted_to_gui_frame_keeps_outward_normals():
    from gvwd.io.config import EngineeringFlatConfig, GVWDRunConfig, build_geometry
    from geometry_export import span_frame_to_gui

    m = build_geometry(GVWDRunConfig(geometry=EngineeringFlatConfig())).mesh
    g = span_frame_to_gui(m)
    assert np.array_equal(g.vertices[:, 1], m.vertices[:, 2])          # y_gui = z_gvwd (up)
    assert np.array_equal(g.vertices[:, 2], m.vertices[:, 1])          # z_gui = y_gvwd (span)
    _, vol_m = _closed_and_volume(m)
    closed, vol_g = _closed_and_volume(g)
    assert closed and vol_g == pytest.approx(vol_m) and vol_g > 0.0


def test_select_stations_keeps_ends():
    from geometry_export import select_stations

    st = [np.full((2, 3), float(i)) for i in range(10)]
    sel = select_stations(st, 4)
    assert sel[0][0, 0] == 0.0 and sel[-1][0, 0] == 9.0 and len(sel) == 4


@pytest.fixture(scope="module")
def liu_waverider():
    from liu2019 import PAPER_PARAMS, build_liu2019_waverider

    return build_liu2019_waverider(PAPER_PARAMS, n_z=60, n_x=40)


def test_liu2019_closed_mesh_matches_its_volume(liu_waverider):
    from geometry_export import grid_to_streams

    S = liu_waverider.surfaces
    mesh = streams_to_mesh(grid_to_streams(S.X_upper, S.Y_upper, S.Z_upper),
                           grid_to_streams(S.X_lower, S.Y_lower, S.Z_lower), full_span=True)
    closed, vol = _closed_and_volume(mesh)
    assert closed
    # volume() also exercises the np.trapz -> np.trapezoid fix (NumPy >= 2.4).
    assert vol == pytest.approx(liu_waverider.volume(), rel=0.02)


def test_shadow_waverider_builds_with_current_numpy():
    from shadow_waverider import ShadowWaverider

    wr = ShadowWaverider(mach=6.0, shock_angle=12.0, poly_coeffs=[-1.0, 0.0, 0.5])
    assert np.isfinite(wr.planform_area) and wr.volume > 0.0


def test_liu2019_bspline_step_has_true_size(tmp_path, liu_waverider):
    cq = pytest.importorskip("cadquery")
    from liu2019 import PAPER_PARAMS

    path = tmp_path / "liu.step"
    liu_waverider.export_step(str(path), scale=1000.0)
    bb = cq.importers.importStep(str(path)).val().BoundingBox()
    assert bb.xlen == pytest.approx(1000.0 * PAPER_PARAMS["L_w"], rel=1e-2)
