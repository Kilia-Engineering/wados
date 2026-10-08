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
