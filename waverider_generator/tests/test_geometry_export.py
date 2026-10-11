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


# ---------------------------------------------------------------------------
#  Reference (planform) area for the Aero Analysis tab
# ---------------------------------------------------------------------------

def test_stl_planform_area_counts_a_closed_surface_once(oc_waverider):
    """A_ref from an STL: the upper and lower surfaces of a closed body both
    project onto the planform, so the projection is halved; an open single
    skin is counted once."""
    from gvwd.geometry.mesh import Mesh

    from geometry_export import write_temp_stl
    from reference_area_calculator import (calculate_planform_area_from_waverider,
                                           calculate_reference_area_from_stl)

    mesh = streams_to_mesh(oc_waverider.upper_surface_streams,
                           oc_waverider.lower_surface_streams, full_span=True)
    area, method = calculate_reference_area_from_stl(write_temp_stl(mesh))
    planform, _ = calculate_planform_area_from_waverider(oc_waverider)
    assert "closed" in method and area == pytest.approx(planform, rel=1e-3)

    plate = Mesh(np.array([[0, 0, 0], [2, 0, 0], [2, 0, 3], [0, 0, 3]], float),
                 np.array([[0, 1, 2], [0, 2, 3]]))
    area, method = calculate_reference_area_from_stl(write_temp_stl(plate))
    assert "open" in method and area == pytest.approx(6.0)


def test_planar_waverider_with_current_numpy():
    """np.cross no longer takes 2-D vectors (NumPy 2); the planar waverider's
    planform area must still compute. Its surface grids (x streamwise,
    y span, z up; one stream per row) give the closed mesh sent to the Aero
    Analysis tab, whose STL projection is the same planform area."""
    from geometry_export import grid_to_streams, span_frame_to_gui, write_temp_stl
    from planar_waverider import PlanarWaverider
    from reference_area_calculator import calculate_reference_area_from_stl

    wr = PlanarWaverider(length=1.0, width=2.0, n=0.5, beta_deg=10.0, M_inf=6.0)
    wr.generate(nx=20, ny=15)
    mesh = span_frame_to_gui(streams_to_mesh(
        grid_to_streams(wr.upper_surface_x, wr.upper_surface_y, wr.upper_surface_z),
        grid_to_streams(wr.lower_surface_x, wr.lower_surface_y, wr.lower_surface_z),
        span_axis=1))
    closed, vol = _closed_and_volume(mesh)
    assert closed and vol > 0.0
    assert mesh.vertices[:, 1].max() <= 1e-12                 # y up: the body hangs below the top
    area, _ = calculate_reference_area_from_stl(write_temp_stl(mesh))
    assert wr.planform_area() == pytest.approx(area, rel=5e-3)        # two discretisations


def test_shadow_waverider_stream_mesh_is_closed():
    """The cone-derived waverider's surface grids (full span, one stream per
    row) give the closed mesh sent to the Aero Analysis tab."""
    from shadow_waverider import create_second_order_waverider

    wr = create_second_order_waverider(mach=6.0, shock_angle=12.0, A2=-2.0, A0=-0.15)  # tab default
    mesh = streams_to_mesh(list(np.asarray(wr.upper_surface)), list(np.asarray(wr.lower_surface)))
    closed, vol = _closed_and_volume(mesh)
    assert closed and vol > 0.0
    assert mesh.vertices[:, 2].min() == pytest.approx(-mesh.vertices[:, 2].max())


# ---------------------------------------------------------------------------
#  Volume
# ---------------------------------------------------------------------------

def _sliced_volume(upper, lower, nx=400):
    """Independent volume of a half model (x streamwise, y up, z span):
    cross-sections at common x stations, each stream interpolated there
    (zero thickness before its leading edge), integrated across the span
    and along x; times 2 for the full vehicle."""
    x0 = min(float(s[:, 0].min()) for s in upper)
    x1 = max(float(s[:, 0].max()) for s in upper)
    xs = np.linspace(x0, x1, nx)
    z = np.array([float(s[0, 2]) for s in upper])
    order = np.argsort(z)
    areas = []
    for x in xs:
        th = np.zeros(len(upper))
        for k, (su, sl) in enumerate(zip(upper, lower)):
            if su[-1, 0] - su[0, 0] > 1e-12 and x >= su[0, 0]:
                th[k] = max(np.interp(x, su[:, 0], su[:, 1]) - np.interp(x, sl[:, 0], sl[:, 1]), 0.0)
        areas.append(np.trapezoid(th[order], z[order]))
    return 2.0 * float(np.trapezoid(areas, xs))


def test_stream_volume_is_exact_for_a_swept_wedge():
    """Half model: flat top, lower surface y = -t (x - x_le(z)), straight
    swept leading edge x_le = z L / b. Volume t L^2 b / 3."""
    from geometry_export import stream_volume

    L, b, t = 2.0, 0.8, 0.15
    upper, lower = [], []
    for z in np.linspace(0.0, b, 11):
        x = np.linspace(z * L / b, L, 30)
        upper.append(np.c_[x, np.zeros_like(x), np.full_like(x, z)])
        lower.append(np.c_[x, -t * (x - z * L / b), np.full_like(x, z)])
    assert stream_volume(upper, lower, full_span=True) == pytest.approx(t * L**2 * b / 3, rel=1e-12)


def test_oc_waverider_volume_against_cross_sections(oc_waverider):
    """Each stream starts at its own leading-edge x. The volume must not
    assume a common x per point index (that overestimated it by about 20 %)."""
    from geometry_export import stream_volume
    from optimization_engine import calculate_volume

    U, L = oc_waverider.upper_surface_streams, oc_waverider.lower_surface_streams
    exact = stream_volume(U, L, full_span=True)
    assert calculate_volume(oc_waverider) == pytest.approx(exact, rel=1e-12)
    assert exact == pytest.approx(_sliced_volume(U, L), rel=0.01)


def test_shadow_and_planar_volumes_are_their_closed_mesh_volumes():
    from geometry_export import grid_to_streams, mesh_volume, span_frame_to_gui
    from planar_waverider import PlanarWaverider
    from shadow_waverider import create_second_order_waverider

    wr = create_second_order_waverider(mach=6.0, shock_angle=12.0, A2=-2.0, A0=-0.15)
    mesh = streams_to_mesh(list(np.asarray(wr.upper_surface)), list(np.asarray(wr.lower_surface)))
    assert wr.volume == pytest.approx(mesh_volume(mesh), rel=1e-12)

    pw = PlanarWaverider(length=1.0, width=2.0, n=0.5, beta_deg=10.0, M_inf=6.0)
    pw.generate(nx=20, ny=15)
    upper = grid_to_streams(pw.upper_surface_x, pw.upper_surface_y, pw.upper_surface_z)
    lower = grid_to_streams(pw.lower_surface_x, pw.lower_surface_y, pw.lower_surface_z)
    gui = span_frame_to_gui(streams_to_mesh(upper, lower, span_axis=1))
    assert pw.volume() == pytest.approx(mesh_volume(gui), rel=1e-12)
    # independent check on the half span y >= 0 (grid rows are the streamwise
    # lines), in the (x, up, span) frame
    rows_u = np.stack([pw.upper_surface_x, pw.upper_surface_z, pw.upper_surface_y], axis=-1)
    rows_l = np.stack([pw.lower_surface_x, pw.lower_surface_z, pw.lower_surface_y], axis=-1)
    keep = rows_u[:, 0, 2] >= -1e-12
    assert pw.volume() == pytest.approx(_sliced_volume(list(rows_u[keep]), list(rows_l[keep])), rel=0.02)
