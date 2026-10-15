"""LE blunting must work on the OC tab's default design.

Near the wingtip the lower streams are much shorter than the upper ones (4 to
13 points against 30). The tangent sample index was taken from the upper
stream and used on both, which raised IndexError in the STEP export's G2
Bezier blunting and in the LE preview. Each stream now gets its own index, at
about a quarter of its length.
"""
import numpy as np
import pytest

from waverider_generator.generator import waverider
from waverider_generator.leading_edge_blunting import (_tangent_index, compute_blunted_le_preview,
                                                       compute_pre_blunted_streams)


@pytest.fixture(scope="module")
def oc_default():
    return waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                     n_upper_surface=1000, n_shockwave=1000, n_planes=40, n_streamwise=30,
                     delta_streamwise=0.1)


def test_tangent_index_stays_in_the_stream():
    assert [_tangent_index(n) for n in (1, 2, 3, 4, 8, 13, 30)] == [0, 1, 2, 2, 2, 3, 7]


def test_oc_default_streams_differ_in_length(oc_default):
    lengths = {(len(u), len(l)) for u, l in zip(oc_default.upper_surface_streams,
                                                oc_default.lower_surface_streams)}
    assert any(nl < 8 for _, nl in lengths)                # the case that failed


def test_pre_blunted_streams_and_preview(oc_default):
    U, L = oc_default.upper_surface_streams, oc_default.lower_surface_streams
    out = compute_pre_blunted_streams(U, L, 0.01)
    assert len(out["modified_upper"]) == len(U) and len(out["modified_lower"]) == len(L)
    for us, ls in zip(out["modified_upper"], out["modified_lower"]):
        assert np.all(np.isfinite(us)) and np.all(np.isfinite(ls))
    assert np.all(np.isfinite(out["blunted_le"]))
    blunted, original = compute_blunted_le_preview(oc_default, 0.01)
    assert np.all(np.isfinite(blunted)) and blunted.shape == original.shape
    # the rounded edge sits behind the sharp one, by less than a few radii
    setback = blunted[:, 0] - original[:, 0]
    assert np.all(setback >= -1e-9) and setback.max() < 0.05


def test_g2_blunted_step_export(tmp_path):
    """The STEP export with G2 Bezier blunting builds one valid solid that keeps
    the Bezier profile (interpPlate could not fit it on this design)."""
    cq = pytest.importorskip("cadquery")
    from waverider_generator.cad_export import to_CAD

    wr = waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                   n_upper_surface=1000, n_shockwave=1000, n_planes=20, n_streamwise=20,
                   delta_streamwise=0.1)
    path = str(tmp_path / "g2.step")
    to_CAD(waverider=wr, sides="both", export=True, filename=path, scale=1000.0,
           blunting_radius=0.010, blunting_method="pre_blunted")
    shape = cq.importers.importStep(path)
    solid = shape.val()
    assert solid.isValid() and len(shape.solids().vals()) == 1
    # the blunt tip at a mid-span station lies on the exported surface
    out = compute_pre_blunted_streams(wr.upper_surface_streams, wr.lower_surface_streams, 0.010,
                                      n_bezier=24)
    i = len(wr.upper_surface_streams) // 2
    tip = out["blunted_le"][i] * 1000.0
    assert tip[0] > wr.upper_surface_streams[i][0][0] * 1000.0 + 1.0     # behind the sharp LE
    d = solid.distance(cq.Vertex.makeVertex(*map(float, tip)))
    assert d < 0.5                                                        # mm
