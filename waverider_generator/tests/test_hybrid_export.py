"""Headless check that the Hybrid (GOC) waverider exports STL and TRI."""

import numpy as np
import pytest

from waverider_generator.hybrid_generator import GOCWaverider


@pytest.fixture(scope="module")
def goc():
    return GOCWaverider(
        M_inf=5.0, beta_OC=15.0, beta_CD=15.0, height=1.34, width=3.0,
        dp=[0.11, 0.63, 0.0, 0.46], h0=0.0, h1=1.0,
        n_planes=12, n_streamwise=15,
    )


def test_export_stl(goc, tmp_path):
    path = tmp_path / "hybrid.stl"
    goc.export_stl(str(path))
    lines = path.read_text().splitlines()
    _, triangles = goc._build_mesh()

    assert lines[0] == "solid hybrid_waverider"
    assert lines[-1] == "endsolid hybrid_waverider"
    assert sum(l.startswith("  facet normal") for l in lines) == len(triangles)
    assert sum(l.startswith("      vertex") for l in lines) == 3 * len(triangles)
    coords = np.array([l.split()[1:] for l in lines if l.startswith("      vertex")],
                      dtype=float)
    assert np.all(np.isfinite(coords))


def test_export_tri(goc, tmp_path):
    path = tmp_path / "hybrid.tri"
    goc.export_tri(str(path))
    lines = path.read_text().splitlines()
    n_v, n_t = int(lines[0]), int(lines[1])

    assert n_v > 0 and n_t > 0
    assert len(lines) == 2 + n_v + n_t
    verts = np.array([l.split() for l in lines[2:2 + n_v]], dtype=float)
    tris = np.array([l.split() for l in lines[2 + n_v:]], dtype=int)
    assert verts.shape == (n_v, 3) and np.all(np.isfinite(verts))
    assert tris.shape == (n_t, 3)
    assert tris.min() == 1 and tris.max() <= n_v      # 1-based indices
