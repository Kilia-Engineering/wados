"""Phase 5 tests: the Qt-free design layer behind the LTOCs GUI tab.

``ltoc.design`` turns the tab's inputs (shock presets, a custom elliptic ruled
shock, an imported OC design, FCT, stations, resolution) into an
``LTOCResult``. These tests run in CI, which has no PyQt5; the tab itself is
tested in ``test_ltoc_gui.py``.
"""
import numpy as np
import pytest

from ltoc.design import PRESETS, EllipticShock, LTOCDesign
from ltoc.ltoc import LTOCError


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_builds_a_closed_waverider(name):
    calls = []
    res = LTOCDesign.from_preset(name, n_stations=9, n_points=20).build(
        progress=lambda k, n: calls.append((k, n)))
    assert res.ok and not res.refusals
    assert calls == [(k, 9) for k in range(1, 10)]
    f = res.forces
    assert f.reference == "planform" and 0.0 < f.CD < f.CL
    assert f.area_planform < f.area_wetted
    # Plotting data are in the GUI frame: half model z >= 0, lower surface below the FCT.
    assert np.all(res.grid.points[..., 2] >= -1e-12)
    assert np.all(res.base_shock()[:, 2] >= -1e-12)
    for x, p in list(res.streamwise_sections().values()) + list(res.crosswise_sections().values()):
        assert x.size > 3 and np.all(p > 1.0)
    assert len(res.station_diagnostics()) == 8                       # the tip has no body
    assert "stations solved" in res.summary() and "planform reference" in res.summary()


def test_gui_defaults_are_the_configuration_validated_against_r1_tables():
    """The defaults (25 clustered stations, 40 points) are those of the V6/V7 tests
    in ``test_ltoc_published.py``, which reproduce R1 Tables 2 and 3 to 0.5 %."""
    from ltoc.tests.test_ltoc_published import r1_waverider

    d = LTOCDesign.from_preset("R1 waverider case II (Eq. 25), M 7")
    assert (d.n_stations, d.n_points, d.clustering) == (25, 40, "symmetry")
    wr = r1_waverider("II", n_stations=25, n_points=40)
    assert np.allclose(d.fct(), wr.fct_yz, atol=1e-15)


def test_fct_stations_are_clustered_at_the_symmetry_plane():
    d = LTOCDesign.from_preset("R1 waverider case I (Eq. 23), M 6", n_stations=11)
    y = d.fct()[:, 0]
    assert y[0] == 0.0 and y[-1] == pytest.approx(d.shock.tip_y(d.fct_z))
    assert np.all(np.diff(np.diff(y)) > 0)                           # spacing grows outward
    d.clustering = "uniform"
    assert np.allclose(np.diff(d.fct()[:, 0]), np.diff(d.fct()[:, 0])[0])


@pytest.mark.parametrize("change, message", [
    (dict(fct_z=0.9), "FCT Z"),
    (dict(M_inf=0.8), "supersonic"),
    (dict(n_stations=2), "at least 3 stations"),
    (dict(shock=EllipticShock(0.5, 0.1, 0.1, 0.4, 0.3, 0.3)), "non-degenerate"),
])
def test_invalid_inputs_are_rejected_before_solving(change, message):
    d = LTOCDesign.from_preset("R1 waverider case I (Eq. 23), M 6")
    for k, v in change.items():
        setattr(d, k, v)
    with pytest.raises(ValueError, match=message):
        d.build()


def test_refused_stations_are_reported_and_no_geometry_is_built():
    # A 12 deg cone at M 4: below the Mach angle (14.5 deg), every station is refused.
    d = LTOCDesign(shock=EllipticShock(0.0, 0.0, 0.0, 1.0, np.tan(np.radians(12.0)),
                                       np.tan(np.radians(12.0))),
                   M_inf=4.0, fct_z=0.1, n_stations=5, n_points=20)
    res = d.build()
    assert not res.ok and res.forces is None and res.grid is None
    statuses = {st for _, st, _ in res.refusals}
    assert statuses <= {"sub_mach", "tip"} and "sub_mach" in statuses
    assert "spec flag 7: refused" in res.summary()


def test_build_can_be_cancelled_from_the_progress_callback():
    d = LTOCDesign.from_preset("R1 waverider case II (Eq. 25), M 7", n_stations=9, n_points=20)
    with pytest.raises(LTOCError, match="cancelled"):
        d.build(progress=lambda k, n: k < 3)


def test_oc_design_import_reproduces_the_oc_waverider():
    from waverider_generator.generator import waverider

    oc = waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                   n_upper_surface=10000, n_shockwave=10000, n_planes=10, n_streamwise=40,
                   delta_streamwise=0.1)
    res = LTOCDesign(oc_waverider=oc, n_points=30).build()
    assert res.ok and res.waverider.length == pytest.approx(oc.length)
    assert np.allclose(res.waverider.leading_edge, np.asarray(oc.leading_edge), atol=1e-9)
    # Same lower surface as the OC generator, within that generator's own accuracy (V5).
    for k, st in enumerate(res.waverider.stations):
        if st.status == "tip":
            continue
        S = np.asarray(oc.lower_surface_streams[k])
        P = st.body.points
        dev = np.hypot(np.interp(S[:, 0], P[:, 0], P[:, 1]) - S[:, 1],
                       np.interp(S[:, 0], P[:, 0], P[:, 2]) - S[:, 2])
        assert np.max(dev) < 2e-3 * oc.length
