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

    d = LTOCDesign.from_preset("Shape-changing elliptic shock, Mach 7 (published case II)")
    assert (d.n_stations, d.n_points, d.clustering) == (25, 40, "symmetry")
    wr = r1_waverider("II", n_stations=25, n_points=40)
    assert np.allclose(d.fct(), wr.fct_yz, atol=1e-15)


def test_fct_stations_are_clustered_at_the_symmetry_plane():
    d = LTOCDesign.from_preset("Elliptic cone shock, Mach 6 (published case I)", n_stations=11)
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
    d = LTOCDesign.from_preset("Elliptic cone shock, Mach 6 (published case I)")
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
    d = LTOCDesign.from_preset("Shape-changing elliptic shock, Mach 7 (published case II)", n_stations=9, n_points=20)
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


# ---------------------------------------------------------------------------
#  Project files: LTOCDesign.to_dict / from_dict (section "ltoc_waverider")
# ---------------------------------------------------------------------------

def test_design_round_trips_through_json():
    import json

    for name in PRESETS:
        d = LTOCDesign.from_preset(name, n_stations=31, clustering="uniform", n_points=55)
        saved = json.loads(json.dumps(d.to_dict(source=name)))
        assert saved["version"] == 1 and saved["source"] == name
        assert LTOCDesign.from_dict(saved) == d
    custom = LTOCDesign(shock=EllipticShock(0.05, 0.1, 0.1, 1.2, 0.5, 0.55), M_inf=8.0, fct_z=0.2)
    assert LTOCDesign.from_dict(json.loads(json.dumps(custom.to_dict()))) == custom


def test_design_dict_defaults_unknown_keys_and_oc_import():
    d = LTOCDesign.from_dict({"shock": {"x_p": 0, "a_p": 0, "b_p": 0, "x_q": 1, "a_q": 0.2, "b_q": 0.2},
                              "future_option": 3})
    assert (d.n_stations, d.n_points, d.clustering, d.M_inf) == (25, 40, "symmetry", 6.0)
    saved = {"version": 1, "source": "oc_import", "shock": None, "M_inf": 5.0}
    assert LTOCDesign.from_dict(saved).oc_waverider is None          # numbers only
    marker = object()
    assert LTOCDesign.from_dict(saved, oc_waverider=marker).oc_waverider is marker


@pytest.mark.parametrize("bad, message", [
    ({"version": 2}, "newer"),
    ({"shock": {"x_p": 0.0, "a_p": 0.0}}, "malformed"),
    ({"shock": {"x_p": "a", "a_p": 0, "b_p": 0, "x_q": 1, "a_q": 1, "b_q": 1}}, "malformed"),
])
def test_malformed_design_dicts_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        LTOCDesign.from_dict(bad)


# ---------------------------------------------------------------------------
#  Shock data files (B-spline input)
# ---------------------------------------------------------------------------

def test_shock_data_csv_round_trip_with_metadata(tmp_path):
    from ltoc.design import ShockData

    shock, M, z, _ = PRESETS["Shape-changing elliptic shock, Mach 7 (published case II)"]
    data = shock.to_data(M_inf=M, fct_z=z)
    path = tmp_path / "shock.csv"
    data.write_csv(path)
    back = ShockData.read_csv(path)
    assert back == data and back.shape == (61, 41)
    assert back.meta == {"x_base": 1.5, "M_inf": 7.0, "fct_z": 0.25}
    assert back.x_range == pytest.approx((0.1, 3.6))
    # Without a header the columns are u, v, X, Y, Z; extra metadata keys are ignored.
    rows = [line for line in path.read_text().splitlines() if line and line[0] not in "#u"]
    plain = tmp_path / "plain.csv"
    plain.write_text("# a comment\n# other = 3\n" + "\n".join(rows) + "\n")
    assert ShockData.read_csv(plain) == data and ShockData.read_csv(plain).meta == {}


def test_shock_data_reproduces_the_analytic_shock():
    """R1 case II through a data file (61 x 41 grid, 1.5 chords past the base)
    gives the analytic result: same FCT tip, same forces."""
    from ltoc.design import fct_tip_y

    name = "Shape-changing elliptic shock, Mach 7 (published case II)"
    shock, M, z, _ = PRESETS[name]
    data = shock.to_data()
    # The spline through the 61 x 41 samples is the ellipse to interpolation accuracy.
    assert fct_tip_y(data.surface(), 1.5, z) == pytest.approx(shock.tip_y(z), abs=1e-7)
    a = LTOCDesign.from_preset(name, n_stations=9, n_points=20).build()
    b = LTOCDesign(shock_data=data, x_base=1.5, M_inf=M, fct_z=z, n_stations=9, n_points=20).build()
    assert b.ok and not b.refusals
    assert b.forces.CL == pytest.approx(a.forces.CL, rel=1e-4)
    assert b.forces.CD == pytest.approx(a.forces.CD, rel=1e-4)
    B = b.base_shock()                                               # GUI frame (x, -Z, Y)
    assert B.shape[0] > 50 and np.allclose(B[:, 0], 1.5, atol=1e-9)
    assert np.allclose((B[:, 2] / 0.65) ** 2 + (B[:, 1] / 0.8) ** 2, 1.0, atol=1e-5)


def test_shock_data_must_extend_past_the_base():
    shock, M, z, _ = PRESETS["Shape-changing elliptic shock, Mach 7 (published case II)"]
    short = shock.to_data(n_max=1.0)                                  # data end at the base
    with pytest.raises(ValueError, match="must lie inside the data"):
        LTOCDesign(shock_data=short, x_base=1.5, M_inf=M, fct_z=z).build()
    just = shock.to_data(n_max=1.2)                                   # 0.2 chords past it
    res = LTOCDesign(shock_data=just, x_base=1.5, M_inf=M, fct_z=z, n_stations=5,
                     n_points=20).build()
    assert not res.ok and res.forces is None
    assert any(st == "undetermined" and "shock data end" in msg for _, st, msg in res.refusals)


@pytest.mark.parametrize("text, message", [
    ("u,v,X,Y\n0,0,0,0\n", "missing z"),
    ("u,v,X,Y,Z\n0,0,0,0,abc\n", "line 2: expected numbers"),
    ("u,v,X,Y,Z\n0,0,0,0,0\n0,1,1,0,0\n1,0,0,1,0\n", "full u x v grid"),
    ("# x_base = one\nu,v,X,Y,Z\n", "line 1: x_base must be a number"),
    ("u,v,X,Y,Z\n" + "\n".join(f"{i},{j},{j},{i},0" for i in range(3) for j in range(5)), "at least 4"),
    ("u,v,X,Y,Z\n", "no data rows"),
])
def test_invalid_shock_data_files_are_rejected_with_the_reason(tmp_path, text, message):
    from ltoc.design import ShockData

    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        ShockData.read_csv(path)


def test_design_dict_embeds_shock_data():
    import json

    shock, M, z, _ = PRESETS["Elliptic cone shock, Mach 6 (published case I)"]
    d = LTOCDesign(shock_data=shock.to_data(nu=9, nv=7), x_base=1.7, M_inf=M, fct_z=z)
    saved = json.loads(json.dumps(d.to_dict()))
    assert saved["source"] == "bspline" and saved["x_base"] == 1.7
    back = LTOCDesign.from_dict(saved)
    assert back.shock_data == d.shock_data and back.x_base == 1.7
    with pytest.raises(ValueError, match="malformed LTOCs shock data"):
        LTOCDesign.from_dict(dict(saved, shock_data={"u": [0, 1]}))


def test_result_info_panel_values():
    from ltoc.design import PRESET_ALIASES

    res = LTOCDesign.from_preset("R1 waverider case I (Eq. 23), M 6",          # earlier name
                                 n_stations=7, n_points=20).build()
    assert "R1 waverider case I (Eq. 23), M 6" in PRESET_ALIASES
    info = res.info("LTOCs (test)")
    assert info["__title__"] == "LTOCs (test)" and info["M_design"] == "6.00"
    V = float(info["Volume"].split()[0])
    S = float(info["Planform"].split()[0])
    assert 0.0 < V and S == pytest.approx(2 * res.forces.area_planform, rel=1e-3)
    assert float(info["Volumetric Efficiency"]) == pytest.approx(V ** (2 / 3) / S, rel=1e-3)
