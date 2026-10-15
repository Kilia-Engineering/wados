"""Phase 5 tests: the LTOCs GUI tab and its hand-off to the Aero Analysis tab.

Run offscreen (``QT_QPA_PLATFORM=offscreen``). Skipped where PyQt5 is not
installed, as in CI; the logic behind the tab is covered there by
``test_ltoc_design.py``.
"""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _run(app, tab, timeout=120.0):
    tab.generate()
    t0 = time.time()
    while tab._worker is not None and tab._worker.isRunning() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


@pytest.fixture(scope="module")
def tab(app):
    from ltoc_waverider_tab import LTOCWaveriderTab

    t = LTOCWaveriderTab()
    t.resize(1280, 820)
    t.show()
    return t


def test_presets_fill_the_inputs_and_editing_makes_a_custom_shock(tab):
    from ltoc.design import PRESETS
    from ltoc_waverider_tab import CUSTOM

    name = "Shape-changing elliptic shock, Mach 7 (published case II)"
    tab.source_combo.setCurrentText(name)
    shock, M, z, _ = PRESETS[name]
    assert tab.shock_spins["a_q"].value() == pytest.approx(shock.a_q)
    assert tab.mach_spin.value() == pytest.approx(M) and tab.fct_spin.value() == pytest.approx(z)
    tab.shock_spins["a_q"].setValue(shock.a_q + 0.01)
    assert tab.source_combo.currentText() == CUSTOM
    tab.source_combo.setCurrentText(name)                            # restore


def test_generate_runs_in_the_background_and_fills_the_views(app, tab, tmp_path):
    tab.source_combo.setCurrentText("Elliptic cone shock, Mach 6 (published case I)")
    tab.stations_spin.setValue(9)
    tab.points_spin.setValue(20)
    _run(app, tab)
    assert tab.waverider is not None and tab.progress.value() == 100
    assert tab.btn_export_stl.isEnabled() and tab.btn_export_step.isEnabled()
    assert "9/9 stations solved" in tab.output_label.text()
    assert "planform reference" in tab.output_label.text()
    path = tab.stl_for_analysis()
    try:
        assert os.path.getsize(path) == 84 + 50 * tab.waverider.mesh().n_faces
    finally:
        os.unlink(path)


def test_refused_stations_are_listed_and_exports_stay_disabled(app, tab):
    from ltoc_waverider_tab import CUSTOM

    tab.source_combo.setCurrentText(CUSTOM)
    tb = np.tan(np.radians(12.0))
    for key, v in dict(x_p=0.0, a_p=0.0, b_p=0.0, x_q=1.0, a_q=tb, b_q=tb).items():
        tab.shock_spins[key].setValue(v)
    tab.mach_spin.setValue(4.0)                                      # Mach angle 14.5 deg > 12 deg
    tab.fct_spin.setValue(0.1)
    tab.stations_spin.setValue(5)
    _run(app, tab)
    assert tab.waverider is None
    assert not tab.btn_export_stl.isEnabled()
    assert "sub_mach" in tab.output_label.text()
    assert "What to change:" in tab.output_label.text()
    text = tab.output_label.text()
    assert "too weak" in text and "too strong" not in text
    assert "Mach number " in text and "(now 4)" in text            # the range that fixes it


def test_limits_box_follows_the_inputs(app, tab):
    from ltoc_waverider_tab import BSPLINE, CUSTOM

    tab.source_combo.setCurrentText("Shape-changing elliptic shock, Mach 7 (published case II)")
    assert tab.limits_group.isVisibleTo(tab)
    assert "Within the recommended limits" in tab.limits_label.text()
    tab.shock_spins["a_q"].setValue(0.55)                          # too weak at the sides
    assert tab.source_combo.currentText() == CUSTOM
    t0 = time.time()                                               # the check waits for a pause
    while "too weak" not in tab.limits_label.text() and time.time() - t0 < 5.0:
        app.processEvents()
        time.sleep(0.02)
    assert "too weak" in tab.limits_label.text()
    assert "base half-width a_q:" in tab.limits_label.text() and "<- now 0.55" in tab.limits_label.text()
    tab.fct_spin.setValue(0.9)                                     # above the base ellipse
    tab.update_limits()
    assert "Error: The FCT height" in tab.limits_label.text()
    tab.generate()
    assert "Input error" in tab.output_label.text() and "Try Z = 0.24" in tab.output_label.text()
    tab.source_combo.setCurrentText(BSPLINE)
    assert not tab.limits_group.isVisibleTo(tab)
    tab.source_combo.setCurrentIndex(0)
    assert tab.limits_group.isVisibleTo(tab)


def test_oc_import_without_an_oc_design_gives_an_input_error(tab):
    from ltoc_waverider_tab import OC_IMPORT

    tab.source_combo.setCurrentText(OC_IMPORT)
    assert not tab.fct_spin.isEnabled() and not tab.shock_spins["x_q"].isEnabled()
    tab.generate()
    assert "generate one in the OC Waverider tab first" in tab.output_label.text()
    tab.source_combo.setCurrentIndex(0)


def test_aero_analysis_mesh_dialog_offers_the_ltoc_waverider(app, tab):
    import waverider_gui as W

    tab.source_combo.setCurrentText("Shape-changing elliptic shock, Mach 7 (published case II)")
    tab.stations_spin.setValue(7)
    tab.points_spin.setValue(20)
    _run(app, tab)
    dlg = W.MeshSelectDialog(None, None, None, title="Select Mesh", ltoc_tab=tab)
    assert dlg.radio_ltoc.isEnabled() and dlg.radio_ltoc.isChecked()
    dlg._on_accept()
    path, source = dlg.get_result()
    try:
        assert source == "ltoc" and os.path.getsize(path) > 84
    finally:
        os.unlink(path)
    # Without an LTOCs tab the dialog is unchanged (no LTOCs entry is enabled).
    plain = W.MeshSelectDialog(None, None, None, title="Select Mesh")
    assert not plain.radio_ltoc.isEnabled() and plain.radio_browse.isChecked()


# ---------------------------------------------------------------------------
#  Project files ("Save/Load Parameters", section "ltoc_waverider")
# ---------------------------------------------------------------------------

def test_tab_params_round_trip_presets_custom_and_oc_import(tab):
    from ltoc_waverider_tab import CUSTOM, OC_IMPORT

    tab.source_combo.setCurrentText("Circular cone shock 12°, Mach 6")   # values rounded in the widgets
    tab.stations_spin.setValue(31)
    tab.points_spin.setValue(55)
    tab.cluster_combo.setCurrentIndex(1)
    saved = tab.get_params_dict()
    tab.source_combo.setCurrentIndex(0)
    tab.stations_spin.setValue(9)
    tab.set_params_dict(saved)
    assert tab.source_combo.currentText() == "Circular cone shock 12°, Mach 6"
    assert (tab.stations_spin.value(), tab.points_spin.value(), tab.cluster_combo.currentIndex()) == (31, 55, 1)
    assert tab.get_params_dict() == saved

    tab.source_combo.setCurrentText("Elliptic cone shock, Mach 6 (published case I)")
    tab.shock_spins["a_q"].setValue(0.61)                                # becomes custom
    saved = tab.get_params_dict()
    assert saved["source"] == "custom"
    tab.source_combo.setCurrentIndex(1)
    tab.set_params_dict(saved)
    assert tab.source_combo.currentText() == CUSTOM and tab.shock_spins["a_q"].value() == 0.61

    tab.source_combo.setCurrentText(OC_IMPORT)
    saved = tab.get_params_dict()
    tab.source_combo.setCurrentIndex(0)
    tab.set_params_dict(saved)
    assert tab.source_combo.currentText() == OC_IMPORT and not tab.fct_spin.isEnabled()
    tab.source_combo.setCurrentIndex(0)


def test_renamed_or_edited_presets_load_as_custom_shocks(tab):
    from ltoc_waverider_tab import CUSTOM

    tab.source_combo.setCurrentText("Shape-changing elliptic shock, Mach 7 (published case II)")
    saved = tab.get_params_dict()
    renamed = dict(saved, source="R1 case II (old name)")
    tab.source_combo.setCurrentIndex(0)
    tab.set_params_dict(renamed)
    assert tab.source_combo.currentText() == CUSTOM and tab.shock_spins["b_q"].value() == 0.8
    edited = dict(saved, shock=dict(saved["shock"], a_q=0.7))           # preset name, other numbers
    tab.set_params_dict(edited)
    assert tab.source_combo.currentText() == CUSTOM and tab.shock_spins["a_q"].value() == 0.7
    tab.source_combo.setCurrentIndex(0)


def test_hub_project_files_save_and_restore_the_ltoc_tab(app, tmp_path, monkeypatch):
    import json

    import waverider_gui as W

    session = tmp_path / "last_session.json"                          # never the user's file
    monkeypatch.setattr(W.WaveriderGUI, "_params_file_path", lambda self: str(session))
    hub = W.WaveriderGUI()
    t = hub.ltoc_waverider_tab
    t.source_combo.setCurrentText("Shape-changing elliptic shock, Mach 7 (published case II)")
    t.stations_spin.setValue(33)
    project = tmp_path / "project.json"
    hub._write_params_to_file(str(project))
    data = json.loads(project.read_text())
    assert data["ltoc_waverider"]["source"] == "Shape-changing elliptic shock, Mach 7 (published case II)"
    assert data["ltoc_waverider"]["n_stations"] == 33 and "oc_waverider" in data

    t.source_combo.setCurrentIndex(0)
    t.stations_spin.setValue(9)
    hub._read_params_from_file(str(project))
    assert t.source_combo.currentText() == "Shape-changing elliptic shock, Mach 7 (published case II)"
    assert t.stations_spin.value() == 33

    # A project saved before LTOCs existed leaves the tab as it is.
    data.pop("ltoc_waverider")
    old = tmp_path / "old_project.json"
    old.write_text(json.dumps(data))
    t.stations_spin.setValue(11)
    hub._read_params_from_file(str(old))
    assert t.stations_spin.value() == 11

    # The session file written on close restores the tab at the next start.
    hub._auto_save_params()
    hub2 = W.WaveriderGUI()
    assert hub2.ltoc_waverider_tab.stations_spin.value() == 11
    assert hub2.ltoc_waverider_tab.source_combo.currentText() == "Shape-changing elliptic shock, Mach 7 (published case II)"


def test_data_file_source_loads_solves_and_round_trips(app, tab, tmp_path):
    from ltoc_waverider_tab import BSPLINE

    tab.source_combo.setCurrentText("Shape-changing elliptic shock, Mach 7 (published case II)")
    path = tmp_path / "case2.csv"
    tab.save_shock_file(path)
    tab.source_combo.setCurrentIndex(0)
    tab.load_shock_file(path)
    assert tab.source_combo.currentText() == BSPLINE
    assert tab.data_box.isVisible() and not tab.elliptic_box.isVisible()
    assert (tab.xbase_spin.value(), tab.mach_spin.value(), tab.fct_spin.value()) == (1.5, 7.0, 0.25)
    assert "61 x 41 grid" in tab.data_label.text()
    tab.stations_spin.setValue(7)
    tab.points_spin.setValue(20)
    _run(app, tab)
    assert tab.waverider is not None and "7/7 stations solved" in tab.output_label.text()
    saved = tab.get_params_dict()
    assert saved["source"] == "bspline" and len(saved["shock_data"]["u"]) == 61
    tab.source_combo.setCurrentIndex(0)
    tab._set_shock_data(None)
    tab.set_params_dict(saved)
    assert tab.source_combo.currentText() == BSPLINE and tab.shock_data is not None
    # Without data, 'Generate' reports an input error instead of solving.
    tab._set_shock_data(None)
    tab.generate()
    assert "load a CSV file first" in tab.output_label.text()
    bad = tmp_path / "bad.csv"
    bad.write_text("u,v,X,Y,Z\n0,0,0,0,x\n")
    with pytest.raises(ValueError, match="line 2"):
        tab.load_shock_file(bad)
    tab.source_combo.setCurrentIndex(0)


def test_projects_with_earlier_preset_names_still_load(tab):
    """Preset names were made plain-language; projects saved with the earlier
    names ("R1 waverider case II (Eq. 25), M 7") still select the preset."""
    tab.source_combo.setCurrentText("Shape-changing elliptic shock, Mach 7 (published case II)")
    saved = dict(tab.get_params_dict(), source="R1 waverider case II (Eq. 25), M 7")
    tab.source_combo.setCurrentIndex(0)
    tab.set_params_dict(saved)
    assert tab.source_combo.currentText() == "Shape-changing elliptic shock, Mach 7 (published case II)"
    tab.source_combo.setCurrentIndex(0)


def test_info_panel_lists_the_waverider_characteristics(app, tab):
    tab.source_combo.setCurrentText("Elliptic cone shock, Mach 6 (published case I)")
    tab.stations_spin.setValue(7)
    tab.points_spin.setValue(20)
    _run(app, tab)
    text = tab.canvas_3d._info_text.get_text()
    assert text.startswith("LTOCs (Elliptic cone shock, Mach 6)")
    for key in ("M_design", "Length", "Span", "Height", "Volume", "Planform", "Volumetric Efficiency", "CL", "L/D"):
        assert key in text
    tab.canvas_3d.placeholder()
    assert tab.canvas_3d._info_text is None
