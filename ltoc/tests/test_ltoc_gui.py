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

    name = "R1 waverider case II (Eq. 25), M 7"
    tab.source_combo.setCurrentText(name)
    shock, M, z, _ = PRESETS[name]
    assert tab.shock_spins["a_q"].value() == pytest.approx(shock.a_q)
    assert tab.mach_spin.value() == pytest.approx(M) and tab.fct_spin.value() == pytest.approx(z)
    tab.shock_spins["a_q"].setValue(shock.a_q + 0.01)
    assert tab.source_combo.currentText() == CUSTOM
    tab.source_combo.setCurrentText(name)                            # restore


def test_generate_runs_in_the_background_and_fills_the_views(app, tab, tmp_path):
    tab.source_combo.setCurrentText("R1 waverider case I (Eq. 23), M 6")
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


def test_oc_import_without_an_oc_design_gives_an_input_error(tab):
    from ltoc_waverider_tab import OC_IMPORT

    tab.source_combo.setCurrentText(OC_IMPORT)
    assert not tab.fct_spin.isEnabled() and not tab.shock_spins["x_q"].isEnabled()
    tab.generate()
    assert "generate one in the OC Waverider tab first" in tab.output_label.text()
    tab.source_combo.setCurrentIndex(0)


def test_aero_analysis_mesh_dialog_offers_the_ltoc_waverider(app, tab):
    import waverider_gui as W

    tab.source_combo.setCurrentText("R1 waverider case II (Eq. 25), M 7")
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
