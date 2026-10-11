"""Every design tab hands its geometry to the Aero Analysis tab.

For each tab of the hub: generate a design, *Send to Aero Analysis*, and
check what arrives: a closed full-span mesh with outward normals in the
analysis frame (x streamwise, y up, z span), A_ref equal to its planform
area, and the tab listed in the Aero Analysis mesh dialog.

Run offscreen (``QT_QPA_PLATFORM=offscreen``); skipped where PyQt5 is not
installed, as in CI.
"""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

# tab attribute in the hub -> (generate method, settings to keep it quick)
TABS = {
    "shadow_waverider_tab": ("generate", {}),
    "planar_waverider_tab": ("generate_waverider", {}),
    "hybrid_waverider_tab": ("generate_waverider", {}),
    "vmn_waverider_tab": ("generate_waverider", {}),
    "vmof_waverider_tab": ("generate_waverider", {}),
    "vmplo_waverider_tab": ("generate_waverider", {}),
    "mfof_waverider_tab": ("_on_generate", {"n_z_spin": 40, "n_x_spin": 30}),
    "pswr_waverider_tab": ("generate_waverider", {}),
    "gvwd_waverider_tab": ("generate_geometry", {}),
    "ltoc_waverider_tab": ("generate", {"stations_spin": 7, "points_spin": 20}),
}


@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    import waverider_gui as W

    app = QApplication.instance() or QApplication([])
    mp = pytest.MonkeyPatch()
    for name in ("information", "warning", "critical", "question"):    # no modal dialogs
        mp.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Ok))
    session = tmp_path_factory.mktemp("session") / "last_session.json"   # never the user's file
    mp.setattr(W.WaveriderGUI, "_params_file_path", lambda self: str(session))
    gui = W.WaveriderGUI()
    gui._app = app
    yield gui
    mp.undo()


def _mesh(path):
    from stl import mesh as stl_mesh

    tri = stl_mesh.Mesh.from_file(path).vectors.astype(float)
    _, idx = np.unique(np.round(tri.reshape(-1, 3), 9), axis=0, return_inverse=True)
    f = idx.ravel().reshape(-1, 3)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    return tri.reshape(-1, 3), bool(np.all(counts == 2)), float(volume)


def _generate(hub, attr):
    import aero_handoff

    tab = getattr(hub, attr)
    method, settings = TABS[attr]
    for spin, value in settings.items():
        getattr(tab, spin).setValue(value)
    getattr(tab, method)()
    t0 = time.time()
    while not aero_handoff.has_geometry(tab) and time.time() - t0 < 300:
        hub._app.processEvents()
        time.sleep(0.05)
    assert aero_handoff.has_geometry(tab), f"{attr}: no design after generate"
    return tab


@pytest.mark.parametrize("attr", list(TABS))
def test_tab_sends_a_closed_full_span_mesh(hub, attr):
    from reference_area_calculator import calculate_reference_area_from_stl

    tab = _generate(hub, attr)
    hub.tab_widget.setCurrentWidget(tab)
    hub.send_to_aero_analysis(tab)
    assert hub.tab_widget.currentWidget() is hub._aero_tab
    name = hub.tab_widget.tabText(hub.tab_widget.indexOf(tab))
    assert hub.last_stl_source == name and name in hub.aero_geo_info.text()

    v, closed, volume = _mesh(hub.last_stl_file)
    assert closed and volume > 0.0                                   # closed, normals outward
    span = v[:, 2].max() - v[:, 2].min()
    assert v[:, 2].min() == pytest.approx(-v[:, 2].max(), abs=1e-6 * span)   # full span along z
    assert np.ptp(v[:, 1]) < span                                    # y is up, not the span
    area, _ = calculate_reference_area_from_stl(hub.last_stl_file)
    assert hub.aref_spin.value() == pytest.approx(area, rel=1e-3)


def test_oc_tab_sends_its_design(hub):
    hub.generate_waverider()
    assert hub.waverider is not None
    hub._send_oc_to_aero_analysis()
    assert hub.last_stl_source == "OC Waverider"
    _, closed, volume = _mesh(hub.last_stl_file)
    assert closed and volume > 0.0


def test_mesh_dialog_lists_every_design_tab(hub):
    import aero_handoff
    import waverider_gui as W

    for attr in TABS:
        if not aero_handoff.has_geometry(getattr(hub, attr)):
            _generate(hub, attr)
    dlg = W.MeshSelectDialog(hub, hub.last_stl_file, hub.shadow_waverider_tab,
                             ltoc_tab=hub.ltoc_waverider_tab,
                             sources=hub._analysis_sources(),
                             last_stl_source=hub.last_stl_source)
    names = {name: dlg._btn_group.button(i) for i, (name, _) in dlg._tab_sources.items()}
    assert len(names) == len(TABS) + 1                               # + the OC tab
    assert all(radio.isEnabled() for radio in names.values())
    names["PSWR-1 Waverider"].setChecked(True)
    dlg._on_accept()
    path, source = dlg.get_result()
    assert source == "PSWR-1 Waverider" and _mesh(path)[1]
    names["Cone-derived Waverider"].setChecked(True)                # earlier source keys kept
    dlg._on_accept()
    assert dlg.get_result()[1] == "shadow"
