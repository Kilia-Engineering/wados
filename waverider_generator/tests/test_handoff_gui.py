"""Every design tab hands its geometry to the Aero Analysis and Optimization tabs.

For each tab of the hub: generate a design, *Send to Aero Analysis*, and
check what arrives: a closed full-span mesh with outward normals in the
analysis frame (x streamwise, y up, z span), A_ref equal to its planform
area, and the tab listed in the Aero Analysis mesh dialog.

*Send to Optimization* must open the tab's inputs as a case in the
Optimization tab, an optimized design must load back into the tab, and a
short run must go through the Optimization tab's worker.

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
    import handoff

    tab = getattr(hub, attr)
    method, settings = TABS[attr]
    for spin, value in settings.items():
        getattr(tab, spin).setValue(value)
    getattr(tab, method)()
    t0 = time.time()
    while not handoff.has_geometry(tab) and time.time() - t0 < 300:
        hub._app.processEvents()
        time.sleep(0.05)
    assert handoff.has_geometry(tab), f"{attr}: no design after generate"
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
    import handoff
    import waverider_gui as W

    for attr in TABS:
        if not handoff.has_geometry(getattr(hub, attr)):
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


# ---------------------------------------------------------------------------
#  Optimization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attr", list(TABS))
def test_tab_sends_its_design_to_optimization(hub, attr):
    tab = getattr(hub, attr)
    name = hub.tab_widget.tabText(hub.tab_widget.indexOf(tab))
    hub.send_to_optimization(tab)
    opt = hub.optimization_tab
    assert hub.tab_widget.currentWidget() is hub._opt_hub_tab
    assert opt.method_combo.currentText() == name and not opt.is_oc_method()
    assert opt.design_vars_group.isHidden() and not opt.method_settings.isHidden()
    case = tab.optimization_case()
    assert opt.var_table.rowCount() == len(case.variables)
    sent = opt.current_case()
    assert sent.params() == case.params() and sent.mach == pytest.approx(case.mach)
    assert [v.enabled for v in sent.variables] == [v.enabled for v in case.variables]


@pytest.mark.parametrize("attr", list(TABS))
def test_optimized_design_loads_back_into_its_tab(hub, attr):
    """Each enabled variable moved to 3/4 of its range: the tab must take
    the values (to its widgets' precision) and generate the design."""
    import handoff

    tab = getattr(hub, attr)
    name = hub.tab_widget.tabText(hub.tab_widget.indexOf(tab))
    method, settings = TABS[attr]
    case = tab.optimization_case()
    x = [v.lower + 0.75 * (v.upper - v.lower) for v in case.active()]
    hub.apply_optimized_design(name, case.params(x))
    t0 = time.time()
    while time.time() - t0 < 300:                       # threaded generators finish here
        hub._app.processEvents()
        if handoff.has_geometry(tab) and not getattr(getattr(tab, "_worker", None), "isRunning", lambda: False)() \
                and not getattr(getattr(tab, "_geom_worker", None), "isRunning", lambda: False)() \
                and not getattr(getattr(tab, "worker", None), "isRunning", lambda: False)():
            break
        time.sleep(0.05)
    assert hub.tab_widget.currentWidget() is tab and handoff.has_geometry(tab)
    from method_optimization import get_param

    now = tab.optimization_case().params()
    for v, xv in zip(case.active(), x):
        # widgets keep 2-3 decimals (some sliders truncate to 0.01)
        assert get_param(now, v.name) == pytest.approx(xv, abs=0.01 * (v.upper - v.lower) + 0.011), v.name


def test_short_optimization_run_and_apply(hub, tmp_path, monkeypatch):
    """GVWD caret through the Optimization tab's worker: designs are logged,
    the Pareto designs written, and the best one applied to the GVWD tab."""
    pytest.importorskip("pymoo")
    monkeypatch.chdir(tmp_path)                          # results/ goes here
    tab = hub.gvwd_waverider_tab
    tab.mode_combo.setCurrentIndex(2)                    # caret
    hub.send_to_optimization(tab)
    opt = hub.optimization_tab
    opt.pop_size_spin.setValue(10)
    opt.n_gen_spin.setValue(5)
    opt.start_optimization()
    t0 = time.time()
    while opt.optimization_worker.isRunning() and time.time() - t0 < 600:
        hub._app.processEvents()
        time.sleep(0.05)
    hub._app.processEvents()
    log = opt.console.toPlainText()
    assert "OPTIMIZATION COMPLETE" in log and "Starting design: CL/CD=" in log
    # the recommended base pressure and skin friction are on by default
    assert "Base pressure: Igra 2018 correlation; skin friction: turbulent" in log
    assert " friction" in log.split("Starting design: CL/CD=")[1].splitlines()[1]
    folder = opt.last_results_folder
    import pandas as pd

    designs = pd.read_csv(os.path.join(folder, "designs.csv"))
    assert len(designs) == 50 and designs["Success"].all()
    assert list(designs.columns[3:5]) == ["theta_d_deg", "Lambda_deg"]
    assert len(os.listdir(os.path.join(folder, "pareto_designs"))) >= 3   # stl, json, info
    start = float(log.split("Starting design: CL/CD=")[1].split(",")[0])
    assert designs["CL/CD"].max() > start
    assert opt.apply_best_btn.isEnabled()
    opt.apply_best_design()
    best = designs.loc[designs["CL/CD"].idxmax()]
    p = tab.optimization_case().params()
    assert p["theta_d_deg"] == pytest.approx(best["theta_d_deg"], abs=0.01)
    assert hub.tab_widget.currentWidget() is tab


def test_aero_analysis_applies_the_corrections(hub, tmp_path, monkeypatch):
    """The Aero Analysis worker applies the tab's base-pressure and friction
    settings (the recommended ones by default) on top of PySAGAS."""
    import waverider_gui as W
    from aero_corrections import INVISCID, AeroCorrections

    monkeypatch.chdir(tmp_path)                          # the worker writes its VTK file here
    tab = _generate(hub, "gvwd_waverider_tab")
    hub.send_to_aero_analysis(tab)
    assert hub.aero_corrections_group.corrections() == AeroCorrections()
    assert hub.import_corrections_group.corrections() == AeroCorrections()
    flow = {"mach": 6.0, "pressure": 2549.0, "temperature": 221.55}

    def run(corrections):
        out = []
        worker = W.AnalysisWorker(hub.last_stl_file, flow, 0.0, hub.aref_spin.value(),
                                  corrections=corrections)
        worker.finished.connect(out.append)
        worker.error.connect(out.append)
        worker.run()                                     # in this thread
        assert isinstance(out[0], dict), out[0]
        return out[0]

    raw, corr = run(INVISCID), run(hub.aero_corrections_group.corrections())
    assert raw["CD_base"] == raw["CD_friction"] == 0.0
    assert corr["CD_inviscid"] == pytest.approx(raw["CD"], rel=1e-12)
    assert corr["CD_base"] < 0.0 < corr["CD_friction"]
    assert corr["CL/CD"] == pytest.approx(corr["CL"] / corr["CD"])
    text = W._drag_breakdown_text(corr)
    assert "Skin friction" in text and "Igra 2018" in text

    hub.aero_corrections_group.set_corrections(AeroCorrections("freestream", "off"))
    assert hub.aero_corrections_group.corrections() == AeroCorrections("freestream", "off")
    assert not hub.aero_corrections_group.wall_spin.isEnabled()
    hub.aero_corrections_group.set_corrections(AeroCorrections())
