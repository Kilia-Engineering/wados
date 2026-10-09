"""LTOCs GUI tab: local-turning osculating cones waverider (spec Phase 5).

Method: Zheng, Hu, Li, Zhu, You, Song, "Local-Turning Osculating Cones Method
for Waverider Design", AIAA J 58(8):3499-3513, 2020 (R1). Library: ``ltoc``.
User guide: ``docs/ltoc/user_guide.md``.

The tab follows the GVWD/PSWR layout: a left scroll panel with parameter
groups and actions, and on the right a 3-D view over a strip of analysis
views (front view, wall pressure, diagnostics). The stream surfaces are
solved in a QThread with a per-station progress bar and can be cancelled.
Stations the method refuses (spec flags 4 and 7) are listed with their
reason, and the geometry is then not exported. Export writes STL in metres
and STEP in millimetres, in the GUI frame (x streamwise, y up, z span), as
the other tabs do. The Aero Analysis tab can take the waverider directly
("LTOCs Waverider" in its mesh dialog).
"""
from __future__ import annotations

import sys

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                             QScrollArea, QSpinBox, QSplitter, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ltoc import viz
from ltoc.design import PRESETS, EllipticShock, LTOCDesign
from ltoc.ltoc import LTOCError

CUSTOM = "Custom elliptic ruled shock (R1 Eq. 15)"
OC_IMPORT = "Import the current OC design (OC Waverider tab)"
_SHOCK_FIELDS = (("x_p", "Upstream curve p(m): X"), ("a_p", "p(m): Y semi-axis a_p"),
                 ("b_p", "p(m): Z semi-axis b_p"), ("x_q", "Base curve q(m): X (base plane)"),
                 ("a_q", "q(m): Y semi-axis a_q"), ("b_q", "q(m): Z semi-axis b_q"))
def _ink() -> str:
    """Neutral line/text colour of the active theme (the hub sets a dark theme)."""
    import matplotlib

    return matplotlib.rcParams["text.color"]


_MUTED = "#888888"
_OUT_STYLE = ("font-size: 10px; font-family: monospace; background-color: #1A1A1A; "
              "padding: 6px; border-radius: 3px; color: {}")


# ======================================================================
#  Worker
# ======================================================================

class _LTOCWorker(QThread):
    """Solves the stream surfaces off the GUI thread."""

    progress = pyqtSignal(int, int)          # stations done, total
    finished_ok = pyqtSignal(object)         # ltoc.design.LTOCResult
    finished_err = pyqtSignal(str)

    def __init__(self, design: LTOCDesign, parent=None):
        super().__init__(parent)
        self.design = design
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _on_progress(self, done, total):
        self.progress.emit(done, total)
        return not self._cancel

    def run(self):
        try:
            self.finished_ok.emit(self.design.build(progress=self._on_progress))
        except LTOCError as e:
            self.finished_err.emit(str(e))
        except Exception as e:                       # report, never crash the hub
            import traceback
            traceback.print_exc()
            self.finished_err.emit(f"{type(e).__name__}: {e}")


# ======================================================================
#  Canvases
# ======================================================================

class _Canvas3D(FigureCanvas):
    """Lower surface coloured by p/p_inf, upper surface, shock curves, LE."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 6))
        self.fig.patch.set_alpha(0.0)                # dark Qt background shows through
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("none")
        super().__init__(self.fig)
        self.setParent(parent)
        self._cbar = None
        self.placeholder()

    def placeholder(self, text="Click 'Generate LTOCs Waverider'"):
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self.ax.clear()
        self.ax.set_title(text, color=_ink())
        self.ax.tick_params(colors=_MUTED)
        self.draw()

    def plot(self, res, full_span: bool = True):
        wr, g = res.waverider, res.grid
        P, p = g.points, g.p
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        ax = self.ax
        ax.clear()
        norm = Normalize(float(p.min()), float(p.max()))
        U = np.asarray(wr.upper_surface_streams)
        for sgn in ((1.0, -1.0) if full_span else (1.0,)):
            ax.plot_surface(P[..., 0], sgn * P[..., 2], P[..., 1], facecolors=viz.SEQ_BLUE(norm(p)),
                            shade=False, linewidth=0, antialiased=False)
            ax.plot_surface(U[..., 0], sgn * U[..., 2], U[..., 1], color="#b8b7b2", alpha=0.35,
                            shade=False, linewidth=0)
            LE = wr.leading_edge
            ax.plot(LE[:, 0], sgn * LE[:, 2], LE[:, 1], color=_ink(), lw=2.0)
        every = max(1, len(wr.stations) // 8)
        for k, C in enumerate(res.shock_curves(every=every)):
            ax.plot(C[:, 0], C[:, 2], C[:, 1], color=viz.SERIES[1], lw=1.4,
                    label="Shock curves (LE to base)" if k == 0 else None)
        for setter, label in ((ax.set_xlabel, "x"), (ax.set_ylabel, "z (span)"),
                              (ax.set_zlabel, "y (up)")):
            setter(label, color=_ink())
        ax.tick_params(colors=_MUTED)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False                   # no grey box on the dark theme
            axis.pane.set_edgecolor(_MUTED)
        span = 2.0 * np.ptp(P[..., 2]) if full_span else np.ptp(P[..., 2])
        ax.set_box_aspect((np.ptp(P[..., 0]), max(span, 1e-9), max(np.ptp(U[..., 1]) +
                                                                    np.ptp(P[..., 1]), 1e-9)))
        ax.view_init(elev=-20, azim=-60)
        f = res.forces
        ax.set_title(f"LTOCs waverider   CL {f.CL:.4f}   CD {f.CD:.4f}   L/D {f.L_over_D:.3f}"
                     "   (planform reference)", color=_ink())
        ax.legend(loc="upper left", fontsize=9, framealpha=0.6)
        sm = viz.mpl.cm.ScalarMappable(cmap=viz.SEQ_BLUE, norm=norm)
        self._cbar = self.fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08)
        self._cbar.set_label("Wall pressure p / p∞", color=_ink())
        self._cbar.ax.tick_params(colors=_MUTED)
        self.draw()


class _Canvas2D(FigureCanvas):
    """Plain 2-D views with a placeholder."""

    def __init__(self, title: str, nrows: int = 1, ncols: int = 1, parent=None):
        self.fig = Figure(figsize=(9, 4), layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self._title, self._shape = title, (nrows, ncols)
        self.placeholder()

    def axes(self):
        self.fig.clear()
        return self.fig.subplots(*self._shape, squeeze=False)

    def placeholder(self, text: str = "Generate a waverider to populate."):
        ax = self.axes()[0, 0]
        ax.text(0.5, 0.5, text, ha="center", va="center", color="gray", transform=ax.transAxes)
        ax.set_title(self._title)
        ax.set_xticks([])
        ax.set_yticks([])
        self.draw()


# ======================================================================
#  Tab
# ======================================================================

class LTOCWaveriderTab(QWidget):
    """Local-turning osculating cones (LTOCs) waverider tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hub = parent                           # main window (OC design for the import)
        self.result = None                           # ltoc.design.LTOCResult
        self.waverider = None                        # LTOCWaverider when every station solved
        self._worker = None
        self._filling = False
        self._init_ui()
        self._on_source_changed(0)

    # ------------------------------------------------------------------
    #  UI
    # ------------------------------------------------------------------
    def _init_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(410)
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.addWidget(self._create_shock_group())
        left_l.addWidget(self._create_fct_group())
        left_l.addWidget(self._create_actions_group())
        self.output_label = QLabel("Choose a shock and click 'Generate LTOCs Waverider'.")
        self.output_label.setWordWrap(True)
        self.output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output_label.setStyleSheet(_OUT_STYLE.format("#aaaaaa"))
        left_l.addWidget(self.output_label)
        left_l.addStretch()
        left_scroll.setWidget(left_w)
        splitter.addWidget(left_scroll)

        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        self.canvas_3d = _Canvas3D()
        right_l.addWidget(NavigationToolbar(self.canvas_3d, self))
        right_l.addWidget(self.canvas_3d, 5)
        self.bottom_tabs = QTabWidget()
        self.canvas_front = _Canvas2D("Front view (base plane)")
        self.canvas_pressure = _Canvas2D("Wall pressure", 1, 2)
        self.canvas_diag = _Canvas2D("Diagnostics", 2, 2)
        self.bottom_tabs.addTab(self.canvas_front, "Front view")
        self.bottom_tabs.addTab(self.canvas_pressure, "Wall pressure")
        self.bottom_tabs.addTab(self.canvas_diag, "Diagnostics")
        right_l.addWidget(self.bottom_tabs, 4)
        splitter.addWidget(right_w)
        splitter.setSizes([450, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main.addWidget(splitter)

    def _create_shock_group(self) -> QGroupBox:
        g = QGroupBox("Shock surface")
        lay = QGridLayout()
        self.source_combo = QComboBox()
        for name in list(PRESETS) + [CUSTOM, OC_IMPORT]:
            self.source_combo.addItem(name)
        self.source_combo.setToolTip(
            "Prescribed shock surface.\n"
            "  R1 presets: the published waverider and test cases (R1 Eqs. 18, 23, 25).\n"
            "  Custom: any elliptic ruled shock S = (1-n) p(m) + n q(m) (R1 Eqs. 12, 15),\n"
            "    in R1's frame: X freestream, Y span, Z down.\n"
            "  Import: the shock of the OC design in the OC Waverider tab; LTOCs then\n"
            "    reproduces that design (validation case V5).")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        lay.addWidget(self.source_combo, 0, 0, 1, 2)
        self.shock_spins = {}
        for row, (key, label) in enumerate(_SHOCK_FIELDS, start=1):
            sb = QDoubleSpinBox()
            sb.setDecimals(4)
            sb.setRange(0.0, 1000.0)
            sb.setSingleStep(0.01)
            sb.setToolTip(f"{label}. p(m) = (x_p, a_p cos m, b_p sin m), "
                          "q(m) = (x_q, a_q cos m, b_q sin m), m in [0, pi].")
            sb.valueChanged.connect(self._on_shock_edited)
            lay.addWidget(QLabel(label + ":"), row, 0)
            lay.addWidget(sb, row, 1)
            self.shock_spins[key] = sb
        self.reference_label = QLabel("")
        self.reference_label.setWordWrap(True)
        self.reference_label.setStyleSheet("color: #888888;")
        lay.addWidget(self.reference_label, len(_SHOCK_FIELDS) + 1, 0, 1, 2)
        g.setLayout(lay)
        return g

    def _create_fct_group(self) -> QGroupBox:
        g = QGroupBox("Freestream, FCT and resolution")
        lay = QGridLayout()
        self.mach_spin = QDoubleSpinBox()
        self.mach_spin.setRange(1.1, 30.0)
        self.mach_spin.setDecimals(2)
        self.mach_spin.setSingleStep(0.5)
        self.mach_spin.setToolTip("Design Mach number (gamma = 1.4).")
        self.mach_spin.valueChanged.connect(self._on_shock_edited)
        self.fct_spin = QDoubleSpinBox()
        self.fct_spin.setRange(0.0001, 1000.0)
        self.fct_spin.setDecimals(4)
        self.fct_spin.setSingleStep(0.01)
        self.fct_spin.setToolTip(
            "Flow capture tube: the line Z = const in the base plane (R1 Eqs. 24, 26).\n"
            "The leading edge is the FCT projected upstream onto the shock.\n"
            "Must lie between 0 and b_q.")
        self.fct_spin.valueChanged.connect(self._on_shock_edited)
        self.stations_spin = QSpinBox()
        self.stations_spin.setRange(3, 201)
        self.stations_spin.setValue(25)
        self.stations_spin.setToolTip(
            "Stream surfaces across the half span (FCT points).\n"
            "25 reproduces R1 Tables 2 and 3 to 0.4 %; 49 to 0.35 %.")
        self.cluster_combo = QComboBox()
        self.cluster_combo.addItems(["clustered at the symmetry plane", "uniform"])
        self.cluster_combo.setToolTip(
            "Spacing of the FCT stations. Clustering at the symmetry plane resolves\n"
            "the fast spanwise change near the centre (Gate 4 report section 2).")
        self.points_spin = QSpinBox()
        self.points_spin.setRange(10, 320)
        self.points_spin.setValue(40)
        self.points_spin.setToolTip(
            "Shock points per stream surface between the leading edge and the base.\n"
            "40 is enough for forces to < 0.03 %; cost grows with its square.")
        rows = (("Mach number:", self.mach_spin), ("FCT Z (R1 frame, down):", self.fct_spin),
                ("FCT stations (half span):", self.stations_spin),
                ("Station spacing:", self.cluster_combo),
                ("Points per stream surface:", self.points_spin))
        for r, (label, w) in enumerate(rows):
            lay.addWidget(QLabel(label), r, 0)
            lay.addWidget(w, r, 1)
        g.setLayout(lay)
        return g

    def _create_actions_group(self) -> QGroupBox:
        g = QGroupBox("Actions")
        lay = QGridLayout()
        self.btn_generate = QPushButton("Generate LTOCs Waverider")
        self.btn_generate.setStyleSheet("QPushButton { background-color: #2196F3; color: white; "
                                        "font-weight: bold; padding: 6px; }")
        self.btn_generate.setToolTip("Solve one stream surface per FCT station (background thread).")
        self.btn_generate.clicked.connect(self.generate)
        lay.addWidget(self.btn_generate, 0, 0, 1, 2)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        lay.addWidget(self.btn_cancel, 1, 0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setToolTip("Stream surfaces solved / total.")
        lay.addWidget(self.progress, 1, 1)
        self.btn_export_stl = QPushButton("Export STL...")
        self.btn_export_stl.setToolTip("Closed full-span mesh, binary STL in metres,\n"
                                       "GUI frame (x streamwise, y up, z span).")
        self.btn_export_stl.clicked.connect(self.export_stl_dialog)
        self.btn_export_step = QPushButton("Export STEP...")
        self.btn_export_step.setToolTip("Faceted STEP solid via cadquery, millimetres,\n"
                                        "GUI frame (x streamwise, y up, z span).")
        self.btn_export_step.clicked.connect(self.export_step_dialog)
        for b in (self.btn_export_stl, self.btn_export_step):
            b.setEnabled(False)
        lay.addWidget(self.btn_export_stl, 2, 0)
        lay.addWidget(self.btn_export_step, 2, 1)
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(32)
        help_btn.setToolTip("Method, inputs, refusals and references.")
        help_btn.clicked.connect(self.show_help_dialog)
        lay.addWidget(help_btn, 3, 0)
        g.setLayout(lay)
        return g

    # ------------------------------------------------------------------
    #  Inputs
    # ------------------------------------------------------------------
    def _on_source_changed(self, _idx: int):
        name = self.source_combo.currentText()
        is_oc = name == OC_IMPORT
        for sb in self.shock_spins.values():
            sb.setEnabled(not is_oc)
        self.fct_spin.setEnabled(not is_oc)
        self.mach_spin.setEnabled(not is_oc)
        self.stations_spin.setEnabled(not is_oc)
        self.cluster_combo.setEnabled(not is_oc)
        if name in PRESETS:
            shock, M, z, ref = PRESETS[name]
            self._filling = True
            for key, sb in self.shock_spins.items():
                sb.setValue(getattr(shock, key))
            self.mach_spin.setValue(M)
            self.fct_spin.setValue(z)
            self._filling = False
            self.reference_label.setText(f"Reference: {ref}.")
        elif is_oc:
            self.reference_label.setText(
                "Uses the shock, Mach number and leading edge of the design last generated "
                "in the OC Waverider tab (one stream surface per OC plane).")
        else:
            self.reference_label.setText("Any elliptic ruled shock in R1's frame (Z down).")

    def _on_shock_edited(self, _value):
        """Editing a preset's numbers turns it into a custom shock."""
        if self._filling or self.source_combo.currentText() not in PRESETS:
            return
        self._filling = True
        self.source_combo.setCurrentText(CUSTOM)
        self._filling = False
        self.reference_label.setText("Any elliptic ruled shock in R1's frame (Z down).")

    def current_design(self) -> LTOCDesign:
        common = dict(n_points=self.points_spin.value())
        if self.source_combo.currentText() == OC_IMPORT:
            oc = getattr(self._hub, "waverider", None)
            if oc is None:
                raise ValueError("no OC design: generate one in the OC Waverider tab first")
            return LTOCDesign(oc_waverider=oc, **common)
        shock = EllipticShock(**{k: sb.value() for k, sb in self.shock_spins.items()})
        clustering = "symmetry" if self.cluster_combo.currentIndex() == 0 else "uniform"
        return LTOCDesign(shock=shock, M_inf=self.mach_spin.value(), fct_z=self.fct_spin.value(),
                          n_stations=self.stations_spin.value(), clustering=clustering, **common)

    # ------------------------------------------------------------------
    #  Run
    # ------------------------------------------------------------------
    def generate(self):
        try:
            design = self.current_design()
            design.validate()
        except ValueError as e:
            self._show_output(f"Input error: {e}", "#ef4444")
            return
        self.result, self.waverider = None, None
        for b in (self.btn_export_stl, self.btn_export_step):
            b.setEnabled(False)
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self._show_output("Solving stream surfaces ...", "#aaaaaa")
        self._worker = _LTOCWorker(design, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.finished_err.connect(self._on_error)
        self._worker.start()

    def cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _on_progress(self, done: int, total: int):
        self.progress.setValue(int(round(100.0 * done / max(total, 1))))

    def _finish_run(self):
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_error(self, msg: str):
        self._finish_run()
        self._show_output(f"Not generated: {msg}", "#ef4444")
        self.canvas_3d.placeholder("No waverider")

    def _on_done(self, res):
        self._finish_run()
        self.result = res
        if not res.ok:
            self._show_output("Some stream surfaces were refused; no geometry was built.\n"
                              + res.summary(), "#f59e0b")
            self.canvas_3d.placeholder("Stations refused: see the list on the left")
            for c in (self.canvas_front, self.canvas_pressure, self.canvas_diag):
                c.placeholder("Refused stations: see the output panel.")
            return
        self.waverider = res.waverider
        for b in (self.btn_export_stl, self.btn_export_step):
            b.setEnabled(True)
        self._show_output(res.summary(), "#22c55e")
        self.refresh_views()

    def _show_output(self, text: str, color: str):
        self.output_label.setText(text)
        self.output_label.setStyleSheet(_OUT_STYLE.format(color))

    # ------------------------------------------------------------------
    #  Views
    # ------------------------------------------------------------------
    def refresh_views(self):
        res = self.result
        if res is None or not res.ok:
            return
        self.canvas_3d.plot(res)
        self._plot_front(res)
        self._plot_pressure(res)
        self._plot_diagnostics(res)

    def _plot_front(self, res):
        ax = self.canvas_front.axes()[0, 0]
        P = res.grid.points
        B = res.base_shock()
        ax.plot(B[:, 2], B[:, 1], color=_ink(), lw=2.0, label="Shock at the base")
        U = np.asarray(res.waverider.upper_surface_streams)
        ax.plot(U[:, -1, 2], U[:, -1, 1], color=_MUTED, ls="--", lw=1.4, label="FCT (upper surface)")
        x0, x1 = float(P[..., 0].min()), res.waverider.x_base
        for k, xc in enumerate(np.linspace(x0 + 0.25 * (x1 - x0), x1, 4)):
            sec = [[np.interp(xc, row[:, 0], row[:, q]) for q in (2, 1)]
                   for row in P if row[0, 0] <= xc <= row[-1, 0] and row[-1, 0] > row[0, 0]]
            if sec:
                sec = np.asarray(sec)
                ax.plot(sec[:, 0], sec[:, 1], color=viz.SERIES[0], alpha=0.4 + 0.2 * k, lw=1.8,
                        label="Lower surface, x = const" if k == 0 else None)
        every = max(1, len(res.waverider.stations) // 10)
        for k, C in enumerate(res.shock_curves(every=every)):
            ax.plot(C[:, 2], C[:, 1], color=viz.SERIES[1], lw=1.2,
                    label="Shock curves, LE to base" if k == 0 else None)
        ax.set_aspect("equal")
        ax.set_xlabel("z (span)")
        ax.set_ylabel("y (up)")
        ax.set_title("Front view: base plane and x-projected shock curves")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        self.canvas_front.draw()

    def _plot_pressure(self, res):
        axes = self.canvas_pressure.axes()[0]
        for k, (z, (x, p)) in enumerate(res.streamwise_sections().items()):
            axes[0].plot(x, p, color=viz.SERIES[k], marker=viz.MARKERS[k], markevery=0.1,
                         label=f"z = {z:.3g}")
        for k, (xc, (z, p)) in enumerate(res.crosswise_sections().items()):
            axes[1].plot(z, p, color=viz.SERIES[k], marker=viz.MARKERS[k], markevery=0.1,
                         label=f"x = {xc:.3g}")
        for ax, xl, t in ((axes[0], "x", "Streamwise planes, z = const"),
                          (axes[1], "z (span)", "Crosswise planes, x = const")):
            ax.set_xlabel(xl)
            ax.set_ylabel("p / p∞")
            ax.set_title(t)
            ax.legend(fontsize=8)
        self.canvas_pressure.draw()

    def _plot_diagnostics(self, res):
        rows = res.station_diagnostics()
        axes = self.canvas_diag.axes()
        f = [r["fct_fraction"] for r in rows]
        panels = ((axes[0, 0], "azimuth_drift_deg", "Azimuth drift, LE to base (deg)"),
                  (axes[0, 1], "axis_drift_over_L", "Axis-centre drift / L"),
                  (axes[1, 0], "extension_needed", "Shock extension past base (chords)"),
                  (axes[1, 1], "body_step_ratio", "Max / min body step"))
        for ax, key, label in panels:
            ax.plot(f, [r[key] for r in rows], color=viz.SERIES[0], marker="o", ms=4,
                    label="needed" if key == "extension_needed" else None)
            if key == "extension_needed":
                ax.plot(f, [r["extension_used"] for r in rows], color=viz.SERIES[1], ls="--",
                        label="used")
                ax.legend(fontsize=8)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("FCT station (arc-length fraction)", fontsize=9)
        self.canvas_diag.draw()

    # ------------------------------------------------------------------
    #  Export
    # ------------------------------------------------------------------
    def export_stl_dialog(self):
        if self.waverider is None:
            QMessageBox.warning(self, "No geometry", "Generate an LTOCs waverider first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export STL", "ltoc_waverider.stl",
                                              "STL Files (*.stl)")
        if not path:
            return
        try:
            self.waverider.export_stl(path, full_span=True)
            QMessageBox.information(self, "Saved", f"STL written:\n{path}\n\n"
                                    "Units: METRES; frame x streamwise, y up, z span")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"{type(e).__name__}: {e}")

    def export_step_dialog(self):
        if self.waverider is None:
            QMessageBox.warning(self, "No geometry", "Generate an LTOCs waverider first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export STEP", "ltoc_waverider.step",
                                              "STEP Files (*.step *.stp)")
        if not path:
            return
        try:
            from geometry_export import StepUnavailableError
        except ImportError:                          # pragma: no cover
            StepUnavailableError = RuntimeError
        try:
            self.waverider.export_step(path, full_span=True)
            QMessageBox.information(self, "Saved", f"STEP written:\n{path}\n\n"
                                    "Units: MILLIMETRES; frame x streamwise, y up, z span")
        except StepUnavailableError as e:
            QMessageBox.critical(self, "STEP export unavailable",
                                 f"cadquery is required for STEP export.\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"{type(e).__name__}: {e}")

    def stl_for_analysis(self) -> str:
        """Write the current waverider to a temporary STL for the Aero Analysis tab."""
        import tempfile

        if self.waverider is None:
            raise ValueError("no LTOCs waverider generated")
        tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
        tmp.close()
        self.waverider.export_stl(tmp.name, full_span=True)
        return tmp.name

    # ------------------------------------------------------------------
    #  Help
    # ------------------------------------------------------------------
    _HELP_TEXT = (
        "LTOCs: Local-Turning Osculating Cones waverider\n"
        "===============================================\n\n"
        "METHOD (R1: Zheng et al., AIAA J 58(8), 2020)\n"
        "  A waverider is designed from a prescribed 3-D shock surface and a flow\n"
        "  capture tube (FCT) in the base plane. The leading edge is the FCT\n"
        "  projected upstream onto the shock. For every leading-edge point:\n"
        "   A. the shock curve follows the freestream projected onto the shock\n"
        "      (it turns locally, unlike an osculating plane);\n"
        "   B. the flow between the shock and the body is solved by a noncoaxial\n"
        "      method of characteristics in a meridian plane, with the local\n"
        "      crosswise radius r = cos(beta) / kappa_b;\n"
        "   C. the meridian solution is mapped back to 3-D (R1 Eqs. 10-11).\n"
        "  The lower surface is the set of body streamlines; the upper surface is\n"
        "  the freestream surface through the leading edge.\n\n"
        "INPUTS\n"
        "  Shock: R1 presets, a custom elliptic ruled shock\n"
        "    S(m, n) = (1-n) p(m) + n q(m), p = (x_p, a_p cos m, b_p sin m),\n"
        "    q = (x_q, a_q cos m, b_q sin m), in R1's frame (X freestream, Y span,\n"
        "    Z down), or the shock of the current OC design.\n"
        "  FCT: the line Z = const in the base plane X = x_q.\n"
        "  Stations: stream surfaces across the half span. Points: shock points per\n"
        "    stream surface between the leading edge and the base.\n\n"
        "REFUSALS (no geometry is built if any station is refused)\n"
        "  concave       shock cross-section curves away from the body (kappa_b < 0)\n"
        "  sub_mach      shock angle at or below the Mach angle\n"
        "  detached      shock angle beyond detachment\n"
        "  undetermined  the body is not determined up to the base by the shock\n"
        "                (the shock must extend past the base plane)\n"
        "  outside_shock the FCT point lies outside the shock at the base\n"
        "  limit_surface characteristics of one family crossed (limit surface)\n\n"
        "OUTPUTS\n"
        "  CL, CD, L/D: inviscid lower-surface forces (R1 Eqs. 19-22), absolute\n"
        "    pressure. Planform reference area (this reproduces R1 Tables 2 and 3);\n"
        "    wetted-area values are also listed.\n"
        "  Views: 3-D (lower surface coloured by p/p_inf), front view, wall\n"
        "    pressure on streamwise and crosswise planes, per-station diagnostics.\n"
        "  Export: STL (metres) and STEP (millimetres), frame x streamwise, y up,\n"
        "    z span. The Aero Analysis tab can load the waverider directly.\n\n"
        "REFERENCES\n"
        "  R1 Zheng, Hu, Li, Zhu, You, Song, Local-Turning Osculating Cones Method\n"
        "     for Waverider Design, AIAA J 58(8):3499-3513, 2020, doi:10.2514/1.J059139.\n"
        "  R2 Zheng, Li, Zhu, You, Multiple Osculating Cones' Waverider Design Method\n"
        "     for Ruled Shock Surfaces, AIAA J 58(2):854-866, 2020, doi:10.2514/1.J058640.\n"
        "  R3 Zheng, Zhu, You, Design of multistage compression waverider based on the\n"
        "     local-turning osculating cones method, Chin. J. Theor. Appl. Mech.\n"
        "     54(3):601-611, 2022, doi:10.6052/0459-1879-21-357 (in Chinese).\n"
        "  R4 Qian, Sobieczky, Waverider Design with Parametric Flow Quality Control by\n"
        "     Inverse Method of Characteristics, ICAS 2002.\n"
        "  R5 Zucrow, Hoffman, Gas Dynamics Vol. 2, Wiley, 1976 (unit processes).\n"
        "  Details: docs/ltoc/user_guide.md and the gate reports in docs/ltoc.\n"
    )

    def show_help_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("LTOCs Waverider - Help")
        dlg.resize(720, 620)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Consolas", 9))
        text.setPlainText(self._HELP_TEXT)
        layout.addWidget(text)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        layout.addWidget(bb)
        dlg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    tab = LTOCWaveriderTab()
    tab.setWindowTitle("LTOCs Waverider Tab (standalone)")
    tab.resize(1280, 820)
    tab.show()
    sys.exit(app.exec_())
