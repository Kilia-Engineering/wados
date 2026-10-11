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
the other tabs do. "Send to Aero Analysis" loads the waverider into the Aero
Analysis tab, whose mesh dialog also lists it ("LTOCs Waverider").
"""
from __future__ import annotations

import sys

import numpy as np
from PyQt5.QtCore import QThread, Qt, pyqtSignal

import aero_handoff
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
from ltoc.design import PRESET_ALIASES, PRESETS, EllipticShock, LTOCDesign, ShockData
from ltoc.ltoc import LTOCError

CUSTOM = "Custom elliptic shock (two ellipses)"
BSPLINE = "Shock from a data file (CSV)"
OC_IMPORT = "Import the current OC design (OC Waverider tab)"
_COORDS = "Coordinates: X downstream, Y across the span, Z downward."
_CUSTOM_HELP = ("Straight lines join matching points of an upstream ellipse and a base ellipse; "
                "the base ellipse's X is the base plane. " + _COORDS)
_DATA_HELP = ("Shock points on a grid, read from a CSV file and joined into a smooth surface "
              "(bicubic spline). " + _COORDS + " The points must continue past the base "
              "plane (typically 0.5-1.5 chords), because the body at the base depends on the "
              "shock downstream of it.")
_SHOCK_FIELDS = (("x_p", "Upstream ellipse: X position"),
                 ("a_p", "Upstream ellipse: half-width (Y)"),
                 ("b_p", "Upstream ellipse: half-height (Z)"),
                 ("x_q", "Base ellipse: X position (base plane)"),
                 ("a_q", "Base ellipse: half-width (Y)"),
                 ("b_q", "Base ellipse: half-height (Z)"))
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
        self._info_text = None
        self.placeholder()

    def _clear_overlays(self):
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        if self._info_text is not None:
            self._info_text.remove()
            self._info_text = None

    def _draw_info_panel(self, info: dict):
        """Monospaced information box, top left, as in the GVWD and PSWR-1 tabs."""
        info = dict(info)
        title = info.pop("__title__", "LTOCs")
        width = max([12] + [len(str(k)) for k in info])
        lines = [title] + [f"  {str(k):<{width}}  {v}" for k, v in info.items()]
        self._info_text = self.fig.text(
            0.02, 0.98, "\n".join(lines), transform=self.fig.transFigure, fontsize=8,
            fontfamily="monospace", verticalalignment="top", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1A1A1A", edgecolor="#D97706",
                      alpha=0.85))

    def placeholder(self, text="Click 'Generate LTOCs Waverider'"):
        self._clear_overlays()
        self.ax.clear()
        self.ax.set_title(text, color=_ink())
        self.ax.tick_params(colors=_MUTED)
        self.draw()

    def plot(self, res, full_span: bool = True, title: str = "LTOCs"):
        wr, g = res.waverider, res.grid
        P, p = g.points, g.p
        self._clear_overlays()
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
        self._draw_info_panel(res.info(title))
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
        self.shock_data = None                       # ltoc.design.ShockData (data-file source)
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
        lay = QVBoxLayout()
        self.source_combo = QComboBox()
        for name in list(PRESETS) + [CUSTOM, BSPLINE, OC_IMPORT]:
            self.source_combo.addItem(name)
        self.source_combo.setToolTip(
            "The shock wave the waverider is designed to produce.\n"
            "  Presets: the shocks published by Zheng et al. (2020) and a circular cone.\n"
            "  Custom elliptic shock: straight lines joining an upstream ellipse to a\n"
            "    base ellipse.\n"
            "  Data file: shock points from a CSV file ('Save shock as CSV...' writes a\n"
            "    template).\n"
            "  Import: the shock of the design in the OC Waverider tab; LTOCs then\n"
            "    reproduces that design.\n"
            "Coordinates: X downstream, Y across the span, Z downward.")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        lay.addWidget(self.source_combo)

        # Elliptic ruled shock (presets and custom)
        self.elliptic_box = QWidget()
        el = QGridLayout(self.elliptic_box)
        el.setContentsMargins(0, 0, 0, 0)
        self.shock_spins = {}
        for row, (key, label) in enumerate(_SHOCK_FIELDS):
            sb = QDoubleSpinBox()
            sb.setDecimals(4)
            sb.setRange(0.0, 1000.0)
            sb.setSingleStep(0.01)
            sb.setToolTip(f"{label}. The shock is made of straight lines joining matching "
                          "points of the upstream ellipse and the base ellipse.\n" + _COORDS)
            sb.valueChanged.connect(self._on_shock_edited)
            el.addWidget(QLabel(label + ":"), row, 0)
            el.addWidget(sb, row, 1)
            self.shock_spins[key] = sb
        lay.addWidget(self.elliptic_box)

        # Shock data file (B-spline)
        self.data_box = QWidget()
        dl = QGridLayout(self.data_box)
        dl.setContentsMargins(0, 0, 0, 0)
        self.data_label = QLabel("No data file loaded.")
        self.data_label.setWordWrap(True)
        self.btn_load_data = QPushButton("Load CSV...")
        self.btn_load_data.setToolTip(
            "Read shock points from a CSV file: a header 'u,v,X,Y,Z' and one row per point\n"
            "of a full (u, v) grid, with u across the span and v running downstream.\n"
            "Coordinates: X downstream, Y across the span, Z downward. Optional lines\n"
            "'# x_base = ...', '# M_inf = ...', '# fct_z = ...' fill the inputs below.")
        self.btn_load_data.clicked.connect(self._browse_shock_file)
        self.xbase_spin = QDoubleSpinBox()
        self.xbase_spin.setDecimals(4)
        self.xbase_spin.setRange(-1.0e4, 1.0e4)
        self.xbase_spin.setSingleStep(0.05)
        self.xbase_spin.setValue(1.0)
        self.xbase_spin.setToolTip("Base plane X of the waverider. The shock data must extend\n"
                                   "past it far enough to determine the body there.")
        dl.addWidget(self.data_label, 0, 0, 1, 2)
        dl.addWidget(self.btn_load_data, 1, 0, 1, 2)
        dl.addWidget(QLabel("Base plane X:"), 2, 0)
        dl.addWidget(self.xbase_spin, 2, 1)
        lay.addWidget(self.data_box)

        self.btn_save_data = QPushButton("Save shock as CSV...")
        self.btn_save_data.setToolTip(
            "Write the current shock as a data file (u, v, X, Y, Z grid). For an elliptic\n"
            "shock the grid extends 1.5 chords past the base. Useful as a template for\n"
            "your own shock data, which then loads with 'Shock from a data file'.")
        self.btn_save_data.clicked.connect(self._browse_save_shock_file)
        lay.addWidget(self.btn_save_data)

        self.reference_label = QLabel("")
        self.reference_label.setWordWrap(True)
        self.reference_label.setStyleSheet("color: #888888;")
        lay.addWidget(self.reference_label)
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
            "Flow capture tube (FCT): the straight line Z = const in the base plane that\n"
            "outlines the top of the waverider. The leading edge is this line projected\n"
            "upstream onto the shock. Z is measured downward; the body hangs below the line.")
        self.fct_spin.valueChanged.connect(self._on_shock_edited)
        self.stations_spin = QSpinBox()
        self.stations_spin.setRange(3, 201)
        self.stations_spin.setValue(25)
        self.stations_spin.setToolTip(
            "Stream surfaces across the half span (FCT points).\n"
            "25 reproduces the published lift and drag of Zheng et al. (2020) to 0.4 %.")
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
        rows = (("Mach number:", self.mach_spin), ("FCT height Z (downward):", self.fct_spin),
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
        self.btn_send_aero = aero_handoff.send_button(self)
        for b in (self.btn_export_stl, self.btn_export_step, self.btn_send_aero):
            b.setEnabled(False)
        lay.addWidget(self.btn_export_stl, 2, 0)
        lay.addWidget(self.btn_export_step, 2, 1)
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(32)
        help_btn.setToolTip("Method, inputs, refusals and references.")
        help_btn.clicked.connect(self.show_help_dialog)
        lay.addWidget(help_btn, 3, 0)
        lay.addWidget(self.btn_send_aero, 3, 1)
        g.setLayout(lay)
        return g

    # ------------------------------------------------------------------
    #  Inputs
    # ------------------------------------------------------------------
    def _on_source_changed(self, _idx: int):
        name = self.source_combo.currentText()
        is_oc = name == OC_IMPORT
        is_data = name == BSPLINE
        self.elliptic_box.setVisible(not is_data)
        self.data_box.setVisible(is_data)
        self.btn_save_data.setEnabled(not is_oc)
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
            self.reference_label.setText(ref)
        elif is_oc:
            self.reference_label.setText(
                "Uses the shock, Mach number and leading edge of the design last generated "
                "in the OC Waverider tab (one stream surface per OC plane).")
        elif is_data:
            self.reference_label.setText(_DATA_HELP)
        else:
            self.reference_label.setText(_CUSTOM_HELP)

    def _on_shock_edited(self, _value):
        """Editing a preset's numbers turns it into a custom shock."""
        if self._filling or self.source_combo.currentText() not in PRESETS:
            return
        self._filling = True
        self.source_combo.setCurrentText(CUSTOM)
        self._filling = False
        self.reference_label.setText(_CUSTOM_HELP)

    def current_design(self) -> LTOCDesign:
        common = dict(n_points=self.points_spin.value())
        if self.source_combo.currentText() == OC_IMPORT:
            oc = getattr(self._hub, "waverider", None)
            if oc is None:
                raise ValueError("no OC design: generate one in the OC Waverider tab first")
            return LTOCDesign(oc_waverider=oc, **common)
        if self.source_combo.currentText() == BSPLINE and self.shock_data is None:
            raise ValueError("no shock data: load a CSV file first")
        return self._widget_design()

    def _widget_design(self) -> LTOCDesign:
        """The design the widgets describe (elliptic or data shock; never the OC import)."""
        data = self.source_combo.currentText() == BSPLINE
        clustering = "symmetry" if self.cluster_combo.currentIndex() == 0 else "uniform"
        return LTOCDesign(shock=EllipticShock(**{k: sb.value() for k, sb in self.shock_spins.items()}),
                          M_inf=self.mach_spin.value(), fct_z=self.fct_spin.value(),
                          n_stations=self.stations_spin.value(), clustering=clustering,
                          n_points=self.points_spin.value(),
                          shock_data=self.shock_data if data else None,
                          x_base=self.xbase_spin.value() if data else None)

    # ------------------------------------------------------------------
    #  Shock data files
    # ------------------------------------------------------------------
    def load_shock_file(self, path) -> ShockData:
        """Read a shock data file, select the data source and fill the inputs
        the file carries (x_base, M_inf, fct_z). Raises ``ValueError`` with the
        reason (and line) if the file is not valid."""
        data = ShockData.read_csv(path)
        self._set_shock_data(data)
        self._filling = True
        try:
            if "x_base" in data.meta:
                self.xbase_spin.setValue(data.meta["x_base"])
            if "M_inf" in data.meta:
                self.mach_spin.setValue(data.meta["M_inf"])
            if "fct_z" in data.meta:
                self.fct_spin.setValue(data.meta["fct_z"])
        finally:
            self._filling = False
        self.source_combo.setCurrentText(BSPLINE)
        return data

    def _set_shock_data(self, data):
        self.shock_data = data
        if data is None:
            self.data_label.setText("No data file loaded.")
            return
        (nu, nv), (x0, x1) = data.shape, data.x_range
        self.data_label.setText(f"{data.name or 'shock data'}: {nu} x {nv} grid, "
                                f"X = {x0:.4g} to {x1:.4g}")

    def save_shock_file(self, path) -> ShockData:
        """Write the current shock (elliptic or loaded data) as a data file."""
        meta = dict(M_inf=self.mach_spin.value(), fct_z=self.fct_spin.value())
        if self.source_combo.currentText() == BSPLINE:
            if self.shock_data is None:
                raise ValueError("no shock data loaded")
            d = self.shock_data
            data = ShockData(d.u, d.v, d.points, name=d.name,
                             meta=dict(meta, x_base=self.xbase_spin.value()))
        else:
            shock = EllipticShock(**{k: sb.value() for k, sb in self.shock_spins.items()})
            data = shock.to_data(**meta)
        data.write_csv(path)
        return data

    def _browse_shock_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load shock data", "",
                                              "CSV files (*.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            self.load_shock_file(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Shock data not loaded", str(e))

    def _browse_save_shock_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save shock as CSV", "ltoc_shock.csv",
                                              "CSV files (*.csv)")
        if not path:
            return
        try:
            self.save_shock_file(path)
            QMessageBox.information(self, "Saved", f"Shock data written:\n{path}")
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Not saved", str(e))

    # ------------------------------------------------------------------
    #  Project files (WADOS "Save/Load Parameters", section "ltoc_waverider")
    # ------------------------------------------------------------------
    def get_params_dict(self) -> dict:
        """The tab's inputs as a JSON-serialisable dict (``LTOCDesign.to_dict``).

        Reads the widgets only, so it works whatever their state. With the OC
        import, the OC design itself is saved in the project's "oc_waverider"
        section; regenerate it after loading, then generate here.
        """
        name = self.source_combo.currentText()
        source = {CUSTOM: "custom", BSPLINE: "bspline", OC_IMPORT: "oc_import"}.get(name, name)
        d = self._widget_design().to_dict(source=source)
        if name == BSPLINE:                              # keep the base even without data
            d["x_base"] = float(self.xbase_spin.value())
        return d

    def set_params_dict(self, d: dict) -> None:
        """Restore inputs saved by :meth:`get_params_dict`.

        A preset that no longer exists, or whose numbers differ from the saved
        ones, is restored as a custom shock with the saved numbers. Values
        outside a widget's range are clipped by the widget. The current result
        is kept until the next 'Generate'.
        """
        design = LTOCDesign.from_dict(d)                 # ValueError for a malformed section
        source = str(d.get("source", "custom"))
        name = {"custom": CUSTOM, "bspline": BSPLINE, "oc_import": OC_IMPORT}.get(source, source)
        name = PRESET_ALIASES.get(name, name)            # names used by earlier versions
        if self.source_combo.findText(name) < 0:
            name = CUSTOM
        self.source_combo.setCurrentText(name)           # enables/fills the widgets for the source
        self._filling = True
        try:
            if design.shock is not None:
                for key, sb in self.shock_spins.items():
                    sb.setValue(getattr(design.shock, key))
            self.mach_spin.setValue(design.M_inf)
            self.fct_spin.setValue(design.fct_z)
            self.stations_spin.setValue(design.n_stations)
            self.cluster_combo.setCurrentIndex(0 if design.clustering == "symmetry" else 1)
            self.points_spin.setValue(design.n_points)
            if design.x_base is not None:
                self.xbase_spin.setValue(design.x_base)
        finally:
            self._filling = False
        if name == BSPLINE:
            self._set_shock_data(design.shock_data)
        if name in PRESETS and not self._matches_preset(name):
            self.source_combo.setCurrentText(CUSTOM)

    def _matches_preset(self, name: str) -> bool:
        """True when the shock, Mach and FCT widgets hold the preset's values."""
        shock, M, z, _ = PRESETS[name]
        pairs = [(sb, getattr(shock, k)) for k, sb in self.shock_spins.items()]
        pairs += [(self.mach_spin, M), (self.fct_spin, z)]
        return all(abs(sb.value() - round(v, sb.decimals())) < 0.5 * 10 ** -sb.decimals()
                   for sb, v in pairs)

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
        for b in (self.btn_export_stl, self.btn_export_step, self.btn_send_aero):
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
        for b in (self.btn_export_stl, self.btn_export_step, self.btn_send_aero):
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
        self.canvas_3d.plot(res, title=self._panel_title(res))
        self._plot_front(res)
        self._plot_pressure(res)
        self._plot_diagnostics(res)

    def _panel_title(self, res) -> str:
        d = res.design
        if d.oc_waverider is not None:
            return "LTOCs (OC design import)"
        if d.shock_data is not None:
            return f"LTOCs (data: {d.shock_data.name or 'shock data'})"
        name = self.source_combo.currentText()
        return f"LTOCs ({name.split(' (')[0] if name in PRESETS else 'custom elliptic shock'})"

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

    def analysis_summary(self) -> str:
        wr = self.waverider
        return f"M={wr.M_inf:g}, {len(wr.stations)} stream surfaces, L={wr.length:.3g}"

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
        "WHAT IT DOES (Zheng et al., AIAA J 58(8), 2020)\n"
        "  You choose the shock wave the waverider should produce and the line that\n"
        "  outlines its top in the base plane (the flow capture tube, FCT). The\n"
        "  leading edge is the FCT projected upstream onto the shock. For every\n"
        "  leading-edge point the method:\n"
        "   1. follows the shock downstream along its local surface direction (the\n"
        "      path may turn sideways, unlike the osculating-cones method);\n"
        "   2. computes the flow between the shock and the body with the method of\n"
        "      characteristics, using the local curvature of the shock;\n"
        "   3. maps that flow back to 3-D and traces the body streamline.\n"
        "  The body streamlines form the lower surface; the upper surface is flat\n"
        "  along the free stream above them.\n\n"
        "COORDINATES\n"
        "  X downstream, Y across the span (symmetry plane Y = 0), Z downward.\n"
        "  Exports and the 3-D view use the WADOS frame: x downstream, y up, z span.\n\n"
        "INPUTS\n"
        "  Shock: a preset, a custom elliptic shock (straight lines joining an\n"
        "    upstream ellipse to a base ellipse; the base ellipse sits in the base\n"
        "    plane), a data file (CSV; 'Save shock as CSV...' writes a template), or\n"
        "    the shock of the current OC design.\n"
        "  FCT height Z: the top of the waverider in the base plane.\n"
        "  Stations: stream surfaces across the half span. Points: shock points per\n"
        "    stream surface from the leading edge to the base.\n\n"
        "WHEN A STATION IS REFUSED (no geometry is built)\n"
        "  concave       the shock curves away from the body across the span\n"
        "  sub_mach      the shock is too weak: at or below the Mach angle\n"
        "  detached      the shock is too strong: it would detach\n"
        "  undetermined  the shock does not reach far enough past the base plane\n"
        "  outside_shock the FCT lies outside the shock in the base plane\n"
        "  limit_surface the flow solution folds over (limit surface)\n\n"
        "OUTPUTS\n"
        "  CL, CD, L/D: inviscid lower-surface forces, referenced to the planform\n"
        "    area (this reproduces the published values); wetted-area values too.\n"
        "  Views: 3-D (lower surface coloured by pressure), front view, wall\n"
        "    pressure along and across the body, per-station diagnostics.\n"
        "  Export: STL (metres) and STEP (millimetres). Send to Aero Analysis\n"
        "    loads the waverider into the Aero Analysis tab.\n\n"
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
