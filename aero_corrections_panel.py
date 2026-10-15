"""Controls for the base-pressure, skin-friction and blunt-LE corrections (aero_corrections).

The Aero Analysis and Optimization tabs show the same group; ``corrections()``
gives the ``AeroCorrections`` to pass on. The GVWD tab, which has its own
friction settings, uses ``base_pressure_combo`` alone.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QLabel

from aero_corrections import BASE_MODELS, FRICTION_MODELS, AeroCorrections

_BASE_ITEMS = [
    ("igra", "Igra 2018 correlation (recommended)",
     "Igra 2018 Eq. (6), the correlation of Guidi et al. 2003. p_b is about\n"
     "0.25 p∞ at Mach 5 and 0.11 p∞ at Mach 14."),
    ("vacuum", "Vacuum, p_b = 0 (PySAGAS as is)",
     "PySAGAS leaves faces turned more than 20° from the flow (the base) at\n"
     "zero pressure. Too pessimistic: overpredicts base drag."),
    ("maxwell", "p∞ / M∞", "The Planar tab's model (Maxwell slender-body base pressure)."),
    ("freestream", "Freestream, p_b = p∞",
     "No base drag. Too optimistic for a glider (Igra 2018, Table 2)."),
]
_FRICTION_ITEMS = [
    ("turbulent", "Turbulent (recommended)",
     "Turbulent flat-plate skin friction from the nose. Waveriders fly at\n"
     "Re_L of 1e7 to 1e8, where the boundary layer is mostly turbulent."),
    ("laminar", "Laminar", "Laminar flat-plate skin friction (wind-tunnel scale)."),
    ("transition", "Transition at Re_x", "Laminar up to the transition Reynolds number, then turbulent."),
    ("off", "Off (inviscid)", "No skin friction: PySAGAS's inviscid drag."),
]
assert {k for k, *_ in _BASE_ITEMS} == set(BASE_MODELS)
assert {k for k, *_ in _FRICTION_ITEMS} == set(FRICTION_MODELS)


def base_pressure_combo(model: str = "igra") -> QComboBox:
    """The base-pressure choice; ``currentData()`` is the model key."""
    combo = QComboBox()
    for key, text, tip in _BASE_ITEMS:
        combo.addItem(text, key)
        combo.setItemData(combo.count() - 1, tip, Qt.ToolTipRole)
    combo.setToolTip(
        "Pressure on the base (the faces turned more than 20° from the flow,\n"
        "which PySAGAS leaves at vacuum). Going from p_b = p∞ to the\n"
        "correlation lowered L/D by 6 % at Mach 14 and by 19 % at Mach 8\n"
        "(Igra 2018, Table 2; 2019, Table 1).")
    combo.setCurrentIndex(combo.findData(model))
    return combo


class AeroCorrectionsGroup(QGroupBox):
    """Base pressure, skin friction (transition Re, wall temperature) and a
    rounded leading edge (radius, spanwise rule, pressure bleed)."""

    def __init__(self, title="Base Pressure, Skin Friction and Blunt LE", parent=None):
        super().__init__(title, parent)
        layout = QGridLayout(self)

        layout.addWidget(QLabel("Base pressure:"), 0, 0)
        self.base_combo = base_pressure_combo()
        layout.addWidget(self.base_combo, 0, 1)

        layout.addWidget(QLabel("Skin friction:"), 1, 0)
        self.friction_combo = QComboBox()
        for key, text, tip in _FRICTION_ITEMS:
            self.friction_combo.addItem(text, key)
            self.friction_combo.setItemData(self.friction_combo.count() - 1, tip, Qt.ToolTipRole)
        self.friction_combo.setToolTip(
            "Eckert reference-temperature skin friction on every wetted face,\n"
            "at the face's PySAGAS edge state. The mesh must be in metres.")
        layout.addWidget(self.friction_combo, 1, 1)

        layout.addWidget(QLabel("Transition Re_x:"), 2, 0)
        self.transition_spin = QDoubleSpinBox()
        self.transition_spin.setRange(1.0e4, 1.0e9)
        self.transition_spin.setDecimals(0)
        self.transition_spin.setSingleStep(1.0e5)
        self.transition_spin.setToolTip("Reynolds number (reference state, from the nose) at transition")
        layout.addWidget(self.transition_spin, 2, 1)

        layout.addWidget(QLabel("Wall temperature T_w (K):"), 3, 0)
        self.wall_spin = QDoubleSpinBox()
        self.wall_spin.setRange(100.0, 3000.0)
        self.wall_spin.setDecimals(0)
        self.wall_spin.setSingleStep(50.0)
        self.wall_spin.setToolTip("Wall temperature for the Eckert reference temperature")
        layout.addWidget(self.wall_spin, 3, 1)

        layout.addWidget(QLabel("LE radius R (m):"), 4, 0)
        self.le_radius_spin = QDoubleSpinBox()
        self.le_radius_spin.setRange(0.0, 0.5)
        self.le_radius_spin.setDecimals(4)
        self.le_radius_spin.setSingleStep(0.001)
        self.le_radius_spin.setSpecialValueText("Sharp")
        self.le_radius_spin.setToolTip(
            "Radius of a rounded leading edge; the mesh stays sharp. The leading\n"
            "edge is found on the mesh. Its arc gets modified Newtonian pressure,\n"
            "the compression side gets the pressure bleed, and the results show\n"
            "the peak stagnation-line heat flux (Tauber-Sutton, swept).")
        layout.addWidget(self.le_radius_spin, 4, 1)

        layout.addWidget(QLabel("LE radius spanwise:"), 5, 0)
        self.le_sweep_combo = QComboBox()
        self.le_sweep_combo.addItems(["Uniform radius", "Sweep-scaled (cos\u00b2\u00b7\u00b2)"])
        self.le_sweep_combo.setToolTip(
            "Uniform: the same radius along the leading edge.\n"
            "Sweep-scaled: R cos^2.2 of the planform sweep, as the OC tab's blunting.")
        layout.addWidget(self.le_sweep_combo, 5, 1)

        self.le_bleed_check = QCheckBox("Pressure bleed behind the LE (Jessen 2026)")
        self.le_bleed_check.setToolTip(
            "Overpressure behind the rounded LE on the compression side\n"
            "(Jessen et al. 2026, Eq. 12, as the Planar tab). Fitted to planar\n"
            "waveriders; an estimate for other shapes.")
        layout.addWidget(self.le_bleed_check, 6, 0, 1, 2)

        self.friction_combo.currentIndexChanged.connect(self._update_enabled)
        self.le_radius_spin.valueChanged.connect(self._update_enabled)
        self.set_corrections(AeroCorrections())

    def _update_enabled(self):
        friction = self.friction_combo.currentData()
        self.transition_spin.setEnabled(friction == "transition")
        self.wall_spin.setEnabled(friction != "off")
        blunt = self.le_radius_spin.value() > 0.0
        self.le_sweep_combo.setEnabled(blunt)
        self.le_bleed_check.setEnabled(blunt)

    def corrections(self) -> AeroCorrections:
        return AeroCorrections(base_pressure=self.base_combo.currentData(),
                               friction=self.friction_combo.currentData(),
                               wall_temperature=self.wall_spin.value(),
                               transition_re=self.transition_spin.value(),
                               le_radius=self.le_radius_spin.value(),
                               le_radius_sweep_scaled=self.le_sweep_combo.currentIndex() == 1,
                               le_bleed=self.le_bleed_check.isChecked())

    def set_corrections(self, corrections: AeroCorrections):
        self.base_combo.setCurrentIndex(self.base_combo.findData(corrections.base_pressure))
        self.friction_combo.setCurrentIndex(self.friction_combo.findData(corrections.friction))
        self.wall_spin.setValue(corrections.wall_temperature)
        self.transition_spin.setValue(corrections.transition_re)
        self.set_le_radius(corrections.le_radius, corrections.le_radius_sweep_scaled)
        self.le_bleed_check.setChecked(corrections.le_bleed)
        self._update_enabled()

    def set_le_radius(self, radius: float, sweep_scaled: bool = False):
        self.le_radius_spin.setValue(radius)
        self.le_sweep_combo.setCurrentIndex(1 if sweep_scaled else 0)
