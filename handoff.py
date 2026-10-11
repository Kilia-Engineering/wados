"""Hand a design tab's geometry to the Aero Analysis and Optimization tabs.

**Aero Analysis.**

Every design tab provides ``stl_for_analysis()``: its closed full-span mesh
written to a temporary binary STL in metres, frame x streamwise, y up,
z span (the frame PySAGAS and the Aero Analysis tab use). A tab has a design
when ``has_analysis_geometry()`` says so or, without that method, when its
``waverider`` is set.

Two ways in:

* the tab's *Send to Aero Analysis* button (``send_button``), which loads
  the mesh into the Aero Analysis tab and switches to it;
* the mesh dialog of the Aero Analysis tab, which lists every tab that has
  a design.

**Optimization.** A tab provides ``optimization_case()``: its current inputs
as a ``method_optimization.OptimizationCase`` (the parameters that may vary,
with bounds, and the rest fixed), and ``apply_optimized_params(params)`` to
load an optimized design back. Its *Send to Optimization* button
(``optimize_button``) opens the case in the Optimization tab.
"""
from PyQt5.QtWidgets import QMessageBox, QPushButton

SEND_TEXT = "Send to Aero Analysis"
SEND_TIP = ("Load this design into the Aero Analysis tab (mesh preview, PySAGAS,\n"
            "AeroDeck sweep): closed full-span mesh, metres, x streamwise, y up, z span.\n"
            "A_ref is set to its planform area.")

# Source keys the Aero Analysis tab already used for these tabs
_SOURCE_KEYS = {"Cone-derived Waverider": "shadow", "LTOCs Waverider": "ltoc"}


def has_geometry(tab) -> bool:
    check = getattr(tab, "has_analysis_geometry", None)
    if check is not None:
        return bool(check())
    return getattr(tab, "waverider", None) is not None


def summary(tab) -> str:
    """One line describing the tab's design, for the mesh dialog."""
    describe = getattr(tab, "analysis_summary", None)
    if describe is not None:
        return describe()
    length = getattr(getattr(tab, "waverider", None), "length", None)
    return f"L = {length:.3g} m" if isinstance(length, (int, float)) else "Design generated"


def source_key(name: str) -> str:
    return _SOURCE_KEYS.get(name, name)


def send(tab):
    """Load ``tab``'s design into the Aero Analysis tab of the main window."""
    hub = tab.window()
    if not hasattr(hub, "send_to_aero_analysis"):
        QMessageBox.information(
            tab, SEND_TEXT,
            "The Aero Analysis tab is part of the WADOS main window "
            "(python waverider_gui.py).")
        return
    hub.send_to_aero_analysis(tab)


def send_button(tab, text: str = SEND_TEXT) -> QPushButton:
    btn = QPushButton(text)
    btn.setToolTip(SEND_TIP)
    btn.clicked.connect(lambda: send(tab))
    return btn


OPT_TEXT = "Send to Optimization"
OPT_TIP = ("Optimize this design in the Optimization tab (genetic algorithm with\n"
           "PySAGAS): the current inputs are the starting point; choose the\n"
           "variables, their ranges, objectives and constraints there.")


def send_to_optimization(tab):
    """Open ``tab``'s design as a case in the Optimization tab of the main window."""
    hub = tab.window()
    if not hasattr(hub, "send_to_optimization"):
        QMessageBox.information(
            tab, OPT_TEXT,
            "The Optimization tab is part of the WADOS main window "
            "(python waverider_gui.py).")
        return
    hub.send_to_optimization(tab)


def optimize_button(tab, text: str = OPT_TEXT) -> QPushButton:
    btn = QPushButton(text)
    btn.setToolTip(OPT_TIP)
    btn.clicked.connect(lambda: send_to_optimization(tab))
    return btn
