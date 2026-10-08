"""Figure helpers for LTOCs gate deliverables.

Spec rule 7: axis labels, tick labels and legends at 14 pt or larger, and
centred titles. Figures are built on ``matplotlib.figure.Figure`` with the Agg
canvas, so they work headless and leave the global pyplot state untouched.

Colours are the first three slots of the validated categorical palette
(blue, orange, aqua), in that fixed order. Mesh lines and guides use
neutral greys. Each series also has its own marker, so colour is never the
only cue.
"""
from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = ["SERIES", "MARKERS", "INK", "INK_2", "GRID", "MESH", "SURFACE",
           "style", "new_figure", "save"]

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")      # palette slots 1-3
MARKERS = ("o", "s", "^")
INK = "#0b0b0b"                                  # primary text
INK_2 = "#52514e"                                # secondary text, guides
GRID = "#d9d8d4"
MESH = "#9d9c97"
SURFACE = "#fcfcfb"

_RC = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.titlelocation": "center",
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
}


@contextmanager
def style():
    """Apply the LTOCs figure style inside the block only."""
    with mpl.rc_context(_RC):
        yield


def new_figure(nrows: int = 1, ncols: int = 1, width: float = 7.0, height: float = 5.5, **kw):
    """Create a ``Figure`` with an Agg canvas and its axes array.

    Call this inside :func:`style` so every artist picks up the style.
    """
    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False, **kw)
    return fig, axes


def save(fig, path) -> None:
    fig.savefig(path)
