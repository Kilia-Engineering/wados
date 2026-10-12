"""Skin friction counts every wetted panel (GVWD §4.8 viscous).

At alpha = 0 the caret's upper surface is parallel to the flow. It is wetted
and carries friction like the lower surface; only the base does not. Counting
windward panels only (theta_local > 0) left the upper surface out and halved
the friction drag.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gvwd.aero.viscous import panel_viscous_drag
from gvwd.geometry import Caret


def _caret():
    return Caret(M_design=6.0, theta_d=math.radians(14.0), Lambda=math.radians(70.0), L=10.0).mesh


def test_every_panel_but_the_base_is_wetted():
    mesh = _caret()
    res = panel_viscous_drag(mesh, 6.0, 0.0, altitude_km=25.0)
    wetted = res.state_per_panel["wetted"]
    is_base = np.array([lab.startswith("base") for lab in mesh.labels])
    assert np.array_equal(wetted, ~is_base)


def test_upper_surface_carries_friction():
    """Every caret triangle runs from the apex to the base, so all centroids sit
    at x = 2L/3 with the same (freestream) edge state: friction is proportional
    to wetted area, and removing the upper surface removes its share."""
    from gvwd.geometry.mesh import Mesh

    mesh = _caret()
    upper = np.array([lab.startswith("upper") for lab in mesh.labels])
    no_upper = Mesh(mesh.vertices, mesh.faces[~upper], mesh.labels[~upper])
    S = 36.0
    full = panel_viscous_drag(mesh, 6.0, 0.0, altitude_km=25.0, Re_x_tr=1e30, S_ref=S)
    part = panel_viscous_drag(no_upper, 6.0, 0.0, altitude_km=25.0, Re_x_tr=1e30, S_ref=S)
    A = mesh.face_areas()
    lower = np.array([lab.startswith("lower") for lab in mesh.labels])
    ratio = (A[upper].sum() + A[lower].sum()) / A[lower].sum()
    assert ratio > 1.9
    assert full.CD_friction == pytest.approx(part.CD_friction * ratio, rel=1e-9)
