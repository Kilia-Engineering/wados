"""Base pressure in the panel method (GVWD §4.8, Igra 2018 Eq. 6).

Faces turned more than BASE_TURN_DEG from the flow are the base. The default
("freestream") keeps the shadow Cp = 0, so earlier results are unchanged;
another model gives them Cp_b = 2 (p_b/p_inf - 1) / (gamma M^2), which adds
-Cp_b A_base / S_ref to CD on a base facing straight downstream.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aero_corrections import base_pressure_ratio
from gvwd.aero.coefficients import aero_coefficients_full
from gvwd.aero.panel_method import panel_aero_coefficients
from gvwd.aero.sweep import SweepConfig, mach_alpha_sweep
from gvwd.geometry import Caret
from gvwd.io.config import SweepRunConfig, config_to_dict, load_config_yaml


def _caret():
    return Caret(M_design=6.0, theta_d=math.radians(14.0), Lambda=math.radians(70.0), L=10.0).mesh


def _is_base(mesh):
    return np.array([lab.startswith("base") for lab in mesh.labels])


def test_freestream_default_is_the_shadow_cp():
    mesh = _caret()
    default = panel_aero_coefficients(mesh, 6.0, 0.0)
    assert default.base_pressure == "freestream" and default.CD_base == 0.0
    assert np.all(default.Cp[_is_base(mesh)] == 0.0)


@pytest.mark.parametrize("model", ["igra", "maxwell", "vacuum"])
def test_base_faces_get_the_base_pressure(model):
    mesh = _caret()
    M, g = 6.0, 1.4
    free = panel_aero_coefficients(mesh, M, 0.0)
    res = panel_aero_coefficients(mesh, M, 0.0, base_pressure=model)
    base = _is_base(mesh)
    cp_b = 2.0 * (base_pressure_ratio(model, M, g) - 1.0) / (g * M * M)
    assert np.allclose(res.Cp[base], cp_b) and np.array_equal(res.Cp[~base], free.Cp[~base])
    A_base = float(mesh.face_areas()[base].sum())             # the caret's base is normal to x
    assert res.CD_base == pytest.approx(-cp_b * A_base / res.S_ref, rel=1e-12)
    assert res.CD == pytest.approx(free.CD + res.CD_base, rel=1e-12)
    assert res.CL == pytest.approx(free.CL, abs=1e-12)


def test_base_is_found_at_angle_of_attack():
    mesh = _caret()
    res = panel_aero_coefficients(mesh, 6.0, math.radians(10.0), base_pressure="igra")
    assert res.CD_base > 0.0
    assert np.all(res.Cp[_is_base(mesh)] < 0.0)


def test_on_design_and_sweep_carry_the_model():
    mesh = _caret()
    free = aero_coefficients_full(mesh, 6.0, altitude_km=25.0)
    igra = aero_coefficients_full(mesh, 6.0, altitude_km=25.0, base_pressure="igra")
    assert free["CD_base"] == 0.0 and igra["CD_base"] > 0.0
    assert igra["CD_total"] == pytest.approx(free["CD_total"] + igra["CD_base"], rel=1e-12)
    assert igra["LD"] < free["LD"] and igra["base_pressure"] == "igra"

    cfg = SweepConfig(M_grid=(5.0, 8.0, 2), alpha_grid_deg=(0.0, 4.0, 2), base_pressure="igra")
    df = mach_alpha_sweep(mesh, cfg)
    assert (df["CD_base"] > 0.0).all()
    df0 = mach_alpha_sweep(mesh, SweepConfig(M_grid=(5.0, 8.0, 2), alpha_grid_deg=(0.0, 4.0, 2)))
    assert (df["LD"] < df0["LD"]).all() and (df0["CD_base"] == 0.0).all()


def test_run_config_round_trip(tmp_path):
    import json

    assert SweepRunConfig().base_pressure == "freestream"
    path = tmp_path / "caret.json"
    path.write_text(json.dumps({
        "geometry": {"mode": "caret", "M_design": 6.0, "theta_d_deg": 14.0, "Lambda_deg": 70.0, "L": 10.0},
        "sweep": {"base_pressure": "igra"}}))
    cfg = load_config_yaml(path)
    assert cfg.sweep.base_pressure == "igra"
    assert config_to_dict(cfg)["sweep"]["base_pressure"] == "igra"
