"""Tests for ``method_optimization``: optimizing any design tab's waverider.

Every method's builder must turn a tab's parameter dict into a closed,
outward-oriented, full-span mesh in the analysis frame (x streamwise, y up,
z span). The evaluation must run PySAGAS on it and report the objectives;
a design whose geometry fails must come back as a failure, not raise. A
short NSGA-II run (needs pymoo) must improve on the starting design.
"""
import numpy as np
import pytest

import method_optimization as M
from method_optimization import DesignVariable, OptimizationCase

FLOW = dict(pressure=2549.0, temperature=221.55, aoa=0.0)

# Each tab's default inputs, as its get_params_dict (or optimization_case) gives them
PARAMS = {
    "pswr": dict(M_inf=6.0, altitude_km=30.0, beta0_deg=12.0, beta1_deg=14.0, beta2_deg=16.0,
                 lambda_deg=70.0, length_m=10.0, flat_fraction=0.0, n_span=41, n_chord=20),
    "gvwd": dict(mode="caret", M_design=6.0, theta_d_deg=14.0, Lambda_deg=70.0, L=10.0, gamma=1.4),
    "planar": dict(mach=6.85, length=1.0, width=2.0, n=0.5, beta=9.0, epsilon=0.0, p1=1.0, p2=1.0,
                   p3=1.0, le_radius=0.0, nx=20, ny=15, blunting_method="none"),
    "shadow": dict(mach=6.0, shock_angle=12.0, order="2nd Order", a3=0.0, a2=-2.0, a0=-0.15,
                   n_le=21, n_stream=20, length=1.0, top_surface_a=0.0),
    "hybrid": dict(mach=7.0, beta_oc=12.0, beta_cd=12.0, height=1.0, width=2.0, x1=0.2, x2=0.2,
                   x3=0.5, x4=0.5, alpha_root=1.0, alpha_tip=0.0, blend_exp=1.0, x_t=0.5,
                   dx_t=0.3, n_planes=20, n_streamwise=20),
    "vmn": dict(ma_min=6.0, ma_max=10.0, beta=13.5, length=0.6, s=0.5, a0=2.0, direction=0,
                n_points=20, n_streamwise=20, blunt_radius_mm=0.0),
    "vmof": dict(ma_min=6.0, ma_max=10.0, beta=13.5, height=1.0, width=2.0, x1=0.1, x2=0.2,
                 x3=0.5, x4=0.5, direction=0, n_points=20, n_streamwise=20, blunt_radius_mm=0.0),
    "vmplo": dict(ma_center=6.0, ma_tip=10.0, ma_dist=1, ma_ncp=4, n_center=1.0, n_tip=1.0,
                  n_dist=0, beta=13.0, length=3.0, height=1.0, width=2.0, x_LE=0.05,
                  icc_center=0.3, icc_tip=0.3, icc_dist=0, icc_ncp=4, n_planes=15,
                  n_streamwise=15),
    "mfof": dict(params=dict(beta_deg=13.0, L_w=6.0, W=3.0, L_s=0.3, y5=0.1608, z5=1.5, y6=1.608,
                             z6=0.0, delta5=0.0, delta6=0.0, Ma_center=6.0, Ma_tip=13.0, gamma=1.4),
                 n_z=40, n_x=40, flowfield="cone", power_law_n=1.0),
}


def _ltoc_params():
    from ltoc.design import LTOCDesign

    d = LTOCDesign.from_preset("Shape-changing elliptic shock, Mach 7 (published case II)")
    d.n_stations, d.n_points = 7, 20
    return d.to_dict()


def _check_mesh(mesh):
    from geometry_export import mesh_planform_area, mesh_volume

    f = np.asarray(mesh.faces)
    v = np.asarray(mesh.vertices)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), "mesh is not closed"
    assert mesh_volume(mesh) > 0.0, "normals point inward"
    span = np.ptp(v[:, 2])
    assert v[:, 2].min() == pytest.approx(-v[:, 2].max(), abs=1e-6 * span), "not full span about z = 0"
    assert np.ptp(v[:, 1]) < span, "y should be up, not the span"
    assert mesh_planform_area(mesh) > 0.0


@pytest.mark.parametrize("method", sorted(PARAMS) + ["ltoc"])
def test_builders_give_closed_full_span_meshes(method):
    params = _ltoc_params() if method == "ltoc" else PARAMS[method]
    geom = M.build(method, params)
    _check_mesh(geom.mesh)
    assert geom.length > 0.0


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="no optimization builder"):
        M.build("nope", {})


def test_design_variables_and_cases():
    v = DesignVariable.around("a", "A", 2.0, rel=0.25, upper=2.2)
    assert (v.lower, v.upper) == (1.5, 2.2)
    with pytest.raises(ValueError, match="empty range"):
        DesignVariable.around("z", "Z", 0.0)                 # rel of 0 is no range
    case = OptimizationCase("ltoc", "LTOCs Waverider", {"fct_z": 0.25, "shock": {"a_q": 0.65}},
                            [DesignVariable("fct_z", "FCT", 0.25, 0.2, 0.3),
                             DesignVariable("shock.a_q", "a_q", 0.65, 0.6, 0.7),
                             DesignVariable("shock.b_q", "b_q", 0.8, 0.7, 0.9, enabled=False)], 7.0)
    p = case.params([0.22, 0.61])
    assert p == {"fct_z": 0.22, "shock": {"a_q": 0.61}}
    assert case.fixed["shock"]["a_q"] == 0.65                # the case itself is unchanged
    assert OptimizationCase.from_dict(case.to_dict()) == case


def test_evaluation_reports_objectives_and_failures():
    r = M.evaluate_design("gvwd", PARAMS["gvwd"], dict(FLOW, mach=6.0))
    assert r["success"], r.get("error")
    # Caret: the lower surface alone has L/D = cot(theta_d) = 4.0; the base and
    # the upper surface lower it
    assert 2.0 < r["CL"] / r["CD"] < 4.0
    assert r["VolEff"] == pytest.approx(r["Volume"] ** (2 / 3) / r["planform_area"])
    assert M.objective_value(r, "CL/CD") == pytest.approx(r["CL"] / r["CD"], rel=1e-6)
    assert M.constraint_value(r, "Volume_min", r["Volume"] + 1.0) == pytest.approx(1.0)

    bad = M.evaluate_design("pswr", dict(PARAMS["pswr"], beta0_deg=80.0), dict(FLOW, mach=6.0))
    assert not bad["success"] and bad["error"].startswith("Geometry failed")


def test_nsga2_improves_on_the_starting_design():
    pytest.importorskip("pymoo")
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize

    case = OptimizationCase("gvwd", "GVWD Waverider", PARAMS["gvwd"], [
        DesignVariable.around("theta_d_deg", "theta_d", 14.0, span=3.0),
        DesignVariable.around("Lambda_deg", "Lambda", 70.0, span=5.0)], 6.0)
    problem = M.MethodProblem(case, [{"name": "CL/CD", "mode": "maximize"}],
                              [{"name": "Volume_min", "value": 10.0}], flow=FLOW)
    start = M.evaluate_design("gvwd", case.params(), dict(FLOW, mach=6.0))
    res = minimize(problem, NSGA2(pop_size=8), ("n_gen", 4), seed=1, verbose=False)
    assert -float(np.min(res.F)) > start["CL"] / start["CD"]
    assert problem.n_eval >= 8 and len(problem.last_eval_X) == len(problem.last_eval_results)
    best = case.params(np.atleast_2d(res.X)[0])
    assert M.evaluate_design("gvwd", best, dict(FLOW, mach=6.0))["Volume"] >= 10.0


def test_pareto_design_files(tmp_path):
    case = OptimizationCase("pswr", "PSWR-1 Waverider", PARAMS["pswr"],
                            [DesignVariable.around("lambda_deg", "Lambda", 70.0, span=5.0)], 6.0)
    M.write_design_files(case, case.params([72.0]), tmp_path, "design_0001", {"CL/CD": 5.5})
    import json

    saved = json.loads((tmp_path / "design_0001.json").read_text())
    assert saved["params"]["lambda_deg"] == 72.0 and saved["method"] == "pswr"
    assert (tmp_path / "design_0001.stl").stat().st_size > 84
    assert "Lambda = 72" in (tmp_path / "design_0001_info.txt").read_text()
