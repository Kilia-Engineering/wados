"""Genetic-algorithm optimization of any design tab's waverider.

The OC Waverider tab keeps its own optimization (``optimization_engine``:
X1-X4, CAD export and Gmsh mesh). Every other design tab hands the
Optimization tab an :class:`OptimizationCase`:

* its current design, as the tab's own parameter dict (``fixed``);
* the numbers that may vary, with bounds (``variables``); each is a key of
  that dict, dotted for a nested entry (``shock.a_q``);
* the design Mach number, the default flight Mach number.

``BUILDERS[case.method]`` turns a parameter dict into the closed full-span
surface mesh, in metres and the analysis frame (x streamwise, y up, z span),
and a reference length. Each design is then evaluated like an OC design:
enclosed volume, planform area (used as A_ref), and PySAGAS C_L, C_D, C_m at
the flight condition, on that mesh directly (no CAD or Gmsh).

No Qt here: the builders run in worker processes and in the tests.
"""
from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# Objectives and constraints, as the OC optimization names them
OBJECTIVES = ("CL", "CD", "Cm", "Volume", "VolEff", "CL/CD")
CONSTRAINTS = ("CL_min", "CD_max", "Cm_max", "Volume_min")


# ---------------------------------------------------------------------------
#  Case: what a tab hands over
# ---------------------------------------------------------------------------

@dataclass
class DesignVariable:
    """One parameter the optimizer may vary, within [lower, upper]."""
    name: str            # key in the parameter dict, dotted for nested entries
    label: str
    value: float         # current value in the tab
    lower: float
    upper: float
    enabled: bool = True

    @classmethod
    def around(cls, name: str, label: str, value: float, *, rel: float = 0.25,
               span: Optional[float] = None, lower: Optional[float] = None,
               upper: Optional[float] = None, enabled: bool = True) -> "DesignVariable":
        """Bounds ``value -/+ span`` (or ``-/+ rel |value|``), clipped to [lower, upper]."""
        d = span if span is not None else rel * abs(value)
        lo, hi = value - d, value + d
        if lower is not None:
            lo = max(lo, lower)
        if upper is not None:
            hi = min(hi, upper)
        if not hi > lo:
            raise ValueError(f"{label}: empty range around {value:g}")
        return cls(name, label, float(value), float(lo), float(hi), enabled)


@dataclass
class OptimizationCase:
    method: str                      # key of BUILDERS
    title: str                       # the tab's name, e.g. "PSWR-1 Waverider"
    fixed: dict                      # the tab's parameter dict
    variables: List[DesignVariable]
    mach: float                      # design Mach number
    notes: str = ""                  # shown in the Optimization tab

    def active(self) -> List[DesignVariable]:
        return [v for v in self.variables if v.enabled]

    def params(self, x=None) -> dict:
        """The parameter dict with the active variables set to ``x``."""
        p = copy.deepcopy(self.fixed)
        if x is not None:
            for var, val in zip(self.active(), x):
                set_param(p, var.name, float(val))
        return p

    def to_dict(self) -> dict:
        return {"method": self.method, "title": self.title, "fixed": self.fixed,
                "variables": [asdict(v) for v in self.variables], "mach": self.mach,
                "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizationCase":
        return cls(method=d["method"], title=d.get("title", d["method"]), fixed=d["fixed"],
                   variables=[DesignVariable(**v) for v in d["variables"]],
                   mach=float(d["mach"]), notes=d.get("notes", ""))


def get_param(d: dict, name: str):
    for key in name.split(".")[:-1]:
        d = d[key]
    return d[name.split(".")[-1]]


def set_param(d: dict, name: str, value) -> None:
    keys = name.split(".")
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
#  Builders: parameter dict -> closed full-span mesh
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    mesh: object        # gvwd.geometry.mesh.Mesh, closed, analysis frame, metres
    length: float       # reference length for C_m


BUILDERS: Dict[str, Callable[[dict], Geometry]] = {}


def builder(key: str):
    def register(fn):
        BUILDERS[key] = fn
        return fn
    return register


def build(method: str, params: dict) -> Geometry:
    try:
        fn = BUILDERS[method]
    except KeyError:
        raise ValueError(f"no optimization builder for method {method!r}") from None
    return fn(params)


def _length(mesh) -> float:
    return float(np.ptp(np.asarray(mesh.vertices)[:, 0]))


@builder("ltoc")
def _build_ltoc(p: dict) -> Geometry:
    from ltoc.design import LTOCDesign

    if p.get("source") == "oc_import":
        raise ValueError("an imported OC design is optimized in the OC Waverider tab")
    res = LTOCDesign.from_dict(p).build()
    if res.refusals:
        k, status, _ = res.refusals[0]
        raise ValueError(f"{len(res.refusals)} stream surfaces refused (station {k}: {status})")
    wr = res.waverider
    return Geometry(wr.mesh(full_span=True), float(wr.length))


GVWD_MODES = ("engineering_flat", "engineering_shallow_v", "caret", "flat_delta", "multi_wedge")


def gvwd_config(p: dict):
    """The GVWD geometry config dataclass for parameter dict ``p`` (its fields plus ``mode``)."""
    from dataclasses import fields as _fields

    from gvwd.io import config as C

    classes = {c().mode: c for c in (C.EngineeringFlatConfig, C.EngineeringShallowVConfig,
                                     C.CaretConfig, C.FlatDeltaConfig, C.MultiWedgeConfig)}
    cls = classes[p["mode"]]
    names = {f.name for f in _fields(cls)}
    return cls(**{k: v for k, v in p.items() if k in names})


@builder("gvwd")
def _build_gvwd(p: dict) -> Geometry:
    import warnings

    from geometry_export import span_frame_to_gui
    from gvwd.io.config import FinsConfig, GVWDRunConfig, build_geometry

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        geom = build_geometry(GVWDRunConfig(geometry=gvwd_config(p), fins=FinsConfig(n_fins=0)))
    mesh = span_frame_to_gui(geom.mesh)          # GVWD works in x, y span, z up
    return Geometry(mesh, _length(mesh))


@builder("pswr")
def _build_pswr(p: dict) -> Geometry:
    from geometry_export import streams_to_mesh
    from pswr.geometry.variable_wedge import VariableWedgeWaverider, to_gui_frame

    wr = VariableWedgeWaverider(
        M_inf=float(p["M_inf"]),
        beta_knots=tuple(math.radians(float(p[k])) for k in ("beta0_deg", "beta1_deg", "beta2_deg")),
        Lambda=math.radians(float(p["lambda_deg"])),
        body_length=float(p["length_m"]),
        flat_fraction=float(p["flat_fraction"]),
        n_span=int(p["n_span"]),
        n_chord=int(p["n_chord"]),
    )
    upper = [to_gui_frame(s) for s in wr.upper_surface]
    lower = [to_gui_frame(s) for s in wr.lower_surface_streams]
    return Geometry(streams_to_mesh(upper, lower, kind="pswr1_waverider"), float(wr.body_length))


@builder("planar")
def _build_planar(p: dict) -> Geometry:
    from geometry_export import grid_to_streams, span_frame_to_gui, streams_to_mesh
    from planar_waverider import PlanarWaverider

    wr = PlanarWaverider(length=float(p["length"]), width=float(p["width"]), n=float(p["n"]),
                         beta_deg=float(p["beta"]), epsilon=float(p["epsilon"]),
                         p1=float(p["p1"]), p2=float(p["p2"]), p3=float(p["p3"]),
                         R=float(p.get("le_radius", 0.0)),
                         blunting_method=p.get("blunting_method", "none"),
                         M_inf=float(p["mach"]))
    wr.generate(nx=int(p["nx"]), ny=int(p["ny"]))
    upper = grid_to_streams(wr.upper_surface_x, wr.upper_surface_y, wr.upper_surface_z)
    lower = grid_to_streams(wr.lower_surface_x, wr.lower_surface_y, wr.lower_surface_z)
    mesh = span_frame_to_gui(streams_to_mesh(upper, lower, span_axis=1, kind="planar_waverider"))
    return Geometry(mesh, float(p["length"]))


@builder("shadow")
def _build_shadow(p: dict) -> Geometry:
    """Cone-derived (SHADOW) waverider, as the tab's Generate builds it."""
    from geometry_export import streams_to_mesh
    from shadow_waverider import create_second_order_waverider, create_third_order_waverider

    dome = None
    if p.get("dome_enabled"):
        dome = [(0.0, p["dome_h1"]), (p["dome_s2"], p["dome_h2"]), (p["dome_s3"], p["dome_h3"])]
    loft = None
    if p.get("vol_loft_enabled"):
        loft = [(0.0, p["vol_loft_center"]),
                (p["vol_loft_cp1_span"], p["vol_loft_cp1_height"]),
                (p["vol_loft_cp2_span"], p["vol_loft_cp2_height"])]
    growth = "smooth" if "smooth" in str(p.get("vol_loft_growth", "")).lower() else "linear"
    common = dict(mach=float(p["mach"]), shock_angle=float(p["shock_angle"]),
                  A2=float(p["a2"]), A0=float(p["a0"]),
                  n_leading_edge=int(p["n_le"]), n_streamwise=int(p["n_stream"]),
                  length=float(p["length"]), top_surface_control=float(p["top_surface_a"]),
                  upper_surface_spline=dome, volume_loft_spline=loft, volume_loft_growth=growth)
    if str(p.get("order", "2nd")).startswith("3"):
        wr = create_third_order_waverider(A3=float(p["a3"]), **common)
    else:
        wr = create_second_order_waverider(**common)
    mesh = streams_to_mesh(list(np.asarray(wr.upper_surface)), list(np.asarray(wr.lower_surface)),
                           kind="cone_derived_waverider")
    return Geometry(mesh, float(wr.length))


def _oc_family_mesh(wr, kind: str) -> Geometry:
    """Half model with station 0 on the symmetry plane, mirrored."""
    from geometry_export import streams_to_mesh

    mesh = streams_to_mesh(wr.upper_surface_streams, wr.lower_surface_streams,
                           full_span=True, kind=kind)
    return Geometry(mesh, _length(mesh))


@builder("hybrid")
def _build_hybrid(p: dict) -> Geometry:
    from waverider_generator.hybrid_generator import GOCWaverider

    extra = {}
    if p.get("per_plane_beta"):
        extra["use_per_plane_beta"] = True
    if p.get("independent_profiles"):
        extra["h0_nose"] = float(p["h0_nose"])
        extra["h1_nose"] = float(p["h1_nose"])
    wr = GOCWaverider(
        M_inf=float(p["mach"]), beta_OC=float(p["beta_oc"]),
        beta_CD=float(p.get("beta_cd", p["beta_oc"])),
        height=float(p["height"]), width=float(p["width"]),
        dp=[float(p[k]) for k in ("x1", "x2", "x3", "x4")],
        h0=float(p["alpha_root"]), h1=float(p["alpha_tip"]),
        x_t=float(p.get("x_t", 0.5)), dx_t=float(p.get("dx_t", 0.3)),
        n_planes=int(p["n_planes"]), n_streamwise=int(p["n_streamwise"]),
        blend_exp=float(p["blend_exp"]),
        dome_height=float(p.get("dome_height", 0.0)), dome_taper=float(p.get("dome_taper", 2.0)),
        **extra)
    return _oc_family_mesh(wr, "hybrid_waverider")


def _fillet_dl(ma_max: float, beta: float, radius_mm: float) -> float:
    """Length added so a leading-edge fillet of ``radius_mm`` keeps the
    nominal length (as the VMN/VMOF tabs do)."""
    if radius_mm <= 0.01:
        return 0.0
    from waverider_generator.flowfield import cone_angle

    theta_min = cone_angle(ma_max, beta, 1.4)
    return radius_mm / 1000.0 * (1.0 / math.sin(math.radians(theta_min)) - 1.0)


def _direction(p: dict) -> str:
    d = p.get("direction", 0)
    if isinstance(d, str):
        return "increasing" if d.lower().startswith("inc") else "decreasing"
    return "decreasing" if int(d) == 0 else "increasing"


@builder("vmn")
def _build_vmn(p: dict) -> Geometry:
    from waverider_generator.vmn_generator import VMNWaverider

    length = float(p["length"]) + _fillet_dl(float(p["ma_max"]), float(p["beta"]),
                                              float(p.get("blunt_radius_mm", 0.0)))
    wr = VMNWaverider(Ma_min=float(p["ma_min"]), Ma_max=float(p["ma_max"]),
                      beta_deg=float(p["beta"]), L0=1.0, S=float(p["s"]), A0=float(p["a0"]),
                      length=length, direction=_direction(p),
                      n_points=int(p["n_points"]), n_streamwise=int(p["n_streamwise"]))
    return _oc_family_mesh(wr, "vmn_waverider")


@builder("vmof")
def _build_vmof(p: dict) -> Geometry:
    from waverider_generator.vmof_generator import VMOFWaverider

    height, width = float(p["height"]), float(p["width"])
    dp = [float(p[k]) for k in ("x1", "x2", "x3", "x4")]
    dl = _fillet_dl(float(p["ma_max"]), float(p["beta"]), float(p.get("blunt_radius_mm", 0.0)))
    if dl > 0.0:                      # the OC length scales with the height: scale the body
        from waverider_generator.generator import waverider as _OCW

        L_nom = _OCW(float(p["ma_max"]), float(p["beta"]), height, width, dp,
                     n_upper_surface=20, n_shockwave=20, n_planes=5, n_streamwise=5).length
        height *= (L_nom + dl) / L_nom
        width *= (L_nom + dl) / L_nom
    wr = VMOFWaverider(M_inf_min=float(p["ma_min"]), M_inf_max=float(p["ma_max"]),
                       beta_deg=float(p["beta"]), height=height, width=width, dp=dp,
                       direction=_direction(p), n_planes=int(p["n_points"]),
                       n_streamwise=int(p["n_streamwise"]))
    return _oc_family_mesh(wr, "vmof_waverider")


def vmplo_waverider(p: dict):
    """The VMPLO waverider for the VMPLO tab's parameter dict."""
    from waverider_generator.vmplo.bspline import BSpline1D
    from waverider_generator.vmplo.geometry import VMPLOWaverider
    from waverider_generator.vmplo.osculating import OsculatingAssembly

    L, H, W = float(p["length"]), float(p["height"]), float(p["width"])

    def dist(kind, center, tip, n_knots):
        if int(kind) == 0:
            return BSpline1D.constant(center, 0.0, W, n_internal_knots=n_knots)
        if int(kind) == 1:
            return BSpline1D.linear(center, tip, 0.0, W, n_internal_knots=n_knots)
        return BSpline1D.quadratic_liu(center, tip, 0.0, W, n_internal_knots=n_knots)

    Ma_sp = dist(p["ma_dist"], float(p["ma_center"]), float(p["ma_tip"]), max(int(p["ma_ncp"]) - 4, 1))
    n_sp = dist(min(int(p["n_dist"]), 1), float(p["n_center"]), float(p["n_tip"]), 4)
    icc_sp = dist(p["icc_dist"], float(p["icc_center"]) * H, float(p["icc_tip"]) * H,
                  max(int(p["icc_ncp"]) - 4, 1))
    assembly = OsculatingAssembly(Ma_spline=Ma_sp, n_spline=n_sp, ICC_spline=icc_sp,
                                  US_spline=None, beta_design=float(p["beta"]),
                                  L=L, W=W, H=H, x_LE=float(p["x_LE"]))
    return VMPLOWaverider(assembly, n_planes=int(p["n_planes"]),
                          n_streamwise=int(p["n_streamwise"]))


@builder("vmplo")
def _build_vmplo(p: dict) -> Geometry:
    return _oc_family_mesh(vmplo_waverider(p), "vmplo_waverider")


@builder("mfof")
def _build_mfof(p: dict) -> Geometry:
    """MFOF waverider: ``params`` (the Liu 2019 parameter dict), ``n_z``,
    ``n_x``, ``flowfield`` (cone, wedge or power-law) and ``power_law_n``."""
    from geometry_export import grid_to_streams, streams_to_mesh
    from mfof.cone_flowfield import ConeFlowfield
    from mfof.geometry import build_mfof_waverider
    from mfof.power_law_flowfield import PowerLawFlowfield
    from mfof.wedge_flowfield import WedgeFlowfield

    params = dict(p["params"])
    params["z5"] = float(params["W"]) / 2.0          # follows the width, as in the tab
    beta, gamma = float(params["beta_deg"]), float(params.get("gamma", 1.4))
    kind = p.get("flowfield", "cone")
    if kind == "wedge":
        def factory(z, Ma_z):
            return WedgeFlowfield(Ma_z, beta, gamma)
    elif kind == "power-law":
        n_pl, L_w = float(p.get("power_law_n", 1.0)), float(params["L_w"])

        def factory(z, Ma_z):
            return PowerLawFlowfield(Ma_z, beta, n=n_pl, L=L_w, gamma=gamma)
    else:
        def factory(z, Ma_z):
            return ConeFlowfield(Ma_z, beta, gamma)
    wr = build_mfof_waverider(params, factory, n_z=int(p["n_z"]), n_x=int(p["n_x"]))
    S = wr.surfaces
    mesh = streams_to_mesh(grid_to_streams(S.X_upper, S.Y_upper, S.Z_upper),
                           grid_to_streams(S.X_lower, S.Y_lower, S.Z_lower),
                           full_span=True, kind="mfof_waverider")
    return Geometry(mesh, float(params["L_w"]))


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def pysagas_coefficients(mesh, *, mach: float, pressure: float, temperature: float,
                         aoa: float, A_ref: float, c_ref: float):
    """PySAGAS (oblique shock / Prandtl-Meyer) C_L, C_D, C_m of a triangle mesh."""
    from pysagas.cfd import OPM
    from pysagas.flow import FlowState
    from pysagas.geometry import Cell, Vector

    v = np.asarray(mesh.vertices, dtype=float)
    cells = [Cell.from_points([Vector(x=float(a), y=float(b), z=float(c)) for a, b, c in v[tri]])
             for tri in np.asarray(mesh.faces, dtype=int)]
    work = tempfile.mkdtemp(prefix="wados_pysagas_")      # PySAGAS writes temporary files
    cwd = os.getcwd()
    os.chdir(work)
    try:
        solver = OPM(cells, FlowState(mach=mach, pressure=pressure, temperature=temperature))
        solver.solve(aoa=aoa)
        CL, CD, Cm = solver.flow_result.coefficients(A_ref=A_ref, c_ref=c_ref)
        return float(CL), float(CD), float(Cm)
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)


def evaluate_design(method: str, params: dict, flow: dict) -> dict:
    """Build and analyse one design.

    ``flow``: mach, pressure (Pa), temperature (K), aoa (deg). Returns
    ``success`` with CL, CD, Cm, Volume, planform_area, VolEff and time, or
    ``success: False`` with ``error``.
    """
    from geometry_export import mesh_planform_area, mesh_volume

    t0 = time.time()
    try:
        geom = build(method, params)
        volume = abs(mesh_volume(geom.mesh))
        planform = mesh_planform_area(geom.mesh)
        if not planform > 0.0:
            raise ValueError("zero planform area")
    except Exception as e:
        return {"success": False, "error": f"Geometry failed: {e}", "time": time.time() - t0}
    try:
        CL, CD, Cm = pysagas_coefficients(
            geom.mesh, mach=flow["mach"], pressure=flow["pressure"],
            temperature=flow["temperature"], aoa=flow.get("aoa", 0.0),
            A_ref=planform, c_ref=geom.length)
    except Exception as e:
        return {"success": False, "error": f"PySAGAS analysis failed: {e}", "time": time.time() - t0}
    if not all(np.isfinite([CL, CD, Cm])):
        return {"success": False, "error": "PySAGAS returned NaN/Inf", "time": time.time() - t0}
    return {"success": True, "CL": CL, "CD": CD, "Cm": Cm, "Volume": volume,
            "planform_area": planform, "VolEff": volume ** (2.0 / 3.0) / planform,
            "length": geom.length, "time": time.time() - t0}


def _evaluate_args(args) -> dict:
    """Module-level, so worker processes can run it."""
    return evaluate_design(*args)


def objective_value(result: dict, name: str) -> float:
    if name in ("CL/CD", "LD", "L/D"):
        return result["CL"] / (result["CD"] + 1e-10)
    if name == "VolEff":
        return result["Volume"] ** (2.0 / 3.0) / result["planform_area"]
    return result[name]


def constraint_value(result: dict, name: str, value: float) -> float:
    """g <= 0 when satisfied."""
    if name == "CL_min":
        return value - result["CL"]
    if name == "CD_max":
        return result["CD"] - value
    if name == "Cm_max":
        return abs(result["Cm"]) - value
    if name == "Volume_min":
        return value - result["Volume"]
    raise ValueError(f"unknown constraint {name!r}")


# ---------------------------------------------------------------------------
#  pymoo problem
# ---------------------------------------------------------------------------

try:
    from pymoo.core.problem import Problem as _Problem
except ImportError:          # the builders and the evaluation work without pymoo
    _Problem = object


class MethodProblem(_Problem):
    """NSGA-II problem over a case's enabled design variables.

    Objectives and constraints as in the OC optimization: ``objectives`` are
    dicts ``name`` (CL, CD, Cm, Volume, VolEff, CL/CD) and ``mode``
    (minimize, maximize or target with ``target``); ``constraints`` are dicts
    ``name`` (CL_min, CD_max, Cm_max, Volume_min) and ``value``. A failed
    design gets penalty values. After each batch ``last_eval_X`` and
    ``last_eval_results`` (with ``last_eval_success``, ``last_eval_errors``)
    hold the designs and their outcomes, as the Optimization tab's worker
    expects.
    """

    def __init__(self, case: OptimizationCase, objectives: List[dict],
                 constraints: Optional[List[dict]] = None, flow: Optional[dict] = None,
                 n_cores: int = 1, verbose: bool = False):
        if _Problem is object:
            raise ImportError("pymoo is required for optimization (pip install pymoo)")
        active = case.active()
        if not active:
            raise ValueError("no design variable is enabled")
        if not objectives:
            raise ValueError("no objective selected")
        self.case = case
        self.objectives = objectives
        self.constraints = [c for c in (constraints or [])
                            if c.get("active", True) and c["name"] != "design_space"]
        self.flow = dict(flow or {})
        self.flow.setdefault("mach", case.mach)
        self.n_cores = max(1, int(n_cores))
        self.verbose = verbose
        self.n_eval = 0
        self.last_eval_results = self.last_eval_success = self.last_eval_errors = None
        self.last_eval_X = None
        super().__init__(n_var=len(active), n_obj=len(objectives),
                         n_ieq_constr=len(self.constraints),
                         xl=np.array([v.lower for v in active], dtype=float),
                         xu=np.array([v.upper for v in active], dtype=float))

    def evaluate_designs(self, X) -> List[dict]:
        args = [(self.case.method, self.case.params(x), self.flow) for x in np.atleast_2d(X)]
        if self.n_cores > 1 and len(args) > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(processes=self.n_cores) as pool:
                return pool.map(_evaluate_args, args)
        return [_evaluate_args(a) for a in args]

    def _evaluate(self, X, out, *args, **kwargs):
        results = self.evaluate_designs(X)
        F = np.full((len(X), self.n_obj), 1e6)
        G = np.full((len(X), len(self.constraints)), 1e6)
        for i, r in enumerate(results):
            if not r["success"]:
                continue
            for j, obj in enumerate(self.objectives):
                v = objective_value(r, obj["name"])
                mode = obj.get("mode", "minimize").lower()
                F[i, j] = -v if mode == "maximize" else (
                    abs(v - obj.get("target", 0.0)) if mode == "target" else v)
            for j, c in enumerate(self.constraints):
                G[i, j] = constraint_value(r, c["name"], c["value"])
        self.last_eval_results = results
        self.last_eval_X = np.array(X, dtype=float, copy=True)
        self.last_eval_success = np.array([r["success"] for r in results])
        self.last_eval_errors = [r.get("error", "") for r in results]
        self.n_eval += len(X)
        out["F"] = F
        if self.constraints:
            out["G"] = G


# ---------------------------------------------------------------------------
#  Output
# ---------------------------------------------------------------------------

def write_design_files(case: OptimizationCase, params: dict, folder, stem: str,
                       info: Optional[dict] = None) -> None:
    """``stem.stl`` (closed mesh, metres), ``stem.json`` (the tab's parameter
    dict, loadable in the tab) and ``stem_info.txt`` for one design."""
    import json
    from pathlib import Path

    from geometry_export import write_stl

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_stl(build(case.method, params).mesh, folder / f"{stem}.stl")
    (folder / f"{stem}.json").write_text(json.dumps(
        {"method": case.method, "title": case.title, "params": params}, indent=2, default=float))
    lines = [f"{case.title}: {stem}", "=" * 40, "", "Design variables:"]
    lines += [f"  {v.label} = {get_param(params, v.name):.6g}" for v in case.active()]
    if info:
        lines += ["", "Performance:"] + [f"  {k} = {val}" for k, val in info.items()]
    (folder / f"{stem}_info.txt").write_text("\n".join(lines) + "\n")
