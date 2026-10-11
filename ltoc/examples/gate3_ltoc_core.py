"""Gate 3 deliverable: LTOCs core figures and numbers (Phase 3).

Run from the repository root::

    python -m ltoc.examples.gate3_ltoc_core --out docs/ltoc/figures

The script writes four PNG figures and ``gate3_results.json``.

* V4: LTOCs on a circular conical shock against the tight Taylor-Maccoll
  reference and the existing cone-derived waverider (``ShadowWaverider``).
* V5: LTOCs on the shock of a WADOS osculating-cone design against the exact
  per-plane reference and the OC generator.
* Convergence: V4 and V5 (planar stream surfaces), and the self-convergence
  of twisted stream surfaces on the R1 case II shock (Eq. 25).
* Diagnostics (spec section 6) for the two R1 waverider shocks, case I
  (Eq. 23, M 6, FCT Z = 0.12) and case II (Eq. 25, M 7, FCT Z = 0.25). These
  are core runs only; forces and pressures against R1 Tables 2 and 3 are
  Phase 4.

R1 waverider shocks are given in R1's frame, where Z points down (R1 Figs. 20,
21 and 24).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

from ltoc import viz
from ltoc.ltoc import find_leading_edge, r1_frame_to_gui, solve_stream_surface
from ltoc.reference import ConicalFlowReference, osculating_cone_streamline
from ltoc.shock_surface import ConeShock, OsculatingConeShock, RuledShock
from ltoc.waverider import LTOCWaverider

R1_CASES = {
    "I": {"shock": dict(x_p=0.0, a_p=0.0, b_p=0.0, x_q=1.7, a_q=0.595, b_q=0.68),
          "M": 6.0, "fct_z": 0.12},
    "II": {"shock": dict(x_p=0.1, a_p=0.3, b_p=0.2, x_q=1.5, a_q=0.65, b_q=0.8),
           "M": 7.0, "fct_z": 0.25},
}


# ---------------------------------------------------------------------------
#  V4 and V5
# ---------------------------------------------------------------------------

def v4(n_points):
    from shadow_waverider import ShadowWaverider

    with contextlib.redirect_stdout(io.StringIO()):
        sh = ShadowWaverider(mach=6.0, shock_angle=12.0, poly_coeffs=[-1.0, 0.0, 0.5],
                             n_leading_edge=11, n_streamwise=40)
    le = sh.leading_edge
    half = le[:, 2] >= -1e-12
    order = np.argsort(le[half, 2])
    le_half, lower_half = le[half][order], sh.lower_surface[half][order]
    fct = np.column_stack([le_half[:, 2], le_half[:, 1]])        # shock frame (X, z_span, y_vert)
    to_gui = lambda P: np.stack([P[..., 0], P[..., 2], P[..., 1]], axis=-1)
    wr = LTOCWaverider(ConeShock(np.radians(12.0), x_range=(0.0, 1.2 * sh.x_end)), fct,
                       sh.x_end, 6.0, n_points=n_points, to_gui=to_gui, strict=True)
    ref = ConicalFlowReference(6.0, np.radians(12.0))
    length = sh.x_end - sh.x_start
    span, dev_tm, dev_sh = [], [], []
    for s, S in zip(wr.stations, lower_half):
        if s.status == "tip":
            continue
        P = s.body.points
        rho = np.hypot(P[:, 1], P[:, 2])
        y_ref = ref.streamline(s.le_point[0], np.hypot(*s.le_point[1:]), sh.x_end)(P[:, 0])
        span.append(s.le_point[1] / le_half[-1, 2])
        dev_tm.append(float(np.max(np.abs(rho - y_ref))) / length)
        dev_sh.append(float(np.max(np.abs(np.interp(S[:, 0], P[:, 0], rho)
                                          - np.hypot(S[:, 1], S[:, 2])))) / length)
    return {"span": span, "vs_tm": dev_tm, "vs_shadow": dev_sh}


def oc_design():
    from waverider_generator.generator import waverider

    return waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                     n_upper_surface=10000, n_shockwave=10000, n_planes=10, n_streamwise=40,
                     delta_streamwise=0.1)


def v5(oc_wr, n_points):
    shock = OsculatingConeShock.from_oc_waverider(oc_wr)
    fct = np.asarray(oc_wr.leading_edge)[:, [1, 2]]
    lt = LTOCWaverider(shock, fct, oc_wr.length, 5.0, n_points=n_points, strict=True)
    L = oc_wr.length
    span, d_ref, d_oc, oc_ref, flat = [], [], [], [], []
    for k, s in enumerate(lt.stations):
        if s.status == "tip":
            continue
        P = s.body.points
        S = np.asarray(oc_wr.lower_surface_streams[k])
        ref_P = osculating_cone_streamline(shock, s.le_uv, s.le_point, P[:, 0], 5.0)
        ref_S = osculating_cone_streamline(shock, s.le_uv, s.le_point, S[:, 0], 5.0)
        yi, zi = np.interp(S[:, 0], P[:, 0], P[:, 1]), np.interp(S[:, 0], P[:, 0], P[:, 2])
        span.append(s.le_point[2] / oc_wr.width)
        d_ref.append(float(np.max(np.linalg.norm(P - ref_P, axis=1))) / L)
        d_oc.append(float(np.max(np.hypot(yi - S[:, 1], zi - S[:, 2]))) / L)
        oc_ref.append(float(np.max(np.linalg.norm(S - ref_S, axis=1))) / L)
        flat.append(bool(s.diagnostics["r_min"] >= 1e5 * L))
    return {"span": span, "vs_ref": d_ref, "vs_oc": d_oc, "oc_vs_ref": oc_ref, "flat": flat}


# ---------------------------------------------------------------------------
#  R1 shocks: twisted stream surfaces
# ---------------------------------------------------------------------------

def r1_shock(case):
    c = R1_CASES[case]
    s = RuledShock.elliptic(**c["shock"], m_range=(0.0, np.pi), n_range=(0.0, 1.0))
    y_tip = c["shock"]["a_q"] * np.sqrt(1.0 - (c["fct_z"] / c["shock"]["b_q"]) ** 2)
    return s, c["shock"]["x_q"], c["M"], c["fct_z"], y_tip


def twisted_convergence(fractions=(0.3, 0.6), ns=(20, 40, 80, 160)):
    s, x_base, M, zf, y_tip = r1_shock("II")
    uv, _, _ = find_leading_edge(s, np.array([[f * y_tip, zf] for f in fractions]))
    out = {}
    for k, f in enumerate(fractions):
        runs = {n: solve_stream_surface(s, uv[k], x_base, M, n_points=n) for n in ns}
        base = [runs[n].body.points[-1] for n in ns]
        out[f] = {
            "n": list(ns),
            "diff_next": [float(np.linalg.norm(base[i] - base[i + 1])) / x_base
                          for i in range(len(ns) - 1)],
            "tangency_deg": [runs[n].diagnostics["stepc_tangency_deg"] for n in ns],
        }
    return out


def r1_waverider(case, n_stations=13, n_points=40):
    s, x_base, M, zf, y_tip = r1_shock(case)
    ys = y_tip * np.linspace(0.0, 1.0, n_stations)
    wr = LTOCWaverider(s, np.column_stack([ys, np.full_like(ys, zf)]), x_base, M,
                       n_points=n_points, to_gui=r1_frame_to_gui, strict=True)
    return wr, ys / y_tip


def station_diagnostics(wr, frac):
    rows = []
    for st, f in zip(wr.stations, frac):
        if st.status == "tip":
            continue
        cv, d = st.curve, st.diagnostics
        m_base = np.interp(wr.x_base, cv.x, cv.uv[:, 0])
        ya_base = np.interp(wr.x_base, cv.x, cv.y_axis)
        rows.append({
            "fct_fraction": float(f),
            "x_le": float(st.le_point[0]),
            "azimuth_drift_le_to_base_deg": float(np.degrees(abs(m_base - cv.uv[0, 0]))),
            "axis_drift_le_to_base_over_L": float(abs(ya_base - cv.y_axis[0]) / wr.x_base),
            "extension_used_chords": d["extension_chords"],
            "extension_needed_chords": d["extension_needed_chords"],
            "body_step_ratio": d["body_step_ratio"],
            "stepc_tangency_deg": d["stepc_tangency_deg"],
            "beta_deg": d["beta_deg"],
            "r_min": d["r_min"],
            "moc_solves": d["moc_solves"],
            "stepc_fallbacks": d["stepc_fallbacks"],
        })
    return rows


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------

def fig_v4_v5(d4, d5, path):
    with viz.style():
        fig, axes = viz.new_figure(1, 2, width=16.0, height=6.0)
        ax = axes[0, 0]
        ax.semilogy(d4["span"], d4["vs_tm"], marker=viz.MARKERS[0], color=viz.SERIES[0],
                    label="vs tight Taylor–Maccoll")
        ax.semilogy(d4["span"], d4["vs_shadow"], marker=viz.MARKERS[1], color=viz.SERIES[1],
                    label="vs ShadowWaverider")
        ax.axhline(1e-4, color=viz.INK_2, ls="--", lw=1.5, label="V4 criterion, 1e-4 L")
        ax.set_xlabel("Leading-edge station, z / (half span)")
        ax.set_ylabel("Max body deviation / L")
        ax.set_title("V4: cone shock, M 6, β 12°, 40 points")
        ax.legend(loc="lower left")
        ax.set_ylim(1e-10, 1e-2)

        ax = axes[0, 1]
        ax.semilogy(d5["span"], d5["vs_ref"], marker=viz.MARKERS[0], color=viz.SERIES[0],
                    label="LTOCs vs exact per-plane flow")
        ax.semilogy(d5["span"], d5["vs_oc"], marker=viz.MARKERS[1], color=viz.SERIES[1],
                    label="LTOCs vs OC generator")
        oc = np.asarray(d5["oc_vs_ref"])
        curved = ~np.asarray(d5["flat"])
        ax.semilogy(np.asarray(d5["span"])[curved], oc[curved], marker=viz.MARKERS[2],
                    color=viz.SERIES[2], ls=":", label="OC generator vs exact per-plane flow")
        ax.axhline(1e-4, color=viz.INK_2, ls="--", lw=1.5, label="1e-4 L")
        ax.axvspan(-0.02, 0.5 * (d5["span"][int(np.sum(~curved)) - 1] + d5["span"][int(np.sum(~curved))]),
                   color=viz.GRID, alpha=0.5, lw=0)
        ax.text(0.01, 2e-10, "flat shock:\nwedge flow,\nOC generator\nexact", color=viz.INK_2,
                fontsize=14, va="bottom")
        ax.set_xlabel("Leading-edge station, z / width")
        ax.set_ylabel("Max body deviation / L")
        ax.set_title("V5: osculating-cone shock, M 5, β 15°, 40 points")
        ax.legend(loc="lower right", fontsize=13)
        ax.set_ylim(1e-10, 1e-2)
        viz.save(fig, path)


def _n_ticks(ax):
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    ax.set_xlim(15, 260)
    ax.xaxis.set_major_locator(FixedLocator([20, 40, 80, 160]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())


def fig_convergence(conv, tw, path):
    with viz.style():
        fig, axes = viz.new_figure(1, 2, width=16.0, height=7.4)
        ax = axes[0, 0]
        ax.loglog(conv["n"], conv["v4"], marker=viz.MARKERS[0], color=viz.SERIES[0],
                  label="V4 vs Taylor–Maccoll")
        ax.loglog(conv["n5"], conv["v5"], marker=viz.MARKERS[1], color=viz.SERIES[1],
                  label="V5 vs exact per-plane flow")
        for k, (f, t) in enumerate(tw.items()):
            ax.loglog(t["n"][:-1], t["diff_next"], marker=viz.MARKERS[2], color=viz.SERIES[2],
                      ls="-" if k == 0 else "--",
                      label=f"Case II, station {f:.1f}: |P(n) − P(2n)| at base")
        n = np.array([20.0, 160.0])
        for order, y0, lab in ((2, 3e-6, "slope −2"), (1, 2e-3, "slope −1")):
            ax.loglog(n, y0 * (n / 20.0) ** -order, color=viz.INK_2, lw=1.2, ls=":")
            ax.text(n[1] * 1.04, y0 * (n[1] / 20.0) ** -order, lab, color=viz.INK_2,
                    va="center", fontsize=14)
        ax.set_xlabel("Shock points from the LE to the base, n")
        ax.set_ylabel("Body error / L")
        ax.set_title("Planar stream surfaces: 2nd order; twisted: 1st")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=14)
        _n_ticks(ax)

        ax = axes[0, 1]
        for k, (f, t) in enumerate(tw.items()):
            ax.loglog(t["n"], t["tangency_deg"], marker=viz.MARKERS[k], color=viz.SERIES[k],
                      label=f"R1 case II, station {f:.1f}")
        ax.loglog(n, 0.1 * (n / 20.0) ** -1, color=viz.INK_2, lw=1.2, ls=":")
        ax.text(n[1] * 1.05, 0.1 * (n[1] / 20.0) ** -1, "slope −1", color=viz.INK_2,
                va="center", fontsize=14)
        ax.set_xlabel("Shock points from the LE to the base, n")
        ax.set_ylabel("Max angle, body velocity to\nbody streamline tangent (deg)")
        ax.set_title("Step C: 3-D velocity tangent to the streamline")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=14)
        _n_ticks(ax)
        viz.save(fig, path)


def fig_stream_surface(path, frac=0.6, n_points=20):
    s, x_base, M, zf, y_tip = r1_shock("II")
    uv, _, _ = find_leading_edge(s, np.array([[frac * y_tip, zf]]))
    st = solve_stream_surface(s, uv[0], x_base, M, n_points=n_points)
    sol, cv, P3 = st.moc, st.curve, st.points3
    v = sol.valid
    with viz.style():
        fig, axes = viz.new_figure(1, 2, width=17.0, height=6.6)
        ax = axes[0, 0]
        for d in range(1, sol.x.shape[0], 2):
            ok = v[d]
            ax.plot(sol.x[d, ok], sol.y[d, ok], color=viz.MESH, lw=0.6)
        for j in range(0, sol.x.shape[1], 4):
            ok = v[:, j]
            ax.plot(sol.x[ok, j], sol.y[ok, j], color=viz.MESH, lw=0.6)
        ax.plot(cv.x, cv.y, color=viz.INK, lw=2.5, label="Shock curve (intrinsic meridian)")
        ax.plot(sol.x[v[:, 0], 0], sol.y[v[:, 0], 0], color=viz.SERIES[0], lw=2.5,
                label="Body streamline")
        ax.plot(cv.x, cv.y_axis, color=viz.SERIES[1], lw=2.0, ls="--",
                label="Local axis y − r (noncoaxial)")
        ax.axvline(x_base, color=viz.INK_2, ls=":", lw=1.5, label="Base plane")
        ax.set_xlabel("x = X")
        ax.set_ylabel("Meridian ordinate y")
        ax.set_title(f"Step B: meridian mesh ({n_points} points LE to base)")
        ax.legend(loc="upper left", fontsize=13)

        ax = axes[0, 1]
        # Frame of the LE's osculating plane: X and the in-plane shock normal n_cs.
        e3 = np.cross([1.0, 0.0, 0.0], cv.geometry.n_cs[0])
        e3 /= np.linalg.norm(e3)
        out = lambda P: (P - st.le_point) @ e3
        keep = cv.x <= x_base
        ax.plot(cv.x[keep], out(cv.points[keep]), color=viz.SERIES[1], lw=2.5,
                label="Shock curve")
        ax.plot(cv.x[~keep], out(cv.points[~keep]), color=viz.SERIES[1], lw=1.5, ls="--",
                label="Shock curve, extension past the base")
        ax.plot(st.body.x, out(st.body.points), color=viz.SERIES[0], lw=2.5,
                label="Body streamline")
        for k, xc in enumerate(np.linspace(st.le_point[0], x_base, 7)[1:]):
            sec = []
            for j in range(P3.shape[1]):
                c = P3[v[:, j], j]
                if c.shape[0] >= 2 and c[0, 0] <= xc <= c[-1, 0]:
                    sec.append(np.array([np.interp(xc, c[:, 0], c[:, q]) for q in range(3)]))
            if sec:
                sec = np.asarray(sec)
                ax.plot(sec[:, 0], out(sec), "o", color=viz.MESH, ms=3,
                        label="Stream surface at X = const" if k == 0 else None)
        ax.axhline(0.0, color=viz.INK_2, lw=1.2)
        ax.axvline(x_base, color=viz.INK_2, ls=":", lw=1.5, label="Base plane")
        ax.set_xlabel("X")
        ax.set_ylabel("Distance from the LE's\nosculating plane")
        ax.set_title("Step C: the stream surface turns out of plane")
        ax.legend(loc="upper left", fontsize=13)
        fig.suptitle(f"R1 case II shock (Eq. 25, M 7): stream surface from FCT station {frac}",
                     fontsize=18)
        viz.save(fig, path)


def fig_diagnostics(diag, path):
    panels = (
        ("azimuth_drift_le_to_base_deg", "Azimuth drift, LE to base (deg)", "Local turning of the shock curve"),
        ("axis_drift_le_to_base_over_L", "Axis-centre drift, LE to base / L", "Noncoaxial axis drift"),
        ("extension_needed_chords", "Shock extension past the base (chords)", "Determinacy (flag 4)"),
        ("body_step_ratio", "Max / min body step", "Body mesh stretching (flag 5)"),
    )
    with viz.style():
        fig, axes = viz.new_figure(2, 2, width=15.0, height=10.5)
        for ax, (key, ylab, title) in zip(axes.ravel(), panels):
            for k, (case, rows) in enumerate(diag.items()):
                f = [r["fct_fraction"] for r in rows]
                ax.plot(f, [r[key] for r in rows], marker=viz.MARKERS[k], color=viz.SERIES[k],
                        label=f"Case {case}" + (", needed" if key.startswith("extension") else ""))
                if key.startswith("extension"):
                    ax.plot(f, [r["extension_used_chords"] for r in rows], color=viz.SERIES[k],
                            ls="--", lw=1.5, label=f"Case {case}, used")
            ax.set_xlabel("FCT station, Y / Y_tip")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.legend(loc="best", fontsize=13)
        fig.suptitle("LTOCs diagnostics on the R1 waverider shocks (13 stations, 40 points)",
                     fontsize=18)
        viz.save(fig, path)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/ltoc/figures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {}

    d4 = v4(40)
    oc_wr = oc_design()
    d5 = v5(oc_wr, 40)
    results["v4"] = {"worst_vs_tm_over_L": max(d4["vs_tm"]),
                     "worst_vs_shadow_over_L": max(d4["vs_shadow"]), **d4}
    results["v5"] = {"worst_vs_ref_over_L": max(d5["vs_ref"]), "worst_vs_oc_over_L": max(d5["vs_oc"]),
                     "worst_oc_vs_ref_over_L": max(d5["oc_vs_ref"]), **d5}

    conv = {"n": [20, 40, 80, 160], "n5": [20, 40, 80]}
    conv["v4"] = [max(v4(n)["vs_tm"]) for n in conv["n"]]
    conv["v5"] = [max(v5(oc_wr, n)["vs_ref"]) for n in conv["n5"]]
    tw = twisted_convergence()
    results["convergence"] = {**conv, "case_II_twisted": {str(k): v for k, v in tw.items()}}

    diag, smooth, geom = {}, {}, {}
    for case in R1_CASES:
        wr, frac = r1_waverider(case)
        diag[case] = station_diagnostics(wr, frac)
        wr25, _ = r1_waverider(case, n_stations=25)
        smooth[case] = {"13_stations": wr.spanwise_smoothness(),
                        "25_stations": wr25.spanwise_smoothness()}
        mesh = wr.mesh(full_span=True)
        f = mesh.faces
        e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        geom[case] = {"length": wr.length, "width_half": wr.width, "height": wr.height,
                      "closed_mesh": bool(np.all(counts == 2)), "n_faces": int(mesh.n_faces)}
    results["r1_diagnostics"] = diag
    results["r1_spanwise_smoothness_over_L"] = smooth
    results["r1_geometry_gui_frame"] = geom

    fig_v4_v5(d4, d5, out / "gate3_v4_v5.png")
    fig_convergence(conv, tw, out / "gate3_convergence.png")
    fig_stream_surface(out / "gate3_stream_surface_case2.png")
    fig_diagnostics(diag, out / "gate3_diagnostics.png")
    results["run_time_s"] = time.time() - t0
    (out / "gate3_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("convergence", "r1_spanwise_smoothness_over_L",
                                              "r1_geometry_gui_frame", "run_time_s")}, indent=1))
    print("V4 worst:", results["v4"]["worst_vs_tm_over_L"], results["v4"]["worst_vs_shadow_over_L"])
    print("V5 worst:", results["v5"]["worst_vs_ref_over_L"], results["v5"]["worst_vs_oc_over_L"],
          results["v5"]["worst_oc_vs_ref_over_L"])


if __name__ == "__main__":
    main()
