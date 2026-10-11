"""Gate 4 deliverable: published cases V6-V9 (LTOCs Phase 4).

Run from the repository root::

    python -m ltoc.examples.gate4_published_cases --out docs/ltoc/figures

The script writes six PNG figures and ``gate4_results.json`` (about 4 min).

* V6: R1 waverider case I, Eq. (23), M 6, FCT Z = 0.12 (Eq. 24): R1 Table 2 and
  Fig. 23.
* V7: R1 waverider case II, Eq. (25), M 7, FCT Z = 0.25 (Eq. 26): R1 Table 3 and
  Fig. 27.
* V8: R1 test case II shock generator, Eqs. (16)-(18), M 7: R1 Figs. 16 and 17.
* V9: robustness inputs and the messages they produce.

R1 shapes are given in R1's frame (X freestream, Y span, Z down). Wall
pressures are compared with R1's own LTOCs curves, digitised into
``ltoc/data/r1_wall_pressure_digitised.json``. Forces follow R1 Eqs. (19)-(22)
with the planform reference area (see ``ltoc.forces``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ltoc import viz
from ltoc.forces import (body_grid, crosswise_section, lower_surface_grid, panel_forces,
                         streamwise_section)
from ltoc.ltoc import find_leading_edge, r1_frame_to_gui, solve_stream_surface
from ltoc.shock_surface import BSplineShock, ConeShock, RuledShock
from ltoc.waverider import LTOCWaverider

DATA = Path(__file__).resolve().parents[1] / "data" / "r1_wall_pressure_digitised.json"
R1 = {
    "I": {"shock": dict(x_p=0.0, a_p=0.0, b_p=0.0, x_q=1.7, a_q=0.595, b_q=0.68), "M": 6.0,
          "fct_z": 0.12, "table": {"CL": 0.2205, "CD": 0.0750, "L/D": 2.9378},
          "cfd": {"CL": 0.2201, "CD": 0.0749, "L/D": 2.9377}, "fig": "fig23",
          "label": "Waverider case I (R1 Eq. 23), M 6, FCT Z = 0.12"},
    "II": {"shock": dict(x_p=0.1, a_p=0.3, b_p=0.2, x_q=1.5, a_q=0.65, b_q=0.8), "M": 7.0,
           "fct_z": 0.25, "table": {"CL": 0.2034, "CD": 0.0648, "L/D": 3.1395},
           "cfd": {"CL": 0.2024, "CD": 0.0645, "L/D": 3.1386}, "fig": "fig27",
           "label": "Waverider case II (R1 Eq. 25), M 7, FCT Z = 0.25"},
}
STATIONS = (13, 25, 49, 97)


def shock_of(case):
    c = R1[case]
    s = RuledShock.elliptic(**c["shock"], m_range=(0.0, np.pi), n_range=(0.0, 1.0))
    y_tip = c["shock"]["a_q"] * np.sqrt(1.0 - (c["fct_z"] / c["shock"]["b_q"]) ** 2)
    return s, y_tip


def r1_waverider(case, n_stations, n_points=40):
    c = R1[case]
    s, y_tip = shock_of(case)
    u = np.linspace(0.0, 1.0, n_stations)
    ys = y_tip * (1.0 - np.cos(0.5 * np.pi * u))             # clustered at the symmetry plane
    return LTOCWaverider(s, np.column_stack([ys, np.full_like(ys, c["fct_z"])]),
                         c["shock"]["x_q"], c["M"], n_points=n_points, to_gui=r1_frame_to_gui,
                         strict=True)


# ---------------------------------------------------------------------------
#  Studies
# ---------------------------------------------------------------------------

def force_study(case):
    rows, best = [], None
    for nst, npt in [(n, 40) for n in STATIONS] + [(49, 80)]:
        t = time.time()
        wr = r1_waverider(case, nst, npt)
        f = panel_forces(lower_surface_grid(wr, 101), wr.M_inf)
        rows.append({"stations": nst, "n_points": npt, "CL": f.CL, "CD": f.CD, "L/D": f.L_over_D,
                     "CL_wetted": f.CL_wetted, "CD_wetted": f.CD_wetted,
                     "area_planform": f.area_planform, "area_wetted": f.area_wetted,
                     "seconds": time.time() - t})
        if nst == 49 and npt == 40:
            best = wr
    return rows, best


def sections(points, p, fig, digitised):
    out = {}
    for key, kind in ((fig + "a", "stream"), (fig + "b", "cross")):
        out[key] = {}
        for name, c in digitised[key].items():
            val = float(name.split("=")[1])
            fn = streamwise_section if kind == "stream" else crosswise_section
            xs, ps = fn(points, p, val)
            xd, pd = np.asarray(c["x"]), np.asarray(c["p"])
            inside = (xd >= xs.min()) & (xd <= xs.max())
            dev = np.interp(xd[inside], xs, ps) / pd[inside] - 1.0
            out[key][name] = {"x": xs.tolist(), "p": ps.tolist(), "r1_x": c["x"], "r1_p": c["p"],
                              "max_dev": float(np.max(np.abs(dev))),
                              "rms_dev": float(np.sqrt(np.mean(dev ** 2)))}
    return out


def v8_generator(n_stations=49):
    s = RuledShock.elliptic(0.1, 0.2, 0.3, 1.5, 0.8, 0.65, m_range=(0.0, np.pi), n_range=(0.0, 1.0))
    stations = [solve_stream_surface(s, (m, 0.0), 1.5, 7.0, n_points=40, station=k)
                for k, m in enumerate(np.linspace(0.5 * np.pi, 0.0, n_stations))]
    return s, stations


def v9_cases():
    rows = []

    def add(name, st):
        rows.append({"input": name, "status": st.status, "message": st.message})

    tb = np.tan(np.radians(15.0))
    vec = lambda f: (lambda m: np.stack(np.broadcast_arrays(*f(m)), -1))
    d1, d2 = vec(lambda m: (0.0, 1.0, 0.0)), vec(lambda m: (0.0, 0.0, 0.0))
    plane = RuledShock(vec(lambda m: (0.0, m, 0.0)), d1, d2, vec(lambda m: (1.0, m, tb)), d1, d2,
                       m_range=(-1.0, 1.0), n_range=(0.0, 1.0))
    uv, _, _ = find_leading_edge(plane, np.array([[0.2, 0.1]]))
    add("flat shock (plane, beta 15 deg, M 6)", solve_stream_surface(plane, uv[0], 1.0, 6.0, n_points=20))
    zero = lambda m: np.zeros(np.shape(m) + (3,))
    concave = RuledShock(zero, zero, zero, vec(lambda m: (1.0, m, 0.3 + 0.6 * m ** 2)),
                         vec(lambda m: (0.0, 1.0, 1.2 * m)), vec(lambda m: (0.0, 0.0, 1.2)),
                         m_range=(-0.4, 0.4))
    uv, _, _ = find_leading_edge(concave, np.array([[0.1, 0.25]]))
    add("concave cross-section", solve_stream_surface(concave, uv[0], 1.0, 6.0, n_points=20))
    for b, M, z, name in ((8.0, 6.0, 0.1, "cone beta 8 deg at M 6 (below the Mach angle)"),
                          (75.0, 3.0, 2.0, "cone beta 75 deg at M 3 (beyond detachment)")):
        cone = ConeShock(np.radians(b), x_range=(0.0, 2.0))
        uv, _, _ = find_leading_edge(cone, np.array([[0.0, z]]))
        add(name, solve_stream_surface(cone, uv[0], 1.0, M, n_points=20))
    cone = ConeShock(np.radians(20.0), x_range=(0.0, 2.0))
    uv, _, _ = find_leading_edge(cone, np.array([[0.0, 0.2]]))
    add("shock extension capped at 0.1 chords",
        solve_stream_surface(cone, uv[0], 1.0, 6.0, n_points=20, max_extension=0.1))
    short = BSplineShock.from_surface(ConeShock(np.radians(20.0), x_range=(0.0, 1.0)),
                                      u_range=(0.5, 2.5), v_range=(0.0, 1.0))
    uv, _, _ = find_leading_edge(short, np.array([[0.0, 0.2]]))
    add("B-spline shock data ending at the base plane",
        solve_stream_surface(short, uv[0], 1.0, 6.0, n_points=20))
    wr = LTOCWaverider(cone, np.array([[0.0, 0.5 * np.tan(np.radians(20.0))],
                                       [0.0, 3.0 * np.tan(np.radians(20.0))]]), 1.0, 6.0, n_points=20)
    st = wr.stations[1]
    rows.append({"input": "FCT point outside the shock at the base", "status": st.status,
                 "message": st.message})
    return rows


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------

def fig_geometry(case, wr, grid, path):
    c = R1[case]
    s, y_tip = shock_of(case)
    x_base = c["shock"]["x_q"]
    P, p = grid.points_native, grid.p
    with viz.style():
        fig = viz.new_figure(1, 1, width=17.0, height=8.2)[0]
        fig.clear()
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        # 3-D view, GUI frame (x streamwise, z span, y up).
        G = lambda Q: r1_frame_to_gui(Q)
        mm, nn = np.meshgrid(np.linspace(0.0, 0.5 * np.pi, 25), np.linspace(0.0, 1.0, 13))
        S = G(s.point(mm, nn))
        ax3.plot_wireframe(S[..., 0], S[..., 2], S[..., 1], color=viz.MESH, lw=0.4, alpha=0.6)
        Q = G(P)
        surf = ax3.plot_surface(Q[..., 0], Q[..., 2], Q[..., 1],
                                facecolors=viz.SEQ_BLUE((p - p.min()) / np.ptp(p)), linewidth=0,
                                antialiased=False, shade=False)
        for k, st in enumerate(wr.stations[::4]):
            if st.status == "tip":
                continue
            cv = st.curve
            keep = cv.x <= x_base
            C = G(cv.points[keep])
            ax3.plot(C[:, 0], C[:, 2], C[:, 1], color=viz.SERIES[1], lw=1.6,
                     label="Shock curves, LE to base" if k == 0 else None)
        LE = wr.leading_edge
        ax3.plot(LE[:, 0], LE[:, 2], LE[:, 1], color=viz.INK, lw=2.0, label="Leading edge")
        ax3.set_xlabel("x", labelpad=10)
        ax3.set_ylabel("z (span)", labelpad=10)
        ax3.set_zlabel("y (up)", labelpad=10)
        ax3.set_box_aspect((np.ptp(S[..., 0]), np.ptp(S[..., 2]), np.ptp(S[..., 1])))
        from matplotlib.ticker import MaxNLocator
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.set_major_locator(MaxNLocator(3))
        ax3.view_init(elev=-22, azim=-58)
        ax3.set_title("Shock, shock curves and lower surface")
        ax3.legend(loc="upper left", fontsize=13)
        sm = viz.mpl.cm.ScalarMappable(cmap=viz.SEQ_BLUE,
                                       norm=viz.mpl.colors.Normalize(p.min(), p.max()))
        cb = fig.colorbar(sm, ax=ax3, shrink=0.7, pad=0.08)
        cb.set_label("Wall pressure p / p∞")

        # Front view of the base plane, R1 frame (Z down).
        m = np.linspace(0.0, np.pi, 300)
        B = s.point(m, np.ones_like(m))
        ax2.plot(B[:, 1], B[:, 2], color=viz.INK, lw=2.5, label="Shock at the base (SWPC)")
        ax2.axhline(c["fct_z"], color=viz.INK_2, ls="--", lw=1.5, label=f"FCT, Z = {c['fct_z']}")
        x_le0 = float(P[0, 0, 0])
        for k, xc in enumerate(np.linspace(x_le0 + 0.25 * (x_base - x_le0), x_base, 4)):
            sec = []
            for row in P:
                if row[0, 0] <= xc <= row[-1, 0] and row[-1, 0] > row[0, 0]:
                    sec.append([np.interp(xc, row[:, 0], row[:, q]) for q in (1, 2)])
            sec = np.asarray(sec)
            ax2.plot(sec[:, 0], sec[:, 1], color=viz.SERIES[0], lw=2.0, alpha=0.4 + 0.2 * k,
                     label="Lower surface at X = const" if k == 0 else None)
            ax2.text(sec[0, 0] - 0.01, sec[0, 1], f"X = {xc:.2f}", ha="right", va="center",
                     fontsize=14, color=viz.INK_2)
        for k, st in enumerate(wr.stations[::4]):
            if st.status == "tip":
                continue
            cv = st.curve
            keep = cv.x <= x_base
            ax2.plot(cv.points[keep, 1], cv.points[keep, 2], color=viz.SERIES[1], lw=1.4)
        ax2.set_aspect("equal")
        ax2.set_xlim(-0.2, 1.05 * c["shock"]["a_q"] + 0.05)
        ax2.set_ylim(c["shock"]["b_q"] + 0.1, c["fct_z"] - 0.12)
        ax2.set_xlabel("Y")
        ax2.set_ylabel("Z (down, as in R1)")
        ax2.set_title("Front view, X projected")
        h3, l3 = ax3.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax3.get_legend().remove()
        fig.legend(h3 + h2, l3 + l2, loc="outside lower center", ncol=5, fontsize=13)
        fig.suptitle(c["label"] + ": LTOCs waverider", fontsize=18)
        viz.save(fig, path)


def _pressure_panels(axes, secs, keys, xlabels, titles, curves_order):
    for ax, key, xl, title in zip(axes, keys, xlabels, titles):
        for k, name in enumerate(curves_order[key]):
            d = secs[key][name]
            ax.plot(d["x"], d["p"], color=viz.SERIES[k], lw=2.0, label=f"LTOCs (this work): {name}")
            ax.plot(d["r1_x"][::2], d["r1_p"][::2], viz.MARKERS[k], color=viz.SERIES[k], mfc="none",
                    mew=1.6, ms=8, label=f"LTOCs (R1, digitised): {name}")
        ax.set_xlabel(xl)
        ax.set_ylabel("p / p∞")
        ax.set_title(title)


def fig_wall_pressure(case, secs, path):
    c = R1[case]
    fig_key = c["fig"]
    order = {k: list(secs[k].keys()) for k in secs}
    with viz.style():
        fig, axes = viz.new_figure(1, 2, width=17.0, height=7.4)
        _pressure_panels(axes[0], secs, [fig_key + "a", fig_key + "b"], ["X", "Y"],
                         ["a) Streamwise planes, Y = const", "b) Crosswise planes, X = const"], order)
        for ax in axes[0]:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=13)
        fig.suptitle(f"{c['label']}: wall pressure (laid out like R1 Fig. {fig_key[3:]})",
                     fontsize=18)
        viz.save(fig, path)


def fig_v8(secs, stations, path):
    order = {k: list(secs[k].keys()) for k in secs}
    with viz.style():
        fig, axes = viz.new_figure(1, 3, width=21.0, height=7.6)
        _pressure_panels(axes[0, :2], secs, ["fig16a", "fig16b"], ["X", "Y"],
                         ["a) Streamwise planes, Y = const", "b) Crosswise planes, X = const"], order)
        for ax in axes[0, :2]:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=13)
        ax = axes[0, 2]
        m = np.linspace(0.0, 0.5 * np.pi, 100)
        for X, (a, b), lab in ((0.1, (0.2, 0.3), "Shock at X = 0.1 (n = 0)"),
                               (1.5, (0.8, 0.65), "Shock at the base, X = 1.5")):
            ax.plot(a * np.cos(m), b * np.sin(m), color=viz.INK, lw=2.0 if X > 1 else 1.4,
                    ls="-" if X > 1 else ":", label=lab)
        for k, st in enumerate(stations[::3]):
            P = st.body.points
            ax.plot(P[:, 1], P[:, 2], color=viz.SERIES[0], lw=1.6,
                    label="Generator streamlines (LTOCs)" if k == 0 else None)
            ax.plot(P[-1, 1], P[-1, 2], "o", color=viz.SERIES[0], ms=5)
        ax.set_aspect("equal")
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_title("c) Streamlines on the generator (cf. R1 Fig. 17)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=13)
        fig.suptitle("V8: R1 test case II shock generator (Eq. 18), M 7", fontsize=18)
        viz.save(fig, path)


def fig_forces(study, path):
    with viz.style():
        fig, axes = viz.new_figure(1, 3, width=19.0, height=6.4)
        for ax, key in zip(axes[0], ("CL", "CD", "L/D")):
            for k, case in enumerate(("I", "II")):
                rows = [r for r in study[case] if r["n_points"] == 40]
                n = [r["stations"] for r in rows]
                err = [100.0 * (r[key] / R1[case]["table"][key] - 1.0) for r in rows]
                ax.plot(n, err, marker=viz.MARKERS[k], color=viz.SERIES[k],
                        label=f"Case {case} vs R1 LTOCs")
                cfd = 100.0 * (R1[case]["cfd"][key] / R1[case]["table"][key] - 1.0)
                ax.axhline(cfd, color=viz.SERIES[k], ls=":", lw=1.5,
                           label=f"Case {case}: R1 CFD vs R1 LTOCs")
            ax.axhspan(-1.0, 1.0, color=viz.GRID, alpha=0.45, lw=0)
            ax.axhline(0.0, color=viz.INK_2, lw=1.0)
            ax.set_xscale("log")
            ax.set_xticks(list(STATIONS))
            ax.set_xticklabels([str(n) for n in STATIONS])
            ax.minorticks_off()
            ax.set_ylim(-2.0, 2.0)
            ax.set_xlabel("FCT stations (half span)")
            ax.set_ylabel(f"{key} error vs R1 (%)")
            ax.set_title(f"{key}: grey band = 1 % target")
        axes[0, 0].legend(loc="lower right", fontsize=13)
        fig.suptitle("V6, V7: lift and drag against R1 Tables 2 and 3 (40 points per stream surface)",
                     fontsize=18)
        viz.save(fig, path)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/ltoc/figures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digitised = json.loads(DATA.read_text())
    t0 = time.time()
    results = {"forces": {}, "sections": {}, "tables": {}}

    for case in R1:
        rows, wr = force_study(case)
        results["forces"][case] = rows
        best = [r for r in rows if r["stations"] == 97][0]
        results["tables"][case] = {
            "r1_ltocs": R1[case]["table"], "r1_cfd": R1[case]["cfd"],
            "this_work_97_stations": {k: best[k] for k in ("CL", "CD", "L/D")},
            "error_percent": {k: 100.0 * (best[k] / R1[case]["table"][k] - 1.0)
                              for k in ("CL", "CD", "L/D")},
            "wetted_reference": {"CL": best["CL_wetted"], "CD": best["CD_wetted"]},
        }
        grid = lower_surface_grid(wr, 201)
        secs = sections(grid.points_native, grid.p, R1[case]["fig"], digitised)
        results["sections"][case] = {k: {n: {q: v[q] for q in ("max_dev", "rms_dev")}
                                         for n, v in d.items()} for k, d in secs.items()}
        fig_geometry(case, wr, grid, out / f"gate4_case{'1' if case == 'I' else '2'}_geometry.png")
        fig_wall_pressure(case, secs, out / f"gate4_case{'1' if case == 'I' else '2'}_wall_pressure.png")

    shock8, stations8 = v8_generator()
    pts, p = body_grid(stations8, 201)
    secs8 = sections(pts, p, "fig16", digitised)
    results["sections"]["V8"] = {k: {n: {q: v[q] for q in ("max_dev", "rms_dev")}
                                     for n, v in d.items()} for k, d in secs8.items()}
    results["v8_statuses"] = sorted({st.status for st in stations8})
    fig_v8(secs8, stations8, out / "gate4_v8_generator.png")
    fig_forces(results["forces"], out / "gate4_forces_convergence.png")
    results["v9"] = v9_cases()
    results["run_time_s"] = time.time() - t0
    (out / "gate4_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("tables", "sections", "v9", "run_time_s")}, indent=1))


if __name__ == "__main__":
    main()
