"""Gate 2 deliverable: shock-geometry figures and numbers (LTOCs Phase 2).

Run from the repository root::

    python -m ltoc.examples.gate2_shock_geometry --out docs/ltoc/figures

The script writes three PNG figures and ``gate2_results.json``.

The R1 waverider shocks are drawn in R1's frame, in which Z points down (R1
Figs. 20, 21 and 24). The flow capture tube (FCT) is the line Z = const in
the base plane, and the leading edge is the FCT projected upstream onto the
shock (R1 Sec. IV.A). In this script the leading-edge points are found with a
2x2 root solve, purely to start the shock curves for the figures; Phase 3
builds the leading edge properly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve

from ltoc import viz
from ltoc.shock_geometry import local_geometry, trace_shock_curves
from ltoc.shock_surface import OsculatingConeShock, RuledShock

CASES = {
    "I": {"shock": dict(x_p=0.0, a_p=0.0, b_p=0.0, x_q=1.7, a_q=0.595, b_q=0.68),
          "M": 6.0, "fct_z": 0.12, "label": "R1 waverider case I (Eq. 23), M = 6"},
    "II": {"shock": dict(x_p=0.1, a_p=0.3, b_p=0.2, x_q=1.5, a_q=0.65, b_q=0.8),
           "M": 7.0, "fct_z": 0.25, "label": "R1 waverider case II (Eq. 25), M = 7"},
}


def leading_edge(surface, fct_z, y_values, guess_n=0.5):
    """(m, n) where the freestream line through (Y, fct_z) meets the shock."""
    out = []
    for y in y_values:
        def f(q):
            P = surface.point(q[0], q[1])
            return [P[1] - y, P[2] - fct_z]

        m, n = fsolve(f, [np.arctan2(fct_z, max(y, 1e-3)), guess_n], xtol=1e-13)
        out.append((m, n))
    return np.array(out)


def study(case):
    c = CASES[case]
    s = RuledShock.elliptic(**c["shock"], m_range=(0.0, np.pi), n_range=(0.0, 1.0))
    x_base = c["shock"]["x_q"]
    a_base, b_base = c["shock"]["a_q"], c["shock"]["b_q"]
    y_tip = a_base * np.sqrt(1.0 - (c["fct_z"] / b_base) ** 2)
    ys = np.linspace(0.0, 0.97 * y_tip, 9)
    le = leading_edge(s, c["fct_z"], ys)
    curves = trace_shock_curves(s, le[:, 0], le[:, 1], x_base, n_steps=200, M_inf=c["M"])
    # beta and r maps over the parameter plane (m, n) -> (azimuth, X).
    m = np.linspace(0.02, np.pi - 0.02, 181)
    n = np.linspace(0.05, 1.0, 120)
    Mg, Ng = np.meshgrid(m, n)
    g = local_geometry(s, Mg, Ng)
    return s, curves, (Mg, g.point[..., 0], np.degrees(g.beta), g.r), ys


def fig_maps(data, path):
    with viz.style():
        fig, axes = viz.new_figure(2, 2, width=14.0, height=10.5)
        for j, case in enumerate(("I", "II")):
            _, curves, (Mg, X, beta, r), _ = data[case]
            for i, (field, label, cmap) in enumerate(((beta, "Shock angle β (deg)", "Blues"),
                                                      (r, "Local crosswise radius r", "Blues"))):
                ax = axes[i, j]
                cs = ax.contourf(np.degrees(Mg), X, field, levels=14, cmap=cmap)
                cb = fig.colorbar(cs, ax=ax)
                cb.set_label(label)
                for cv in curves:
                    ax.plot(np.degrees(cv.uv[:, 0]), cv.x, color=viz.SERIES[1], lw=1.6)
                ax.set_xlabel("Azimuth m (deg)")
                ax.set_ylabel("X")
                ax.set_title(f"Case {case}: {label.split(' (')[0]}")
        axes[0, 0].plot([], [], color=viz.SERIES[1], lw=1.6, label="Shock curves from the LE")
        axes[0, 0].legend(loc="lower right")
        fig.suptitle("β and r on the R1 waverider shocks (R1 frame; m = 90° is the Z axis)",
                     fontsize=18)
        viz.save(fig, path)


def fig_curves(data, path):
    s, curves, _, ys = data["II"]
    with viz.style():
        fig, axes = viz.new_figure(1, 3, width=19.0, height=6.2)
        ax = axes[0, 0]
        mm = np.linspace(0.0, np.pi, 200)
        for nn, lab in ((1.0, "Shock at the base plane"),):
            P = s.point(mm, np.full_like(mm, nn))
            ax.plot(P[:, 1], P[:, 2], color=viz.INK, lw=2.5, label=lab)
        ax.axhline(CASES["II"]["fct_z"], color=viz.INK_2, ls="--", lw=1.5, label="FCT, Z = 0.25")
        for k, cv in enumerate(curves):
            P = cv.points
            ax.plot(P[:, 1], P[:, 2], color=viz.SERIES[1], lw=2.0,
                    label="Shock curves (LTOCs)" if k == 0 else None)
            m0, n0 = cv.uv[0]
            ruling = s.point(np.full(2, m0), np.array([n0, 1.0]))
            ax.plot(ruling[:, 1], ruling[:, 2], color=viz.SERIES[0], lw=1.2, ls=":",
                    label="Straight ruling through the LE" if k == 0 else None)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("Y")
        ax.set_ylabel("Z (down, as in R1)")
        ax.set_title("Front view, X projected")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=13, framealpha=0.9,
                  frameon=True)
        ax.set_ylim(0.85, -0.45)

        ax = axes[0, 1]
        for k, cv in enumerate(curves):
            ax.plot(cv.x, np.degrees(cv.uv[:, 0] - cv.uv[0, 0]), color=viz.SERIES[0], lw=1.8)
        ax.set_xlabel("X")
        ax.set_ylabel("Azimuth drift m − m_LE (deg)")
        ax.set_title("Local turning of the shock curves")

        ax = axes[0, 2]
        for k, cv in enumerate(curves):
            ax.plot(cv.x, cv.y_axis - cv.y_axis[0], color=viz.SERIES[0], lw=1.8)
        ax.set_xlabel("X (meridian plane)")
        ax.set_ylabel("Axis-centre height y_O − y_O,LE")
        ax.set_title("Noncoaxial axis-centre drift")
        fig.suptitle("R1 waverider case II shock: 9 stream surfaces from the leading edge",
                     fontsize=18)
        viz.save(fig, path)


def oc_study():
    from waverider_generator.generator import waverider

    wr = waverider(M_inf=5.0, beta=15.0, height=1.34, width=3.0, dp=[0.11, 0.63, 0.0, 0.46],
                   n_upper_surface=10000, n_shockwave=10000, n_planes=20, n_streamwise=30,
                   delta_streamwise=0.1)
    oc = OsculatingConeShock.from_oc_waverider(wr)
    L = wr.length
    z = np.asarray(wr.z_local_shockwave)
    g = local_geometry(oc, z, np.full_like(z, L), length_scale=L)
    curved = np.isfinite(g.r)
    centres = np.asarray(wr.cone_centers)
    err = float(np.max(np.abs(g.axis_centre[curved, 1:] - centres[curved, 1:])))
    return wr, oc, g, curved, centres, err


def fig_oc(oc_data, path):
    wr, oc, g, curved, centres, err = oc_data
    L = wr.length
    with viz.style():
        fig, axes = viz.new_figure(1, 1, width=13.0, height=6.5)
        ax = axes[0, 0]
        zz = np.linspace(0.0, wr.width, 300)
        P = oc.point(zz, np.full_like(zz, L))
        ax.plot(P[:, 2], P[:, 1], color=viz.INK, lw=2.5, label="Shock-wave profile curve (base)")
        ax.plot(centres[curved, 2], centres[curved, 1], "o", color=viz.SERIES[1], ms=11,
                label="OC generator cone centres")
        ax.plot(g.axis_centre[curved, 2], g.axis_centre[curved, 1], "^", color=viz.SERIES[0], ms=8,
                label="LTOCs axis centres (r = cos β / κ_b)")
        for k in np.flatnonzero(curved):
            ax.plot([g.point[k, 2], g.axis_centre[k, 2]], [g.point[k, 1], g.axis_centre[k, 1]],
                    color=viz.MESH, lw=0.8)
        H = wr.height
        ax.set_xlim(-0.05 * wr.width, 1.05 * wr.width)
        ax.set_ylim(-1.3 * H, 2.2 * H)
        ax.set_xlabel("z (span)")
        ax.set_ylabel("y (up)")
        ax.set_aspect("equal")
        ax.set_title(f"OC waverider shock (M 5, β 15°): max centre difference {err:.1e} m")
        ax.text(0.02 * wr.width, 2.05 * H, "Centres near the flat-to-curved junction lie\n"
                "far above the view (curvature → 0)", va="top", color=viz.INK_2, fontsize=14)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        viz.save(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/ltoc/figures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {case: study(case) for case in CASES}
    oc_data = oc_study()
    fig_maps(data, out / "gate2_beta_r_maps.png")
    fig_curves(data, out / "gate2_shock_curves_case2.png")
    fig_oc(oc_data, out / "gate2_oc_shock.png")

    results = {}
    for case, (s, curves, (Mg, X, beta, r), ys) in data.items():
        drift = [float(np.degrees(cv.uv[-1, 0] - cv.uv[0, 0])) for cv in curves]
        ax_drift = [float(cv.y_axis[-1] - cv.y_axis[0]) for cv in curves]
        results[case] = {
            "beta_deg_range": [float(np.min(beta)), float(np.max(beta))],
            "r_range": [float(np.min(r)), float(np.max(r))],
            "le_Y": ys.tolist(),
            "azimuth_drift_deg": drift,
            "axis_centre_drift": ax_drift,
            "flags": {k: int(sum(cv.flags[k].sum() for cv in curves)) for k in curves[0].flags},
        }
    results["oc_centre_error"] = oc_data[5]
    (out / "gate2_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
