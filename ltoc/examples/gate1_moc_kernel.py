"""Gate 1 deliverable: verification figures and tables for the MOC kernel.

Run from the repository root::

    python -m ltoc.examples.gate1_moc_kernel --out docs/ltoc/figures

This writes four PNG figures and ``gate1_results.json``, and prints the tables
used in ``docs/ltoc/gate1_report.md``. Runtime is about two minutes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ltoc import viz
from ltoc.moc_noncoaxial import (
    SCHEMES, AxisRule, CHARACTERISTIC_FOLD, UNDETERMINED, shock_initial_line, solve_inverse,
)
from ltoc.reference import ConicalFlowReference

CONES = [(6.0, 20.0), (10.0, 14.0)]
X0, X1 = 0.25, 1.0
NS = [25, 50, 100, 200, 400, 800]
LABEL = {"streamline": "Streamline mesh", "characteristic": "Characteristic net"}


def cone_line(M, beta_deg, N, y_shift=0.0):
    b = np.radians(beta_deg)
    x = np.linspace(X0, X1, N + 1)
    return shock_initial_line(x, x * np.tan(b) + y_shift, b, M)


def curved_shock(N, b0, b1, ramp="smooth", x0=X0, x1=X1, y0=0.09):
    if ramp == "smooth":
        def beta(xx):
            s = (xx - x0) / (x1 - x0)
            return np.radians(b0 + (b1 - b0) * (3 * s ** 2 - 2 * s ** 3))
    else:
        def beta(xx):
            return np.radians(b0 + (b1 - b0) * 0.5 * (1 + np.tanh((xx - 0.5) / 0.025)))
    x = np.linspace(x0, x1, N + 1)
    xf = np.linspace(x0, x1, 40001)
    yf = y0 + cumulative_trapezoid(np.tan(beta(xf)), xf, initial=0.0)
    return x, np.interp(x, xf, yf), beta(x)


def body(sol):
    return sol.streamline_column(0) if sol.scheme == "streamline" else sol.trace_streamline(0, 0)


# ---------------------------------------------------------------------------
#  Studies
# ---------------------------------------------------------------------------

def study_v1():
    out = {}
    for M, bdeg in CONES:
        ref = ConicalFlowReference(M, np.radians(bdeg))
        for scheme in SCHEMES:
            rows = []
            for N in NS:
                il = cone_line(M, bdeg, N)
                t = time.perf_counter()
                sol = solve_inverse(il, AxisRule.coaxial(), scheme=scheme)
                dt = time.perf_counter() - t
                _, _, Mn, _ = sol.derived()
                st = ref.state_at(sol.x, sol.y)
                ok = sol.valid & np.isfinite(st["p"])
                sl = body(sol)
                y_ref = ref.streamline(sl.x[0], sl.y[0], sl.x[-1])(sl.x)
                rows.append({
                    "N": N, "time_s": dt, "points": int(ok.sum()),
                    "err_p": float(np.max(np.abs(sol.p[ok] / st["p"][ok] - 1))),
                    "err_M": float(np.max(np.abs(Mn[ok] / st["M"][ok] - 1))),
                    "err_theta": float(np.max(np.abs(sol.theta[ok] / st["theta"][ok] - 1))),
                    "body_dy": float(np.max(np.abs(sl.y - y_ref))),
                    "body_dp": float(np.max(np.abs(sl.p / ref.state_at(sl.x, y_ref)["p"] - 1))),
                    "body_x_end": float(sl.x[-1]),
                })
            out[f"M{M:g}_b{bdeg:g}_{scheme}"] = rows
    return out


def study_v2():
    out = {}
    for scheme in SCHEMES:
        il = cone_line(6.0, 20.0, 200)
        sol = solve_inverse(il, AxisRule.planar(), scheme=scheme, strict=True)
        v = sol.valid
        out[scheme] = {
            "points": int(v.sum()),
            "max_dp": float(np.max(np.abs(sol.p[v] / il.p[0] - 1))),
            "max_dtheta": float(np.max(np.abs(sol.theta[v] - il.theta[0]))),
        }
    return out


def study_rotational():
    out = {}
    for name, rule in (("planar", AxisRule.planar()), ("coaxial", AxisRule.coaxial())):
        curves = {s: [] for s in SCHEMES}
        mids = {s: [] for s in SCHEMES}
        for N in [50, 100, 200, 400, 800, 1600]:
            x, y, beta = curved_shock(N, 22.0, 18.0)
            il = shock_initial_line(x, y, beta, 6.0)
            sols = {s: solve_inverse(il, rule, scheme=s, strict=True) for s in SCHEMES}
            b_s = sols["streamline"].streamline_column(0)
            xs = np.linspace(0.3, 0.9 * b_s.x[-1], 40)
            yq = 0.5 * (np.interp(xs, b_s.x, b_s.y) + np.interp(xs, x, y))
            for s in SCHEMES:
                curves[s].append((N, body(sols[s])))
                mids[s].append(sols[s].interpolate("p", xs, yq))
        x_end = min(c.x[-1] for s in SCHEMES for _, c in curves[s])
        xg = np.linspace(0.26, 0.97 * x_end, 200)
        res = {}
        for s in SCHEMES:
            P = [np.interp(xg, c.x, c.p) for _, c in curves[s]]
            Yb = [np.interp(xg, c.x, c.y) for _, c in curves[s]]
            res[s] = {
                "N": [n for n, _ in curves[s]][:-1],
                "body_p_change": [float(np.max(np.abs(P[i + 1] / P[i] - 1))) for i in range(len(P) - 1)],
                "body_y_change": [float(np.max(np.abs(Yb[i + 1] - Yb[i]))) for i in range(len(P) - 1)],
                "mid_p_change": [float(np.nanmax(np.abs(mids[s][i + 1] / mids[s][i] - 1)))
                                 for i in range(len(P) - 1)],
            }
        res["scheme_difference_mid_p_N1600"] = float(np.nanmax(np.abs(mids["streamline"][-1]
                                                                     / mids["characteristic"][-1] - 1)))
        res["scheme_difference_body_p_N1600"] = float(np.max(np.abs(
            np.interp(xg, curves["streamline"][-1][1].x, curves["streamline"][-1][1].p)
            / np.interp(xg, curves["characteristic"][-1][1].x, curves["characteristic"][-1][1].p) - 1)))
        out[name] = res
    return out


def study_limit_surface():
    x, y, beta = curved_shock(200, 28.0, 14.0, ramp="steep")
    il = shock_initial_line(x, y, beta, 6.0)
    out = {}
    for s in SCHEMES:
        f = solve_inverse(il, AxisRule.planar(), scheme=s).first_failure()
        out[s] = {"x": f["x"], "y": f["y"], "reason": f["reason"]}
    return out


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------

def _n_ticks(ax, ns):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(n) for n in ns])


def _slope2(ax, ns, e0):
    ns = np.asarray(ns, dtype=float)
    ax.plot(ns, e0 * (ns / ns[0]) ** -2, ls="--", color=viz.INK_2, lw=1.4, label="Slope −2")


def fig_v1_convergence(v1, path):
    with viz.style():
        fig, axes = viz.new_figure(2, 2, width=13.0, height=10.0, sharex=True, sharey=True)
        for ci, (M, bdeg) in enumerate(CONES):
            for ri, scheme in enumerate(("streamline", "characteristic")):
                ax = axes[ri, ci]
                rows = v1[f"M{M:g}_b{bdeg:g}_{scheme}"]
                ns = [r["N"] for r in rows]
                for k, (key, lab) in enumerate((("err_p", "Pressure"), ("err_M", "Mach number"),
                                               ("err_theta", "Flow angle"))):
                    ax.loglog(ns, [r[key] for r in rows], marker=viz.MARKERS[k], color=viz.SERIES[k],
                              label=lab)
                _slope2(ax, ns, 1.6 * rows[0]["err_p"])
                ax.set_title(f"{LABEL[scheme]}, M = {M:g}, β = {bdeg:g}°")
                if ri == 1:
                    ax.set_xlabel("Shock points N")
                if ci == 0:
                    ax.set_ylabel("Max relative error vs Taylor–Maccoll")
                ax.axhline(1e-3, color=viz.INK_2, lw=1.0, ls=":")
                ax.text(ns[-1], 7e-4, "V1 limit 0.1 %", ha="right", va="top", color=viz.INK_2,
                        fontsize=14)
                ax.set_ylim(3e-9, 2e-3)
                _n_ticks(ax, ns)
        axes[0, 0].legend(loc="lower left")
        fig.suptitle("V1: coaxial MOC against Taylor–Maccoll, conical shock from x = 0.25 to 1",
                     fontsize=18)
        viz.save(fig, path)


def _draw_mesh(ax, sol, ref_line=None, cone_angle=None):
    """Draw the mesh as height below the shock, y - y_shock(x), to open up the layer."""
    v = sol.valid
    n_rows, n_cols = sol.x.shape
    xs0, ys0 = sol.x[0], sol.y[0]

    def h(x, y):
        return y - np.interp(x, xs0, ys0)

    # Columns: streamlines (streamline mesh) or C- lines (characteristic net).
    for k in range(n_cols):
        d = np.arange(min(n_rows, n_cols - k))
        m = v[d, k]
        ax.plot(np.where(m, sol.x[d, k], np.nan), np.where(m, h(sol.x[d, k], sol.y[d, k]), np.nan),
                color=viz.MESH, lw=0.9)
    # Anti-diagonals: C+ lines for both schemes.
    for i in range(n_cols):
        d = np.arange(0, min(i, n_rows - 1) + 1)
        k = i - d
        m = v[d, k]
        ax.plot(np.where(m, sol.x[d, k], np.nan), np.where(m, h(sol.x[d, k], sol.y[d, k]), np.nan),
                color=viz.MESH, lw=0.9, ls=(0, (3, 2)))
    und = sol.exists & (sol.status == UNDETERMINED)
    if und.any():
        ax.plot(sol.x[und], h(sol.x[und], sol.y[und]), "x", color=viz.SERIES[1], ms=8, mew=2,
                label="Entropy undetermined")
    ax.plot(xs0, 0.0 * xs0, color=viz.INK, lw=3.0, label="Shock")
    b = body(sol)
    ax.plot(b.x, h(b.x, b.y), color=viz.SERIES[0], lw=3.0, label="Body streamline (MOC)")
    if ref_line is not None:
        xr, yr = ref_line
        ax.plot(xr, h(xr, yr), color=viz.SERIES[2], lw=2.0, ls="--", label="Body streamline (T–M)")
    if cone_angle is not None:
        xx = np.linspace(xs0[0], xs0[-1], 50)
        ax.plot(xx, h(xx, xx * np.tan(cone_angle)), color=viz.INK_2, lw=1.5, ls=":",
                label="Cone surface")
    return h


def fig_meshes(path):
    M, bdeg = 6.0, 20.0
    ref = ConicalFlowReference(M, np.radians(bdeg))
    with viz.style():
        fig, axes = viz.new_figure(1, 2, width=15.0, height=6.2, sharey=True)
        for ax, scheme in zip(axes[0], ("streamline", "characteristic")):
            sol = solve_inverse(cone_line(M, bdeg, 12), AxisRule.coaxial(), scheme=scheme)
            b = body(sol)
            xr = np.linspace(b.x[0], b.x[-1], 200)
            _draw_mesh(ax, sol, (xr, ref.streamline(b.x[0], b.y[0], b.x[-1])(xr)), ref.cone_angle)
            ax.set_title(LABEL[scheme])
            ax.set_xlabel("x (meridian plane)")
            ax.set_xlim(0.22, 1.02)
            ax.set_ylim(-0.085, 0.006)
        axes[0, 0].set_ylabel("Height below shock, y − y_shock(x)")
        axes[0, 0].legend(*axes[0, 1].get_legend_handles_labels(), loc="lower left")
        fig.suptitle("Inverse MOC meshes, Mach 6, β = 20° conical shock, N = 12 "
                     "(solid: streamlines | C−, dashed: C+)", fontsize=17)
        viz.save(fig, path)


def fig_rotational(rot, path):
    with viz.style():
        fig, axes = viz.new_figure(1, 2, width=13.0, height=5.8, sharey=True)
        for ax, name in zip(axes[0], ("planar", "coaxial")):
            for k, scheme in enumerate(("streamline", "characteristic")):
                r = rot[name][scheme]
                ax.loglog(r["N"], r["body_p_change"], marker=viz.MARKERS[k], color=viz.SERIES[k],
                          label=f"{LABEL[scheme]}, body p")
            r = rot[name]["streamline"]
            ax.loglog(r["N"], r["mid_p_change"], marker=viz.MARKERS[2], color=viz.SERIES[2],
                      label="Streamline mesh, mid-layer p")
            _slope2(ax, r["N"], 1.6 * max(rot[name]["streamline"]["body_p_change"][0],
                                           rot[name]["characteristic"]["body_p_change"][0]))
            ax.set_title("Planar (δ = 0)" if name == "planar" else "Axisymmetric (coaxial)")
            _n_ticks(ax, r["N"])
            ax.set_xlabel("Shock points N")
        axes[0, 0].set_ylabel("Max |p(2N)/p(N) − 1|")
        axes[0, 0].legend(loc="lower left", fontsize=14)
        fig.suptitle("Rotational flow: curved shock, β 22° → 18°, Mach 6 (self-convergence)",
                     fontsize=18)
        viz.save(fig, path)


def fig_limit_surface(path):
    x, y, beta = curved_shock(60, 28.0, 14.0, ramp="steep")
    il = shock_initial_line(x, y, beta, 6.0)
    sol = solve_inverse(il, AxisRule.planar(), scheme="streamline")
    f = sol.first_failure()
    with viz.style():
        fig, axes = viz.new_figure(1, 2, width=15.0, height=6.0, gridspec_kw={"width_ratios": [1, 2]})
        ax = axes[0, 0]
        xf = np.linspace(X0, X1, 400)
        _, _, bf = curved_shock(399, 28.0, 14.0, ramp="steep")
        ax.plot(xf, np.degrees(bf), color=viz.INK, lw=2.5)
        ax.set_xlabel("x")
        ax.set_ylabel("Shock angle β (deg)")
        ax.set_title("Prescribed shock")
        ax = axes[0, 1]
        h = _draw_mesh(ax, sol)
        fold = sol.exists & (sol.status == CHARACTERISTIC_FOLD)
        ax.plot(sol.x[fold], h(sol.x[fold], sol.y[fold]), "X", color=viz.SERIES[1], ms=11, mew=1.5,
                label="Limit surface detected")
        hf = float(h(f["x"], f["y"]))
        ax.annotate(f"first crossing\nx = {f['x']:.3f}", (f["x"], hf),
                    xytext=(0.40, -0.0165), color=viz.INK,
                    arrowprops={"arrowstyle": "->", "color": viz.INK_2})
        ax.set_xlabel("x")
        ax.set_ylabel("Height below shock, y − y_shock(x)")
        ax.set_title("Streamline mesh, planar, N = 60")
        ax.legend(loc="lower right")
        fig.suptitle("Flag 6: a shock that weakens too fast needs crossing C+ characteristics",
                     fontsize=18)
        viz.save(fig, path)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/ltoc/figures")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from an existing gate1_results.json")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.figures_only and (out / "gate1_results.json").exists():
        results = json.loads((out / "gate1_results.json").read_text())
    else:
        results = {"v1": study_v1(), "v2": study_v2(), "rotational": study_rotational(),
                   "limit_surface": study_limit_surface()}
        (out / "gate1_results.json").write_text(json.dumps(results, indent=1))

    fig_v1_convergence(results["v1"], out / "gate1_v1_convergence.png")
    fig_meshes(out / "gate1_meshes.png")
    fig_rotational(results["rotational"], out / "gate1_rotational_convergence.png")
    fig_limit_surface(out / "gate1_limit_surface.png")

    print("V1 (max relative error vs tight Taylor-Maccoll):")
    for key, rows in results["v1"].items():
        print(f"  {key}")
        for r in rows:
            print(f"    N={r['N']:4d} p {r['err_p']:.2e} M {r['err_M']:.2e} theta {r['err_theta']:.2e}"
                  f"  body dy {r['body_dy']:.2e} dp {r['body_dp']:.2e}  {r['time_s']:.2f} s")
    print("V2:", results["v2"])
    print("Rotational:", json.dumps(results["rotational"], indent=1))
    print("Limit surface:", results["limit_surface"])


if __name__ == "__main__":
    main()
