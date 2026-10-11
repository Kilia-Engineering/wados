# LTOCs waverider: user guide

The **Local-Turning Osculating Cones (LTOCs)** method (Zheng et al. 2020, R1) designs a waverider from a prescribed three-dimensional shock surface. The osculating-cone method assumes the flow behind each leading-edge point stays in one plane. LTOCs instead follows the shock curve as it turns locally. That makes shocks possible that osculating methods cannot handle, such as an FCT below some of the local curvature centres (R1 Sec. IV.B).

In WADOS it is the **LTOCs Waverider** tab (`ltoc_waverider_tab.py`) and the `ltoc` library.

---

## 1. Using the tab

**Inputs (left panel)**

| Input | Meaning |
|---|---|
| Shock surface | One of four kinds:<br>• **Presets:** the shocks published by Zheng et al. (2020): *Elliptic cone shock, Mach 6 (published case I)*, *Shape-changing elliptic shock, Mach 7 (published case II)* and *(published test shock)*. Also *Circular cone shock 12°, Mach 6*. The text under the list describes each one and gives its published lift and drag.<br>• **Custom elliptic shock (two ellipses):** straight lines join matching points of an upstream ellipse and a base ellipse. You set each ellipse's X position, half-width (Y) and half-height (Z); the base ellipse's X is the base plane (R1 Eqs. 12, 15).<br>• **Shock from a data file (CSV):** any shock given as points; see §1.1.<br>• **Import the current OC design** from the OC Waverider tab. |
| Coordinates | X downstream, Y across the span (symmetry plane Y = 0), **Z downward**: the body hangs below the FCT. Exports and the 3-D view use the WADOS frame (x downstream, y up, z span). |
| Mach number | Design Mach number (γ = 1.4). |
| FCT height Z | The flow capture tube (FCT) is the straight line Z = const in the base plane that outlines the top of the waverider (R1 Eqs. 24, 26). The leading edge is the FCT projected upstream onto the shock. It must cross the shock trace in the base plane. |
| FCT stations | Stream surfaces across the half span. 25 (default) reproduces R1 Tables 2 and 3 to 0.4 %. |
| Station spacing | Clustered at the symmetry plane (default) or uniform. Clustering matters when the FCT meets the shock near its apex (R1 case I). |
| Points per stream surface | Shock points between the leading edge and the base. 40 (default) converges the forces to < 0.03 %; the cost grows with its square. |

Editing a preset's numbers turns it into a custom shock. With the OC import, the shock, Mach number and leading edge come from the OC design, and LTOCs reproduces that design (validation V5).

### 1.1 Shock data files

*Shock from a data file (CSV)* reads shock points on a structured grid and joins them into a smooth surface (an interpolating bicubic spline, `BSplineShock`). *Save shock as CSV...* writes the current shock in this format, so a preset makes a good starting template.

```
# LTOCs shock surface data (WADOS)
# x_base = 1.5        optional: fills "Base plane X"
# M_inf = 7           optional: fills "Mach number"
# fct_z = 0.25        optional: fills "FCT height Z"
u,v,X,Y,Z
0.0,0.0,0.1,0.3,0.0
0.0,0.0625,0.1875,0.3219,0.0
...
```

- **Grid.** One row per point of a full u × v grid, at least 4 × 4. u runs across the span and v downstream; their values only label the grid (any increasing numbers).
- **Columns.** The header names the columns in any order. Without a header the order is u, v, X, Y, Z. Lines starting with `#` are comments.
- **Coordinates** as above: X downstream, Y across the span, Z downward.
- **Base plane.** Set *Base plane X* inside the data. The points must continue **past** the base plane: the body at the base depends on the shock downstream of it, typically 0.5 to 1.5 local chords. Data are never extrapolated; a station whose shock data end too early is refused as `undetermined`.
- **Errors** name the line and the problem (a missing column, a non-number, an incomplete grid).

Shock data are embedded in project files, so a project does not depend on the CSV file. R1 case II through a 61 × 41 data file reproduces the analytic result to better than 0.01 % in C_L and C_D.

**Run.** *Generate LTOCs Waverider* solves one stream surface per station in a background thread, with a progress bar; *Cancel* stops it. A typical design takes 5–20 s.

**Outputs**

- **Summary** (left panel):
  - number of stations solved;
  - length, half span, height;
  - **C_L, C_D, L/D** with the planform reference area, plus the wetted-area values (see §3);
  - spanwise smoothness.
- **3-D view:** the lower surface coloured by p/p∞, the upper (freestream) surface, the shock curves from the leading edge, and the leading edge. An information panel (as in the GVWD and PSWR-1 tabs) lists Mach number, length, span, height, volume, planform area, volumetric efficiency (V^(2/3) / planform area), C_L, C_D, L/D and the resolution.
- **Front view:** the shock trace in the base plane, the FCT, lower-surface cross-sections and the x-projected shock curves. The curves turn in azimuth, which is the "local turning".
- **Wall pressure:** p/p∞ on streamwise planes (z = const) and crosswise planes (x = const), laid out like R1 Figs. 23 and 27.
- **Diagnostics** per station:
  - azimuth drift of the shock curve;
  - drift of the local axis centre (the noncoaxial effect);
  - shock extension past the base, needed vs used (determinacy);
  - body mesh step ratio.

**Refusals.** If any station is refused, no geometry is built, and every refused station is listed with its reason:

| Status | Meaning | What to change |
|---|---|---|
| `concave` | the shock cross-section curves away from the body (κ_b < 0) | use a shock that wraps around the body |
| `sub_mach` | shock angle at or below the Mach angle | raise the Mach number or the shock angle |
| `detached` | shock angle beyond detachment | lower the shock angle |
| `undetermined` | the shock does not determine the body up to the base. Analytic shocks are extended automatically; data shocks are never extrapolated | extend the shock data past the base |
| `outside_shock` | the FCT lies outside the shock at the base | move the FCT inside the shock trace |
| `limit_surface` | characteristics of one family crossed | a different shock or FCT |

**Export and analysis**

- **STL:** closed full-span mesh, metres.
- **STEP:** faceted solid via cadquery, millimetres.

Both use the GUI frame (x streamwise, y up, z span).

**Project files.** *File → Save/Load Parameters* and the session restored at start-up (`last_session.json`) include the tab's inputs, in the section `ltoc_waverider`:
- the shock source (preset name, `custom`, `bspline` or `oc_import`) and the six shock numbers; for a data file, the points themselves and the base plane X;
- Mach number, FCT Z, stations, spacing and points.

The shock numbers are always stored. A preset whose numbers differ from the saved ones loads as a custom shock with the saved numbers; projects saved with the earlier preset names ("R1 waverider case I (Eq. 23), M 6" and so on) still select the renamed presets. For the OC import, the OC design is stored in the project's own `oc_waverider` section: after loading, generate it in the OC Waverider tab first. A loaded design is not solved automatically; click *Generate*.

*Send to Aero Analysis* loads the waverider into the **Aero Analysis** tab (closed full-span mesh, A_ref set to its planform area) for the preview, the PySAGAS analysis or the AeroDeck sweep. That tab's mesh dialog also offers *LTOCs Waverider* directly.

*Send to Optimization* opens the current inputs in the **Optimization** tab (genetic algorithm with PySAGAS): the FCT height and the base ellipse size vary by default, and the upstream ellipse can be added. Each design solves every stream surface, so fewer stations make the optimization faster; designs with refused stations count as failed. *Apply best design to its tab* loads the result back. See `Optimization_Guide.md`.

---

## 2. Using the library

```python
from ltoc.design import LTOCDesign, EllipticShock

# R1 waverider case II (R1 Eqs. 25-26)
design = LTOCDesign(shock=EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 0.8),   # x, half-width, half-height of the upstream and base ellipses
                    M_inf=7.0, fct_z=0.25, n_stations=25, n_points=40)
res = design.build()
print(res.summary())               # CL 0.2029, CD 0.0645, L/D 3.143 (R1: 0.2034, 0.0648, 3.1395)
res.waverider.export_stl("case2.stl")
```

Lower-level pieces, for other shock surfaces:

| Module | Content |
|---|---|
| `ltoc.shock_surface` | `RuledShock`, `ConeShock`, `OsculatingConeShock`, `BSplineShock` |
| `ltoc.ltoc` | `find_leading_edge`, `solve_stream_surface` (Steps A–C) |
| `ltoc.waverider` | `LTOCWaverider`: the WADOS stream protocol (`upper_surface_streams`, `lower_surface_streams`, `leading_edge`, `length`, `width`, `height`), STL/STEP export, `aerodynamics()` |
| `ltoc.forces` | `panel_forces` (R1 Eqs. 19–22), `streamwise_section`, `crosswise_section` |
| `ltoc.moc_noncoaxial` | the noncoaxial inverse method of characteristics, usable on its own |

---

## 3. What to know

- **Forces** are inviscid, from the absolute lower-surface pressure only (R1 Sec. IV.A). The upper surface is a freestream surface, and there is no base contribution.
- **Reference area.** C_L and C_D use the **planform** area. R1 Eq. (22) says "total wetted area", but R1's own Tables 2 and 3 are matched by the planform area, to 0.05–0.33 %. With the wetted area they come out 17–23 % low (Gate 4 report §2). The wetted-area values are listed too.
- **Accuracy:**
  - Second order on planar stream surfaces (cones, osculating cones, the symmetry plane).
  - First order where the stream surfaces twist: about 1e−4 L at 40–80 points on R1 case II (Gate 3 report §3).
  - Forces are converged to < 0.03 % at 40 points.
- **Not covered in version 1:** concave shock cross-sections, sideslip (R1 test case I) and viscous forces.

Validation: [`ltoc/docs/validation.md`](../../ltoc/docs/validation.md) (V1–V9), and the gate reports [0](gate0_report.md), [1](gate1_report.md), [2](gate2_report.md), [3](gate3_report.md), [4](gate4_report.md) and [5](gate5_report.md).

---

## 4. References

| Tag | Reference | Used for |
|---|---|---|
| R1 | Zheng, X., Hu, Z., Li, Y., Zhu, C., You, Y., Song, W., "Local-Turning Osculating Cones Method for Waverider Design," *AIAA Journal* 58(8):3499–3513, 2020, [doi:10.2514/1.J059139](https://doi.org/10.2514/1.J059139) | the method (Sec. II), test cases and published coefficients (Secs. III–IV) |
| R2 | Zheng, X., Li, Y., Zhu, C., You, Y., "Multiple Osculating Cones' Waverider Design Method for Ruled Shock Surfaces," *AIAA Journal* 58(2):854–866, 2020, [doi:10.2514/1.J058640](https://doi.org/10.2514/1.J058640) | origin of the noncoaxial method of characteristics (Sec. III.B) |
| R3 | Zheng, X., Zhu, C., You, Y., "Design of multistage compression waverider based on the local-turning osculating cones method," *Chinese Journal of Theoretical and Applied Mechanics* 54(3):601–611, 2022 (in Chinese), [doi:10.6052/0459-1879-21-357](https://doi.org/10.6052/0459-1879-21-357) | LTOCs restated (Sec. 1.1); shock extension past the base (Sec. 1.4) |
| R4 | Qian, Y., Sobieczky, H., "Waverider Design with Parametric Flow Quality Control by Inverse Method of Characteristics," ICAS 2002 | the inverse method of characteristics and its limit surfaces |
| R5 | Zucrow, M. J., Hoffman, J. D., *Gas Dynamics*, Vol. 2, Wiley, 1976, pp. 187–192 | compatibility equations and unit processes (restated in R1 Eqs. 4–9) |

The PDFs are not in the repository; see [`docs/refs/README.md`](../refs/README.md).
