# LTOCs waverider: user guide

The **Local-Turning Osculating Cones (LTOCs)** method (Zheng et al. 2020, R1) designs a waverider from a prescribed three-dimensional shock surface. The osculating-cone method assumes the flow behind each leading-edge point stays in one plane. LTOCs instead follows the shock curve as it turns locally. That makes shocks possible that osculating methods cannot handle, such as an FCT below some of the local curvature centres (R1 Sec. IV.B).

In WADOS it is the **LTOCs Waverider** tab (`ltoc_waverider_tab.py`) and the `ltoc` library.

---

## 1. Using the tab

**Inputs (left panel)**

| Input | Meaning |
|---|---|
| Shock surface | One of three kinds:<br>• **R1 presets**: waverider cases I and II (R1 Eqs. 23, 25), test case II (Eq. 18), and a circular cone.<br>• **Custom elliptic ruled shock** S(m, n) = (1 − n) p(m) + n q(m), with p = (x_p, a_p cos m, b_p sin m) and q = (x_q, a_q cos m, b_q sin m) (R1 Eqs. 12, 15).<br>• **Import the current OC design** from the OC Waverider tab. |
| Frame | R1's: X freestream, Y span, **Z down**. The base plane is X = x_q. |
| Mach number | Design Mach number (γ = 1.4). |
| FCT Z | The flow capture tube is the line Z = const in the base plane (R1 Eqs. 24, 26). It must lie between 0 and b_q. The leading edge is the FCT projected upstream onto the shock. |
| FCT stations | Stream surfaces across the half span. 25 (default) reproduces R1 Tables 2 and 3 to 0.4 %. |
| Station spacing | Clustered at the symmetry plane (default) or uniform. Clustering matters when the FCT meets the shock near its apex (R1 case I). |
| Points per stream surface | Shock points between the leading edge and the base. 40 (default) converges the forces to < 0.03 %; the cost grows with its square. |

Editing a preset's numbers turns it into a custom shock. With the OC import, the shock, Mach number and leading edge come from the OC design, and LTOCs reproduces that design (validation V5).

**Run.** *Generate LTOCs Waverider* solves one stream surface per station in a background thread, with a progress bar; *Cancel* stops it. A typical design takes 5–20 s.

**Outputs**

- **Summary** (left panel):
  - number of stations solved;
  - length, half span, height;
  - **C_L, C_D, L/D** with the planform reference area, plus the wetted-area values (see §3);
  - spanwise smoothness.
- **3-D view:** the lower surface coloured by p/p∞, the upper (freestream) surface, the shock curves from the leading edge, and the leading edge.
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

The **Aero Analysis** tab can load the waverider directly: choose *LTOCs Waverider* in its mesh dialog for the preview, the PySAGAS analysis or the AeroDeck sweep.

---

## 2. Using the library

```python
from ltoc.design import LTOCDesign, EllipticShock

# R1 waverider case II (R1 Eqs. 25-26)
design = LTOCDesign(shock=EllipticShock(0.1, 0.3, 0.2, 1.5, 0.65, 0.8),
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
- **Not covered in version 1:** concave shock cross-sections, sideslip (R1 test case I), viscous forces, and saving LTOCs designs in WADOS project files.

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
