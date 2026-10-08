# WADOS: Local-Turning Osculating Cones (LTOCs) generator

Implementation spec for Claude Code, version 2 (reconciled against the papers). Read this whole file before touching any code.

## 1. Goal

Add a new waverider generation method to WADOS: the local-turning osculating cones method, which generates a waverider from a **prescribed three-dimensional shock surface** instead of a shock restricted to osculating planes normal to a base-plane shock curve.

Original source: Zheng, X., Hu, Z., Li, Y., Zhu, C., You, Y., Song, W., "Local-Turning Osculating Cones Method for Waverider Design," AIAA Journal 58(8):3499-3513, 2020, doi:10.2514/1.J059139.

Out of scope for this step: viscous correction, blunting, RCS, multistage shocks, variable Mach, sideslip, GUI polish, CFD runs.

## 2. Ground rules

1. **Recon first.** Before writing anything, read the repository and report: package layout, how existing methods (osculating cone, cone-derived, wedge-derived, osculating flowfield, VMPLO, multi-flowfield framework) are structured, what the common geometry output object is, where the Taylor-Maccoll solver and oblique-shock relations live, and what the test setup is. LTOCs must reuse those, not duplicate them.
2. **Do not modify existing methods.** LTOCs is additive. If a shared utility needs a change, propose it and wait.
3. **Stop at every gate** (section 8). Show results and plots, then wait for approval. Do not chain phases.
4. **No long or external solver runs.** Everything here runs in seconds to minutes in pure Python. CFD confirmation is a later step done together with the user.
5. **Dependencies:** NumPy, SciPy, Matplotlib only, plus whatever WADOS already uses. Ask before adding anything else.
6. **Cite the original source** of every equation in the docstring of the function implementing it (with equation number), not a secondary source.
7. **Plots:** every gate deliverable includes figures. Axis labels, tick labels and legends at 14 pt or larger, titles centred.
8. **The papers win over this spec.** Where section 5 flags an ambiguity, follow the instruction given there; anything else that looks inconsistent, stop and report.

## 3. Reference material (PDFs in `docs/refs/`)

| Tag | Reference | Use |
|---|---|---|
| R1 | Zheng, Hu, Li, Zhu, You, Song, AIAA J 58(8):3499-3513, 2020, doi:10.2514/1.J059139 | The method: Sec. II (algorithm), Sec. III-IV (test cases and numbers) |
| R2 | Zheng, Li, Zhu, You, "Multiple Osculating Cones' Waverider Design Method for Ruled Shock Surfaces," AIAA J 58(2):854-866, 2020, doi:10.2514/1.J058640 | Origin of the noncoaxial MOC (Sec. III.B, Fig. 15); extra geometry benchmark |
| R3 | Zheng, Zhu, You, Chin. J. Theor. Appl. Mech. 54(3):601-611, 2022, doi:10.6052/0459-1879-21-357 (in Chinese) | Restates LTOCs (Sec. 1.1); shock-extension and grid-refinement remarks (Sec. 1.4) |
| R4 | Qian, Sobieczky, "Waverider Design with Parametric Flow Quality Control by Inverse Method of Characteristics," ICAS 2002 | Inverse MOC domain picture (Fig. 1), limit-surface warning |
| R5 | Zucrow, Hoffman, Gas Dynamics Vol. 2, Wiley (cited by R1 as pp. 187-192) | Compatibility equations and unit processes (not in the folder; equations are restated in R1) |

## 4. The method as published (R1, Sec. II)

Conventions: freestream along +X, perfect gas with gamma = 1.4, shock surface given by an exact analytic expression so that slopes are exact. R1 parametrises shocks as ruled surfaces `S(m,n) = (1-n) p(m) + n q(m)` (Eq. 12), but the algorithm only needs point, normal and cross-section curvature.

### 4.1 Step A: shock curve of each stream surface (R1 Sec. II.A, Figs. 1-2)

- Each strip is a **stream surface**, one per leading-edge point, made of many local osculating planes.
- The local osculating plane at a shock point contains the freestream vector `V_inf` and the shock normal `n` (the post-shock velocity is the freestream velocity plus a change along `n`).
- `C = n x V_inf` is normal to that plane and tangent to the shock. `D = C x n` lies in the plane and in the shock tangent plane, so `D` is the direction of the shock curve.
- March from the leading-edge point along `D` with fourth-order Runge-Kutta in steps of `dX`. R1 writes this for a surface `Z(X,Y)` as Eqs. (1)-(2).
- Shock angle (Eq. 3): `beta = pi/2 - acos( V_inf . n / (|V_inf| |n|) )`.
- Post-shock state at every shock point from the Rankine-Hugoniot relations with `M_inf`, `gamma`, `beta`.

### 4.2 Step B: flowfield in the fictional meridian plane (R1 Sec. II.B, Figs. 3-4)

- All shock points of the stream surface are rotated about the X axis into one fictional meridian plane (R3 states the rotation axis explicitly). The axial coordinate is unchanged: `x = X`.
- Each shock point has a **local axis centre** `O`, set by the local radius of the shock in its crosswise plane (X = const). These centres drift away from the x axis downstream, so the flow in this plane is axisymmetric-like but **noncoaxial**.
- Governing relations (R1 Eqs. 4-8, after Zucrow and Hoffman):
  - along streamlines, `dy/dx = tan(theta)`: `rho V dV + dP = 0` and `dP - c^2 d(rho) = 0`
  - along Mach lines, `dy/dx = tan(theta +/- mu)`: `( sqrt(Ma^2 - 1) / (rho V^2) ) dP +/- d(theta) + delta [ sin(theta) dx / ( r Ma cos(theta +/- mu) ) ] = 0`, with `delta = 1`; `r -> infinity` recovers planar flow (Eq. 9).
- **The noncoaxial rule for r**: `r` is not the y coordinate. It is the distance from the point to the local axis centre.
  - Unit process starting at shock point `A1` (triangle `A1 A2 C2`): `r = |A1 O2|`, the local crosswise radius of the shock at `A1`.
  - Unit process starting at interior point `D1` (triangle `D1 C2 D2`): the axis centre of an interior point is assumed equal to the axis centre of the shock point **at the same axial position**, so `r = |D1 O1|`.
- Marching pattern (Fig. 3): interior point `C2` is located at the intersection of the streamline through `A1` with the left-running Mach line through `A2`, and so on row by row from the shock toward the body.
- R1 attributes its residual errors (below 2.2 %) to the same-axial-position axis assumption.

### 4.3 Step C: back to Cartesian coordinates (R1 Sec. II.C, Fig. 4, Eqs. 10-11)

For interior point `C2` in triangle `A1 A2 C2`:

- Drop the perpendicular from `C2` onto shock segment `A1 A2` in the meridian plane; the foot is `C2'`. Properties and Cartesian coordinates at `C2'` by linear interpolation along the segment.
- `i` = shock segment: `(Xi, Yi, Zi) = A2 - A1` in Cartesian coordinates.
- `j` = from `C2'` to `C2`, with `Xj = x_C2 - x_C2'` known from the meridian plane, and
  - `Yj = [ ( v (Xi^2 + Zi^2) - u Xi Yi - w Yi Zi ) / ( u (Yi^2 + Zi^2) - v Xi Yi - w Xi Zi ) ] Xj` (Eq. 10)
  - `Zj = -( Xi Xj + Yi Yj ) / Zi` (Eq. 11)
  - `(u, v, w)` is the Cartesian post-shock velocity at `C2'`.
- Meaning: `j` is the component of the post-shock velocity perpendicular to `i`, so the interior point is displaced from the shock **inside the local osculating plane, normal to the shock line**. Implement it in that vector form, `j ∥ V - (V.i) i / |i|^2`, scaled to match `Xj`, which avoids the division by `Zi` in Eq. 11.
- The speed `V` at the interior point is then decomposed into `(u, v, w)` from the geometry of neighbouring points.

### 4.4 Waverider construction (R1 Sec. IV.A, Fig. 18)

- Inputs: shock surface, flow capture tube (FCT) curve on the base plane.
- Leading edge: FCT projected upstream along the freestream onto the shock.
- Lower surface: streamlines from the leading edge to the base plane.
- Upper surface: freestream surface through the leading edge.
- Inviscid forces (Eqs. 19-22): per quadrilateral panel, area from the cross product of its diagonals, pressure as the mean of the four corners, `L = P_avg A_w n_z/|n|`, `D = P_avg A_w n_x/|n|`, coefficients referenced to `0.5 rho_inf V_inf^2 A` with `A` the total wetted area of the lower surface. The published coefficients are consistent with absolute lower-surface pressure and no upper-surface or base contribution (inferred from the numbers, confirm when reproducing).

## 5. Flags: where to deviate from, or add to, the printed text

1. **Sign of R1 Eq. (1).** For `Z = Z(X,Y)` with freestream along +X, the direction `D = (n x V_inf) x n` gives `dY/dX = - Z_X Z_Y / (1 + Z_Y^2)`; a circular cone confirms it (the shock curve must be the generator through the apex, `dY/dX = Y/X`). The printed equation has the opposite sign, which may be an axis-orientation convention or a typo. **Implement the vector form** on a general `ShockSurface`, and unit-test it on a cone.
2. **The printed unit process is one relation short.** Locating `C2` on the streamline from `A1` and the left-running Mach line into `A2` gives one compatibility relation for the pair `(P, theta)`; the streamline relations only link `V` and `rho` to `P`. **Implement the standard two-family inverse scheme** (Zucrow and Hoffman; R4 Fig. 1 shows the full two-family mesh marched from the shock), carrying entropy and total enthalpy along streamlines, and apply the noncoaxial `r` rule of 4.2 in both compatibility relations. If a reading of R1 or R2 closes the system with one family only, report it at Gate 0 before coding.
3. **Local crosswise radius.** R1 defines it through the curvature of the shock cross-section at X = const. Equivalent closed form for any surface: `r = cos(beta) / kappa_b`, where `kappa_b` is the normal curvature of the shock along `b = n x t` (`t` = unit `D`). Both must agree; test on a cone (`r` = cone radius) and on an elliptic cone.
4. **Domain of determinacy.** Shock data from the leading-edge point to the base plane does not determine the body streamline all the way to the base plane. R3 Sec. 1.4 states this and extends the shock beyond the design plane. Evaluate the analytic shock past the base plane as far as needed and report an under-determined streamline instead of extrapolating silently.
5. **Characteristic mesh stretching.** R3 Sec. 1.4 inserts points by linear interpolation along streamlines where the streamwise pressure gradient is large. Version 1: uniform refinement control plus a reported maximum step ratio on the body; adaptive insertion only if Gate 3 shows it is needed.
6. **Limit surfaces.** R4 warns that inverse MOC can give non-physical solutions with limit surfaces. Detect same-family characteristic crossing and abort that stream surface with a diagnostic.
7. **Not covered by the papers, handle defensively:** flat regions (`kappa_b -> 0`, switch smoothly to the planar relation; a planar shock must reproduce the wedge-derived result); concave cross-sections (`kappa_b < 0`, refuse with a clear message in version 1); `beta` at or below the Mach angle or beyond detachment (refuse; R3 requires the shock angle to exceed the Mach angle everywhere).
8. **Neighbouring stream surfaces** are solved independently. Report a spanwise smoothness metric of the lower surface; do not smooth silently.

## 6. Software design (adapt names to the repo conventions found in recon)

- `ShockSurface` abstraction: `point`, `normal`, cross-section curvature or second fundamental form. Version 1 implementations: `ConeShock`, `RuledShock` (R1 Eq. 12 with user curves `p(m)`, `q(m)`, covers every published case), `OsculatingConeShock` (the shock implied by an existing WADOS osculating-cone design), `BSplineShock`.
- `shock_geometry.py`: `beta`, `t`, `b`, `kappa_b`, `r`, axis centre, shock-curve integrator (RK4).
- `moc_noncoaxial.py`: two-family rotational MOC kernel with a pluggable `r` rule (coaxial rule = ordinary axisymmetric MOC, for testing), independent of LTOCs so it can be reused later.
- `ltoc.py`: orchestration, one stream surface per leading-edge point; Step C mapping; streamline extraction.
- Output: the same geometry object the other methods return, so export, PySAGAS and the GUI work unchanged. Also a panel-force routine following R1 Eqs. 19-22 for comparison with the published tables.
- Diagnostics per design: maps of `beta`, `r`, axis-centre drift, flags from section 5, determinacy margin, smoothness metric.

## 7. Verification plan

All at gamma = 1.4. Conditions at 27 km in R1: `P_inf = 1880 Pa`, `T_inf = 223.54 K`.

| ID | Case | Pass criterion |
|---|---|---|
| V1 | MOC kernel, coaxial rule, circular conical shock, Mach 6 and 10 | Matches the existing Taylor-Maccoll solver to 0.1 % with refinement; show convergence order |
| V2 | MOC kernel, planar shock | Uniform post-shock flow to round-off |
| V3 | Shock geometry | Cone: shock curve is the generator, `r` equals the cone radius; flag 1 and flag 3 identities hold |
| V4 | LTOCs with `ConeShock` | Identical to the existing cone-derived waverider (deviation below 1e-4 of length) |
| V5 | LTOCs with `OsculatingConeShock` | Reproduces the existing osculating-cone waverider |
| V6 | **R1 waverider case I**, Mach 6: shock `X = 1.7 n, Y = 0.595 n cos m, Z = 0.68 n sin m` (Eq. 23), FCT `Z = 0.12` (Eq. 24) | R1 Table 2 (LTOCs row): `CL = 0.2205, CD = 0.0750, L/D = 2.9378`. Target within 1 %; wall-pressure curves compared with R1 Fig. 23 |
| V7 | **R1 waverider case II**, Mach 7: shock `X = 0.1 + 1.4 n, Y = (0.3 + 0.35 n) cos m, Z = (0.2 + 0.6 n) sin m` (Eq. 25), FCT `Z = 0.25` (Eq. 26) | R1 Table 3 (LTOCs row): `CL = 0.2034, CD = 0.0648, L/D = 3.1395`. Target within 1 % |
| V8 | R1 test case II generator, Mach 7: shock `X = 0.1 + 1.4 n, Y = (0.2 + 0.6 n) cos m, Z = (0.3 + 0.35 n) sin m` (Eq. 18) | Wall pressure against R1 Fig. 16 (digitised curves, qualitative) |
| V9 | Robustness | Flat, concave, sub-Mach-angle and under-determined inputs give the specified messages, never a silent bad geometry |

Optional geometry cross-check from R2 (different method, so expect closeness, not identity): ruled shock `X = 1.3 n, Y = m n, Z = (5.28 m^4 + 0.16 m^2 - 0.37) n`, FCT `Z = 0.09`, Mach 6; R2 Table 2 gives `W = 0.9439`, `H = 0.1817`, `A_w = 0.1166` normalised by length.

R1 test case I (elliptic cone at 5 degrees sideslip) is deferred with sideslip.

V1 to V9 become automated tests. Euler confirmation of V6 and V7 is a later session with the user.

## 8. Phases and gates

**Gate 0: recon and reconciliation.** Written report only: repo findings (rule 1), confirmation or correction of flags 1 to 3 after reading R1 Sec. II and R2 Sec. III.B, proposed file layout. No code.

**Phase 1: MOC kernel.** `moc_noncoaxial.py`, tests V1 and V2, convergence plots. Gate 1.

**Phase 2: shock geometry.** `ShockSurface` classes, `shock_geometry.py`, test V3, maps of `beta` and `r` on the R1 shocks. Gate 2.

**Phase 3: LTOCs core.** Steps A to C, diagnostics, tests V4 and V5. Gate 3.

**Phase 4: published cases.** V6 to V9, with figures: shock with shock curves, lower surface, wall-pressure distributions laid out like R1 Figs. 23 and 27, and a table against R1 Tables 2 and 3. Gate 4.

**Phase 5: integration.** Method registry and GUI entry following the existing pattern; short user documentation page with the references of section 3. Gate 5.

At each gate report: what was done, test results, figures, open questions, and anything in this spec that turned out to be wrong.
