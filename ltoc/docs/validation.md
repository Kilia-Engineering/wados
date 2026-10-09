# LTOCs validation record

This page records what was checked, against which reference, at what tolerance, and by which test. It is filled in gate by gate. Figures and the raw numbers come from `python -m ltoc.examples.gate<N>_*`, which write `docs/ltoc/figures/gate<N>_*.png` and `gate<N>_results.json`.

```
pytest ltoc/tests/ -q        # 84 passed, ~2 min (the 5 GUI tests need PyQt5 and are skipped in CI)
```

All cases use γ = 1.4. Pressure is reported as p/p∞. The meridian-plane frame and nondimensionalisation are those of `ltoc.moc_noncoaxial`.

---

## Phase 1: MOC kernel (`ltoc/moc_noncoaxial.py`)

The kernel has two meshes, which are compared throughout. Both use the C+ and the C− compatibility relations (R1 Eq. 6):

- **streamline mesh**: R1's own topology, with the C− relation added;
- **characteristic net**: the standard net of R4 Fig. 1.

### V2: planar shock gives uniform flow

Mach 6, β = 20°, straight shock, planar rule, N = 200 (`test_v2_planar_shock_gives_uniform_flow`).

| Mesh | Points | max \|p/p₂ − 1\| | max \|θ − θ₂\| (rad) |
|---|---|---|---|
| Streamline | 20 301 | 5.2e−15 | 8.3e−16 |
| Characteristic | 14 928 | 5.7e−15 | 8.6e−16 |

The test passes at 1e−13. A noncoaxial axis at y = −10⁸ gives the planar result to < 1e−7 (`test_v2_large_noncoaxial_radius_tends_to_planar`).

### V1: conical shock, coaxial rule, against Taylor–Maccoll

The shock is y = x tan β for 0.25 ≤ x ≤ 1, with N + 1 equally spaced shock points. Errors are the maximum over every valid mesh point.

**Against the existing solver** (`liu2019.shock.taylor_maccoll_cone_field`), at N = 100 (`test_v1_matches_existing_taylor_maccoll_solver`, limit 1e−3):

| Case | Mesh | Points | p | M | θ |
|---|---|---|---|---|---|
| M 6, β 20° | Streamline | 5 140 | 3.3e−5 | 6.0e−6 | 6.3e−5 |
| M 6, β 20° | Characteristic | 3 262 | 3.3e−5 | 6.0e−6 | 6.8e−5 |
| M 10, β 14° | Streamline | 5 151 | 3.8e−5 | 6.1e−6 | 6.5e−5 |
| M 10, β 14° | Characteristic | 3 064 | 3.7e−5 | 5.9e−6 | 6.5e−5 |

At this level the error is dominated by the existing solver itself. Its splines are built at rtol 1e−6 on sparse knots and are about 7e−5 off in θ and 3e−5 in p (`test_tight_reference_agrees_with_existing_solver`).

**Against the tight reference** (`ltoc.reference.ConicalFlowReference`: the same Taylor–Maccoll right-hand side at rtol 1e−12), maximum relative error in p (`test_v1_second_order_convergence` requires order > 1.8 and error < 2e−5 at N = 200):

| N | Streamline, M 6 | Streamline, M 10 | Characteristic, M 6 | Characteristic, M 10 |
|---|---|---|---|---|
| 25 | 5.9e−5 | 4.7e−5 | 1.5e−4 | 1.4e−4 |
| 100 | 3.8e−6 | 2.8e−6 | 7.1e−6 | 6.0e−6 |
| 200 | 9.4e−7 | 7.1e−7 | 1.6e−6 | 1.4e−6 |
| 800 | 5.9e−8 | 4.4e−8 | 9.4e−8 | 8.1e−8 |
| observed order | 2.00 | 2.00 | 2.03 | 2.03 |

M and θ behave the same way (figure `gate1_v1_convergence.png`).

**Body streamline** (the streamline from the first shock point) against the Taylor–Maccoll streamline, N = 200 (`test_body_streamline_matches_taylor_maccoll`):

| Case | Mesh | max \|Δy\| / L | max \|Δp/p\| | Reaches end of determined region |
|---|---|---|---|---|
| M 6 | Streamline (native column) | 1.7e−7 | 8.7e−7 | yes |
| M 6 | Characteristic (traced) | 2.1e−7 | 1.0e−5 | yes |
| M 10 | Streamline (native column) | 1.0e−7 | 6.5e−7 | yes |
| M 10 | Characteristic (traced) | 1.2e−7 | 1.1e−5 | stops 0–10 % early, depending on N |

### Rotational flow (curved shock)

Mach 6. β goes smoothly from 22° to 18° along 0.25 ≤ x ≤ 1, so the entropy varies across streamlines. Self-convergence of the body pressure, max |p(2N)/p(N) − 1| (`test_rotational_flow_schemes_agree_and_converge`):

| N → 2N | Planar, streamline | Planar, characteristic | Coaxial, streamline | Coaxial, characteristic |
|---|---|---|---|---|
| 50 → 100 | 1.3e−4 | 2.4e−4 | 2.6e−4 | 2.2e−4 |
| 200 → 400 | 8.3e−6 | 1.5e−5 | 1.3e−5 | 1.3e−5 |
| 800 → 1600 | 5.7e−7 | 1.0e−6 | 9.7e−7 | 9.9e−7 |

The body position converges at order 2.0. The two meshes are independent discretisations. At N = 1600 they agree to 4.0e−8 (planar) and 6.9e−8 (coaxial) in mid-layer pressure, and to 2.5e−7 and 1.1e−7 in body pressure.

### Axis-rule plumbing

- `test_axis_rule_translation_invariance`: moving the cone and the axis by the same offset changes nothing (< 1e−11).
- `test_noncoaxial_rule_with_axis_centres_on_axis_is_coaxial`: the noncoaxial rule built from a circular cone's own crosswise radii is bit-identical to the coaxial rule.
- The genuinely noncoaxial case is verified in Phase 3 (V5) and Phase 4 (V6, V7).

### Limit surfaces (spec flag 6)

Mach 6, planar. β drops from 28° to 14° over Δx ≈ 0.1 around x = 0.5. Both meshes report crossing same-family characteristics, at x = 0.483 (characteristic) and x = 0.487 (streamline), N = 200. With `strict=True` they raise `LimitSurfaceError` (`test_limit_surface_detected_by_both_schemes`). The smooth 22° → 18° shock raises nothing (`test_smooth_curved_shock_has_no_failures`).

### Input validation

The following are refused with a clear exception (`test_initial_line_must_be_space_like`, `test_shock_below_mach_angle_or_detached_is_refused`, `test_initial_line_below_axis_is_refused`):

- an initial line that is not space-like (steeper than C+ or shallower than C−);
- a shock angle at or below the freestream Mach angle;
- subsonic post-shock flow;
- an initial line at or below the axis.

---

## Phase 2: shock geometry (`ltoc/shock_surface.py`, `ltoc/shock_geometry.py`)

Tests are in `ltoc/tests/test_shock_geometry.py`. Figures and numbers come from `python -m ltoc.examples.gate2_shock_geometry`.

### V3: cone, and the flag 1 and flag 3 identities

| Check | Reference | Worst error | Test |
|---|---|---|---|
| Cone (β = 12°, 20°, 35°): the shock curve stays on its generator | azimuth constant | 0 (exact) | `test_v3_cone_shock_curve_is_generator_and_r_is_cone_radius` |
| Cone: β along the curve | β | 1.1e−16 rad | same |
| Cone: r = cone radius X tan β | analytic | 1.0e−15 | same |
| Cone: axis centre on the X axis | 0 | 6.7e−16 | same |
| Cone: intrinsic meridian ordinate y rises at tan β; y_axis constant | analytic | < 1e−13 | same |
| Flag 1: dY/dX = −Z_X Z_Y / (1 + Z_Y²) on an elliptic cone | closed form | 2.2e−16 | `test_v3_flag1_shock_curve_slope_identity` |
| Flag 1: R1 Eq. (1) as printed (opposite sign) | — | differs by 0.58 | same |
| Flag 3: r = cos β / κ_b equals the cross-section radius, elliptic cones a/b = 1, 0.875, 0.42 | ellipse formula | 1.1e−15 relative | `test_v3_flag3_radius_equals_cross_section_radius_elliptic_cone` |
| Flag 3 on the R1 Eq. (18) shock | finite differences of the X = const section | < 1e−7 relative | `test_v3_flag3_on_variable_elliptic_shock_by_finite_differences` |

### Other shock surfaces

| Check | Result | Test |
|---|---|---|
| `OsculatingConeShock` base-plane curve against the OC generator's SWPC samples (M 5, β 15°) | 1.1e−9 m | `test_osculating_cone_shock_matches_oc_generator` |
| `OsculatingConeShock` axis centres against the OC generator's `cone_centers` | 1.1e−9 m | same |
| `OsculatingConeShock`: shock curves stay in their osculating planes, β constant, axis centre fixed, dr/dx = tan β | to round-off | same |
| `BSplineShock` (60 × 30 grid) against its source, R1 Eq. (25) | point 4.5e−9, β 5.5e−8 rad, r 8.4e−5 relative | `test_bspline_shock_reproduces_source_surface` |
| R1 Eqs. (18), (23), (25): convex, attached, above the Mach angle along the shock curves | no flags raised | `test_published_r1_shocks_are_convex_and_attached` |
| R2 Eq. (13) quartic shock: κ_b > 0 (centres on the body side) | yes | `test_r2_quartic_shock_curvature_is_positive_toward_the_body` |
| R1 Eq. (25): the shock curves turn (azimuth drift) and the axis centres drift | 12° and 0.46 at most | `test_shock_curves_turn_on_elliptic_shocks` |

---

## Phase 3: LTOCs core (`ltoc/ltoc.py`, `ltoc/waverider.py`)

Tests are in `ltoc/tests/test_ltoc_core.py`. Figures and numbers come from `python -m ltoc.examples.gate3_ltoc_core` (about 2 min). Deviations are divided by the waverider length L. "40 points" means 40 shock points from the leading edge to the base plane.

### Leading edge and Step C identities

| Check | Reference | Worst error | Test |
|---|---|---|---|
| FCT projected onto the shock (R1 Sec. IV.A), R1 Eq. (25) shock | the point lies on the FCT line | 1e−12 | `test_leading_edge_projection_lies_on_shock_and_fct` |
| Cone: each stream surface stays in its meridian plane (Step C, R1 Eqs. 10–11) | azimuth constant | < 1e−12 rad | `test_step_c_on_cone_keeps_stream_surface_in_meridian_plane` |
| Cone: 3-D distance to the axis equals the meridian radius y − y_axis | coaxial meridian | < 1e−12 | same |
| Cone: R1's X-component scaling of j never ill-conditioned | `stepc_fallbacks` = 0 | 0 | same |

### V4: cone-derived waverider (cone shock, M 6, β 12°, 11 stations)

| Check | Reference | Worst deviation / L | Criterion | Test |
|---|---|---|---|---|
| Body streamlines, 40 points | tight Taylor–Maccoll streamline (`ConicalFlowReference.streamline`) | 1.8e−7 | 1e−4 | `test_v4_cone_waverider_matches_taylor_maccoll` |
| Body streamlines, 40 points | `ShadowWaverider` lower surface | 4.1e−7 | 1e−4 | `test_v4_cone_waverider_matches_shadow_waverider` |
| Convergence, n = 20, 40, 80, 160 | Taylor–Maccoll | 7.3e−7, 1.8e−7, 4.6e−8, 1.1e−8 (order 2.0) | — | example script |

### V5: osculating-cone waverider (WADOS OC design, M 5, β 15°, 10 planes)

| Check | Reference | Worst deviation / L | Criterion | Test |
|---|---|---|---|---|
| Body streamlines, 40 points | exact per-plane flow: Taylor–Maccoll in each local cone, wedge flow where the shock is flat (`osculating_cone_streamline`) | 8.8e−7 | 1e−4 | `test_v5_oc_waverider_matches_tight_reference` |
| Body streamlines, 40 points | OC generator's `lower_surface_streams` | 9.0e−4 | equals the OC generator's own error (next row) to 1e−5 | `test_v5_oc_waverider_matches_oc_generator_within_its_accuracy` |
| OC generator against the exact per-plane flow | — | 9.0e−4 (its cone angle is 9.2276° against 9.2318°: default `solve_ivp` tolerances in `waverider_generator/flowfield.py`) | — | same |
| Convergence, n = 20, 40, 80 | exact per-plane flow | 3.5e−6, 8.8e−7, 2.2e−7 (order 2.0) | — | example script |
| Stream protocol: equal-length streams, station 0 on z = 0, upper surface ends at the base; full-span mesh closed; STL size | — | exact | — | `test_stream_protocol_and_closed_stl` |

### Twisted stream surfaces (R1 case II shock, Eq. 25, M 7, FCT Z = 0.25)

On the R1 shocks the shock curves turn by up to 12° in azimuth between the leading edge and the base, so the stream surfaces are not planar. There is no exact reference, so these checks use self-convergence and a consistency test.

| Check | Result | Test |
|---|---|---|
| Step C velocity tangent to the 3-D body streamline (max angle, station 0.3 / 0.6), n = 20 → 160 | 0.090° → 0.011° / 0.032° → 0.0042°, first order | `test_step_c_on_twisted_stream_surface_is_consistent_and_first_order` |
| Base-plane body point, \|P(n) − P(2n)\| / L, station 0.3, n = 20, 40, 80 | 2.4e−4, 1.2e−4, 6.1e−5: first order | same |
| Same, station 0.6 | 4.0e−5, 2.0e−5, 1.0e−5: first order | example script |
| Determinacy (flag 4): shock extension needed ≤ extension used | yes, all stations of cases I and II | same, and example script |

Step C is exact on planar stream surfaces, so V4 and V5 keep the kernel's second order. On twisted surfaces R1's linear interpolation along the row chord gives an O(h²) error per row, so the body converges at first order. The error is about 1.2e−4 L at 80 points on the worst station (see the `ltoc.ltoc` module docstring and the Gate 3 report).

### Refusals (flag 7; see also V9 below)

| Input | Status | Test |
|---|---|---|
| FCT point outside the shock at the base plane | `outside_shock`, and the stream-protocol attributes raise "waverider incomplete" | `test_fct_outside_shock_and_concave_shock_are_refused` |
| Shock cross-section curving away from the body (κ_b < 0) | `concave`, message "... (spec flag 7: refused)" | same |

---

## Phase 4: published cases (`ltoc/forces.py`, R1)

Tests are in `ltoc/tests/test_ltoc_published.py`. Figures and numbers come from `python -m ltoc.examples.gate4_published_cases` (about 4 min). Forces follow R1 Eqs. (19)–(22) with the planform reference area (see the V6/V7 note). Wall pressures are compared with R1's own LTOCs curves, digitised into `ltoc/data/r1_wall_pressure_digitised.json` (about 0.03 in p/p∞ and 0.003 in X or Y). FCT stations are clustered toward the symmetry plane, Y = Y_tip (1 − cos(πu/2)), with 40 points per stream surface.

### V6 and V7: R1 waverider cases I and II, R1 Tables 2 and 3

| Case | Stations | C_L | C_D | L/D | Test |
|---|---|---|---|---|---|
| I (Eq. 23, M 6, FCT Z = 0.12): R1 LTOCs | — | 0.2205 | 0.0750 | 2.9378 | |
| I: this work | 25 | 0.22033 (−0.08 %) | 0.07482 (−0.24 %) | 2.9448 (+0.24 %) | `test_v6_v7_lift_and_drag_match_r1_tables[I]` (1 % target, 0.5 % margin) |
| I: this work | 97 | 0.22038 (−0.05 %) | 0.07496 (−0.05 %) | 2.9400 (+0.07 %) | example script |
| I: R1 CFD, for scale | — | 0.2201 (−0.18 %) | 0.0749 (−0.13 %) | 2.9377 | |
| II (Eq. 25, M 7, FCT Z = 0.25): R1 LTOCs | — | 0.2034 | 0.0648 | 3.1395 | |
| II: this work | 25 | 0.20288 (−0.26 %) | 0.06454 (−0.40 %) | 3.1434 (+0.12 %) | `test_v6_v7_lift_and_drag_match_r1_tables[II]` |
| II: this work | 97 | 0.20290 (−0.24 %) | 0.06459 (−0.33 %) | 3.1415 (+0.06 %) | example script |
| II: R1 CFD, for scale | — | 0.2024 (−0.49 %) | 0.0645 (−0.46 %) | 3.1386 | |

Going from 40 to 80 points per stream surface changes C_L and C_D by less than 0.03 %.

**Reference area.** R1 Eq. (22) says A is "the total wetted area". R1's tables are reproduced with the planform (projected) area of the lower surface. With the wetted area, C_L = 0.1702 and 0.1685 and C_D = 0.0579 and 0.0536: 17–23 % low, with L/D unchanged. This is asserted in `test_r1_coefficients_are_referenced_to_planform_not_wetted_area`. R1 Eq. (19) also leaves out the 1/2 of the quadrilateral area, which cancels in the coefficients.

### Wall pressure: R1 Figs. 23 and 27 (V6, V7) and Fig. 16 (V8)

Deviation of this work from R1's LTOCs curves, in %, as max / rms over each digitised curve:

| Case | Streamwise planes | Crosswise planes | Test |
|---|---|---|---|
| V6, Fig. 23 | Y = 0: 0.19 / 0.09; Y = 0.1: 0.90 / 0.20; Y = 0.2: 1.95 / 0.42 | X = 0.80: 1.69 / 0.68; X = 1.20: 1.07 / 0.46; X = 1.60: 0.74 / 0.43 | `test_v6_v7_wall_pressure_matches_r1_figures` (rms < 1.5 %, max < 4 %) |
| V7, Fig. 27 | Y = 0: 0.16 / 0.09; Y = 0.1: 0.38 / 0.21; Y = 0.2: 1.09 / 0.71 | X = 0.60: 2.06 / 1.21; X = 1.00: 0.65 / 0.31; X = 1.40: 2.57 / 0.92 | same |
| V8, Fig. 16: test case II shock generator (Eq. 18), streamlines from the shock's upstream edge n = 0 | Y = 0.1: 2.19 / 0.59; Y = 0.2: 3.12 / 1.43; Y = 0.3: 1.51 / 1.16 | X = 0.52: 3.28 / 1.44; X = 1.00: 2.35 / 1.00; X = 1.48: 2.33 / 1.16 | `test_v8_test_case_2_generator_wall_pressure_matches_r1_fig16` (rms < 2 %, max < 4 %) |

The maxima occur on the steep parts of the curves, where 0.003 of position error alone is worth 2–3 % of pressure. R1 reports its own LTOCs-vs-CFD wall-pressure differences as up to 1.4–1.9 %.

### V9: robustness

| Input | Result | Test |
|---|---|---|
| Planar shock (β 15°, M 6) with a curved FCT | the wedge-derived waverider: body slope tan θ to 2e−8, p = p₂ to 3e−7, planar stream surfaces. Panel forces C_L = p₂/q and L/D = 1/tan θ to 1e−5 | `test_v9_flat_shock_gives_the_wedge_derived_waverider` |
| Cross-section curving away from the body (κ_b < 0) | `concave`, "... (spec flag 7: refused)" | `test_fct_outside_shock_and_concave_shock_are_refused` |
| Cone β 8° at M 6 (below the Mach angle) | `sub_mach`, "... (spec flag 7: refused)" | `test_v9_sub_mach_and_detached_shocks_are_refused` |
| Cone β 75° at M 3 (beyond detachment) | `detached`, "... (spec flag 7: refused)" | same |
| Shock extension capped at 0.1 chords | `undetermined`, "... (spec flag 4)" | `test_v9_under_determined_body_is_reported` |
| B-spline (data) shock ending at the base plane | `undetermined`, "the shock data end at X = ..." (spec flag 4); data are never extrapolated | `test_v9_data_shock_is_never_extrapolated` |
| B-spline shock with data past the base | same body as the analytic cone to 9e−7 | same |
| FCT point outside the shock at the base | `outside_shock`; the stream-protocol attributes raise | `test_fct_outside_shock_and_concave_shock_are_refused` |

---

## Phase 5: integration (`ltoc/design.py`, `ltoc_waverider_tab.py`)

Tests are in `ltoc/tests/test_ltoc_design.py` (CI) and `ltoc/tests/test_ltoc_gui.py` (offscreen Qt; skipped without PyQt5).

| Check | Result | Test |
|---|---|---|
| Every preset (R1 cases I, II, test case II, cone) builds a closed waverider with planform-referenced forces, GUI-frame plotting data and per-station progress | yes | `test_every_preset_builds_a_closed_waverider` |
| GUI defaults are the configuration that reproduces R1 Tables 2 and 3 (25 clustered stations, 40 points) | same FCT to 1e−15 | `test_gui_defaults_are_the_configuration_validated_against_r1_tables` |
| Invalid inputs (FCT outside the shock, subsonic, too few stations, degenerate shock) rejected before solving | `ValueError` with the reason | `test_invalid_inputs_are_rejected_before_solving` |
| Refused stations reported, no geometry built | statuses and flag messages in the summary | `test_refused_stations_are_reported_and_no_geometry_is_built` |
| Build cancellable from the progress callback | `LTOCError("cancelled")` | `test_build_can_be_cancelled_from_the_progress_callback` |
| OC-design import reproduces the OC waverider | LE to 1e−9; lower surface within the OC generator's accuracy (V5) | `test_oc_design_import_reproduces_the_oc_waverider` |
| Tab: presets fill inputs; editing makes a custom shock; background run fills the views; refusals listed with exports disabled; OC import without a design gives an input error | yes | `test_ltoc_gui.py` (4 tests) |
| Aero Analysis mesh dialog offers the LTOCs waverider and returns its STL; unchanged without an LTOCs tab | yes; the STL reads in meshio (PySAGAS) and numpy-stl | `test_aero_analysis_mesh_dialog_offers_the_ltoc_waverider` |
