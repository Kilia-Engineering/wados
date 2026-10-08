# LTOCs validation record

This page records what was checked, against which reference, at what tolerance, and by which test. It is filled in gate by gate. Figures and the raw numbers come from `python -m ltoc.examples.gate1_moc_kernel`, which writes `docs/ltoc/figures/gate1_*.png` and `gate1_results.json`.

```
pytest ltoc/tests/ -q        # 28 passed, ~30 s
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
