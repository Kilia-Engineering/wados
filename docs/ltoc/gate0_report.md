# LTOCs — Gate 0 report: recon and reconciliation

Spec: [`LTOC_implementation_spec.md`](LTOC_implementation_spec.md) (version 2).
References: [`docs/refs/`](../refs/README.md) (R1–R4, tags as in spec §3).
Status: **written report only, no code** (spec §8). Waiting for approval before Phase 1.

---

## 1. Repository findings (spec rule 1)

### 1.1 Layout and how the existing methods are structured

- **Hub.** `waverider_gui.py` (PyQt5) plus one `*_tab.py` per method.
- **Registration.** There is no method registry. Each method is registered by copy-paste:
  - a `try/except ImportError` import with an `<X>_AVAILABLE` flag (`waverider_gui.py:149-262`);
  - an `addTab` block in `create_visualization_panel` (`:2491-2583`);
  - optionally project save/load (`:1365-1401`).
- **Generator libraries** (numpy/scipy only, none import PyQt5):

| Spec name | WADOS code | Entry point | Output |
|---|---|---|---|
| Osculating cone | `waverider_generator/generator.py` | `waverider(M_inf, beta, height, width, dp, …)` | OC stream protocol (§1.3) |
| Cone-derived | `shadow_waverider.py` | `ShadowWaverider(mach, shock_angle, poly_coeffs, …)` | `upper/lower_surface` arrays (n_le, n_s, 3), full span; `get_mesh()`, `export_stl/tri` |
| Wedge-derived | `gvwd/` | dataclass generators (`Caret`, `EngineeringFlat`, …) | `gvwd.geometry.Mesh` (vertices, faces, labels), frame x/**y span**/z up |
| Osculating flowfield | `waverider_generator/vmof_generator.py` | `VMOFWaverider(…)` (re-traces an internal OC instance) | OC stream protocol + `to_CAD`, `export_stl/tri` |
| VMPLO | `waverider_generator/vmplo/` | `OsculatingAssembly` → `VMPLOWaverider` | OC stream protocol, contract documented at `vmplo/geometry.py:21-41` |
| Multi-flowfield framework | `mfof/` (subclasses `liu2019/`) | `build_mfof_waverider(params, flowfield_factory)` | `(n_x, n_z)` surface grids, half span |

### 1.2 Shared physics: where it lives, and what LTOCs will reuse

| Need | Reuse | Notes |
|---|---|---|
| Rankine–Hugoniot, θ–β–M, detachment, p₀₂/p₀₁, Mach angle | `gvwd.thermo.oblique_shock` (`rankine_hugoniot`, `mach_angle`, `theta_from_beta_M` re-exported from `pswr.thermo.oblique_shock`; adds `obtain_beta`, `theta_max`, `stagnation_pressure_ratio`) | Radians. **The only shock set tested against tables in CI** (`gvwd/tests/test_oblique_shock.py`, 0.1 %). Five other copies exist (liu2019, vmplo, pysagas, planar, inline); not used. |
| Taylor–Maccoll reference | `liu2019.shock.taylor_maccoll_cone_field(Ma, beta_deg, gamma)` | Finds δc by event, rtol 1e-6, **interpolating** splines on [δc, β]. Returns V′ components only; M, p, ρ are derived from V′ plus the post-shock state. The most-used solver (`waverider_generator/flowfield.py`) has default rtol 1e-3 and *smoothing* splines, so it is not suitable as a 0.1 % reference. No TM solver in the repo is tested against tables; V1 will be the first such test. |
| MOC | **none reusable** | The only kernel, `waverider_generator/vmplo/moc.py` (wrapped by `mfof/moc.py`): irrotational (Riemann invariants ν ± α), axisymmetric only, `r` hard-wired to the point's own coordinate, no shock-point unit process. Its own tests document a 25–35 % bias (`mfof/test_phase3.py:62-66`). A rotational, noncoaxial kernel is therefore new code, not duplication. |
| OC shock data (for `OsculatingConeShock`) | the OC `waverider` instance | SWPC Bézier evaluators (`Bezier_Shockwave`, `First/Second_Derivative`, `Find_t_Value`, `Calculate_Radius_Curvature`), `cone_centers`, `length`, `beta`. |
| STL / TRI export | `waverider_generator.stream_mesh.build_stream_mesh`, `write_ascii_stl`, `write_tri` | Consume the stream protocol. |
| STEP export | `waverider_generator.cad_export.build_waverider_solid` | Needs equal-length streams. This is the robust route used by Shadow and VMPLO. |

### 1.3 The common geometry output object

There is **no shared class or Protocol** (spec §6 assumed one). What export, LE blunting and
compensation, volume and reference-area code actually consume is the duck-typed **OC stream protocol**:

- `upper_surface_streams`, `lower_surface_streams`: lists of `(n, 3)` arrays, one per spanwise station.
  Use equal length (`build_waverider_solid` needs it).
- Half model `z ≥ 0`, station 0 = symmetry plane; point 0 = leading edge (shared); last point = base,
  at `x = length`.
- `leading_edge (n, 3)`, `length`, `width`, `height`. For the OC-style canvases and planform
  `A_ref`: `upper_surface_x/_y/_z (n_span, n_stream)`.
- Frame: **x streamwise, y up, z span**. This is the frame PySAGAS expects (lift from `w.y`).
  GVWD/PSWR use y span, z up, and convert at the boundary.

LTOCs will return an `LTOCWaverider` exposing this protocol, with `export_stl`, `export_tri` and
`to_CAD` methods following the VMOF/VMPLO pattern. R1's frame `(X, Y, Z_down)` maps to
`(x, y, z) = (X, −Z, Y)`.

### 1.4 Tests and CI

- CI (`.github/workflows/tests.yml`): Python 3.11/3.12, numpy/scipy/matplotlib/pytest, **no PyQt5,
  no cadquery**.
- It runs three named directories only: `gvwd/tests`, `pysagas/tests`, `waverider_generator/tests`.
- A new `ltoc/tests/` runs in CI only after a step is added (a shared file; see §5).
- Both `ltoc/` and `ltoc/tests/` need an `__init__.py`, so plain `pytest` puts the repo root on
  `sys.path` (commit `626b68e`).
- House style:
  - plain `assert` and `pytest.approx`;
  - module docstrings that cite the reference and the pass criterion;
  - figures written to `tmp_path` through `matplotlib.figure.Figure`, which is headless-safe;
  - `pytest.skip` when cadquery is absent.
- No figure style in the repo meets the spec's 14 pt rule: `pswr.viz.style` "slide" has 12 pt ticks
  and mutates global rcParams. LTOCs gets its own local style.

### 1.5 GUI integration facts relevant to Phase 5

- Recent tabs (GVWD, PSWR) build their own 3-D canvas, thread only long jobs, and have **no
  in-memory hand-off to the Aero Analysis tab**. `MeshSelectDialog` hard-codes the cone-derived tab;
  every other method hands off via an exported STL/STEP file.
- `_on_main_tab_changed` hides the OC parameter panel only for the cone-derived tab.
- Project save/load covers only five methods.
- These are Phase 5 choices; nothing in Phases 1–4 touches GUI code.

---

## 2. Flags 1–3: confirmation after reading R1 Sec. II and R2 Sec. III.B

All three flags are **confirmed**. The identities below were checked symbolically
(sympy) and numerically in a scratch script outside the repository.

### Flag 1 — sign of R1 Eq. (1): confirmed, it is a typo

- With `n = (Z_X, Z_Y, −1)` and `V∞ = (1, 0, 0)`, `D = (n × V∞) × n` gives
  `dY/dX = −Z_X Z_Y / (1 + Z_Y²)`. On a circular cone this is `Y/X` (the generator),
  while the printed Eq. (1) gives `−Y/X`.
- The sign is the same for all four choices of normal orientation (±n) and freestream sense (±V∞).
  It is also unchanged by flipping Y or Z, because `D` is quadratic in `n`. So it cannot be an
  axis-orientation convention. The printed sign is a typo.
- Cleaner vector form, used in the implementation: `D = V∞ − (V∞·n̂) n̂`, the projection of the
  freestream onto the shock tangent plane. Verified identical to `(n × V∞) × n` up to the factor `|n|²`.
  The angle between `D` and `V∞` is exactly `β` of Eq. (3).

### Flag 2 — the printed unit process is one relation short: confirmed

- **R1 Sec. II.B** (Fig. 3): C2 = streamline through A1 ∩ left-running Mach line through A2.
  The text then says "(P, ρ, V) are obtained by solving the compatibility equations".
- **R2 Sec. III.B** (Fig. 15b), where the noncoaxial MOC originates, gives the same wording for D1
  (streamline through A0 ∩ left-running Mach line through A1).
- Unknowns at the new point: `x, y, θ, P, ρ, V` (6).
  - Available relations: two characteristic directions; Eqs. (4)–(5) along the streamline (link
    `V, ρ` to `P`); Eq. (6) along the one Mach line (links `P, θ`). That is 5.
  - `θ` (equivalently `P`) is left undetermined.
- R3 Sec. 1.1 and Fig. 5 use the same streamline × C+ mesh and cite R1/R2 for the kernel; it adds
  no closure.
- **Neither R1 nor R2 closes the system with one family.** As spec flag 2 instructs, I will implement
  the standard two-family rotational inverse scheme:
  - Zucrow & Hoffman unit processes; R4 Fig. 1 shows the mesh.
  - Interior point = C− from the upstream known point ∩ C+ traced back from the downstream known point.
  - The streamline through the new point is traced back to the previous row to carry entropy;
    total enthalpy is uniform.
  - The noncoaxial `r` rule is applied in both compatibility relations.
- Body streamlines (the waverider lower surface) are traced through that net. The streamline from
  the leading-edge shock point A0 is the body.
- Alternative, not proposed: keep the papers' streamline × C+ mesh and add the C− relation from the
  previous row as the missing equation. It is equivalent in accuracy. I mention it only because it
  reproduces the R1 figure topology exactly.

### Flag 3 — local crosswise radius `r = cos β / κ_b`: confirmed, exact

- `b = n̂ × t̂` is parallel to `n̂ × V∞` (R1's vector **C**), so `b` is exactly the tangent of the
  crosswise section `X = const` (computed `|b_x| < 1e−16`).
- Meusnier's theorem gives `κ_b = κ_cs cos φ`, where `φ` is the angle between the section's principal
  normal and `n̂`. In the plane spanned by `X̂` and `n̂`, `φ = β`, hence `r = 1/κ_cs = cos β / κ_b`
  for any surface.
- Checked on a circular cone and two elliptic cones (a/b = 0.875 as in R1 case I, and 0.42) at
  several azimuths: agreement to round-off (≤ 6e−16 relative).
- The local axis centre is `O = A − r N̂_cs`. It lies in the local osculating plane (span of `X̂`
  and `n̂`), which is why the noncoaxial picture is consistent.

---

## 3. Further reconciliation items (papers vs spec)

### 3.1 Meridian-plane coordinate — needs a decision (recommendation below)

R1 Sec. II.B only says the shock points are "rotated to a same fictional meridian plane" with
`x = X`. R3 Sec. 1.1 says explicitly 绕X轴旋转 ("rotated about the X axis"). Read literally, the
meridian ordinate is `y = √(Y² + Z²)`. Three problems:

1. It depends on where the frame's X axis is put. A shock that is not centred on the X axis (any
   `BSplineShock`, or an osculating-cone shock) maps differently under a transverse translation of
   the frame.
2. For an osculating-cone shock (V5) the stream surfaces are planes that do not contain the X axis.
   The literal mapping bends the straight shock line into a curve, so V5 cannot reproduce the
   osculating-cone waverider.
3. With the literal mapping the meridian shock slope `dy/dx ≠ tan β` (the 3-D shock angle used in
   R-H), and the 3-D speed is not preserved, because the azimuthal velocity component is dropped.

**Recommendation: intrinsic mapping.**
- Keep `x = X`. Along each shock curve set `dy/dx = tan β`, i.e. preserve arc length and the angle to
  the freestream.
- Set the axis centre at `y_O = y_A − r_A`.
- This is exactly the local osculating plane developed along the stream surface. It is
  frame-invariant and coincides with the literal mapping in the symmetry plane and for any shock
  curve lying in a plane through the X axis (circular cones).
- It is also what makes R1's Step C consistent (§3.3).

Quantified on R1's own waverider shocks, along the shock curves from leading edge to base:

| Case | FCT station Y | Azimuth drift of shock curve | Literal vs intrinsic meridian rise |
|---|---|---|---|
| I (Mach 6) | 0 / 0.2 / 0.4 / 0.55 | 0 / −5.6 / −1.3 / −0.2 deg | 0 / 0.51 / 0.20 / 0.12 % |
| II (Mach 7) | 0 / 0.2 / 0.4 / 0.55 | 0 / −1.7 / −2.2 / −0.7 deg | 0 / 0.73 / 0.65 / 0.75 % |

The choice is therefore unlikely to move the V6/V7 coefficients by more than a fraction of the 1 %
target. If V6/V7 miss, I would add the literal mapping as an option and compare.

### 3.2 Noncoaxial `r` rule and the axis-centre labels

- R2 Fig. 15b draws each axis centre directly below a shock point: `O_{k+1}` is the centre of `A_k`,
  so `|A_k O_{k+1}|` is the radius of `A_k`.
- R1's text ("r = |A1O2|, the local radius of shock point A1") uses R2's labels. R1's own Fig. 3
  labels the centres differently. The rule itself is unambiguous.
- I implement it as one continuous function `r(x, y) = y − y_O(x)`:
  - `y_O(x)` is interpolated along the shock data of the stream surface.
  - Shock points get their own radius; an interior point gets the centre of the shock point at the
    same axial position, as R1 states.
  - The paper's index offsets are O(Δx) and vanish with refinement.
- The coaxial rule (`y_O ≡ 0`, i.e. ordinary axisymmetric MOC) and the planar limit (`δ = 0`) are
  the same function with different arguments. This is the pluggable rule the spec asks for.

### 3.3 Step C (R1 Eqs. 10–11)

- Verified symbolically: Eqs. (10)–(11) are exactly `j ∥ V − (V·i) i/|i|²` with `Xj` imposed. The
  spec's vector form is correct.
- With the intrinsic mapping (§3.1), imposing `Xj` from the meridian plane is the same as imposing
  the meridian length `|C2 C2′|`. With the literal mapping it is not.
- R1 Fig. 4 shows deeper rows use the previous-row segment (D2′ lies on D1–C2), not the shock.
  I apply Step C row by row in the same way.

### 3.4 Force routine (R1 Eqs. 19–22)

- Eq. (19) prints `A_w = |AC·DB|` with no ½. The true area of a quadrilateral is `½|AC × DB|`.
  The factor cancels in `C_L, C_D` only if the reference area `A` is the sum of the same `A_w`.
- I will implement the correct area and report both readings at Gate 4 against Tables 2 and 3.
- The spec's inference stands: `A` is the lower-surface wetted area. R1 Eq. (22) says "total wetted
  area of the waverider", but Sec. IV.A defines "waverider" as the lower surface. Absolute pressure,
  no upper or base contribution, is to be confirmed in Phase 4.

### 3.5 Frame of the published waverider cases

- R1 Figs. 20, 21 and 24 draw Z pointing down. The FCTs `Z = 0.12` (case I) and `Z = 0.25`
  (case II) lie on the +Z side, between the axis and the base-plane curvature centres.
- I computed those centres at `Z = 0.1594` and `Z = 0.2719`. This is exactly R1's point that these
  FCTs sit "beneath part of the curvature centres", so osculating cones cannot design them.
- So the lower surface is on the +Z side of R1's frame, and lift is −Z in that frame. The adapter
  maps R1 `(X, Y, Z_down)` to the WADOS frame (§1).

### 3.6 Spec numbers checked against the papers

- V6 = R1 Table 2 LTOCs row (0.2205 / 0.0750 / 2.9378) ✓
- V7 = R1 Table 3 LTOCs row (0.2034 / 0.0648 / 3.1395) ✓
- Eqs. (23)–(26) ✓
- V8 = Eq. (18) and Fig. 16 ✓
- R2 Table 2 (W 0.9439, H 0.1817, A_w 0.1166) ✓
- Conditions 1880 Pa / 223.54 K ✓
- Shock-angle ranges on the published shocks, against the Mach angles:
  - case I: 19.3–21.8° (Mach angle 9.6°)
  - case II: 14.0–23.2° (Mach angle 8.2°)
  - So the "β above the Mach angle" requirement (R3 Sec. 1.2) is comfortably met.
- R3 Sec. 1.4 confirms flags 4 (shock extended past the design plane, citing [34] = R2) and 5
  (pressure-gradient-triggered linear insertion along streamlines, Fig. 5).
- R4 Sec. 2 confirms flag 6 (limit surfaces).

### 3.7 V4 and V5: what "identical" can mean

- **V5 is a clean identity test, but only with the intrinsic mapping.**
  - For an osculating-cone shock (constant β), the surface is ruled:
    `S(s, t) = C(s) + t·g(s)`, with `g` in the plane normal to the SWPC.
  - `∂S/∂s = T (1 − t tanβ κ) ∥ T`, so the shock normal lies exactly in the osculating plane.
  - So LTOCs' shock curves *are* the osculating-plane generators.
  - The crosswise section at fixed X is the parallel curve of the SWPC, so `r` equals the local cone
    radius and the axis centre is fixed.
  - The noncoaxial MOC therefore reduces to the coaxial one, i.e. conical flow.
  - With the literal "rotate about X" mapping this fails, because those planes do not contain the
    X axis.
- **The comparison is limited by the existing generators' own Taylor–Maccoll accuracy.**
  - The OC generator uses solve_ivp default tolerances and smoothing splines.
  - `ShadowWaverider` uses max_step 1e-3 rad and linear interpolation, which is probably adequate.
  - I will measure each generator against the tight TM reference first, then apply the spec
    tolerance (1e-4 of length) against that reference. The deviation from each existing generator is
    reported separately.

---

## 4. Proposed file layout

A new package, additive only, numpy/scipy/matplotlib only:

```
ltoc/
├── __init__.py            public API; frame and references in the docstring
├── moc_noncoaxial.py      Phase 1  two-family rotational inverse MOC; pluggable r rule
│                                   r(x,y) = y − y_O(x)  (coaxial: y_O ≡ 0; planar: δ = 0)
├── reference.py           Phase 1  tight conical-flow reference (liu2019 TM + gvwd R-H) for V1/V4
├── shock_surface.py       Phase 2  ShockSurface ABC; ConeShock, RuledShock (R1 Eq. 12),
│                                   OsculatingConeShock (from an OC `waverider`), BSplineShock
├── shock_geometry.py      Phase 2  β, t, b, κ_b, r, axis centre, RK4 shock-curve integrator
├── ltoc.py                Phase 3  per-stream-surface solve: Steps A–C, streamline extraction,
│                                   determinacy margin, limit-surface detection, diagnostics
├── waverider.py           Phase 3  LTOCWaverider: FCT → leading edge → lower/upper surfaces;
│                                   OC stream protocol; export_stl / export_tri / to_CAD
├── forces.py              Phase 4  R1 Eqs. 19–22 panel forces (correct ½ area; both readings reported)
├── cases.py               Phase 4  published cases (R1 Eqs. 18, 23–26; R2 Eq. 13) + reference numbers
├── viz.py                 gate figures: ≥ 14 pt labels/ticks/legends, centred titles, Figure API
├── docs/
│   ├── methodology.md     Phase 5
│   └── validation.md      filled gate by gate (V1–V9 tables)
└── tests/
    ├── __init__.py
    ├── test_moc_kernel.py        V1, V2
    ├── test_shock_geometry.py    V3
    ├── test_ltoc_core.py         V4, V5
    ├── test_published_cases.py   V6, V7, V8
    └── test_robustness.py        V9
ltoc_waverider_tab.py      Phase 5
docs/ltoc/                 spec, gate reports, gate figures
docs/refs/                 papers (R1–R4) + index
```

Existing methods are not modified.

Shared-file changes, each proposed now and made only after approval:
1. **Phase 1:** a `pytest ltoc/tests/ -q` step in `.github/workflows/tests.yml`.
2. **Phase 5:** the standard GUI import and `addTab` block in `waverider_gui.py`; the README methods
   table and testing section.
3. **Phase 5, optional:** an "LTOCs" source in `MeshSelectDialog` for an in-memory aero hand-off.

---

## 5. Decisions needed before Phase 1

1. **Meridian mapping (§3.1).** Intrinsic (recommended) or literal R3 "rotate about X"?
   - Intrinsic is frame-invariant and required for V5.
   - On R1's published cases the two differ by < 0.8 % of the meridian rise.
2. **CI step (§4, item 1).** OK to add `pytest ltoc/tests/ -q` to `tests.yml` in Phase 1?
3. **V4/V5 tolerance basis (§3.7).** OK to apply the 1e-4 L criterion against a tight TM reference,
   and report the deviation from the existing OC and cone-derived generators separately?

### What in the spec turned out to be wrong or incomplete

- §6 "the same geometry object the other methods return": there is no such object. Use the OC stream
  protocol (§1.3).
- §4.2 "rotated about the X axis": R3 does say so, but the literal reading is frame-dependent and
  breaks V5 (§3.1).
- §4.4 / Eq. (19): R1 omits the ½ in the quadrilateral area (§3.4).
- Flag 2: confirmed. The unit-process figure in R1 (streamline × C+) is a valid mesh topology; only
  the closure relation is missing.
- Everything else in §§3–7 checks out against the papers.
