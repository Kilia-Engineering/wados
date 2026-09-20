"""Paper parameters from Liu et al. 2019 Tables 1, 2 and 4."""

# Paper Table 1 — design parameters (SI: metres, degrees, radians)
PAPER_PARAMS = {
    "beta_deg":  13.0,
    "L_w":       6.000,     # m
    "W":         3.000,     # m
    "L_s":       0.300,     # m
    "y5":        0.1608,    # m  (tip upper-surface height)
    "z5":        1.500,     # m  (tip spanwise position = W/2)
    "y6":        1.608,     # m  (centreline upper-surface height)
    "z6":        0.0,       # m
    "delta5":    0.0,       # rad
    "delta6":    0.0,       # rad
    "Ma_center": 6.0,
    "Ma_tip":    13.0,
    "gamma":     1.4,
}

# Paper Table 2 — constant-q trajectory (q ~ 64.2 kPa)
PAPER_TRAJECTORY = [
    {"Ma":  6, "alpha": 0.0, "H_km": 25.0, "T": 221.6, "P": 2549.0, "rho": 4.01e-2, "q": 64.3e3, "a": 298.40},
    {"Ma":  8, "alpha": 0.0, "H_km": 28.8, "T": 225.3, "P": 1443.0, "rho": 2.22e-2, "q": 64.2e3, "a": 300.00},
    {"Ma": 10, "alpha": 0.0, "H_km": 31.8, "T": 228.3, "P":  916.0, "rho": 1.40e-2, "q": 64.1e3, "a": 301.97},
    {"Ma": 13, "alpha": 0.0, "H_km": 35.4, "T": 237.6, "P":  543.0, "rho": 7.96e-3, "q": 64.2e3, "a": 309.02},
]

# Paper Table 4 — reference geometric metrics for waverider_M6-M13
PAPER_REFERENCE_GEOMETRY = {
    "Vol_m3":   3.02,
    "S_wet_m2": 26.23,
    "S_p_m2":    9.59,
    "S_b_m2":    1.42,
    "eta":       0.0797,
}

# Paper Fig. 12 — values as originally read off the graph (+/-5% claimed).
#
# These are kept verbatim for traceability ONLY. Two of the five columns are
# provably mis-scaled: any set of aerodynamic coefficients normalised by a
# common reference area must satisfy
#
#     (1)  CL / CD   == L/D
#     (2)  Cmz / CL  == Xcp          (Cmz about the nose, over L_ref)
#
# Neither holds, and the violation drifts with Mach, so it is not a single
# missing constant:
#
#     Ma      CL/CD   vs L/D      Cmz/CL   vs Xcp
#      6      0.227   19.4x       2.971    4.38x
#      8      0.198   25.3x       3.438    5.06x
#     10      0.181   29.9x       3.692    5.45x
#     13      0.173   33.5x       4.000    5.90x
#
# A wrong CL cannot explain it either: identity (1) would need CL = CD*(L/D)
# = 3.388 at Ma 6, identity (2) would need CL = Cmz/Xcp = 0.767. Those
# disagree, so at least two columns are bad, and the minimal consistent
# reading is that CD and Cmz are the mis-scaled ones.
#
# This is NOT a reference-area problem. Solving for the S_ref that would
# reconcile each column against the evaluator gives 1.837 m^2 from CL,
# 0.067 m^2 from CD and 0.411 m^2 from Cmz -- and a reference area is one
# number. (L/D and Xcp are S_ref-free, so they are unaffected either way.)
PAPER_FIG12_AS_READ = {
    6:  {"CL": 0.175, "CD": 0.77, "L_D": 4.4, "Cmz": 0.52, "Xcp": 0.678},
    8:  {"CL": 0.160, "CD": 0.81, "L_D": 5.0, "Cmz": 0.55, "Xcp": 0.679},
    10: {"CL": 0.130, "CD": 0.72, "L_D": 5.4, "Cmz": 0.48, "Xcp": 0.678},
    13: {"CL": 0.090, "CD": 0.52, "L_D": 5.8, "Cmz": 0.36, "Xcp": 0.678},
}

# Reference values actually used for validation. ``None`` means "no usable
# reference" -- consumers skip those rather than scoring the solver against
# a number known to be wrong. Re-reading CD and Cmz off Fig. 12 (or taking
# them from the paper's text) is the only way to restore them; until then
# the honest comparison is CL, L/D and Xcp.
PAPER_REFERENCE_AERO = {
    Ma: {
        "CL":  row["CL"],
        "CD":  None,          # mis-scaled, see above
        "L_D": row["L_D"],
        "Cmz": None,          # mis-scaled, see above
        "Xcp": row["Xcp"],
    }
    for Ma, row in PAPER_FIG12_AS_READ.items()
}

# Aerodynamic reference dimensions (paper Section 4.2)
REF_LENGTH_M = 6.0
REF_AREA_M2  = 1.0
MOMENT_REF   = (0.0, 0.0, 0.0)   # pitching moment taken about the nose

# Acceptance tolerances for validation (fractional deviation)
TOLERANCES = {
    "Vol_m3":   0.03,
    "S_wet_m2": 0.02,
    "S_p_m2":   0.01,
    "S_b_m2":   0.03,
    "eta":      0.03,
    "CL":       0.10,
    "CD":       0.10,
    "L_D":      0.10,
    "Cmz":      0.15,
    "Xcp":      0.05,
}
