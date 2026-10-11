#!/usr/bin/env python3
"""Train the Volume surrogate (ensemble_Volume.pkl) on exact OC waverider volumes.

The Off-Design NN and Multi-Mach tabs use ``ensemble_Volume.pkl`` to predict
the internal volume of an OC waverider from its geometry features
(``volume_feature_cols`` in ``config.json``):

    X1, X2, X3, X4, design_Mach, design_beta, width, height

Two steps::

    python surrogate_model/train_volume_model.py generate   # data, ~5 min on 4 cores
    python surrogate_model/train_volume_model.py train      # ensemble + config.json metrics

**Data.** Designs are sampled the way the hunters sample them: design Mach
3-6, shock angle from the Mach angle + 1 deg to 30 deg, X1 0-0.5, X2-X4
0-1, width 0.5-5 m, height 0.1-5 m, inside the OC design space
X2/(1-X1)^4 < 0.9 (7/64) (W/H)^4. Half the shock angles are drawn close to
the lower limit, where the body is thin and its volume changes fastest. The volume is the volume enclosed by the
closed full-span surface mesh (``geometry_export.stream_volume``), as the
OC tab reports it. The OC geometry scales exactly with its size (at a fixed
width/height ratio, V = H^3 V(H=1)), so each shape is generated once at
unit height and gives several samples at other heights.

**Model.** Same file format as before (an ``NNEnsemble``: ``models``,
``scaler_X``, ``scaler_y``), so the tabs load it unchanged. Each model is a
scikit-learn pipeline that learns log V from X1-X4, M, beta, log W, log H and
log(beta - Mach angle), built from scikit-learn and NumPy parts only (so the
file loads without this script); the ensemble's scalers are identities. Train with the scikit-learn version
the tabs run (the previous models were 1.7.2).
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "volume_training_data.csv.gz"
MODEL = HERE / "ensemble_Volume.pkl"
CONFIG = HERE / "config.json"
FEATURES = ["X1", "X2", "X3", "X4", "design_Mach", "design_beta", "width", "height"]

W_RANGE, H_RANGE = (0.5, 5.0), (0.1, 5.0)
N_PLANES, N_STREAMWISE = 40, 30          # volume converged to 3e-5 (vs 80 x 60)


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def sample_designs(n: int, seed: int) -> list:
    """Hunter-like samples (X1-X4, M, beta, W, H) inside the OC design space."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        M = rng.uniform(3.0, 6.0)
        beta_min = math.degrees(math.asin(1.0 / M)) + 1.0
        if rng.uniform() < 0.5:
            beta = rng.uniform(beta_min, 30.0)
        else:                                       # dense near the lower limit
            beta = min(beta_min + 10.0 ** rng.uniform(-2.0, 1.0), 30.0)
        X1, X2, X3, X4 = rng.uniform(0.0, 0.5), rng.uniform(0.0, 1.0), rng.uniform(), rng.uniform()
        W, H = rng.uniform(*W_RANGE), rng.uniform(*H_RANGE)
        if X2 / (1.0 - X1) ** 4 < 0.9 * (7.0 / 64.0) * (W / H) ** 4:
            out.append((X1, X2, X3, X4, M, beta, W, H))
    return out


def unit_volume(design) -> float:
    """Exact volume at unit height (width W/H), or NaN if the generator fails."""
    import warnings

    sys.path.insert(0, str(ROOT))
    from geometry_export import stream_volume
    from waverider_generator.generator import waverider

    X1, X2, X3, X4, M, beta, W, H = design
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wr = waverider(M_inf=M, beta=beta, height=1.0, width=W / H, dp=[X1, X2, X3, X4],
                           n_upper_surface=10000, n_shockwave=10000, n_planes=N_PLANES,
                           n_streamwise=N_STREAMWISE, delta_streamwise=0.1)
            return stream_volume(wr.upper_surface_streams, wr.lower_surface_streams, full_span=True)
    except Exception:
        return float("nan")


def generate(n_shapes: int = 6000, per_shape: int = 3, seed: int = 2026, workers: int = 0):
    import multiprocessing as mp

    import pandas as pd

    designs = sample_designs(n_shapes, seed)
    workers = workers or os.cpu_count() or 1
    with mp.get_context("spawn").Pool(workers) as pool:
        v1 = pool.map(unit_volume, designs, chunksize=20)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for d, v in zip(designs, v1):
        if not np.isfinite(v) or v <= 0.0:
            continue
        X1, X2, X3, X4, M, beta, W, H = d
        r = W / H
        h_lo, h_hi = max(H_RANGE[0], W_RANGE[0] / r), min(H_RANGE[1], W_RANGE[1] / r)
        heights = [H] + list(rng.uniform(h_lo, h_hi, per_shape - 1))   # the sample's own size first
        for h in heights:
            rows.append((X1, X2, X3, X4, M, beta, r * h, h, v * h ** 3))
    df = pd.DataFrame(rows, columns=FEATURES + ["Volume"])
    df.to_csv(DATA, index=False, float_format="%.10g")
    n_fail = sum(not (np.isfinite(v) and v > 0) for v in v1)
    print(f"{len(designs)} shapes ({n_fail} failed) -> {len(df)} samples: {DATA.name}")


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

def _model(seed: int):
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import FunctionTransformer as F, StandardScaler

    # log(beta - Mach angle) in radians, clipped above 0 for inputs below the
    # Mach angle (no attached shock; the prediction is then meaningless)
    log_gap = Pipeline([
        ("angles", ColumnTransformer([
            ("mach_angle", Pipeline([("inv", F(np.reciprocal)), ("asin", F(np.arcsin))]), [4]),
            ("beta", F(np.radians), [5])])),                       # -> [mu, beta]
        ("gap", F(np.diff, kw_args={"axis": 1})),                 # -> beta - mu
        ("clip", F(np.clip, kw_args={"a_min": 1e-4, "a_max": None})),
        ("log", F(np.log)),
    ])
    features = FeatureUnion([
        ("base", ColumnTransformer([("log_size", F(np.log), [6, 7])], remainder="passthrough")),
        ("log_gap", log_gap),
    ])
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", alpha=1e-5,
                       learning_rate_init=1e-3, max_iter=3000, early_stopping=True,
                       n_iter_no_change=50, tol=1e-7, random_state=seed)
    net = Pipeline([("features", features), ("scale", StandardScaler()), ("mlp", mlp)])
    return TransformedTargetRegressor(regressor=net, func=np.log, inverse_func=np.exp,
                                      check_inverse=False)


def _identity_scaler(X):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler(with_mean=False, with_std=False).fit(X)


def predict(ensemble: dict, X) -> tuple:
    """Mean and spread, as the tabs' NNEnsemble.predict computes them."""
    Xs = ensemble["scaler_X"].transform(X)
    P = np.array([ensemble["scaler_y"].inverse_transform(m.predict(Xs).reshape(-1, 1)).ravel()
                  for m in ensemble["models"]])
    return P.mean(axis=0), P.std(axis=0)


def metrics(y, p) -> dict:
    err = p - y
    return {"R2": float(1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MAPE": float(100.0 * np.mean(np.abs(err) / y))}


def train(n_models: int = 5, seed: int = 42, test_fraction: float = 0.2):
    import pandas as pd
    import sklearn

    df = pd.read_csv(DATA)
    X, y = df[FEATURES].to_numpy(float), df["Volume"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(test_fraction * len(df))
    test, fit = idx[:n_test], idx[n_test:]
    models = []
    for k in range(n_models):
        boot = rng.choice(fit, size=len(fit), replace=True)        # bagging for the spread
        models.append(_model(seed + k).fit(X[boot], y[boot]))
        print(f"model {k + 1}/{n_models} fitted")
    ensemble = {"n_models": n_models, "hidden_layers": (128, 64, 32), "models": models,
                "scaler_X": _identity_scaler(X[fit]),
                "scaler_y": _identity_scaler(y[fit].reshape(-1, 1)), "is_fitted": True}
    p, _ = predict(ensemble, X[test])
    m = metrics(y[test], p)
    print("held-out:", {k: round(v, 5) for k, v in m.items()})
    with open(MODEL, "wb") as f:
        pickle.dump(ensemble, f)
    config = json.loads(CONFIG.read_text())
    config["metrics"]["Volume"] = m
    config["volume_model"] = {
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "input_file": DATA.name,
        "n_samples": int(len(df)),
        "volume": "exact: closed full-span surface mesh (geometry_export.stream_volume)",
        "model": "log V from X1-X4, design_Mach, design_beta, log width, log height, "
                 "log(beta - Mach angle)",
        "scikit_learn": sklearn.__version__,
        "script": "surrogate_model/train_volume_model.py",
    }
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    print(f"written {MODEL.name} and {CONFIG.name}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "generate":
        generate()
    elif step == "train":
        train()
    else:
        sys.exit(__doc__)
