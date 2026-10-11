"""The shipped Volume surrogate (surrogate_model/ensemble_Volume.pkl) must
predict the exact OC waverider volume, as the OC tab reports it.

It is loaded and evaluated the way the Off-Design NN and Multi-Mach tabs do
(``NNEnsemble``: scaler_X, models, scaler_y). Needs scikit-learn.
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
MODEL_DIR = Path(__file__).resolve().parents[2] / "surrogate_model"


def _predict(ensemble, X):
    Xs = ensemble["scaler_X"].transform(X)
    P = np.array([ensemble["scaler_y"].inverse_transform(m.predict(Xs).reshape(-1, 1)).ravel()
                  for m in ensemble["models"]])
    return P.mean(axis=0), P.std(axis=0)


def _exact(x):
    from geometry_export import stream_volume
    from waverider_generator.generator import waverider

    X1, X2, X3, X4, M, beta, W, H = x
    wr = waverider(M_inf=M, beta=beta, height=H, width=W, dp=[X1, X2, X3, X4],
                   n_upper_surface=10000, n_shockwave=10000, n_planes=40, n_streamwise=30,
                   delta_streamwise=0.1)
    return stream_volume(wr.upper_surface_streams, wr.lower_surface_streams, full_span=True)


def test_volume_surrogate_predicts_exact_volumes():
    with open(MODEL_DIR / "ensemble_Volume.pkl", "rb") as f:
        ensemble = pickle.load(f)
    rng = np.random.default_rng(11)
    designs = [[0.11, 0.63, 0.0, 0.46, 5.0, 15.0, 3.0, 1.34]]          # the OC tab's default
    while len(designs) < 6:                                              # Off-Design hunter samples
        M = float(rng.choice([3.0, 4.0, 5.0, 6.0]))
        x = [rng.uniform(0.05, 0.45), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9),
             rng.uniform(0.1, 0.9), M, rng.uniform(max(15.0, math.degrees(math.asin(1 / M)) + 1), 26.0),
             rng.uniform(1.0, 3.0), rng.uniform(0.5, 2.5)]
        if x[1] / (1 - x[0]) ** 4 < 0.85 * 7 / 64 * (x[6] / x[7]) ** 4:
            designs.append(x)
    mean, std = _predict(ensemble, np.array(designs))
    exact = np.array([_exact(x) for x in designs])
    assert mean == pytest.approx(exact, rel=0.03)
    assert np.all(std < 0.05 * exact)


def test_config_records_the_exact_volume_model():
    config = json.loads((MODEL_DIR / "config.json").read_text())
    assert "exact" in config["volume_model"]["volume"]
    assert config["metrics"]["Volume"]["MAPE"] < 1.0
    assert config["volume_feature_cols"] == ["X1", "X2", "X3", "X4", "design_Mach", "design_beta",
                                             "width", "height"]
