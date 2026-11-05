"""Tests for saving the whole pipeline and loading it back."""

import joblib
import numpy as np
import pandas as pd


def test_reloaded_model_predicts_the_same(tmp_path, fitted_pipeline, split):
    _x_train, x_test, _y_train, _y_test = split
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    loaded = joblib.load(path)
    assert np.array_equal(
        fitted_pipeline.predict_proba(x_test), loaded.predict_proba(x_test)
    )


def test_saved_model_reads_a_raw_order(tmp_path, fitted_pipeline):
    raw_order = pd.DataFrame(
        [
            {
                "distance_km": 6.5,
                "items": 4,
                "hour": 19,
                "courier_months": 3,
                "weather": "rain",
                "day_type": "weekend",
                "vehicle": "bike",
            }
        ]
    )
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    assert joblib.load(path).predict(raw_order).shape == (1,)
