"""Tests for the feature list, the split and the pipeline."""

import numpy as np
import pandas as pd
import pytest

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    split_orders,
)


def make_order(**changes):
    order = {
        "distance_km": 4.0,
        "items": 2,
        "hour": 12,
        "courier_months": 6,
        "weather": "clear",
        "day_type": "weekday",
        "vehicle": "car",
    }
    order.update(changes)
    return pd.DataFrame([order])


def test_feature_list_excludes_leaky_columns_and_target():
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for column in EXCLUDED_FEATURES:
        assert column not in features
    assert TARGET not in features


def test_split_is_stratified_and_repeatable(orders):
    first = split_orders(orders)
    second = split_orders(orders)
    x_train, x_test, y_train, y_test = first
    assert len(x_train) == 2250
    assert len(x_test) == 750
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.005)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[3], second[3])


def test_probabilities_are_valid(fitted_pipeline, split):
    _x_train, x_test, _y_train, _y_test = split
    proba = fitted_pipeline.predict_proba(x_test)
    assert proba.shape == (len(x_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_missing_value_does_not_break_the_pipeline(fitted_pipeline):
    order = make_order(distance_km=np.nan)
    assert fitted_pipeline.predict_proba(order).shape == (1, 2)


def test_unseen_category_does_not_break_the_pipeline(fitted_pipeline):
    order = make_order(weather="hail")
    proba = fitted_pipeline.predict_proba(order)
    assert 0 <= proba[0, 1] <= 1


@pytest.mark.parametrize("weather", ["clear", "rain", "snow"])
def test_each_known_weather_predicts(fitted_pipeline, weather):
    assert fitted_pipeline.predict(make_order(weather=weather)).shape == (1,)
