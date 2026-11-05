"""Fixtures shared by the tests."""

import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import build_pipeline, split_orders


@pytest.fixture
def orders():
    return make_orders()


@pytest.fixture
def split(orders):
    return split_orders(orders)


@pytest.fixture
def fitted_pipeline(split):
    x_train, _x_test, y_train, _y_test = split
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(x_train, y_train)
    return pipe
