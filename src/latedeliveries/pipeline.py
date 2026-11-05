"""Feature lists, the train/test split and the preprocessing pipeline.

The model is used at the moment an order is placed, so a column is only
a legitimate feature if its value is known at that moment. Three columns
fail that test and are left out on purpose:

* order_id is a label for the order, not a property of it.
* prep_minutes is how long the kitchen actually took. Nobody knows that
  when the order is placed.
* customer_rating is given after delivery and is partly made from
  whether the order was late, so using it would leak the answer.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "late"
NUMERIC_FEATURES = ["distance_km", "items", "hour", "courier_months"]
CATEGORICAL_FEATURES = ["weather", "day_type", "vehicle"]
EXCLUDED_FEATURES = ["order_id", "prep_minutes", "customer_rating"]


def split_orders(orders: pd.DataFrame):
    """Return x_train, x_test, y_train, y_test (75/25, stratified on late)."""
    features = orders[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return train_test_split(
        features,
        orders[TARGET],
        test_size=0.25,
        stratify=orders[TARGET],
        random_state=SEED,
    )


def build_pipeline(model) -> Pipeline:
    """Wrap any scikit-learn classifier in the shared preprocessing.

    Numbers get the training median for missing values and are scaled.
    Categories are one-hot encoded; a category never seen in training
    becomes all zeros instead of raising an error.
    """
    preprocess = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("prep", preprocess), ("model", model)])
