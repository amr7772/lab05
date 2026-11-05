"""Train, check and save the late-delivery model.

Run with:  python -m latedeliveries.train

Steps 1 to 4 only ever see the training data. The test set is scored once,
at step 5. Running this again rebuilds every report and the saved model.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from latedeliveries.data import make_orders
from latedeliveries.evaluation import choose_threshold, evaluate_at_threshold
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
    SEED,
    build_pipeline,
    split_orders,
)

ROOT = Path(__file__).resolve().parents[2]

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}

PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}


def make_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def candidate_models() -> dict:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000),
        "logistic_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(random_state=SEED),
        "random_forest_balanced": RandomForestClassifier(
            random_state=SEED, class_weight="balanced"
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def compare_models(x_train, y_train) -> pd.DataFrame:
    """Five-fold stratified cross-validation of every candidate, train only."""
    rows = []
    for name, model in candidate_models().items():
        results = cross_validate(
            build_pipeline(model),
            x_train,
            y_train,
            cv=make_folds(),
            scoring=SCORING,
        )
        row = {"model": name}
        for metric in SCORING:
            scores = results[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(scores.mean()), 4)
            row[f"{metric}_std"] = round(float(scores.std()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def tune_logistic(x_train, y_train) -> GridSearchCV:
    """Grid search on the logistic pipeline, scored by F1 (training only)."""
    search = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=1000)),
        PARAM_GRID,
        scoring="f1",
        cv=make_folds(),
        n_jobs=-1,
    )
    return search.fit(x_train, y_train)


def out_of_fold_probabilities(pipe, x_train, y_train):
    """Probability of late for each training order, from a fold that never saw it."""
    return cross_val_predict(
        pipe, x_train, y_train, cv=make_folds(), method="predict_proba"
    )[:, 1]


def main() -> None:
    orders = make_orders()
    x_train, x_test, y_train, y_test = split_orders(orders)

    # 2. Compare against a baseline, training data only.
    cv_results = compare_models(x_train, y_train)
    print(cv_results.to_string(index=False))

    # 3. Tune the logistic model. F1 is the score because late orders are the
    # minority and both mistakes matter. Accuracy would reward flagging nobody.
    search = tune_logistic(x_train, y_train)
    best_pipe = search.best_estimator_
    print("Best settings:", search.best_params_)
    print("Best cross-validated F1:", round(search.best_score_, 4))

    # 4. Threshold from the costs, using out-of-fold training probabilities.
    train_proba = out_of_fold_probabilities(best_pipe, x_train, y_train)
    threshold, train_cost = choose_threshold(y_train, train_proba)
    print("Chosen threshold:", threshold, "| training cost:", train_cost)

    # 5. Fit on all the training data, score the test set once.
    best_pipe.fit(x_train, y_train)
    test_proba = best_pipe.predict_proba(x_test)[:, 1]
    test_result = evaluate_at_threshold(y_test, test_proba, threshold)
    print("Test set (scored once):", test_result)

    # 6. Save everything.
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    cv_results.to_csv(ROOT / "reports" / "cv_results.csv", index=False)
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps(test_result, indent=2), encoding="utf-8"
    )
    joblib.dump(best_pipe, ROOT / "models" / "late_orders.joblib")
    model_info = {
        "scikit_learn_version": sklearn.__version__,
        "seed": SEED,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_f1": round(float(search.best_score_), 4),
        "train_cost_at_threshold": int(train_cost),
        "test": test_result,
    }
    (ROOT / "models" / "model_info.json").write_text(
        json.dumps(model_info, indent=2), encoding="utf-8"
    )
    print("Saved models/late_orders.joblib and the files in reports/")


if __name__ == "__main__":
    main()
