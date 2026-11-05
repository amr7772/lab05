"""The company cost rule, the threshold choice and the test-set scores.

Every flagged order costs a $4 voucher, whether or not it turns out to
be late. Every late order that was not flagged costs $15 in refunds and
lost repeat business.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

VOUCHER_COST = 4
MISSED_LATE_COST = 15
THRESHOLDS = [round(step * 0.05, 2) for step in range(1, 20)]  # 0.05 .. 0.95


def order_cost(y_true, flagged) -> int:
    """Total cost of a set of flag decisions."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)
    vouchers = int(flagged.sum())
    missed_late = int(((y_true == 1) & ~flagged).sum())
    return vouchers * VOUCHER_COST + missed_late * MISSED_LATE_COST


def choose_threshold(y_true, probabilities):
    """Return (threshold, cost) for the cheapest threshold from 0.05 to 0.95.

    An order is flagged when its probability is at least the threshold.
    On a tie the smaller threshold wins, because it is tried first.
    """
    probabilities = np.asarray(probabilities)
    best_threshold, best_cost = None, None
    for threshold in THRESHOLDS:
        cost = order_cost(y_true, probabilities >= threshold)
        if best_cost is None or cost < best_cost:
            best_threshold, best_cost = threshold, cost
    return best_threshold, best_cost


def cost_curve(y_true, probabilities):
    """Return a list of (threshold, cost) pairs, for plotting."""
    probabilities = np.asarray(probabilities)
    return [(t, order_cost(y_true, probabilities >= t)) for t in THRESHOLDS]


def evaluate_at_threshold(y_true, probabilities, threshold) -> dict:
    """Confusion matrix, scores and the cost of the three policies."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    flagged = probabilities >= threshold
    predicted = flagged.astype(int)
    return {
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "precision": round(precision_score(y_true, predicted, zero_division=0), 4),
        "recall": round(recall_score(y_true, predicted, zero_division=0), 4),
        "f1": round(f1_score(y_true, predicted, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probabilities), 4),
        "flagged": int(flagged.sum()),
        "cost_flag_nobody": order_cost(y_true, np.zeros(len(y_true), dtype=bool)),
        "cost_flag_everybody": order_cost(y_true, np.ones(len(y_true), dtype=bool)),
        "cost_model": order_cost(y_true, flagged),
    }
