"""Tests for the cost function and the threshold choice, checked by hand."""

import pytest

from latedeliveries.evaluation import choose_threshold, order_cost


def test_cost_of_a_small_example():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, True, False, False, False]
    # two vouchers (2 x 4) and one missed late order (1 x 15)
    assert order_cost(y_true, flagged) == 2 * 4 + 1 * 15


def test_flagging_nobody_is_cheapest_when_nothing_is_late():
    threshold, cost = choose_threshold([0, 0, 0], [0.9, 0.8, 0.7])
    assert (threshold, cost) == (0.95, 0)


def test_flagging_everybody_is_cheapest_when_everything_is_late():
    threshold, cost = choose_threshold([1, 1, 1], [0.9, 0.8, 0.7])
    assert (threshold, cost) == (0.05, 3 * 4)


@pytest.mark.parametrize(
    "probabilities, expected_threshold, expected_cost",
    [
        # late, on time, late. At 0.05 all three flagged: 12. At 0.15 the two
        # late ones flagged: 8. At 0.85 only one flagged, one missed: 4 + 15.
        ([0.9, 0.1, 0.8], 0.15, 8),
        # every probability is 0.4: flag all (cost 12) or none (cost 30).
        ([0.4, 0.4, 0.4], 0.05, 12),
    ],
)
def test_threshold_choice_on_worked_examples(
    probabilities, expected_threshold, expected_cost
):
    threshold, cost = choose_threshold([1, 0, 1], probabilities)
    assert threshold == expected_threshold
    assert cost == expected_cost
