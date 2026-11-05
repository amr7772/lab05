# Deliveries

QuickBite wants to know, at the moment an order is placed, whether it will arrive late. If it can tell, it sends the customer a $4 voucher before the complaint comes. This project builds a scikit-learn pipeline that predicts late orders, picks the flagging threshold from the company's own costs, and saves the model so another program can load it.

## Setup

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

I install the pinned requirements first and the package second, so `requirements.txt` never picks up a `-e` line with a path from my own computer.

## Run

```bash
python -m latedeliveries.train
```

One command does all of it: split, compare, tune, choose the threshold, score the test set once, and write `models/late_orders.joblib`, `models/model_info.json`, `reports/cv_results.csv` and `reports/metrics.json`. The `models/` and `reports/` folders are committed so you can see the results without running anything.

## Check it

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Settings

- Seed 42 for the split, the cross-validation folds and the random forest. The data has its own seed (7) inside `data.py`, which I did not touch.
- Split: stratified, 75 percent training (2,250 orders) and 25 percent test (750 orders). Late rate is 0.156 in both.
- Cross-validation: five stratified, shuffled folds.
- Grid search: `model__C` in 0.01, 0.1, 1, 10 and `model__class_weight` in None, "balanced" (8 combinations), scored by F1.
- Thresholds tried: 0.05 to 0.95 in steps of 0.05.
- Costs: $4 per flagged order, $15 per late order that was not flagged.

## Decisions

### Columns used and columns left out

The model is used the moment an order is placed, so I only kept columns whose value is known at that moment.

Used: `distance_km`, `items`, `hour`, `courier_months` (numbers) and `weather`, `day_type`, `vehicle` (categories).

Left out:

- `order_id` is a label. It says nothing about the order itself.
- `prep_minutes` is how long the kitchen really took. We cannot know that when the order comes in. Dropping it costs some accuracy, but using it would mean the model only works on orders that are already finished.
- `customer_rating` is given after delivery and is partly made from whether the order was late. It would leak the answer (see the leakage check below).

### How I chose the model

I compared six models with five-fold stratified cross-validation on the training set only, and put a `DummyClassifier` in as the baseline. Mean ± standard deviation:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| dummy_most_frequent | 0.8440 ± 0.0009 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| logistic | 0.8618 ± 0.0129 | 0.6695 ± 0.1137 | 0.2221 ± 0.0656 | 0.3303 ± 0.0863 | 0.7795 ± 0.0384 |
| logistic_balanced | 0.7307 ± 0.0263 | 0.3233 ± 0.0353 | 0.6551 ± 0.0541 | 0.4323 ± 0.0397 | 0.7793 ± 0.0376 |
| random_forest | 0.8498 ± 0.0094 | 0.5410 ± 0.0729 | 0.2135 ± 0.0547 | 0.3047 ± 0.0673 | 0.7361 ± 0.0302 |
| random_forest_balanced | 0.8231 ± 0.0092 | 0.4259 ± 0.0330 | 0.3872 ± 0.0544 | 0.4045 ± 0.0403 | 0.7385 ± 0.0215 |
| hist_gradient_boosting | 0.8467 ± 0.0090 | 0.5254 ± 0.0582 | 0.2792 ± 0.0259 | 0.3618 ± 0.0197 | 0.7452 ± 0.0116 |

Accuracy is a trap here. The dummy model that never flags anything scores 0.844 and finds no late orders at all. The plain logistic model has the best accuracy (0.8618), but its recall is 0.2221, so it misses most late orders, and every missed late order costs $15 against $4 for a voucher. The balanced logistic model has the best F1 (0.4323) and a recall of 0.6551. Its ROC-AUC is the same as the plain logistic model (0.7793 against 0.7795), so it ranks orders equally well and only the decision rule has moved. I chose it for that reason, and because the costs favour recall.

### What tuning changed

I ran `GridSearchCV` on the logistic pipeline over `model__C` and `model__class_weight`, scored by F1. I used F1 because late orders are the minority and both mistakes matter. ROC-AUC does not choose a decision rule, and accuracy rewards flagging nobody.

The winner was `C = 0.01` with `class_weight = "balanced"`, at a cross-validated F1 of 0.4464. The untuned balanced model scored 0.4323, so tuning added 0.0141. That is smaller than the fold-to-fold spread (0.0397). So tuning did not beat the untuned model by more than the spread. I kept the grid winner anyway, but I would not claim the gain is real.

### How I chose the threshold, and why the cost rule needs it

A probability above 0.5 is not a business decision. Here a missed late order costs almost four times a voucher, so the right cut-off should be set by the costs. I took out-of-fold probabilities on the training set with `cross_val_predict`, tried every threshold from 0.05 to 0.95, and took the one with the lowest total cost. That was **0.65**, at a training cost of $4,194, against $5,265 for flagging nobody. The test set had no part in this. The cost calculation is the function `order_cost` in `evaluation.py`.

### Leakage check

I ran the comparison once more with `customer_rating` added back (in the notebook, not in the model). Cross-validated ROC-AUC went from 0.779 to 0.905. A jump like that from one column is a warning. The rating exists because of the delivery, so the extra score is leakage and the column stays out.

## Result

The final pipeline was fitted on the whole training set and the test set was scored once, at threshold 0.65.

|  | Predicted on time | Predicted late |
| --- | --- | --- |
| Actual on time | 586 | 47 |
| Actual late | 73 | 44 |

Precision 0.4835, recall 0.3761, F1 0.4231, ROC-AUC 0.7747. The model flagged 91 of the 750 test orders.

Cost on the test set: $1,755 if nobody is flagged, $3,000 if everybody is flagged, $1,459 with the model. That saves $296, about 17 percent, compared with flagging nobody. I would trust the direction (flagging with the model is cheaper than flagging nobody) more than the exact $296, because it comes from one test set with only 117 late orders, and the training cost curve is flat near the minimum ($4,273 at 0.60, $4,194 at 0.65, $4,318 at 0.70).

## What the tests cover

15 tests across `tests/`, with shared fixtures in `conftest.py`:

- The feature list leaves out `order_id`, `prep_minutes`, `customer_rating` and the target.
- The split is stratified and repeatable, with 2,250 training and 750 test orders.
- Predicted probabilities are between 0 and 1 and sum to 1.
- The pipeline copes with a missing number and with an unseen weather value, and predicts for each known weather value (parametrized).
- The cost function is checked on a small example I worked out by hand. The threshold function is checked where flagging nobody is cheapest, where flagging everybody is cheapest, and on two parametrized worked examples where a middle threshold wins.
- A saved model gives the same probabilities after it is reloaded (using `tmp_path`) and accepts a raw order without any manual scaling.

I chose not to test the exact test-set metrics. They depend on the data, the seed and the full training run, so a test on them would break easily and would tempt me to bend the model to make a number pass.

## Known limitations

- The data is invented, so nothing here shows how the model would do on real QuickBite orders, weather or couriers.
- Leaving out `prep_minutes` is the honest choice but it throws away a real cause of lateness.
- I assume a voucher always works and a missed late order always costs $15. Real customers are less tidy.
- The saving is small, the tuning gain is within the fold spread, and the cost curve is shallow near 0.65. I would check the threshold again on more data before relying on it.
- The model only sees seven features. It knows nothing about the restaurant, the dish, traffic or the courier's current load.
- Joblib files can run code when loaded, so only load model files you trust. `model_info.json` records the scikit-learn version the model was saved with (1.9.1).

## AI use

I used Claude (Anthropic) to help prepare this lab, working from a draft solution I supplied. Commits it helped write carry a co-author line.
