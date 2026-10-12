# Model notebook workflow

Results stay inside the notebooks. Repeated split, pipeline and metric helpers live in `src/model_utils.py`; learned preprocessing lives in `src/preprocessing.py`. No tables, figures or reports are exported into `docs/`.

## Read each model notebook in order

1. Setup: load the shared split and define initial model settings and the search grid.
2. Constant baselines: compare training-fold mean and median predictions.
3. Without G1/G2: initial model, tuning, score comparison, training gap and risk errors.
4. With G1/G2: repeat those steps for the separate later-stage scenario.
5. Scenario summary: compare information availability and save the two complete fitted pipelines under `models/experiments/`.

Linear Regression compares original versus log absences. Ridge also searches alpha. Tree models search their original depth, leaf-size, feature-sampling or boosting settings. Original search grids and the MAE selection rule are retained.

## Shared evaluation rules

Retain all 395 Mathematics students, including 38 zero final grades. Use the same 316/79 split with random_state=42 and five shuffled training folds with the same seed. Fit preprocessing inside each fold.

MAE is the selection metric; RMSE and R² support interpretation. Compare continuous predicted G3 < 10 with observed G3 < 10 using accuracy, precision, recall, F1, specificity and confusion counts. Undefined precision remains missing. Fold means/std and pooled risk rates are distinct aggregations.

Selected settings were chosen using the same validation folds, so their scores can be optimistic. They are not nested-CV or independent test estimates. G1/G2 requires second-period availability; other input timing, particularly absences, remains unresolved.

## Comparison and final evaluation

Run model notebooks 03a–03e in any order, then Notebook 03. It compares cloned versions of saved pipelines on the same training folds. It saves only the two selected fitted pipelines and small split/feature/fingerprint metadata files under `models/` for Notebook 04.

Member 3 owns baselines, training, tuning and training-fold comparison. Member 4 uses Notebook 04 for final test metrics, residuals, risk cases and recommendations. All outputs stay inside the notebooks.

The 79-row test set has already been inspected and must not guide more tuning or threshold selection. Full-dataset EDA also preceded splitting. Further development requires new untouched evaluation data. Predictions support staff review rather than automatic decisions.
