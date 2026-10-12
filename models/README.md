# Fitted regression models

The selected models are Random Forest Regressors, saved together with their fitted preprocessing.

- `best_without_prior_grades_random_forest_regressor.joblib`: main model, excluding G1 and G2.
- `best_with_prior_grades_random_forest_regressor.joblib`: later-stage comparison, including both G1 and G2.

Each has a matching `_metadata.json` file with input columns, train/test row IDs, validation scores and fingerprints. Notebook 03 creates the selected model files; Notebook 04 loads them through `src/evaluation_utils.py`. These pipelines are already fitted on 316 training students; do not refit them before final evaluation.

The `experiments/` folder contains fitted pipelines from the individual model notebooks. Results stay in notebook outputs rather than exported reports.
