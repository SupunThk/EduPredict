# EduPredict notebooks

1. `01_data_understanding_eda.ipynb`: dataset understanding and EDA.
2. `02_preprocessing_feature_engineering_ipynb.ipynb`: preprocessing explanations and demonstrations.
3. Run each model experiment independently, in any order:
   - `03a_linear_regression.ipynb`
   - `03b_ridge_regression.ipynb`
   - `03c_decision_tree.ipynb`
   - `03d_random_forest.ipynb`
   - `03e_gradient_boosting.ipynb`
4. Run `03_regression_modelling.ipynb` to validate and compare the completed experiments and save the selected pipelines.
5. Run `04_evaluation_recommendations.ipynb` for Member 4's final held-out evaluation, residual analysis, risk flags, recommendations and exported evidence in `docs/evaluation/`.

Every model notebook contains an initial configuration, selection/tuning, cross-validation scores, training/validation diagnostics, and fitted-pipeline exports. Linear Regression searches the absence representation rather than a regularization hyperparameter.

All model notebooks retain 395 students, including the 38 recorded zero grades, and share the same 316/79 split and five validation folds. Each compares two scenarios: without prior grades (main) and with G1/G2 available (later-stage prediction). Notebook 04 has now evaluated the test set; it must not guide further tuning. Availability of other features, including absences, remains unresolved.

Shared preprocessing lives in `src/preprocessing.py`. Change it there, then rerun every model notebook before the central comparison. Fingerprint checks prevent mixing stale experiments. Pipelines are saved under `models/experiments/` and experiment records under `docs/experiments/per_model/`.

The previous combined notebook is preserved in `archive/03_regression_modelling_combined_2026-10-11.ipynb` as a historical snapshot; do not use its scores as the latest experiment record. The earlier positive-grade-only results remain separately archived.
