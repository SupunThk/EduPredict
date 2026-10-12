# Model experiment workflow

Each regression model has an independent notebook containing its initial configuration, cross-validation, selection/tuning, training/validation diagnostics, and fitted-pipeline exports. Model-specific code stays visible in the notebook; shared preprocessing stays in `src/preprocessing.py`.

## Execution order

Run the five model notebooks in any order, then run `notebooks/03_regression_modelling.ipynb` for the central comparison. Each notebook runs independently from a clean kernel. There is no dependency on Notebook 02's in-memory variables.

| Notebook | Search |
|---|---|
| `03a_linear_regression.ipynb` | Original versus log absences; no regularization hyperparameter for OLS |
| `03b_ridge_regression.ipynb` | Alpha and absence representation |
| `03c_decision_tree.ipynb` | Depth, minimum leaf size, and absence representation |
| `03d_random_forest.ipynb` | Depth, minimum leaf size, feature sampling, and absence representation; 300 trees fixed |
| `03e_gradient_boosting.ipynb` | Tree count, learning rate, depth, minimum leaf size, and absence representation |

Every search contains the initial configuration. MAE is the primary selection metric. Both mean and median constant baselines are recorded. All learned encoding and scaling are fitted within each training fold.

## Prediction scenarios

Both scenarios retain all 395 students, including 38 zero grades. They use exactly the same 316 training students, 79 reserved test students, and five shuffled validation folds (random_state=42).

- `no_prior_grades`: excludes G1 and G2. This remains the main project scenario. Availability of other variables, including absences, still needs to be justified at the chosen prediction time.
- `with_prior_grades`: includes both G1 and G2 as numerical features through the preprocessor's explicit `include_prior_grades=True` option. This predicts after the second-period grade is available. It answers a different timing question and must not be presented as an earlier prediction with improved accuracy.

Final-grade correlation alone does not make G1/G2 leakage if those grades are known at the stated prediction time. Using them before they are available would be leakage. This experiment does not resolve timing of the other input variables.

## Results and artifacts

Each notebook saves its experiment record to `docs/experiments/per_model/` and its two fitted pipelines to `models/experiments/`. Records include fold scores, selected settings, feature names, row indices, package versions, and data/code/artifact fingerprints.

The central comparison rejects missing or inconsistent records, ranks models separately for each scenario, and saves:

- `models/best_regression_pipeline.joblib` and `models/best_regression_metadata.json` for the main no-prior-grades scenario.
- `models/best_with_prior_grades_pipeline.joblib` and `models/best_with_prior_grades_metadata.json` for the later-stage comparison.

The metadata provides exact input columns and held-out row indices. Member 4 should load the full fitted pipeline, check matching data/code/versions, and evaluate the reserved rows. Do not refit preprocessing separately or retune from test results. Avoid mixing baseline or candidate scores from historical population definitions.

## Evaluation limits

These cross-validation scores were used for selection and may be optimistic. Final held-out performance is now recorded in `notebooks/04_evaluation_recommendations.ipynb` and `docs/evaluation/`. Both frozen selected pipelines were evaluated on the same 79 rows without refitting or retuning. Results include baselines, residuals, the fixed grade-10 risk rule, descriptive group checks and recommendations. The test set is now used and must not guide subsequent tuning. Notebook 01 explored the full dataset before splitting, so the holdout is not fully blind to exploratory analysis. Zero-grade causes remain unknown, and predictions should support staff review rather than determine student outcomes.

`docs/evaluation/evaluation_manifest.json` records the completed evaluation and artifact fingerprints. The `test_performance_evaluated: false` fields in selection metadata and training experiment records describe their pre-evaluation handoff state; they are historical records, not the current evaluation status.

The old combined notebook and its all-student results are preserved under `notebooks/archive/` and `docs/experiments/archive/`. The earlier positive-grade-only experiment remains separately archived; its scores concern a different population.
