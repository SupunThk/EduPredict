# EduPredict notebooks

1. `01_data_understanding_eda.ipynb`: understand the dataset and explore relationships.
2. `02_preprocessing_feature_engineering_ipynb.ipynb`: learn the preprocessing decisions.
3. Run the five model notebooks in any order: `03a_linear_regression.ipynb`, `03b_ridge_regression.ipynb`, `03c_decision_tree.ipynb`, `03d_random_forest.ipynb`, and `03e_gradient_boosting.ipynb`.
4. `03_regression_modelling.ipynb`: compare the saved configurations on training CV and prepare the selected pipelines for evaluation.
5. `04_evaluation_recommendations.ipynb`: inspect final test metrics, residuals, risk errors and recommendations.
6. `05_student_prediction.ipynb`: enter a student's raw details and predict G3 using the saved selected Random Forest model. Choose without G1/G2 or, when both grades are known, with G1/G2. No retraining is needed.

Each model notebook has five sections: setup, constant baselines, without G1/G2, with G1/G2, and a scenario summary with fitted-model saving. Each scenario shows the initial model, tuning and interpretation. Settings and fitting stay visible; repeated metric helpers live in `src/model_utils.py`, and preprocessing lives in `src/preprocessing.py`.

Notebook 04 has ten short code cells: setup/loading, predictions, grade metrics, residuals, error groups, risk metrics, confusion matrices, individual cases and descriptive group checks. Findings and recommendations are readable Markdown. Saved-model checks live in `src/evaluation_utils.py`; no model is fitted again during final evaluation. Individual case views show up to ten rows each; full test results remain in the notebook's `results` variable.

All tables, plots, predictions and recommendations stay inside the notebooks. No results or reports are exported to `docs/`. Complete fitted pipelines remain under `models/experiments/`; the two selected pipelines and small metadata files remain under `models/` for Notebook 04.

Grade metrics are MAE, RMSE and R². The fixed predicted-G3-below-10 rule also has accuracy, precision, recall, F1, specificity and confusion counts. Fold means/std and pooled risk rates are labelled separately. Selection uses MAE and reuses training folds, so selection scores may be optimistic.

All 395 students, including 38 zeros, share the same 316/79 split and five training folds. G1/G2 is a later-stage comparison; availability of other inputs, especially absences, remains unresolved. The test set was already inspected and must not guide new tuning. Rerunning unchanged choices does not create an unseen test.
