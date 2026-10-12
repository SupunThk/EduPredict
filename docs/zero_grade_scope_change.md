# Zero grade scope decision

On 11 October 2026, the main analysis was changed to retain all 395 Mathematics students, including the 38 recorded zero final grades. There is no confirmed evidence that these grades are missing values or recording errors. Keeping them aligns the evaluated population with final-grade prediction and academic-support identification.

This supersedes the previous decision to exclude zero grades. That earlier experiment covered 357 students and is preserved in `experiments/positive_grades_only_results.json`. Its scores describe a different population and split; differences from the new scores cannot be attributed solely to a better or worse model.

## First rerun after the scope change

The main 80/20 split has 316 training and 79 held-out test students, using random_state=42. G1 and G2 remain excluded. The same cross-validation configuration and model-search grids were rerun. The 130 recorded grades below 10 include all 38 zeros.

| Tuned model | CV MAE | CV RMSE | CV R2 |
|---|---:|---:|---:|
| Random Forest | 2.964054 | 3.936634 | 0.208057 |
| Gradient Boosting | 3.046962 | 3.943664 | 0.210082 |
| Ridge | 3.242256 | 4.167685 | 0.125253 |

In this first rerun, Random Forest was selected from the three tuned candidates by mean cross-validation MAE. These historical selection scores are not an independent final evaluation. The expanded per-model experiments and optional later-stage G1/G2 scenario are now documented in `model_experiment_workflow.md`; the central modelling notebook contains the latest comparison.

No held-out test metrics were calculated. Notebook 01 explored the full dataset before splitting, so the test set is not fully blind to exploratory analysis. Prediction-time availability of absences remains unresolved. Zero-grade causes remain unknown; these outcomes should not be interpreted as confirmed withdrawals.

## Reproducibility

Both active notebooks use all recorded students and produce identical training/test row indices. The indices, model settings, and updated results are recorded in `experiments/all_students_results.json`. The reusable preprocessing functions contain no target-based filtering and did not require a change for this population correction. They were subsequently extended with an explicit optional prior-grade scenario for the separate model experiments.
