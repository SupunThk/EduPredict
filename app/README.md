# EduPredict local prediction app

A small Streamlit app that loads the two selected Random Forest pipelines from `models/` and predicts a student's final Mathematics grade (G3, 0–20). It does not train, tune or save anything, and it changes no notebook, model or `src/` file.

## Run it

Use Python 3.11. The pinned versions match the versions the saved models were fitted with.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
streamlit run app/app.py --server.address localhost
```

The app opens at http://localhost:8501. To check that it reproduces the project's results:

```bash
python -m unittest discover -s app
```

## What it does

- **Single student:** enter the raw details in a form and get the predicted G3 and the *potentially at risk* flag (predicted G3 < 10).
- **Batch (CSV):** upload many students at once (comma or semicolon separated, same column names as `data/student-mat.csv`) and download the predictions. Use the template button for the column layout.
- **About the models:** shows the selection rule, training-CV MAE and held-out test metrics, plus the limitations from the reports.

## Which model is used

Notebook 03 compared five regressors with five-fold training CV and selected the lowest MAE in each scenario. Random Forest won both. The app picks between the two saved pipelines only by what is known:

| When | Model | Held-out test MAE | At-risk students found |
|---|---|---|---|
| G1 and G2 not yet known | `best_without_prior_grades_random_forest_regressor.joblib` (30 inputs) | 3.095 | 10 of 27 |
| Both G1 and G2 known | `best_with_prior_grades_random_forest_regressor.joblib` (32 inputs) | 1.170 | 27 of 27 |

These test metrics are recomputed at start-up from the saved models and match Notebook 04.

## Safety checks at start-up

Before loading, the app checks the dataset, `src/preprocessing.py` and both `.joblib` files against the fingerprints in the metadata files, and it checks the installed library versions. A mismatch in scikit-learn, joblib or numpy stops the app, because pickled models are not reliable across those versions.

The preprocessing fingerprint was recorded on a checkout with Windows (CRLF) line endings. The app ignores LF/CRLF differences when it compares that file, so the check passes on macOS and Linux checkouts while still detecting any real code change.

## Responsible use

Use fictional or de-identified students only; nothing entered is stored. A prediction supports staff review. It must not decide grades, discipline or access to help, and an unflagged student can still need support.
