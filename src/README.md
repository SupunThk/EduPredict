# Shared preprocessing

`preprocessing.py` defines feature groups, binary mappings, and `make_preprocessor()`.

- `use_log_absences`: choose original or log-transformed absences.
- `scale_numeric`: standardize numerical/ordinal inputs when appropriate.
- `include_prior_grades`: include G1 and G2 only in the later-stage scenario; defaults to False.

The factory returns a fresh, unfitted pipeline. Attach it to the model before cross-validation. Target filtering, student splits, model settings, fitting, and evaluation stay in the notebooks. The main analysis retains zero final grades.

Model notebooks configure reference-category encoding through `column_processing__nominal__drop="first"`. All fitted-pipeline exports contain both preprocessing and the estimator. Import this module before loading exported pipelines in a fresh environment.
