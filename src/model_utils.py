"""Small helpers shared by the teaching notebooks; no result-file exports."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from preprocessing import make_preprocessor

METRICS = ["MAE", "RMSE", "R2", "Accuracy", "Precision", "Recall", "F1", "Specificity"]
COUNTS = ["TN", "FP", "FN", "TP"]
CV = KFold(n_splits=5, shuffle=True, random_state=42)


def load_data(project_root):
    """Keep all students and reproduce the common 316/79 row split."""
    df = pd.read_csv(Path(project_root) / "data/student-mat.csv", sep=";")
    assert len(df) == 395 and df["G3"].between(0, 20).all()
    assert df["G3"].eq(0).sum() == 38 and not df.isna().any().any()
    train_ids, test_ids = train_test_split(df.index, test_size=0.2, random_state=42)
    assert not set(train_ids) & set(test_ids)
    return df, train_ids, test_ids


def preprocessing_for(use_log=False, scale_numeric=False, include_prior_grades=False):
    """Use the same reference-category encoding for every model."""
    preprocessor = make_preprocessor(use_log, scale_numeric, include_prior_grades)
    return preprocessor.set_params(column_processing__nominal__drop="first")


def model_pipeline(estimator, scale_numeric=False, include_prior_grades=False):
    return Pipeline([
        ("preprocessing", preprocessing_for(False, scale_numeric, include_prior_grades)),
        ("model", clone(estimator)),
    ])


def flag_metric(actual, predicted, metric):
    """Convert grades to the fixed project flag; never round predictions."""
    actual, predicted = np.asarray(actual) < 10, np.asarray(predicted) < 10
    if metric in COUNTS or metric == "Specificity":
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
        if metric == "Specificity":
            return tn / (tn + fp) if tn + fp else np.nan
        return dict(zip(COUNTS, [tn, fp, fn, tp]))[metric]
    if metric == "Accuracy":
        return accuracy_score(actual, predicted)
    function = {"Precision": precision_score, "Recall": recall_score, "F1": f1_score}[metric]
    return function(actual, predicted, zero_division=np.nan)


SCORING = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
SCORING.update({name: make_scorer(flag_metric, metric=name) for name in METRICS[3:] + COUNTS})


def fold_results(scores, search_index=None, partition="test"):
    """Normalize sklearn CV/search scores into one readable fold table."""
    result = {"Fold": np.arange(1, 6)}
    for name in METRICS + COUNTS:
        values = (scores[f"{partition}_{name}"] if search_index is None else
                  [scores[f"split{i}_{partition}_{name}"][search_index] for i in range(5)])
        result[name] = np.asarray(values) * (-1 if name in ["MAE", "RMSE"] else 1)
    return pd.DataFrame(result)


def cv_summary(folds):
    """Fold means/std; undefined precision is omitted and counted explicitly."""
    result = folds[METRICS].agg(["mean", "std"])
    result.loc["defined folds"] = folds[METRICS].notna().sum()
    return result


def pooled_risk(folds):
    """Pool confusion counts; these rates can differ from fold averages."""
    tn, fp, fn, tp = folds[COUNTS].sum().astype(int)
    return pd.Series({
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision": tp / (tp + fp) if tp + fp else np.nan,
        "Recall": tp / (tp + fn) if tp + fn else np.nan,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    })
