import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


binary_features = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic"
]


nominal_features = ["Mjob", "Fjob", "reason", "guardian"]


ordinal_features = [
    "Medu", "Fedu", "traveltime", "studytime", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health"
]


numerical_features = ["age", "failures", "absences"]


binary_mappings = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"F": 0, "M": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
}


yes_no_features = [
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]


for column in yes_no_features:
    binary_mappings[column] = {"no": 0, "yes": 1}


def transform_absences(X, use_log=False):
    X = X.copy()

    if use_log:
        if (X["absences"] < 0).any():
            raise ValueError("Absences cannot be negative.")

        X["absences_log"] = np.log1p(X["absences"])
        X = X.drop(columns="absences")

    return X


def encode_binary(X):
    X = X.copy()

    for column in X.columns:
        mapping = binary_mappings[column]

        if not X[column].isin(mapping.keys()).all():
            raise ValueError(
                f"Missing or unexpected category in {column}"
            )

        X[column] = X[column].map(mapping)

    return X


def make_preprocessor(
    use_log_absences=False,
    scale_numeric=False,
    include_prior_grades=False,
):
    """Create an unfitted pipeline for the chosen prediction scenario.

    Prior grades are excluded by default. Enable include_prior_grades only
    for a later-stage scenario where both G1 and G2 are already available.
    """
    absence_feature = (
        "absences_log" if use_log_absences else "absences"
    )

    numeric_and_ordinal = ordinal_features + [
        "age", "failures", absence_feature
    ]

    if include_prior_grades:
        numeric_and_ordinal += ["G1", "G2"]

    numeric_transformer = (
        StandardScaler() if scale_numeric else "passthrough"
    )

    column_processing = ColumnTransformer(
        transformers=[
            (
                "binary",
                FunctionTransformer(
                    encode_binary,
                    validate=False,
                    feature_names_out="one-to-one"
                ),
                binary_features
            ),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                nominal_features
            ),
            (
                "numeric_ordinal",
                numeric_transformer,
                numeric_and_ordinal
            )
        ],
        remainder="drop"
    )

    return Pipeline([
        (
            "absence_transform",
            FunctionTransformer(
                transform_absences,
                kw_args={"use_log": use_log_absences},
                validate=False
            )
        ),
        ("column_processing", column_processing)
    ])
