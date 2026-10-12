"""Load the two selected EduPredict models and make checked predictions.

Nothing here trains, tunes or saves anything. The fitted pipelines in models/
are used exactly as Notebook 03 exported them.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import io
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The pickled pipelines reference functions in src/preprocessing.py by module name.
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import binary_mappings  # noqa: E402

THRESHOLD = 10
MAIN, LATER = "no_prior_grades", "with_prior_grades"
METADATA_FILES = {
    MAIN: "best_without_prior_grades_random_forest_regressor_metadata.json",
    LATER: "best_with_prior_grades_random_forest_regressor_metadata.json",
}
SCENARIO_LABELS = {
    MAIN: "Main model (without G1/G2)",
    LATER: "Later-stage model (with G1/G2)",
}

# Input fields: (name, label, group, options or (low, high), default).
# Options map the stored code to a readable label.
YES_NO = {"yes": "Yes", "no": "No"}
JOBS = {"teacher": "Teacher", "health": "Health care", "services": "Civil services",
        "at_home": "At home", "other": "Other"}
EDUCATION = {0: "0 – None", 1: "1 – Primary (grade 4)", 2: "2 – Grades 5–9",
             3: "3 – Secondary", 4: "4 – Higher education"}
LOW_HIGH = {1: "1 – Very low", 2: "2 – Low", 3: "3 – Medium", 4: "4 – High", 5: "5 – Very high"}
BAD_GOOD = {1: "1 – Very bad", 2: "2 – Bad", 3: "3 – Fair", 4: "4 – Good", 5: "5 – Very good"}

FIELDS = [
    ("school", "School", "Student and family", {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"}, "GP"),
    ("sex", "Sex", "Student and family", {"F": "Female", "M": "Male"}, "F"),
    ("age", "Age (years)", "Student and family", (15, 22), 17),
    ("address", "Home area", "Student and family", {"U": "Urban", "R": "Rural"}, "U"),
    ("famsize", "Family size", "Student and family", {"LE3": "3 or fewer", "GT3": "More than 3"}, "GT3"),
    ("Pstatus", "Parents' cohabitation", "Student and family", {"T": "Living together", "A": "Living apart"}, "T"),
    ("Medu", "Mother's education", "Student and family", EDUCATION, 3),
    ("Fedu", "Father's education", "Student and family", EDUCATION, 3),
    ("Mjob", "Mother's job", "Student and family", JOBS, "services"),
    ("Fjob", "Father's job", "Student and family", JOBS, "other"),
    ("reason", "Reason for choosing school", "Student and family",
     {"home": "Close to home", "reputation": "School reputation", "course": "Course preference", "other": "Other"}, "course"),
    ("guardian", "Guardian", "Student and family", {"mother": "Mother", "father": "Father", "other": "Other"}, "mother"),
    ("traveltime", "Home-to-school travel time", "Study and support",
     {1: "1 – Under 15 min", 2: "2 – 15 to 30 min", 3: "3 – 30 to 60 min", 4: "4 – Over 1 hour"}, 1),
    ("studytime", "Weekly study time", "Study and support",
     {1: "1 – Under 2 hours", 2: "2 – 2 to 5 hours", 3: "3 – 5 to 10 hours", 4: "4 – Over 10 hours"}, 2),
    ("failures", "Past class failures (dataset code)", "Study and support", (0, 3), 0),
    ("schoolsup", "Extra school support", "Study and support", YES_NO, "no"),
    ("famsup", "Family educational support", "Study and support", YES_NO, "yes"),
    ("paid", "Extra paid classes (Maths)", "Study and support", YES_NO, "no"),
    ("activities", "Extracurricular activities", "Study and support", YES_NO, "yes"),
    ("nursery", "Attended nursery school", "Study and support", YES_NO, "yes"),
    ("higher", "Wants higher education", "Study and support", YES_NO, "yes"),
    ("internet", "Internet access at home", "Study and support", YES_NO, "yes"),
    ("romantic", "In a romantic relationship", "Study and support", YES_NO, "no"),
    ("famrel", "Family relationship quality", "Lifestyle and attendance",
     {1: "1 – Very bad", 2: "2 – Bad", 3: "3 – Fair", 4: "4 – Good", 5: "5 – Excellent"}, 4),
    ("freetime", "Free time after school", "Lifestyle and attendance", LOW_HIGH, 3),
    ("goout", "Going out with friends", "Lifestyle and attendance", LOW_HIGH, 3),
    ("Dalc", "Workday alcohol use", "Lifestyle and attendance", LOW_HIGH, 1),
    ("Walc", "Weekend alcohol use", "Lifestyle and attendance", LOW_HIGH, 1),
    ("health", "Current health", "Lifestyle and attendance", BAD_GOOD, 4),
    ("absences", "School absences", "Lifestyle and attendance", (0, 93), 4),
    ("G1", "First period grade (G1)", "Prior grades", (0, 20), 12),
    ("G2", "Second period grade (G2)", "Prior grades", (0, 20), 13),
]
FIELD_BY_NAME = {f[0]: f for f in FIELDS}
DEFAULT_STUDENT = {f[0]: f[4] for f in FIELDS if f[0] not in ("G1", "G2")}
OBSERVED_MAX_ABSENCES = 75

# Tight checks only where pickles depend on the exact library build.
STRICT_PACKAGES = {"scikit-learn", "joblib", "numpy"}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def text_fingerprint_matches(path, expected):
    """Compare a text file's hash, ignoring LF/CRLF differences.

    The models were exported on a checkout with Windows line endings; git stores
    the same source with LF, so the raw hash differs while the code is identical.
    """
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {_sha256(raw), _sha256(lf), _sha256(lf.replace(b"\n", b"\r\n"))}


def environment_report(expected_versions):
    rows = []
    for package, expected in expected_versions.items():
        if package == "python":
            installed = platform.python_version()
            ok = installed.split(".")[:2] == expected.split(".")[:2]
            strict = True
        else:
            try:
                installed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                installed = "not installed"
            ok = installed == expected
            strict = package in STRICT_PACKAGES
        rows.append({"package": package, "expected": expected, "installed": installed,
                     "ok": ok, "blocking": strict and not ok})
    return rows


def risk_counts(actual, predicted):
    actual, flagged = np.asarray(actual) < THRESHOLD, np.asarray(predicted) < THRESHOLD
    tp = int((actual & flagged).sum()); fp = int((~actual & flagged).sum())
    fn = int((actual & ~flagged).sum()); tn = int((~actual & ~flagged).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def grade_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = predicted - actual
    counts = risk_counts(actual, predicted)
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt((error ** 2).mean())),
        "R2": float(1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "Recall": tp / (tp + fn) if tp + fn else float("nan"),
        "Precision": tp / (tp + fp) if tp + fp else float("nan"),
        "Specificity": tn / (tn + fp) if tn + fp else float("nan"),
        **counts,
    }


@dataclass
class SelectedModel:
    scenario: str
    pipeline: object
    metadata: dict
    test_metrics: dict

    @property
    def label(self):
        return SCENARIO_LABELS[self.scenario]

    @property
    def features(self):
        return self.metadata["input_features"]

    @property
    def estimator_name(self):
        return type(self.pipeline[-1]).__name__


class ModelLoadError(RuntimeError):
    pass


def load_models(root=PROJECT_ROOT):
    """Verify fingerprints and versions, then load both selected pipelines."""
    root = Path(root)
    dataset_path = root / "data/student-mat.csv"
    dataset = pd.read_csv(dataset_path, sep=";")
    models, problems = {}, []
    for scenario, filename in METADATA_FILES.items():
        record = json.loads((root / "models" / filename).read_text(encoding="utf-8"))
        if record["scenario"] != scenario:
            problems.append(f"{filename}: unexpected scenario {record['scenario']}")
        if not text_fingerprint_matches(dataset_path, record["dataset_sha256"]):
            problems.append("data/student-mat.csv does not match the data the models were trained on")
        if not text_fingerprint_matches(root / "src/preprocessing.py", record["preprocessing_sha256"]):
            problems.append("src/preprocessing.py differs from the version used to train the models")
        pipeline_path = root / record["pipeline_path"]
        if _sha256(pipeline_path.read_bytes()) != record["pipeline_sha256"]:
            problems.append(f"{record['pipeline_path']} does not match its saved fingerprint")
        blocking = [r for r in environment_report(record["versions"]) if r["blocking"]]
        for r in blocking:
            problems.append(f"{r['package']} {r['installed']} installed; the models need {r['expected']}")
        if problems:
            raise ModelLoadError("; ".join(dict.fromkeys(problems)))

        pipeline = joblib.load(pipeline_path)
        test = dataset.loc[record["test_indices"]]
        predicted = pipeline.predict(test[record["input_features"]])
        models[scenario] = SelectedModel(scenario, pipeline, record, grade_metrics(test["G3"], predicted))
    return models, dataset


def choose_scenario(has_prior_grades):
    """Use the later-stage model only when both G1 and G2 are known."""
    return LATER if has_prior_grades else MAIN


def validate_student(values, scenario):
    """Return (clean values, errors) for one student, using raw dataset codes."""
    needed = [f[0] for f in FIELDS if scenario == LATER or f[0] not in ("G1", "G2")]
    clean, errors = {}, []
    for name in needed:
        _, label, _, rule, _ = FIELD_BY_NAME[name]
        value = values.get(name)
        if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
            errors.append(f"{label} ({name}) is missing")
            continue
        if isinstance(rule, dict):
            codes = list(rule)
            if isinstance(codes[0], int):
                try:
                    number = float(value)
                    value = int(number) if number.is_integer() else value
                except (TypeError, ValueError):
                    pass
            else:
                value = str(value).strip()
            if value not in rule:
                errors.append(f"{label} ({name}) must be one of {', '.join(map(str, codes))}")
                continue
        else:
            low, high = rule
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{label} ({name}) must be a whole number from {low} to {high}")
                continue
            if isinstance(value, bool) or not number.is_integer() or not low <= number <= high:
                errors.append(f"{label} ({name}) must be a whole number from {low} to {high}")
                continue
            value = int(number)
        clean[name] = value
    return clean, errors


def predict(model, students):
    """Predict G3 for already-validated students (list of dicts)."""
    frame = pd.DataFrame(students, columns=model.features)
    grades = model.pipeline.predict(frame)
    return [{"predicted_G3": float(g), "flag": bool(g < THRESHOLD)} for g in grades]


def read_student_csv(data):
    """Read uploaded CSV bytes (comma or semicolon separated); drop any G3 column."""
    text = data.decode("utf-8-sig")
    header = text.splitlines()[0] if text.strip() else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    table = pd.read_csv(io.StringIO(text), sep=sep)
    table.columns = [str(c).strip() for c in table.columns]
    had_target = "G3" in table.columns
    return table.drop(columns="G3", errors="ignore"), had_target


def predict_batch(models, table):
    """Validate and predict each row; rows with both G1 and G2 use the later-stage model."""
    table = table.copy()
    table.columns = [str(c).strip() for c in table.columns]
    out = []
    for position, (_, row) in enumerate(table.iterrows(), start=1):
        values = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        has_prior = all(values.get(g) is not None for g in ("G1", "G2"))
        scenario = choose_scenario(has_prior)
        clean, errors = validate_student(values, scenario)
        result = {"row": position, "model": SCENARIO_LABELS[scenario]}
        if errors:
            result.update(predicted_G3=None, status="Invalid input", notes="; ".join(errors))
        else:
            pred = predict(models[scenario], [clean])[0]
            notes = []
            if not has_prior and any(values.get(g) is not None for g in ("G1", "G2")):
                notes.append("only one of G1/G2 given, so prior grades were not used")
            if clean["absences"] > OBSERVED_MAX_ABSENCES:
                notes.append("absences above the observed maximum (75)")
            result.update(predicted_G3=round(pred["predicted_G3"], 2),
                          status="Potentially at risk" if pred["flag"] else "Not flagged",
                          notes="; ".join(notes))
        out.append(result)
    return pd.DataFrame(out)


def binary_codes():
    """Exposed for tests: the binary codes the saved pipeline accepts."""
    return {name: set(mapping) for name, mapping in binary_mappings.items()}
