"""Load the selected models safely; evaluation steps stay in Notebook 04."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import joblib


def load_selected_models(project_root, df):
    """Check saved fingerprints and row splits without fitting anything."""
    root = Path(project_root)
    files = {"no_prior_grades": "best_without_prior_grades_random_forest_regressor_metadata.json",
             "with_prior_grades": "best_with_prior_grades_random_forest_regressor_metadata.json"}
    models, metadata = {}, {}
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for scenario, filename in files.items():
        record = json.loads((root / "models" / filename).read_text(encoding="utf-8"))
        assert record["scenario"] == scenario
        assert digest(root / "data/student-mat.csv") == record["dataset_sha256"]
        assert digest(root / "src/preprocessing.py") == record["preprocessing_sha256"]
        path = root / record["pipeline_path"]
        assert digest(path) == record["pipeline_sha256"]
        for package, expected in record["versions"].items():
            version = platform.python_version() if package == "python" else importlib.metadata.version(package)
            assert version == expected, f"Version mismatch for {package}"
        train_ids, test_ids = record["training_indices"], record["test_indices"]
        assert len(train_ids) == len(set(train_ids)) == 316
        assert len(test_ids) == len(set(test_ids)) == 79
        assert not set(train_ids) & set(test_ids)
        assert set(train_ids) | set(test_ids) == set(df.index)
        features = record["input_features"]
        assert "G3" not in features
        assert all((grade in features) == (scenario == "with_prior_grades") for grade in ["G1", "G2"])
        models[scenario], metadata[scenario] = joblib.load(path), record
    for key in ["training_indices", "test_indices"]:
        assert metadata["no_prior_grades"][key] == metadata["with_prior_grades"][key]
    return models, metadata
