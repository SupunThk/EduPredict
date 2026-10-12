"""Checks that the app reproduces the project's saved-model results.

Run from the project root:  python -m unittest discover -s app
"""

import unittest

import pandas as pd

import model_service as ms


class ModelServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.dataset = ms.load_models()

    def test_held_out_results_match_notebook_04(self):
        main, later = self.models[ms.MAIN].test_metrics, self.models[ms.LATER].test_metrics
        self.assertAlmostEqual(main["MAE"], 3.095, places=3)
        self.assertAlmostEqual(main["RMSE"], 3.871, places=3)
        self.assertEqual((main["TP"], main["FN"], main["FP"]), (10, 17, 11))
        self.assertAlmostEqual(later["MAE"], 1.170, places=3)
        self.assertEqual((later["TP"], later["FN"], later["FP"]), (27, 0, 10))

    def test_selected_models_are_random_forests(self):
        for m in self.models.values():
            self.assertEqual(m.estimator_name, "RandomForestRegressor")
        self.assertNotIn("G1", self.models[ms.MAIN].features)
        self.assertIn("G2", self.models[ms.LATER].features)

    def test_choose_scenario(self):
        self.assertEqual(ms.choose_scenario(False), ms.MAIN)
        self.assertEqual(ms.choose_scenario(True), ms.LATER)

    def test_matches_direct_pipeline_prediction(self):
        row = self.dataset.iloc[0]
        for scenario, m in self.models.items():
            clean, errors = ms.validate_student(row.to_dict(), scenario)
            self.assertEqual(errors, [])
            expected = m.pipeline.predict(self.dataset.iloc[[0]][m.features])[0]
            self.assertAlmostEqual(ms.predict(m, [clean])[0]["predicted_G3"], expected)

    def test_default_student_is_valid(self):
        clean, errors = ms.validate_student(ms.DEFAULT_STUDENT, ms.MAIN)
        self.assertEqual(errors, [])
        self.assertEqual(set(clean), set(self.models[ms.MAIN].features))

    def test_rejects_bad_values(self):
        bad = dict(ms.DEFAULT_STUDENT, school="XX", age=30, Medu=2.5)
        del bad["health"]
        _, errors = ms.validate_student(bad, ms.MAIN)
        self.assertEqual(len(errors), 4)
        _, errors = ms.validate_student(ms.DEFAULT_STUDENT, ms.LATER)
        self.assertTrue(any("G1" in e for e in errors))

    def test_batch_routes_rows_by_prior_grades(self):
        table = pd.DataFrame([
            ms.DEFAULT_STUDENT,
            dict(ms.DEFAULT_STUDENT, G1=8, G2=7),
            dict(ms.DEFAULT_STUDENT, sex="X"),
        ])
        out = ms.predict_batch(self.models, table)
        self.assertEqual(list(out["model"][:2]), [ms.SCENARIO_LABELS[ms.MAIN], ms.SCENARIO_LABELS[ms.LATER]])
        self.assertEqual(out["status"][2], "Invalid input")
        self.assertTrue(out["predicted_G3"][:2].notna().all())

    def test_reads_dataset_format_csv(self):
        data = (ms.PROJECT_ROOT / "data/student-mat.csv").read_bytes()
        table, had_target = ms.read_student_csv(data)
        self.assertTrue(had_target)
        out = ms.predict_batch(self.models, table.head(20))
        self.assertTrue((out["model"] == ms.SCENARIO_LABELS[ms.LATER]).all())
        expected = self.models[ms.LATER].pipeline.predict(self.dataset.head(20)[self.models[ms.LATER].features])
        self.assertTrue(((out["predicted_G3"] - expected.round(2)).abs() < 1e-9).all())
        template = pd.DataFrame([{**ms.DEFAULT_STUDENT, "G1": "", "G2": ""}]).to_csv(index=False).encode()
        out = ms.predict_batch(self.models, ms.read_student_csv(template)[0])
        self.assertEqual(out["model"][0], ms.SCENARIO_LABELS[ms.MAIN])
        self.assertEqual(out["status"][0], "Not flagged")

    def test_line_ending_tolerant_fingerprint(self):
        record = ms.json.loads((ms.PROJECT_ROOT / "models" / ms.METADATA_FILES[ms.MAIN]).read_text())
        self.assertTrue(ms.text_fingerprint_matches(ms.PROJECT_ROOT / "src/preprocessing.py",
                                                    record["preprocessing_sha256"]))


if __name__ == "__main__":
    unittest.main()
