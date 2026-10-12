"""Exercise the local web server's API end to end.

Run from the project root:  python -m unittest discover -s app
"""

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer

import model_service as ms
import server


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models, _ = ms.load_models()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(models, server.build_meta(models)))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request(method, path, body=body, headers=headers or {})
        res = conn.getresponse()
        data = res.read()
        conn.close()
        return res.status, data

    def post_json(self, path, payload):
        status, data = self.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        return status, json.loads(data)

    def test_page_and_meta(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"EduPredict", page)
        status, data = self.request("GET", "/api/meta")
        meta = json.loads(data)
        self.assertEqual(len(meta["fields"]), 32)
        self.assertAlmostEqual(meta["models"][0]["test"]["MAE"], 3.095, places=3)

    def test_predict_routes_to_the_right_model(self):
        status, main = self.post_json("/api/predict", {"student": ms.DEFAULT_STUDENT, "priorGrades": False})
        self.assertEqual((status, main["scenario"]), (200, ms.MAIN))
        later_student = dict(ms.DEFAULT_STUDENT, G1=7, G2=6)
        status, later = self.post_json("/api/predict", {"student": later_student, "priorGrades": True})
        self.assertEqual((status, later["scenario"], later["flag"]), (200, ms.LATER, True))

    def test_predict_reports_bad_input(self):
        status, data = self.post_json("/api/predict", {"student": dict(ms.DEFAULT_STUDENT, age=40)})
        self.assertEqual(status, 422)
        self.assertTrue(any("Age" in e for e in data["errors"]))

    def test_batch_csv(self):
        csv = (ms.PROJECT_ROOT / "data/student-mat.csv").read_bytes()
        status, data = self.request("POST", "/api/batch", csv, {"Content-Type": "text/csv"})
        result = json.loads(data)
        self.assertEqual(status, 200)
        self.assertTrue(result["ignoredTarget"])
        self.assertEqual(len(result["rows"]), 395)
        status, _ = self.request("POST", "/api/batch", b"\xff\xfe", {"Content-Type": "text/csv"})
        self.assertEqual(status, 400)

    def test_static_files_stay_inside_the_folder(self):
        status, _ = self.request("GET", "/../server.py")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
