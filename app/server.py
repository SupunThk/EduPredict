"""EduPredict local app: a small web page over the two selected models.

Run from the project root:  python app/server.py
Then open http://127.0.0.1:8000 (the server only listens on this computer).
"""

import argparse
import json
import mimetypes
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_service as ms  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 2 * 1024 * 1024  # 2 MB is plenty for a class-sized CSV


def build_meta(models):
    fields = []
    for name, label, group, rule, default in ms.FIELDS:
        entry = {"name": name, "label": label, "group": group, "default": default}
        if isinstance(rule, dict):
            entry.update(kind="choice", options=[[code, text] for code, text in rule.items()])
        else:
            entry.update(kind="number", min=rule[0], max=rule[1])
        fields.append(entry)
    return {
        "threshold": ms.THRESHOLD,
        "observedMaxAbsences": ms.OBSERVED_MAX_ABSENCES,
        "fields": fields,
        "models": [model_summary(m) for m in models.values()],
    }


def model_summary(m):
    t = m.test_metrics
    return {
        "scenario": m.scenario, "label": m.label, "estimator": m.estimator_name,
        "inputs": len(m.features), "logAbsences": m.metadata["log_absences"],
        "cvMae": m.metadata["selection_cv_scores"]["MAE"],
        "test": {k: (None if v != v else v) for k, v in t.items()},  # NaN -> null
    }


def make_handler(models, meta):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EduPredict"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, status, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("The upload is larger than 2 MB.")
            return self.rfile.read(length)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/meta":
                return self._send(HTTPStatus.OK, meta)
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            file = (STATIC / name).resolve()
            if STATIC not in file.parents or not file.is_file():
                return self._send(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            if kind.startswith("text/") or kind.endswith("javascript"):
                kind += "; charset=utf-8"
            self._send(HTTPStatus.OK, file.read_bytes(), kind)

        def do_POST(self):
            try:
                body = self._body()
                if self.path == "/api/predict":
                    request = json.loads(body or b"{}")
                    scenario = ms.choose_scenario(bool(request.get("priorGrades")))
                    clean, errors = ms.validate_student(request.get("student") or {}, scenario)
                    model = models[scenario]
                    if errors:
                        return self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"errors": errors})
                    result = ms.predict(model, [clean])[0]
                    return self._send(HTTPStatus.OK, {
                        "scenario": scenario, "predicted": result["predicted_G3"], "flag": result["flag"],
                        "absencesAboveObserved": clean["absences"] > ms.OBSERVED_MAX_ABSENCES,
                    })
                if self.path == "/api/batch":
                    table, had_target = ms.read_student_csv(body)
                    results = ms.predict_batch(models, table)
                    return self._send(HTTPStatus.OK, {
                        "ignoredTarget": had_target,
                        "rows": json.loads(results.to_json(orient="records")),
                        "csv": results.to_csv(index=False),
                    })
                return self._send(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except (ValueError, UnicodeDecodeError) as error:
                message = str(error) if isinstance(error, ValueError) and "2 MB" in str(error) else \
                    "Could not read that input. Use a UTF-8 CSV with the template's column names."
                self._send(HTTPStatus.BAD_REQUEST, {"errors": [message]})
            except Exception:  # never leak a stack trace to the page
                self._send(HTTPStatus.BAD_REQUEST, {"errors": ["Could not read that input. Check the file and try again."]})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the EduPredict local app.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    try:
        models, _ = ms.load_models()
    except ms.ModelLoadError as error:
        sys.exit(f"The saved models could not be loaded safely:\n  {error}\n"
                 "Install the pinned versions with: pip install -r app/requirements.txt (Python 3.11)")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(models, build_meta(models)))
    url = f"http://127.0.0.1:{args.port}"
    print(f"EduPredict is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
