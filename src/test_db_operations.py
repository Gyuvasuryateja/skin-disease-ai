"""Integration tests for the PostgreSQL prediction-history storage layer.

Prerequisites:

- A reachable PostgreSQL server configured through DATABASE_URL (or PG* variables).
  Run ``python src/init_database.py`` first to create the database and schema.
- The trained checkpoint (``models/skinvision_efficientnet_b0/best_model.pth``)
  for the API-level tests.

WARNING: these tests DELETE all rows in the predictions table. Point them at a
dedicated local database, not a shared one.

Without DATABASE_URL the script still verifies the not-configured degradation
paths (prediction works, history endpoints return 503) and then exits.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import app, create_app  # noqa: E402
from db import (  # noqa: E402
    PredictionStore,
    PredictionStoreError,
    describe_target,
    resolve_conninfo,
    sanitize_image_filename,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "skinvision_efficientnet_b0" / "best_model.pth"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "prepared" / "split_manifest.csv"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "HAM10000"
ENV_KEYS = ("DATABASE_URL", "PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD")


def pick_test_image() -> Path:
    """Choose one held-out test image (read-only) for real API predictions."""
    import pandas as pd

    manifest = pd.read_csv(MANIFEST_PATH)
    test_rows = manifest.loc[manifest["split"] == "test"]
    if test_rows.empty:
        raise RuntimeError("The prepared manifest contains no test-split images.")
    return DATASET_ROOT / str(test_rows.iloc[0]["relative_path"])


def parse_timestamp(value: object) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    assert parsed.tzinfo is not None, f"Timestamp must carry a timezone: {value}"
    return parsed


def test_sanitize_image_filename() -> None:
    assert sanitize_image_filename(None) == "unnamed-image"
    assert sanitize_image_filename("") == "unnamed-image"
    assert sanitize_image_filename("lesion.jpg") == "lesion.jpg"
    assert sanitize_image_filename(r"C:\Users\someone\private\lesion.jpg") == "lesion.jpg"
    assert sanitize_image_filename("../../etc/passwd") == "passwd"
    # µ is printable and kept; the lone surrogate cannot be UTF-8 encoded into
    # PostgreSQL, so it must be replaced with the placeholder.
    assert sanitize_image_filename("folder/µ\ud83d weird.png") == "µ_ weird.png"
    long_name = "x" * 500 + ".jpg"
    assert len(sanitize_image_filename(long_name)) == 200
    assert sanitize_image_filename("///") == "unnamed-image"
    print("sanitize_image_filename: path stripping, length cap, and fallbacks verified")


def run_store_tests(store: PredictionStore) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    store.delete_all_predictions()
    rows = [
        store.insert_prediction(
            image_filename=sanitize_image_filename(name),
            predicted_class=code,
            predicted_class_name=name,
            confidence=confidence,
            class_probabilities={"akiec": confidence, "nv": 1.0 - confidence},
        )
        for name, code, confidence in (
            ("first.jpg", "nv", 0.91),
            (r"C:\temp\second.png", "mel", 0.5),
            ("third.webp", "bcc", 0.123),
        )
    ]
    assert all(row["id"] >= 1 for row in rows)
    assert rows[1]["image_filename"] == "second.png", rows[1]["image_filename"]
    assert all(isinstance(row["created_at"], datetime) for row in rows)
    print(f"insert: 3 predictions stored, filenames sanitized, ids {[r['id'] for r in rows]}")
    results.append({"check": "insert_predictions", "status": "pass"})

    page, total = store.list_predictions(limit=2, offset=0)
    assert total == 3 and len(page) == 2
    assert [row["id"] for row in page] == sorted([r["id"] for r in rows], reverse=True)
    tail, tail_total = store.list_predictions(limit=2, offset=2)
    assert tail_total == 3 and len(tail) == 1
    assert page[0]["created_at"] >= page[1]["created_at"]
    print("list: pagination (2+1 of 3) newest-first with stable tie-breaking")
    results.append({"check": "list_pagination", "status": "pass"})

    fetched = store.get_prediction(rows[0]["id"])
    assert fetched is not None
    assert fetched["predicted_class"] == "nv"
    assert abs(fetched["confidence"] - 0.91) < 1e-12
    assert fetched["class_probabilities"] == {"akiec": 0.09, "nv": 0.91}
    assert store.get_prediction(999_999_999) is None
    print("get: round-trip values match; unknown id returns None")
    results.append({"check": "get_prediction", "status": "pass"})

    try:
        store.list_predictions(limit=0, offset=0)
        raise AssertionError("limit=0 must be rejected.")
    except ValueError:
        pass
    print("validation: limit=0 rejected before touching the database")
    results.append({"check": "list_validation", "status": "pass"})

    deleted = store.delete_all_predictions()
    assert deleted == 3
    empty_page, empty_total = store.list_predictions(limit=10, offset=0)
    assert empty_total == 0 and empty_page == []
    print("delete: all 3 records removed; history empty afterwards")
    results.append({"check": "delete_all", "status": "pass"})
    return results


class _InsertFailingStore:
    """Delegates to the real store but simulates a database outage on insert."""

    def __init__(self, inner: PredictionStore):
        self._inner = inner

    def insert_prediction(self, **_: object) -> dict[str, object]:
        raise PredictionStoreError("simulated database outage during insert")

    def list_predictions(self, **kwargs: object):
        return self._inner.list_predictions(**kwargs)  # type: ignore[arg-type]


def run_api_tests(configured: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    image_path = pick_test_image()

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        expected_storage = "ready" if configured else "not_configured"
        assert health.json()["history_storage"] == expected_storage, health.text
        print(f"health: history_storage={expected_storage}")

        history = client.get("/predictions")
        if configured:
            assert history.status_code == 200, history.text
            before_total = history.json()["total"]
        else:
            assert history.status_code == 503, history.text
            assert "detail" in history.json()
            before_total = 0
            print("history endpoints: 503 with explanatory detail when not configured")
            results.append({"check": "history_not_configured_503", "status": "pass", "http": 503})

        with open(image_path, "rb") as handle:
            upload = {"file": (image_path.name, handle.read(), "image/jpeg")}
        prediction = client.post("/predict", files=upload)
        assert prediction.status_code == 200, prediction.text
        payload = prediction.json()
        expected_header = "saved" if configured else "not_configured"
        assert prediction.headers.get("X-Prediction-Storage") == expected_header, dict(
            prediction.headers
        )
        print(f"predict: 200 with X-Prediction-Storage={expected_header}")

        if not configured:
            return results

        history = client.get("/predictions", params={"limit": 5, "offset": 0})
        assert history.status_code == 200, history.text
        body = history.json()
        assert body["total"] == before_total + 1
        newest = body["items"][0]
        assert newest["image_filename"] == image_path.name
        assert newest["predicted_class"] == payload["predicted_class"]
        assert newest["predicted_class_name"] == payload["predicted_class_name"]
        assert abs(newest["confidence"] - payload["confidence"]) < 1e-9
        for label, probability in payload["probabilities"].items():
            assert abs(newest["class_probabilities"][label] - probability) < 1e-9
        parse_timestamp(newest["created_at"])
        print(
            f"stored prediction matches API response: {newest['predicted_class']} "
            f"({newest['confidence'] * 100:.2f}%) stored as record #{newest['id']}"
        )
        results.append({"check": "predict_saved_and_matches", "status": "pass", "id": newest["id"]})

        detail = client.get(f"/predictions/{newest['id']}")
        assert detail.status_code == 200, detail.text
        assert detail.json()["predicted_class"] == payload["predicted_class"]
        assert abs(detail.json()["confidence"] - payload["confidence"]) < 1e-9
        missing = client.get("/predictions/999999999")
        assert missing.status_code == 404, missing.text
        invalid_limit = client.get("/predictions", params={"limit": 0})
        assert invalid_limit.status_code == 422, invalid_limit.text
        print("detail: /predictions/{id} 200, unknown id 404, limit=0 422")
        results.append({"check": "get_prediction_endpoints", "status": "pass"})

        corrupt = client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64, "image/jpeg")},
        )
        assert corrupt.status_code == 422, corrupt.text
        total_after_failure = client.get("/predictions").json()["total"]
        assert total_after_failure == before_total + 1, total_after_failure
        print("failure isolation: rejected upload stored nothing")
        results.append({"check": "failed_prediction_not_stored", "status": "pass"})

        real_store = client.app.state.prediction_store
        assert real_store is not None
        client.app.state.prediction_store = _InsertFailingStore(real_store)
        try:
            with open(image_path, "rb") as handle:
                upload = {"file": (image_path.name, handle.read(), "image/jpeg")}
            degraded = client.post("/predict", files=upload)
            assert degraded.status_code == 200, degraded.text
            assert degraded.headers.get("X-Prediction-Storage") == "unavailable"
            total_after_outage = client.get("/predictions").json()["total"]
            assert total_after_outage == before_total + 1
        finally:
            client.app.state.prediction_store = real_store
        print("degraded mode: prediction still succeeds with header 'unavailable' during DB outage")
        results.append({"check": "db_outage_degrades_gracefully", "status": "pass"})

        cleared = client.delete("/predictions")
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["deleted_count"] == before_total + 1
        assert client.get("/predictions").json()["total"] == 0
        print(f"clear history: {cleared.json()['deleted_count']} records deleted, history empty")
        results.append({"check": "clear_history", "status": "pass"})

    return results


def run_not_configured_prediction_test() -> None:
    """A fresh app without any database configuration must still classify."""
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        unconfigured_app = create_app()
        with TestClient(unconfigured_app) as client:
            health = client.get("/health")
            assert health.status_code == 200, health.text
            assert health.json()["history_storage"] == "not_configured"
            assert client.get("/predictions").status_code == 503
            image_path = pick_test_image()
            with open(image_path, "rb") as handle:
                upload = {"file": (image_path.name, handle.read(), "image/jpeg")}
            prediction = client.post("/predict", files=upload)
            assert prediction.status_code == 200, prediction.text
            assert prediction.headers.get("X-Prediction-Storage") == "not_configured"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("unconfigured app: /predict works, /predictions returns 503, header says not_configured")


def main() -> int:
    results: list[dict[str, object]] = [{"check": "sanitize_image_filename", "status": "pass"}]
    test_sanitize_image_filename()

    if not MODEL_PATH.is_file():
        print(f"Trained checkpoint not found: {MODEL_PATH}")
        print("Restore the model artifacts first; API-level tests require trained weights.")
        return 2

    conninfo = resolve_conninfo()
    if conninfo is None:
        print("No DATABASE_URL configured: running only the not-configured degradation tests.")
        run_not_configured_prediction_test()
        results.append({"check": "unconfigured_degradation", "status": "pass"})
        report_path = PROJECT_ROOT / "db_operations_test_results.json"
        report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"DB_TESTS_SKIPPED_NO_CONFIGURATION (results saved to {report_path.name})")
        return 0

    store = PredictionStore.from_environment()
    if store is None:
        print("Unexpected state: DATABASE_URL set but resolve_conninfo returned None.")
        return 1
    try:
        print(f"Running store tests against {describe_target(conninfo)}...")
        results.extend(run_store_tests(store))
        results.extend(run_api_tests(configured=True))
        run_not_configured_prediction_test()
        results.append({"check": "unconfigured_degradation", "status": "pass"})
    finally:
        store.close()

    report_path = PROJECT_ROOT / "db_operations_test_results.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"ALL_DB_TESTS_PASSED ({len(results)} checks; results saved to {report_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
