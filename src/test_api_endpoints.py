"""Endpoint tests for the skin-lesion API, including predicted-class Grad-CAM.

Run after train_mobilenetv2.py has produced its artifacts:

    python src/test_api_endpoints.py

The tests perform inference only. Held-out test images are read from disk and
never modified, and no training or tuning occurs here. Every assertion treats
the classifier as an educational/research tool, not a medical diagnosis system.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import app  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "training_output" / "mobilenetv2_ham10000" / "best_model.pt"
MANIFEST_PATH = PROJECT_ROOT / "dataset" / "prepared" / "split_manifest.csv"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "HAM10000"
EXPECTED_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
FRONTEND_ORIGIN = "http://localhost:3000"


def select_test_images(count_per_class: int = 1) -> list[dict[str, str]]:
    """Choose held-out test images covering every class without modifying them."""
    import pandas as pd

    manifest = pd.read_csv(MANIFEST_PATH)
    test_rows = manifest.loc[manifest["split"] == "test"]
    if test_rows.empty:
        raise RuntimeError("The prepared manifest contains no test-split images.")
    selected: list[dict[str, str]] = []
    for dx, group in test_rows.groupby("dx"):
        for record in group.head(count_per_class).itertuples(index=False):
            selected.append(
                {"image_id": record.image_id, "dx": dx, "path": DATASET_ROOT / record.relative_path}
            )
    return selected


def decode_data_url(data_url: str) -> Image.Image:
    if not data_url.startswith("data:image/jpeg;base64,"):
        raise AssertionError("Explanation images must be inline JPEG data URLs.")
    payload = base64.b64decode(data_url.split(",", 1)[1])
    image = Image.open(io.BytesIO(payload))
    image.load()
    if image.format != "JPEG":
        raise AssertionError(f"Expected JPEG payload, found {image.format}.")
    return image


def upload(client: TestClient, path: Path, endpoint: str):
    return client.post(
        endpoint,
        files={"file": (path.name, path.read_bytes(), "image/jpeg")},
    )


def check_prediction_payload(payload: dict) -> None:
    assert payload["predicted_class"] in EXPECTED_CLASSES, payload["predicted_class"]
    assert 0.0 <= payload["confidence"] <= 1.0
    assert set(payload["probabilities"]) == EXPECTED_CLASSES
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-4
    assert "not a medical diagnosis" in payload["notice"]


def main() -> int:
    if not MODEL_PATH.is_file():
        print(f"Trained checkpoint not found: {MODEL_PATH}")
        print("Run src/train_mobilenetv2.py first; these tests require real trained artifacts.")
        return 2

    results: list[dict[str, object]] = []
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        health_payload = health.json()
        assert health_payload["status"] == "ok" and health_payload["model_loaded"] is True
        print("health: ok, model loaded")
        results.append({"check": "health", "status": "pass"})

        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        paths = openapi.json()["paths"]
        assert "/health" in paths and "/predict" in paths and "/explain" in paths
        print("openapi: /health, /predict, /explain documented")
        results.append({"check": "openapi_documentation", "status": "pass"})

        preflight = client.options(
            "/explain",
            headers={
                "Origin": FRONTEND_ORIGIN,
                "Access-Control-Request-Method": "POST",
            },
        )
        allowed = preflight.headers.get("access-control-allow-origin")
        assert preflight.status_code == 200 and allowed == FRONTEND_ORIGIN, (
            preflight.status_code,
            allowed,
        )
        print(f"cors: preflight from {FRONTEND_ORIGIN} allowed")
        results.append({"check": "cors_preflight", "status": "pass"})

        for record in select_test_images():
            image_path = Path(record["path"])
            prediction_response = upload(client, image_path, "/predict")
            assert prediction_response.status_code == 200, prediction_response.text
            prediction = prediction_response.json()
            check_prediction_payload(prediction)

            explanation_response = upload(client, image_path, "/explain")
            assert explanation_response.status_code == 200, explanation_response.text
            explanation = explanation_response.json()
            check_prediction_payload(explanation)

            assert explanation["predicted_class"] == prediction["predicted_class"], (
                f"/explain class {explanation['predicted_class']} differs from "
                f"/predict class {prediction['predicted_class']}"
            )
            assert abs(explanation["confidence"] - prediction["confidence"]) < 1e-6
            assert explanation["explanation"] == (
                "Highlighted regions indicate areas that influenced the model prediction."
            )

            original = decode_data_url(explanation["original_image"]["data_url"])
            visualization = decode_data_url(explanation["gradcam_visualization"]["data_url"])
            assert original.size == visualization.size == (
                explanation["original_image"]["width"],
                explanation["original_image"]["height"],
            )
            assert max(original.size) <= 1024

            repeat = upload(client, image_path, "/explain").json()
            assert repeat["predicted_class"] == explanation["predicted_class"]
            assert repeat["gradcam_visualization"]["data_url"] == explanation["gradcam_visualization"]["data_url"]

            matches_label = explanation["predicted_class"] == record["dx"]
            results.append(
                {
                    "check": "explain_image",
                    "image_id": record["image_id"],
                    "true_class": record["dx"],
                    "predicted_class": explanation["predicted_class"],
                    "confidence": round(explanation["confidence"], 4),
                    "matches_true_label": matches_label,
                    "visualization_size": list(visualization.size),
                    "deterministic_repeat": True,
                    "status": "pass",
                }
            )
            print(
                f"explain {record['image_id']}: true={record['dx']} "
                f"predicted={explanation['predicted_class']} "
                f"confidence={explanation['confidence'] * 100:.2f}% "
                f"visualization={visualization.size[0]}x{visualization.size[1]}"
            )

        empty = client.post("/explain", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert empty.status_code == 422, (empty.status_code, empty.text)
        results.append({"check": "empty_upload_rejected", "status": "pass", "http": 422})
        print("error handling: empty upload -> 422")

        bad_type = client.post(
            "/explain", files={"file": ("notes.txt", b"not an image", "text/plain")}
        )
        assert bad_type.status_code == 415, (bad_type.status_code, bad_type.text)
        results.append({"check": "unsupported_type_rejected", "status": "pass", "http": 415})
        print("error handling: unsupported type -> 415")

        corrupt = client.post(
            "/explain",
            files={"file": ("corrupt.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64, "image/jpeg")},
        )
        assert corrupt.status_code == 422, (corrupt.status_code, corrupt.text)
        results.append({"check": "corrupt_image_rejected", "status": "pass", "http": 422})
        print("error handling: corrupt image -> 422")

    report_path = PROJECT_ROOT / "api_endpoint_test_results.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"ALL_ENDPOINT_TESTS_PASSED ({len(results)} checks; results saved to {report_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
