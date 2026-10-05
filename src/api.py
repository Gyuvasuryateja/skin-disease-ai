"""FastAPI backend for educational/research skin-lesion image classification.

The service loads one pre-trained classifier checkpoint (the benchmarked
SkinVision EfficientNet-B0 model by default; the v1 MobileNetV2 model remains
available as a backup via SKIN_MODEL_PATH) and its persisted metadata at
startup. It is not a medical diagnostic tool.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Path as PathParam, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn

from gradcam import (
    GradCAMError,
    UnsupportedGradCAMModelError,
    downscale_for_response,
    generate_gradcam_for_image,
    image_to_jpeg_data_url,
    map_cam_to_original_image,
    overlay_heatmap,
    resolve_gradcam_target_layer,
)
from ham10000_data import PreprocessingConfig, normalize_image
from predict_image import (
    CLASS_NAMES,
    build_model,
    load_checkpoint,
    load_json_file,
    parse_inference_preprocessing,
    parse_label_mapping,
    parse_preprocessing,
)


LOGGER = logging.getLogger("skin_lesion_api")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

try:
    from db import PredictionStore, PredictionStoreError, sanitize_image_filename
except ImportError as _db_import_error:
    # History storage is optional at runtime: /predict and /explain keep working
    # without the PostgreSQL driver; only /predictions endpoints report 503.
    LOGGER.warning(
        "Prediction-history storage disabled: %s. Install requirements.txt to enable it.",
        _db_import_error,
    )
    PredictionStore = None  # type: ignore[assignment, misc]
    PredictionStoreError = RuntimeError  # type: ignore[assignment, misc]

    def sanitize_image_filename(filename: str | None) -> str:  # type: ignore[misc]
        return (filename or "unnamed-image")[-200:]

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "skinvision_efficientnet_b0"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "best_model.pth"
DEFAULT_MAPPING_PATH = DEFAULT_MODEL_DIR / "class_label_mapping.json"
DEFAULT_PREPROCESSING_PATH = DEFAULT_MODEL_DIR / "inference_preprocessing.json"
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
NOTICE = (
    "Educational and research use only: this is an AI image-classification prediction, "
    "not a medical diagnosis or a substitute for clinical assessment."
)
EXPLANATION_TEXT = "Highlighted regions indicate areas that influenced the model prediction."


@dataclass(frozen=True)
class Settings:
    model_path: Path
    mapping_path: Path
    preprocessing_path: Path
    cors_origins: tuple[str, ...]
    max_upload_bytes: int
    max_image_pixels: int
    max_explanation_image_dimension: int


def positive_integer_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def load_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
        if origin.strip()
    )
    if not origins:
        raise ValueError("CORS_ORIGINS must contain at least one origin.")
    return Settings(
        model_path=Path(os.getenv("SKIN_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve(),
        mapping_path=Path(os.getenv("SKIN_MAPPING_PATH", str(DEFAULT_MAPPING_PATH))).expanduser().resolve(),
        preprocessing_path=Path(
            os.getenv("SKIN_PREPROCESSING_PATH", str(DEFAULT_PREPROCESSING_PATH))
        ).expanduser().resolve(),
        cors_origins=origins,
        max_upload_bytes=positive_integer_setting("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        max_image_pixels=positive_integer_setting("MAX_IMAGE_PIXELS", 20_000_000),
        max_explanation_image_dimension=positive_integer_setting(
            "MAX_EXPLANATION_IMAGE_DIMENSION", 1024
        ),
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_architecture: str = Field(
        default="",
        description="Architecture of the loaded checkpoint ('' when no model is loaded).",
    )
    history_storage: str = Field(
        description="'ready', 'not_configured', or 'unavailable'. Predictions work in all states."
    )
    message: str


class PredictionResponse(BaseModel):
    notice: str
    predicted_class: str = Field(description="HAM10000 class code predicted by the model.")
    predicted_class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ImageReference(BaseModel):
    """A non-persistent inline image reference returned for one explanation."""

    data_url: str = Field(description="Inline JPEG data URL; the API does not retain uploads.")
    media_type: str = "image/jpeg"
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ExplanationResponse(PredictionResponse):
    original_image: ImageReference
    gradcam_visualization: ImageReference
    explanation: str = Field(
        description=(
            "Model-influence description only; highlighted regions do not prove a disease is present."
        )
    )


class ErrorResponse(BaseModel):
    detail: str


class StoredPrediction(BaseModel):
    """One persisted prediction record. Only a sanitized filename is kept."""

    id: int = Field(ge=1)
    image_filename: str = Field(
        description="Sanitized client-side filename; the image itself is never stored."
    )
    predicted_class: str = Field(description="HAM10000 class code predicted by the model.")
    predicted_class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
    created_at: datetime


class PredictionHistoryResponse(BaseModel):
    items: list[StoredPrediction]
    total: int = Field(ge=0, description="Total number of stored predictions.")
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ClearHistoryResponse(BaseModel):
    deleted_count: int = Field(ge=0)


@dataclass
class SkinLesionClassifier:
    model: nn.Module
    architecture: str
    ordered_labels: list[str]
    preprocessing: PreprocessingConfig
    gradcam_target_layer: nn.Module | None
    gradcam_error: str | None = None
    _explanation_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def _build_prediction_response(self, probabilities: np.ndarray) -> PredictionResponse:
        if (
            probabilities.ndim != 1
            or len(probabilities) != len(self.ordered_labels)
            or not np.isfinite(probabilities).all()
            or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5)
        ):
            raise RuntimeError("The model returned invalid class probabilities.")
        predicted_index = int(probabilities.argmax())
        predicted_class = self.ordered_labels[predicted_index]
        return PredictionResponse(
            notice=NOTICE,
            predicted_class=predicted_class,
            predicted_class_name=CLASS_NAMES[predicted_class],
            confidence=float(probabilities[predicted_index]),
            probabilities={
                label: float(probabilities[index]) for index, label in enumerate(self.ordered_labels)
            },
        )

    def predict(self, image: Image.Image) -> PredictionResponse:
        normalized = normalize_image(image, self.preprocessing)
        image_tensor = torch.from_numpy(np.expand_dims(normalized, axis=0)).to(dtype=torch.float32)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(image_tensor), dim=1).squeeze(0).cpu().numpy()
        return self._build_prediction_response(probabilities)

    def explain(self, image: Image.Image, max_image_dimension: int) -> ExplanationResponse:
        """Return a predicted-class Grad-CAM without retaining the uploaded image."""
        if self.gradcam_target_layer is None:
            raise UnsupportedGradCAMModelError(
                self.gradcam_error
                or "The loaded model does not expose a supported Grad-CAM target layer."
            )
        # Grad-CAM temporarily registers a hook and backpropagates only to obtain
        # influence weights. The lock prevents hook/gradient state from crossing
        # between simultaneous explanation requests on the shared startup model.
        with self._explanation_lock:
            gradcam_result = generate_gradcam_for_image(
                self.model, image, self.preprocessing, self.gradcam_target_layer
            )
        prediction = self._build_prediction_response(gradcam_result.probabilities)
        if prediction.predicted_class != self.ordered_labels[gradcam_result.predicted_index]:
            raise GradCAMError("Grad-CAM class selection does not match the prediction result.")

        source_aligned_cam = map_cam_to_original_image(
            gradcam_result.cam, image.size, self.preprocessing
        )
        display_original = downscale_for_response(image, max_image_dimension)
        display_cam = source_aligned_cam.resize(display_original.size, Image.Resampling.BILINEAR)
        visualization = overlay_heatmap(display_original, display_cam)
        return ExplanationResponse(
            **prediction.model_dump(),
            original_image=ImageReference(
                data_url=image_to_jpeg_data_url(display_original),
                width=display_original.width,
                height=display_original.height,
            ),
            gradcam_visualization=ImageReference(
                data_url=image_to_jpeg_data_url(visualization),
                width=visualization.width,
                height=visualization.height,
            ),
            explanation=EXPLANATION_TEXT,
        )


def load_classifier(settings: Settings) -> SkinLesionClassifier:
    """Load and cross-check all model assets once during application startup."""
    checkpoint = load_checkpoint(settings.model_path)
    checkpoint_mapping, _ = parse_label_mapping(
        checkpoint["label_to_index"], checkpoint["num_classes"]
    )
    checkpoint_preprocessing = parse_preprocessing(checkpoint["preprocessing"])

    saved_mapping, ordered_labels = parse_label_mapping(
        load_json_file(settings.mapping_path, "Class-label mapping"), checkpoint["num_classes"]
    )
    if saved_mapping != checkpoint_mapping:
        raise ValueError("Saved class-label mapping does not match the model checkpoint.")

    preprocessing = parse_inference_preprocessing(
        load_json_file(settings.preprocessing_path, "Inference preprocessing configuration")
    )
    if preprocessing != checkpoint_preprocessing:
        raise ValueError("Saved inference preprocessing does not match the model checkpoint.")

    model = build_model(
        checkpoint["architecture"], checkpoint["num_classes"], checkpoint["model_state_dict"]
    )
    try:
        gradcam_target_layer = resolve_gradcam_target_layer(model)
        gradcam_error = None
    except UnsupportedGradCAMModelError as exc:
        # Keep ordinary prediction available when a future compatible inference
        # model lacks a known Grad-CAM target; /explain will return a clear error.
        gradcam_target_layer = None
        gradcam_error = str(exc)
    return SkinLesionClassifier(
        model=model,
        architecture=checkpoint["architecture"],
        ordered_labels=ordered_labels,
        preprocessing=preprocessing,
        gradcam_target_layer=gradcam_target_layer,
        gradcam_error=gradcam_error,
    )


def decode_upload(contents: bytes, max_image_pixels: int) -> Image.Image:
    """Fully decode a supported image before passing it to the model."""
    try:
        with Image.open(BytesIO(contents)) as opened_image:
            opened_image.load()
            if opened_image.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported decoded image format. Use JPEG, PNG, or WebP.",
                )
            if opened_image.width * opened_image.height > max_image_pixels:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image dimensions exceed the {max_image_pixels:,}-pixel limit.",
                )
            return opened_image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"The uploaded file is not a valid, readable image: {type(exc).__name__}.",
        ) from exc


async def read_and_decode_upload(file: UploadFile, settings: Settings) -> Image.Image:
    """Apply one shared upload-validation path for prediction and explanation."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Upload a JPEG, PNG, or WebP image.",
        )
    try:
        contents = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded image file is empty.",
        )
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded image exceeds the {settings.max_upload_bytes:,}-byte limit.",
        )
    return decode_upload(contents, settings.max_image_pixels)


def persist_prediction(app: FastAPI, filename: str | None, response: PredictionResponse) -> str:
    """Save one completed prediction; never called for failed predictions.

    Returns the storage outcome for the X-Prediction-Storage response header:
    'saved', 'not_configured', or 'unavailable'. Storage failures never
    invalidate an otherwise successful classification.
    """
    store = getattr(app.state, "prediction_store", None)
    if store is None:
        return str(getattr(app.state, "history_status", "not_configured"))
    try:
        store.insert_prediction(
            image_filename=sanitize_image_filename(filename),
            predicted_class=response.predicted_class,
            predicted_class_name=response.predicted_class_name,
            confidence=response.confidence,
            class_probabilities=response.probabilities,
        )
    except PredictionStoreError as exc:
        LOGGER.error("Prediction history could not be saved: %s", exc)
        return "unavailable"
    except Exception:
        LOGGER.exception("Unexpected error while saving prediction history")
        return "unavailable"
    return "saved"


def create_app() -> FastAPI:
    settings = load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = None
        app.state.model_error = None
        app.state.prediction_store = None
        app.state.history_status = "not_configured"
        app.state.history_error = None
        try:
            app.state.classifier = load_classifier(settings)
            LOGGER.info(
                "Skin-lesion model (%s) loaded once during startup from %s",
                app.state.classifier.architecture,
                settings.model_path,
            )
        except Exception as exc:
            app.state.model_error = str(exc)
            LOGGER.exception("Skin-lesion model could not be loaded during startup")

        # History storage degrades gracefully: a missing or unreachable
        # database never blocks classification, only /predictions endpoints.
        if PredictionStore is not None:
            try:
                store = PredictionStore.from_environment()
            except Exception as exc:
                app.state.history_status = "unavailable"
                app.state.history_error = str(exc)
                LOGGER.error("Prediction-history storage could not be initialized: %s", exc)
            else:
                if store is None:
                    app.state.history_error = (
                        "Prediction history storage is not configured. Set DATABASE_URL "
                        "(or PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD) to enable it."
                    )
                    LOGGER.info(
                        "Prediction-history storage is not configured; predictions will not be saved."
                    )
                else:
                    app.state.prediction_store = store
                    app.state.history_status = "ready"
                    LOGGER.info("Prediction-history storage ready (schema migrations applied).")
        else:
            app.state.history_status = "unavailable"
            app.state.history_error = (
                "Prediction history storage requires the psycopg[binary,pool] package."
            )
        yield
        app.state.classifier = None
        if app.state.prediction_store is not None:
            app.state.prediction_store.close()
            app.state.prediction_store = None

    app = FastAPI(
        title="Skin-Lesion Classification API",
        version="1.0.0",
        description=(
            "Educational and research AI image classification only. This API is not a medical "
            "diagnostic tool and must not replace clinical assessment. Model inference uses the "
            "saved preprocessing configuration and performs no training. Prediction history is "
            "stored in PostgreSQL when configured; uploaded images are never persisted — only a "
            "sanitized filename is saved with each record."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Invalid request. Submit one image file using the 'file' form field.", "errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unexpected API error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Unexpected server error while processing the request."},
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["Service"],
        summary="Check service and model readiness",
    )
    async def health() -> HealthResponse | JSONResponse:
        if app.state.classifier is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=HealthResponse(
                    status="unavailable",
                    model_loaded=False,
                    history_storage=app.state.history_status,
                    message="Model assets could not be loaded. Check server configuration and logs.",
                ).model_dump(),
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_architecture=app.state.classifier.architecture,
            history_storage=app.state.history_status,
            message="Educational/research image-classification model is ready.",
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["Prediction"],
        summary="Classify one uploaded skin image",
    )
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        if app.state.classifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is unavailable. Check the server configuration and model assets.",
            )
        image = await read_and_decode_upload(file, settings)
        prediction = app.state.classifier.predict(image)
        # Persist only after the classification succeeded: a failed or invalid
        # upload raises above and is never stored as a successful prediction.
        storage_status = persist_prediction(app, file.filename, prediction)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=prediction.model_dump(mode="json"),
            headers={"X-Prediction-Storage": storage_status},
        )

    @app.post(
        "/explain",
        response_model=ExplanationResponse,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            501: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["Explanation"],
        summary="Classify an image and return predicted-class Grad-CAM",
        description=(
            "Returns the predicted category plus an inline Grad-CAM visualization. Highlighted "
            "regions show model influence only and do not establish the presence of a disease."
        ),
    )
    async def explain(file: UploadFile = File(...)) -> JSONResponse:
        if app.state.classifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is unavailable. Check the server configuration and model assets.",
            )
        image = await read_and_decode_upload(file, settings)
        try:
            explanation = app.state.classifier.explain(
                image, settings.max_explanation_image_dimension
            )
        except UnsupportedGradCAMModelError as exc:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Grad-CAM is unavailable for the loaded model: {exc}",
            ) from exc
        except GradCAMError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Grad-CAM visualization could not be generated: {exc}",
            ) from exc
        # The classification embedded in this explanation succeeded, so it is
        # stored exactly like a /predict result; images are never persisted.
        storage_status = persist_prediction(app, file.filename, explanation)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=explanation.model_dump(mode="json"),
            headers={"X-Prediction-Storage": storage_status},
        )

    def history_unavailable_detail() -> str:
        return app.state.history_error or "Prediction history storage is unavailable."

    @app.get(
        "/predictions",
        response_model=PredictionHistoryResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["History"],
        summary="List stored prediction history (newest first)",
    )
    async def list_predictions(
        limit: int = Query(default=20, ge=1, le=100, description="Page size between 1 and 100."),
        offset: int = Query(default=0, ge=0, description="Number of records to skip."),
    ) -> PredictionHistoryResponse:
        store = app.state.prediction_store
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=history_unavailable_detail(),
            )
        try:
            rows, total = store.list_predictions(limit=limit, offset=offset)
        except PredictionStoreError as exc:
            LOGGER.error("Could not list prediction history: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction history is temporarily unavailable. Check the database.",
            ) from exc
        return PredictionHistoryResponse(
            items=[StoredPrediction(**row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/predictions/{prediction_id}",
        response_model=StoredPrediction,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        tags=["History"],
        summary="Retrieve one stored prediction",
    )
    async def get_prediction(
        prediction_id: int = PathParam(..., ge=1, description="Stored prediction id."),
    ) -> StoredPrediction:
        store = app.state.prediction_store
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=history_unavailable_detail(),
            )
        try:
            row = store.get_prediction(prediction_id)
        except PredictionStoreError as exc:
            LOGGER.error("Could not read prediction %s: %s", prediction_id, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction history is temporarily unavailable. Check the database.",
            ) from exc
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No stored prediction with id {prediction_id}.",
            )
        return StoredPrediction(**row)

    @app.delete(
        "/predictions",
        response_model=ClearHistoryResponse,
        responses={503: {"model": ErrorResponse}},
        tags=["History"],
        summary="Delete all stored prediction history",
        description=(
            "Permanently removes every stored prediction record. Only these records are "
            "deleted; uploaded images were never stored, and the model is untouched."
        ),
    )
    async def clear_predictions() -> ClearHistoryResponse:
        store = app.state.prediction_store
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=history_unavailable_detail(),
            )
        try:
            deleted_count = store.delete_all_predictions()
        except PredictionStoreError as exc:
            LOGGER.error("Could not clear prediction history: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction history is temporarily unavailable. Check the database.",
            ) from exc
        LOGGER.info("Cleared %d stored prediction records.", deleted_count)
        return ClearHistoryResponse(deleted_count=deleted_count)

    return app


app = create_app()
