"""Run one educational/research skin-lesion image classification prediction.

This utility loads the MobileNetV2 checkpoint produced by train_mobilenetv2.py,
then prints the predicted HAM10000 class, its confidence, and probabilities for
all classes. It is not a medical diagnosis tool.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from ham10000_data import PreprocessingConfig, normalize_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "training_output" / "mobilenetv2_ham10000" / "best_model.pt"
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen disease)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
NOTICE = (
    "Educational and research use only: this is an AI image-classification prediction, "
    "not a medical diagnosis or a substitute for clinical assessment."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?", help="Path to one skin-image file.")
    parser.add_argument(
        "--image",
        dest="image_option",
        type=Path,
        help="Path to one skin-image file (alternative to the positional argument).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to best_model.pt from train_mobilenetv2.py.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        help="Path to class_label_mapping.json (defaults next to the model checkpoint).",
    )
    parser.add_argument(
        "--preprocessing-path",
        type=Path,
        help="Path to inference_preprocessing.json (defaults next to the model checkpoint).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path at which to save the prediction JSON result.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def load_json_file(path: Path, description: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {description} '{path}': {exc}") from exc


def load_checkpoint(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    required_fields = {
        "architecture",
        "model_state_dict",
        "num_classes",
        "label_to_index",
        "preprocessing",
    }
    missing_fields = required_fields.difference(checkpoint)
    if missing_fields:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing_fields)}")
    if checkpoint["architecture"] != "MobileNetV2":
        raise ValueError(f"Unsupported checkpoint architecture: {checkpoint['architecture']}")
    if not isinstance(checkpoint["num_classes"], int) or checkpoint["num_classes"] < 1:
        raise ValueError("Checkpoint num_classes must be a positive integer.")
    return checkpoint


def parse_label_mapping(raw_mapping: Any, num_classes: int) -> tuple[dict[str, int], list[str]]:
    if not isinstance(raw_mapping, dict):
        raise ValueError("Checkpoint label_to_index must be an object.")
    mapping = {str(label): int(index) for label, index in raw_mapping.items()}
    if len(mapping) != num_classes or set(mapping) != set(CLASS_NAMES):
        raise ValueError("Checkpoint contains an unexpected HAM10000 label mapping.")
    ordered_labels = [label for label, index in sorted(mapping.items(), key=lambda item: item[1])]
    if [mapping[label] for label in ordered_labels] != list(range(num_classes)):
        raise ValueError("Checkpoint label indices must be contiguous and start at zero.")
    return mapping, ordered_labels


def parse_preprocessing(raw_config: Any) -> PreprocessingConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint preprocessing must be an object.")
    required_fields = {"image_size", "mean", "std", "padding_color"}
    missing_fields = required_fields.difference(raw_config)
    if missing_fields:
        raise ValueError(f"Checkpoint preprocessing is missing: {sorted(missing_fields)}")
    config = PreprocessingConfig(
        image_size=int(raw_config["image_size"]),
        mean=tuple(float(value) for value in raw_config["mean"]),
        std=tuple(float(value) for value in raw_config["std"]),
        padding_color=tuple(int(value) for value in raw_config["padding_color"]),
    )
    if (
        config.image_size < 1
        or len(config.mean) != 3
        or len(config.std) != 3
        or len(config.padding_color) != 3
        or any(value <= 0 for value in config.std)
    ):
        raise ValueError("Checkpoint preprocessing configuration is invalid.")
    return config


def parse_inference_preprocessing(raw_config: Any) -> PreprocessingConfig:
    """Load the separately persisted inference transform and validate its exact policy."""
    if not isinstance(raw_config, dict):
        raise ValueError("Saved inference preprocessing must be an object.")
    required_fields = {
        "color_mode",
        "image_size",
        "resize",
        "interpolation",
        "padding_color",
        "normalization",
        "tensor_layout",
    }
    missing_fields = required_fields.difference(raw_config)
    if missing_fields:
        raise ValueError(f"Saved inference preprocessing is missing: {sorted(missing_fields)}")
    image_size = raw_config["image_size"]
    normalization = raw_config["normalization"]
    if (
        raw_config["color_mode"] != "RGB"
        or raw_config["resize"] != "aspect-ratio-preserving letterbox with black padding"
        or raw_config["interpolation"] != "Pillow Image.Resampling.BILINEAR"
        or raw_config["tensor_layout"] != "C,H,W"
        or not isinstance(image_size, list)
        or len(image_size) != 2
        or image_size[0] != image_size[1]
        or not isinstance(normalization, dict)
    ):
        raise ValueError("Saved inference preprocessing policy is invalid or unsupported.")
    return parse_preprocessing(
        {
            "image_size": image_size[0],
            "mean": normalization.get("mean"),
            "std": normalization.get("std"),
            "padding_color": raw_config["padding_color"],
        }
    )


def build_model(num_classes: int, state_dict: dict[str, Any]) -> nn.Module:
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def validate_image_file(image_path: Path) -> None:
    """Reject a missing, invalid, or corrupted input before model loading."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image was not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Unable to read image '{image_path}': {type(exc).__name__}: {exc}") from exc


def predict_image(image_path: Path, model: nn.Module, preprocessing: PreprocessingConfig) -> np.ndarray:
    validate_image_file(image_path)
    with Image.open(image_path) as image:
        normalized = normalize_image(image, preprocessing)

    image_tensor = torch.from_numpy(np.expand_dims(normalized, axis=0)).to(dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1).squeeze(0).cpu().numpy()
    if not np.isfinite(probabilities).all() or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
        raise RuntimeError("Model produced invalid class probabilities.")
    return probabilities


def build_result(
    image_path: Path,
    model_path: Path,
    ordered_labels: list[str],
    probabilities: np.ndarray,
    preprocessing: PreprocessingConfig,
) -> dict[str, Any]:
    predicted_index = int(probabilities.argmax())
    predicted_code = ordered_labels[predicted_index]
    all_probabilities = [
        {
            "class_code": label,
            "class_name": CLASS_NAMES[label],
            "probability": float(probabilities[index]),
        }
        for index, label in enumerate(ordered_labels)
    ]
    return {
        "notice": NOTICE,
        "image_path": str(image_path.resolve()),
        "model_path": str(model_path.resolve()),
        "predicted_class": {
            "class_code": predicted_code,
            "class_name": CLASS_NAMES[predicted_code],
        },
        "confidence": float(probabilities[predicted_index]),
        "class_probabilities": all_probabilities,
        "preprocessing": asdict(preprocessing),
    }


def display_result(result: dict[str, Any]) -> None:
    """Print a concise, human-readable educational/research prediction result."""
    predicted = result["predicted_class"]
    print(result["notice"])
    print()
    print(f"Predicted Category: {predicted['class_code']} ({predicted['class_name']})")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print()
    print("Class Probabilities:")
    for class_probability in result["class_probabilities"]:
        print(
            f"{class_probability['class_code']} ({class_probability['class_name']}): "
            f"{class_probability['probability'] * 100:.2f}%"
        )


def main() -> int:
    args = parse_args()
    try:
        if (args.image is None) == (args.image_option is None):
            raise ValueError("Provide exactly one image path, either positionally or with --image.")
        image_path = (args.image if args.image is not None else args.image_option).resolve()
        validate_image_file(image_path)
        model_path = args.model_path.resolve()
        checkpoint = load_checkpoint(model_path)
        checkpoint_mapping, _ = parse_label_mapping(
            checkpoint["label_to_index"], checkpoint["num_classes"]
        )
        checkpoint_preprocessing = parse_preprocessing(checkpoint["preprocessing"])

        mapping_path = (
            args.mapping_path.resolve()
            if args.mapping_path is not None
            else model_path.with_name("class_label_mapping.json")
        )
        saved_mapping, ordered_labels = parse_label_mapping(
            load_json_file(mapping_path, "Class-label mapping"), checkpoint["num_classes"]
        )
        if saved_mapping != checkpoint_mapping:
            raise ValueError("Saved class-label mapping does not match the model checkpoint.")

        preprocessing_path = (
            args.preprocessing_path.resolve()
            if args.preprocessing_path is not None
            else model_path.with_name("inference_preprocessing.json")
        )
        preprocessing = parse_inference_preprocessing(
            load_json_file(preprocessing_path, "Inference preprocessing configuration")
        )
        if preprocessing != checkpoint_preprocessing:
            raise ValueError("Saved inference preprocessing does not match the model checkpoint.")

        model = build_model(checkpoint["num_classes"], checkpoint["model_state_dict"])
        probabilities = predict_image(image_path, model, preprocessing)
        result = build_result(image_path, model_path, ordered_labels, probabilities, preprocessing)
        if args.output_json:
            atomic_write_json(args.output_json.resolve(), result)
    except Exception as exc:
        print(f"Prediction error: {exc}", file=sys.stderr)
        return 2

    display_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
