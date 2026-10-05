"""Train and evaluate a MobileNetV2 HAM10000 classifier without test-set tuning.

The script uses a frozen ImageNet-pretrained MobileNetV2 feature extractor and a
new seven-class head. Training uses only train and validation splits. The test
split is loaded exactly once after the best validation checkpoint is selected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from ham10000_data import PreprocessingConfig, create_dataloaders, load_preprocessing_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "HAM10000"
DEFAULT_PREPARED_DIR = PROJECT_ROOT / "dataset" / "prepared"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training_output" / "mobilenetv2_ham10000"
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen disease)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def atomic_write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_prepared_assets(prepared_dir: Path) -> tuple[pd.DataFrame, dict[str, int], PreprocessingConfig]:
    required_files = [
        "split_manifest.csv",
        "label_to_index.json",
        "class_weights.json",
        "preparation_config.json",
        "quality_report.json",
    ]
    missing = [name for name in required_files if not (prepared_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared dataset is incomplete; missing: {missing}")

    manifest = pd.read_csv(prepared_dir / "split_manifest.csv")
    required_columns = {
        "image_id",
        "relative_path",
        "lesion_id",
        "dx",
        "split",
        "component_id",
        "duplicate_content_group",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing_columns)}")
    if len(manifest) != 10015 or manifest["image_id"].duplicated().any():
        raise ValueError("Split manifest must contain 10,015 unique HAM10000 image IDs.")
    if set(manifest["split"]) != {"train", "val", "test"}:
        raise ValueError("Split manifest must contain train, val, and test assignments.")
    if (manifest.groupby("lesion_id")["split"].nunique() > 1).any():
        raise ValueError("Leakage detected: a lesion spans multiple splits.")
    if (manifest.groupby("component_id")["split"].nunique() > 1).any():
        raise ValueError("Leakage detected: a constrained component spans multiple splits.")
    duplicate_rows = manifest.loc[manifest["duplicate_content_group"].fillna("") != ""]
    if not duplicate_rows.empty and (duplicate_rows.groupby("duplicate_content_group")["split"].nunique() > 1).any():
        raise ValueError("Leakage detected: duplicate image content spans multiple splits.")

    with (prepared_dir / "label_to_index.json").open(encoding="utf-8") as handle:
        label_to_index = {str(label): int(index) for label, index in json.load(handle).items()}
    expected_labels = sorted(CLASS_NAMES)
    expected_mapping = {label: index for index, label in enumerate(expected_labels)}
    if label_to_index != expected_mapping:
        raise ValueError(
            f"Unexpected class mapping: {label_to_index}; expected {expected_mapping}."
        )

    preprocessing = load_preprocessing_config(prepared_dir)
    if preprocessing.image_size != 224:
        raise ValueError(
            f"MobileNetV2 run requires the prepared 224x224 configuration; found {preprocessing.image_size}."
        )
    expected_mean = (0.485, 0.456, 0.406)
    expected_std = (0.229, 0.224, 0.225)
    if preprocessing.mean != expected_mean or preprocessing.std != expected_std:
        raise ValueError("Prepared normalization does not match ImageNet MobileNetV2 requirements.")
    return manifest, label_to_index, preprocessing


def verify_source_paths(manifest: pd.DataFrame, dataset_root: Path) -> None:
    missing = [
        relative_path
        for relative_path in manifest["relative_path"]
        if not (dataset_root / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing source images; first paths: {missing[:10]}")


def build_model(num_classes: int) -> nn.Module:
    """Load MobileNetV2 ImageNet weights and replace the classification head."""
    try:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load pretrained MobileNetV2 weights. Training is stopped rather than "
            "silently falling back to a non-pretrained model."
        ) from exc

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        # Head-only transfer learning must not update frozen BatchNorm statistics.
        model.features.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
            labels = torch.from_numpy(batch.labels).to(device=device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            sample_count += batch_size
    if not sample_count:
        raise RuntimeError("An empty loader cannot be used for training or validation.")
    return {"loss": total_loss / sample_count, "accuracy": correct / sample_count}


def predict_all(model: nn.Module, loader: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions; called only for untouched test evaluation."""
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
            logits = model(images)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(batch.labels)
    return np.concatenate(labels), np.concatenate(predictions)


def compute_classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, label_to_index: dict[str, int]
) -> tuple[np.ndarray, list[dict[str, float | int | str]], float]:
    ordered_labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    number_of_classes = len(ordered_labels)
    confusion = np.zeros((number_of_classes, number_of_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    metrics: list[dict[str, float | int | str]] = []
    for index, label in enumerate(ordered_labels):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append(
            {
                "class_code": label,
                "class_name": CLASS_NAMES[label],
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )
    accuracy = float((labels == predictions).mean())
    return confusion, metrics, accuracy


def save_training_plots(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epochs, [item["train_accuracy"] for item in history], marker="o", label="Train")
    axis.plot(epochs, [item["val_accuracy"] for item in history], marker="o", label="Validation")
    axis.set_title("Training and validation accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "training_validation_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epochs, [item["train_loss"] for item in history], marker="o", label="Train")
    axis.plot(epochs, [item["val_loss"] for item in history], marker="o", label="Validation")
    axis.set_title("Training and validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted cross-entropy loss")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "training_validation_loss.png", dpi=180)
    plt.close(figure)


def save_confusion_matrix(
    confusion: np.ndarray, label_to_index: dict[str, int], output_dir: Path
) -> None:
    ordered_labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    pd.DataFrame(confusion, index=ordered_labels, columns=ordered_labels).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true_class"
    )
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title("Untouched test-set confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(len(ordered_labels)), ordered_labels)
    axis.set_yticks(range(len(ordered_labels)), ordered_labels)
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center", fontsize=8)
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def write_final_report(
    output_dir: Path,
    run_summary: dict[str, Any],
    class_metrics: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# MobileNetV2 HAM10000 Training Report",
        "",
        "## Scope and safety",
        "",
        "This is an AI image-classification experiment for educational and research purposes. "
        "It is not a medical diagnosis tool and must not be used as a substitute for clinical assessment.",
        "",
        "## Model and training",
        "",
        f"- Architecture: {run_summary['architecture']}",
        f"- Device: {run_summary['device']}",
        f"- Parameters: {run_summary['parameter_count']['total']:,} total; {run_summary['parameter_count']['trainable']:,} trainable.",
        f"- Epochs completed: {run_summary['epochs_completed']}",
        f"- Best epoch: {run_summary['best_epoch']}",
        f"- Best validation accuracy: {run_summary['best_validation_accuracy']:.4f}",
        f"- Test accuracy: {run_summary['test_accuracy']:.4f}",
        f"- Training time: {run_summary['training_time_seconds']:.1f} seconds",
        "- Imbalance handling: class-weighted cross-entropy, with weights computed exclusively from the training split.",
        "- Test-set policy: the test split was not iterated during model selection, scheduling, or early stopping.",
        "",
        "## Class-wise untouched test performance",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in class_metrics:
        lines.append(
            f"| `{metric['class_code']}` | {metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['f1_score']:.4f} | {metric['support']:,} |"
        )
    lines.extend(
        [
            "",
            "## Saved artifacts",
            "",
            "- `best_model.pt`: MobileNetV2 state dictionary and architecture metadata for FastAPI inference.",
            "- `class_label_mapping.json`: stable class-code-to-index mapping.",
            "- `inference_preprocessing.json`: exact RGB, resize/padding, and normalization configuration.",
            "- `training_history.json`, `training_validation_accuracy.png`, and `training_validation_loss.png`: training history and curves.",
            "- `test_metrics.json`, `classification_report.json`, `confusion_matrix.csv`, and `confusion_matrix.png`: final untouched-test evaluation.",
        ]
    )
    atomic_write_text(output_dir / "training_report.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.early_stopping_patience < 1:
        raise ValueError("epochs, batch-size, and early-stopping-patience must be positive.")
    seed_everything(args.seed)
    dataset_root = args.dataset_root.resolve()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, label_to_index, preprocessing = verify_prepared_assets(prepared_dir)
    verify_source_paths(manifest, dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cpu":
        raise RuntimeError("This CPU-focused configuration does not permit unexpected accelerator changes.")

    with (prepared_dir / "class_weights.json").open(encoding="utf-8") as handle:
        class_weights_by_code = {str(label): float(value) for label, value in json.load(handle).items()}
    ordered_labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    if set(class_weights_by_code) != set(ordered_labels):
        raise ValueError("Class weights do not align with the saved label mapping.")
    class_weights = torch.tensor(
        [class_weights_by_code[label] for label in ordered_labels], dtype=torch.float32, device=device
    )

    loaders = create_dataloaders(
        prepared_dir=prepared_dir,
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        seed=args.seed,
        preprocessing=preprocessing,
    )
    model = build_model(num_classes=len(ordered_labels)).to(device)
    parameter_count = count_parameters(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )

    run_start = time.perf_counter()
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    no_improvement_epochs = 0
    history: list[dict[str, float]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer)
        validation_result = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(validation_result["accuracy"])
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_result["loss"],
                "train_accuracy": train_result["accuracy"],
                "val_loss": validation_result["loss"],
                "val_accuracy": validation_result["accuracy"],
                "learning_rate": current_learning_rate,
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={train_result['loss']:.4f}, "
            f"train_accuracy={train_result['accuracy']:.4f}, val_loss={validation_result['loss']:.4f}, "
            f"val_accuracy={validation_result['accuracy']:.4f}, lr={current_learning_rate:.2e}",
            flush=True,
        )

        if validation_result["accuracy"] > best_validation_accuracy + 1e-4:
            best_validation_accuracy = validation_result["accuracy"]
            best_epoch = epoch
            no_improvement_epochs = 0
            torch.save(
                {
                    "architecture": "MobileNetV2",
                    "weights_source": "torchvision MobileNet_V2_Weights.IMAGENET1K_V2",
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(ordered_labels),
                    "label_to_index": label_to_index,
                    "preprocessing": asdict(preprocessing),
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_validation_accuracy,
                },
                best_model_path,
            )
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}; no validation improvement.", flush=True)
                break

    training_time_seconds = time.perf_counter() - run_start
    if not best_model_path.is_file():
        raise RuntimeError("No validation checkpoint was saved.")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # The first and only test-data iteration occurs after validation-based selection is complete.
    test_labels, test_predictions = predict_all(model, loaders["test"], device)
    confusion, class_metrics, test_accuracy = compute_classification_metrics(
        test_labels, test_predictions, label_to_index
    )

    atomic_write_json(output_dir / "class_label_mapping.json", label_to_index)
    atomic_write_json(
        output_dir / "inference_preprocessing.json",
        {
            "color_mode": "RGB",
            "image_size": [preprocessing.image_size, preprocessing.image_size],
            "resize": "aspect-ratio-preserving letterbox with black padding",
            "interpolation": "Pillow Image.Resampling.BILINEAR",
            "padding_color": list(preprocessing.padding_color),
            "normalization": {"mean": list(preprocessing.mean), "std": list(preprocessing.std)},
            "tensor_layout": "C,H,W",
        },
    )
    atomic_write_json(output_dir / "training_history.json", history)
    save_training_plots(history, output_dir)
    save_confusion_matrix(confusion, label_to_index, output_dir)
    atomic_write_json(output_dir / "classification_report.json", class_metrics)
    run_summary = {
        "architecture": "MobileNetV2 with frozen ImageNet-pretrained feature extractor and 7-class linear head",
        "device": str(device),
        "parameter_count": parameter_count,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time_seconds,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate_scheduler": "ReduceLROnPlateau(validation accuracy, factor=0.5, patience=1)",
        "early_stopping": {"patience": args.early_stopping_patience, "metric": "validation accuracy"},
        "loss": "class-weighted cross entropy",
        "test_images_evaluated": int(len(test_labels)),
    }
    atomic_write_json(output_dir / "test_metrics.json", {**run_summary, "class_metrics": class_metrics})
    write_final_report(output_dir, run_summary, class_metrics)
    shutil.copy2(prepared_dir / "preparation_config.json", output_dir / "prepared_preprocessing_config.json")

    print("Training and untouched-test evaluation completed.")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
