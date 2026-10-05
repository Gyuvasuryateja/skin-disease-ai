"""Prepare HAM10000 for later model training without training a model.

This script validates labels and images, detects duplicate content, creates a
reproducible group-aware 70/15/15 split, writes class weights and reports, and
verifies the reusable preprocessing loader on every image.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from ham10000_data import PreprocessingConfig, create_dataloaders, verify_preprocessed_loaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "HAM10000"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "prepared"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen disease)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
SPLIT_NAMES = ("train", "val", "test")


class UnionFind:
    """Small disjoint-set implementation for leakage constraints."""

    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class CatalogResult:
    image_paths: dict[str, str]
    duplicate_content_groups: list[list[str]]
    formats: dict[str, int]
    dimensions: dict[str, int]
    unreadable_files: list[dict[str, str]]
    zero_byte_files: list[str]


def atomic_write_text(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def catalog_images(dataset_root: Path) -> CatalogResult:
    """Find, fully decode, and hash canonical RGB pixels for all source images."""
    image_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    by_image_id: dict[str, str] = {}
    hash_to_ids: dict[str, list[str]] = defaultdict(list)
    formats: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    unreadable_files: list[dict[str, str]] = []
    zero_byte_files: list[str] = []

    for position, path in enumerate(image_paths, start=1):
        relative_path = path.relative_to(dataset_root).as_posix()
        image_id = path.stem
        if image_id in by_image_id:
            raise RuntimeError(
                f"Multiple extracted image paths use the same image ID: {image_id}."
            )
        try:
            raw = path.read_bytes()
            if not raw:
                zero_byte_files.append(relative_path)
                continue
            with Image.open(BytesIO(raw)) as image:
                image.load()
                formats[image.format or "<unknown>"] += 1
                dimensions[f"{image.width}x{image.height}"] += 1
                rgb_image = image.convert("RGB")
                pixel_hash = hashlib.sha256()
                pixel_hash.update(f"RGB:{rgb_image.width}x{rgb_image.height}:".encode("ascii"))
                pixel_hash.update(rgb_image.tobytes())
                content_hash = pixel_hash.hexdigest()
            by_image_id[image_id] = relative_path
            hash_to_ids[content_hash].append(image_id)
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            unreadable_files.append({"path": relative_path, "error": f"{type(exc).__name__}: {exc}"})
        if position % 1000 == 0:
            print(f"Cataloged {position}/{len(image_paths)} images", flush=True)

    duplicate_content_groups = sorted(
        (sorted(ids) for ids in hash_to_ids.values() if len(ids) > 1),
        key=lambda ids: (-len(ids), ids),
    )
    return CatalogResult(
        image_paths=by_image_id,
        duplicate_content_groups=duplicate_content_groups,
        formats=dict(sorted(formats.items())),
        dimensions=dict(sorted(dimensions.items(), key=lambda item: (-item[1], item[0]))),
        unreadable_files=unreadable_files,
        zero_byte_files=zero_byte_files,
    )


def load_and_validate_metadata(dataset_root: Path, catalog: CatalogResult) -> pd.DataFrame:
    metadata_path = dataset_root / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV was not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required_columns = {"lesion_id", "image_id", "dx"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")
    if metadata.empty:
        raise ValueError("Metadata CSV contains no records.")
    if metadata["image_id"].isna().any() or metadata["image_id"].duplicated().any():
        raise ValueError("Metadata image_id values must be non-empty and unique.")
    if metadata["lesion_id"].isna().any():
        raise ValueError("Metadata contains missing lesion_id values.")
    if metadata["dx"].isna().any() or (metadata["dx"].astype(str).str.strip() == "").any():
        raise ValueError("Metadata contains missing or blank diagnostic labels.")

    metadata["image_id"] = metadata["image_id"].astype(str)
    metadata["dx"] = metadata["dx"].astype(str)
    metadata["lesion_id"] = metadata["lesion_id"].astype(str)
    unexpected_classes = sorted(set(metadata["dx"]).difference(CLASS_NAMES))
    if unexpected_classes:
        raise ValueError(f"Unexpected HAM10000 class labels: {unexpected_classes}")

    metadata_ids = set(metadata["image_id"])
    image_ids = set(catalog.image_paths)
    missing_images = sorted(metadata_ids.difference(image_ids))
    unlabelled_images = sorted(image_ids.difference(metadata_ids))
    if catalog.zero_byte_files or catalog.unreadable_files or missing_images or unlabelled_images:
        details = {
            "zero_byte_files": catalog.zero_byte_files,
            "unreadable_files": catalog.unreadable_files,
            "metadata_ids_missing_images": missing_images,
            "image_ids_missing_metadata": unlabelled_images,
        }
        raise RuntimeError("Dataset validation failed:\n" + json.dumps(details, indent=2))

    metadata["relative_path"] = metadata["image_id"].map(catalog.image_paths)
    return metadata.sort_values("image_id").reset_index(drop=True)


def assign_components_to_splits(
    metadata: pd.DataFrame,
    duplicate_content_groups: list[list[str]],
    ratios: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign image components to splits while keeping lesions and duplicates intact."""
    image_ids = metadata["image_id"].tolist()
    union_find = UnionFind(image_ids)

    for _, lesion_records in metadata.groupby("lesion_id", sort=True):
        ids = lesion_records["image_id"].tolist()
        for image_id in ids[1:]:
            union_find.union(ids[0], image_id)

    duplicate_group_for_image: dict[str, str] = {}
    for index, duplicate_ids in enumerate(duplicate_content_groups, start=1):
        duplicate_group_id = f"content_duplicate_{index:03d}"
        for image_id in duplicate_ids:
            duplicate_group_for_image[image_id] = duplicate_group_id
        for image_id in duplicate_ids[1:]:
            union_find.union(duplicate_ids[0], image_id)

    root_to_ids: dict[str, list[str]] = defaultdict(list)
    for image_id in image_ids:
        root_to_ids[union_find.find(image_id)].append(image_id)
    component_for_image = {
        image_id: f"component_{min(ids)}" for ids in root_to_ids.values() for image_id in ids
    }

    manifest = metadata.copy()
    manifest["component_id"] = manifest["image_id"].map(component_for_image)
    manifest["duplicate_content_group"] = manifest["image_id"].map(duplicate_group_for_image).fillna("")

    classes = sorted(CLASS_NAMES)
    class_totals = manifest["dx"].value_counts().reindex(classes, fill_value=0).astype(float)
    total_images = float(len(manifest))
    target_totals = {split: total_images * ratios[split] for split in SPLIT_NAMES}
    target_classes = {
        split: {label: class_totals[label] * ratios[split] for label in classes} for split in SPLIT_NAMES
    }

    component_records = []
    for component_id, component_rows in manifest.groupby("component_id", sort=True):
        class_counts = component_rows["dx"].value_counts().reindex(classes, fill_value=0).astype(int)
        rarity = sum(
            class_counts[label] / class_totals[label]
            for label in classes
            if class_counts[label] > 0 and class_totals[label] > 0
        )
        component_records.append(
            {
                "component_id": component_id,
                "size": int(len(component_rows)),
                "class_counts": {label: int(class_counts[label]) for label in classes},
                "rarity": float(rarity),
            }
        )

    rng = np.random.default_rng(seed)
    tie_breakers = {record["component_id"]: float(rng.random()) for record in component_records}
    component_records.sort(
        key=lambda record: (
            -record["rarity"],
            -record["size"],
            tie_breakers[record["component_id"]],
            record["component_id"],
        )
    )

    assigned_total = {split: 0.0 for split in SPLIT_NAMES}
    assigned_class = {split: {label: 0.0 for label in classes} for split in SPLIT_NAMES}
    component_split: dict[str, str] = {}

    def objective(candidate_split: str, record: dict[str, object]) -> float:
        totals = dict(assigned_total)
        class_counts = {split: dict(values) for split, values in assigned_class.items()}
        totals[candidate_split] += float(record["size"])
        for label, count in record["class_counts"].items():
            class_counts[candidate_split][label] += float(count)

        total_error = sum(
            ((totals[split] - target_totals[split]) / max(target_totals[split], 1.0)) ** 2
            for split in SPLIT_NAMES
        )
        class_error = sum(
            ((class_counts[split][label] - target_classes[split][label]) / max(target_classes[split][label], 1.0)) ** 2
            for split in SPLIT_NAMES
            for label in classes
        )
        return class_error + 0.2 * total_error

    for record in component_records:
        scores = {split: objective(split, record) for split in SPLIT_NAMES}
        selected_split = min(
            SPLIT_NAMES,
            key=lambda split: (scores[split], assigned_total[split] / max(target_totals[split], 1.0), split),
        )
        component_split[str(record["component_id"])] = selected_split
        assigned_total[selected_split] += float(record["size"])
        for label, count in record["class_counts"].items():
            assigned_class[selected_split][label] += float(count)

    manifest["split"] = manifest["component_id"].map(component_split)
    split_summary = {
        split: {
            "images": int((manifest["split"] == split).sum()),
            "classes": {
                label: int(((manifest["split"] == split) & (manifest["dx"] == label)).sum())
                for label in classes
            },
        }
        for split in SPLIT_NAMES
    }
    return manifest, split_summary


def validate_split_integrity(manifest: pd.DataFrame) -> dict[str, int]:
    """Fail preparation if any leakage or missing split assignment is detected."""
    if manifest["split"].isna().any() or not set(manifest["split"]).issubset(SPLIT_NAMES):
        raise ValueError("Every image must have one valid split assignment.")
    component_leaks = int((manifest.groupby("component_id")["split"].nunique() > 1).sum())
    lesion_leaks = int((manifest.groupby("lesion_id")["split"].nunique() > 1).sum())
    duplicate_rows = manifest.loc[manifest["duplicate_content_group"] != ""]
    duplicate_leaks = int(
        (duplicate_rows.groupby("duplicate_content_group")["split"].nunique() > 1).sum()
    )
    if component_leaks or lesion_leaks or duplicate_leaks:
        raise RuntimeError(
            "Leakage detected: "
            f"components={component_leaks}, lesions={lesion_leaks}, duplicate_groups={duplicate_leaks}."
        )
    return {
        "component_leaks": component_leaks,
        "lesion_leaks": lesion_leaks,
        "duplicate_content_leaks": duplicate_leaks,
    }


def create_distribution_artifacts(manifest: pd.DataFrame, output_dir: Path) -> None:
    """Save a class count CSV, a readable report, and a PNG chart."""
    classes = sorted(CLASS_NAMES)
    totals = manifest["dx"].value_counts().reindex(classes, fill_value=0)
    split_counts = pd.crosstab(manifest["dx"], manifest["split"]).reindex(
        index=classes, columns=SPLIT_NAMES, fill_value=0
    )
    distribution = pd.DataFrame(
        {
            "class_code": classes,
            "class_name": [CLASS_NAMES[label] for label in classes],
            "total": [int(totals[label]) for label in classes],
            "train": [int(split_counts.loc[label, "train"]) for label in classes],
            "val": [int(split_counts.loc[label, "val"]) for label in classes],
            "test": [int(split_counts.loc[label, "test"]) for label in classes],
        }
    )
    distribution["dataset_share_percent"] = (distribution["total"] / len(manifest) * 100).round(2)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    colors = ["#8ecae6", "#219ebc", "#ffb703", "#fb8500", "#90be6d", "#577590", "#f28482"]
    axes[0].bar(distribution["class_code"], distribution["total"], color=colors)
    axes[0].set_title("HAM10000 class distribution")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Images")
    for index, value in enumerate(distribution["total"]):
        axes[0].text(index, value, str(value), ha="center", va="bottom", fontsize=8)

    positions = np.arange(len(classes))
    width = 0.24
    split_colors = {"train": "#219ebc", "val": "#ffb703", "test": "#fb8500"}
    for offset, split in zip((-width, 0, width), SPLIT_NAMES):
        axes[1].bar(
            positions + offset,
            distribution[split],
            width=width,
            label=split,
            color=split_colors[split],
        )
    axes[1].set_title("Class counts by reproducible split")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Images")
    axes[1].set_xticks(positions, classes)
    axes[1].legend()
    figure.savefig(output_dir / "class_distribution_chart.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Prepared HAM10000 Class Distribution",
        "",
        "| Code | Class | Total | Train | Validation | Test | Dataset share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in distribution.itertuples(index=False):
        lines.append(
            f"| `{record.class_code}` | {record.class_name} | {record.total:,} | "
            f"{record.train:,} | {record.val:,} | {record.test:,} | {record.dataset_share_percent:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"The largest-to-smallest class ratio is {totals.max() / totals.min():.2f}:1 "
            f"(`{totals.idxmax()}` versus `{totals.idxmin()}`).",
            "",
            "Training uses class-weighted loss values derived only from the training split. "
            "Validation and test sets are not augmented or rebalanced.",
        ]
    )
    atomic_write_text(output_dir / "class_distribution_report.md", "\n".join(lines) + "\n")


def write_preparation_summary(
    output_dir: Path,
    manifest: pd.DataFrame,
    catalog: CatalogResult,
    integrity: dict[str, int],
    preprocessing_verification: dict[str, dict[str, object]],
) -> None:
    split_counts = manifest["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    duplicate_pairs = [
        [manifest.loc[manifest["image_id"] == image_id, "relative_path"].iloc[0] for image_id in group]
        for group in catalog.duplicate_content_groups
    ]
    lines = [
        "# HAM10000 Data Preparation Summary",
        "",
        "## Prepared artifacts",
        "",
        "- `split_manifest.csv`: immutable image-to-split assignment for reproducible experiments.",
        "- `label_to_index.json`: stable diagnostic class index mapping.",
        "- `class_weights.json`: training-split class weights for weighted cross-entropy.",
        "- `class_distribution.csv`, `class_distribution_report.md`, and `class_distribution_chart.png`: distribution artifacts.",
        "- `preprocessing_verification.json`: full loader validation result.",
        "",
        "## Split design",
        "",
        f"- Train: {split_counts['train']:,} images",
        f"- Validation: {split_counts['val']:,} images",
        f"- Test: {split_counts['test']:,} images",
        "- Target split ratio: 70% / 15% / 15%",
        "- Leakage safeguards: all images from the same `lesion_id`, and all decoded-pixel-identical images, share one split.",
        f"- Leakage validation: component leaks={integrity['component_leaks']}, lesion leaks={integrity['lesion_leaks']}, duplicate-content leaks={integrity['duplicate_content_leaks']}.",
        "",
        "## Preprocessing",
        "",
        "- Input: RGB, aspect-ratio-preserving letterbox to 224 × 224.",
        "- Normalization: ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.",
        "- Training-only augmentation: horizontal and vertical flips, ±20° rotation, brightness adjustment (0.85–1.15), and contrast adjustment (0.85–1.15).",
        "- Validation/test: deterministic resize, padding, and normalization only; no random augmentation.",
        "",
        "## Full preprocessing verification",
        "",
    ]
    for split in SPLIT_NAMES:
        details = preprocessing_verification[split]
        lines.append(
            f"- {split}: {details['images_loaded']:,} images loaded in {details['batches_loaded']} batches; "
            f"shape C×H×W={details['shape']}; finite normalized range "
            f"[{details['value_min']:.4f}, {details['value_max']:.4f}]; "
            f"random augmentation={details['random_augmentation']}."
        )
    lines.extend(["", "## Duplicate content retained", ""])
    if duplicate_pairs:
        for index, paths in enumerate(duplicate_pairs, start=1):
            lines.append(f"{index}. `{'` and `'.join(paths)}` (kept in the same split)")
    else:
        lines.append("No duplicate image content was detected.")
    atomic_write_text(output_dir / "preparation_summary.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if args.image_size < 32:
        raise ValueError("image-size must be at least 32 pixels.")
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    catalog = catalog_images(dataset_root)
    metadata = load_and_validate_metadata(dataset_root, catalog)
    manifest, split_summary = assign_components_to_splits(
        metadata, catalog.duplicate_content_groups, ratios, args.seed
    )
    integrity = validate_split_integrity(manifest)

    manifest = manifest[
        [
            "image_id",
            "relative_path",
            "lesion_id",
            "dx",
            "split",
            "component_id",
            "duplicate_content_group",
        ]
    ].sort_values(["split", "image_id"])
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    labels = sorted(CLASS_NAMES)
    label_to_index = {label: index for index, label in enumerate(labels)}
    training_counts = (
        manifest.loc[manifest["split"] == "train", "dx"].value_counts().reindex(labels, fill_value=0)
    )
    if (training_counts == 0).any():
        raise RuntimeError(f"Training split has empty classes: {training_counts[training_counts == 0].to_dict()}")
    class_weights = {
        label: float(len(manifest.loc[manifest["split"] == "train"]) / (len(labels) * training_counts[label]))
        for label in labels
    }
    write_json(output_dir / "label_to_index.json", label_to_index)
    write_json(output_dir / "class_weights.json", class_weights)
    write_json(
        output_dir / "preparation_config.json",
        {
            "seed": args.seed,
            "dataset_root": str(dataset_root),
            "metadata_csv": "HAM10000_metadata.csv",
            "split_ratios": ratios,
            "input": {
                "color_mode": "RGB",
                "image_size": [args.image_size, args.image_size],
                "resize": "aspect-ratio-preserving letterbox",
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
            "training_only_augmentation": [
                "horizontal flip (p=0.5)",
                "vertical flip (p=0.5)",
                "rotation (-20 to +20 degrees)",
                "brightness (0.85 to 1.15)",
                "contrast (0.85 to 1.15)",
            ],
            "validation_and_test_augmentation": [],
            "class_imbalance_strategy": "class-weighted cross-entropy using class_weights.json from train only",
            "split_constraint": "same lesion_id and decoded-pixel-identical content remain in one split",
        },
    )
    write_json(
        output_dir / "quality_report.json",
        {
            "images_found": len(catalog.image_paths),
            "metadata_rows": len(metadata),
            "formats": catalog.formats,
            "dimensions": catalog.dimensions,
            "zero_byte_files": catalog.zero_byte_files,
            "unreadable_files": catalog.unreadable_files,
            "duplicate_content_groups": catalog.duplicate_content_groups,
            "duplicate_content_group_count": len(catalog.duplicate_content_groups),
            "split_summary": split_summary,
            "integrity": integrity,
        },
    )
    create_distribution_artifacts(manifest, output_dir)

    loaders = create_dataloaders(
        prepared_dir=output_dir,
        dataset_root=dataset_root,
        batch_size=32,
        seed=args.seed,
        preprocessing=PreprocessingConfig(image_size=args.image_size),
    )
    preprocessing_verification = verify_preprocessed_loaders(loaders)
    write_json(output_dir / "preprocessing_verification.json", preprocessing_verification)
    write_preparation_summary(
        output_dir, manifest, catalog, integrity, preprocessing_verification
    )

    print("Preparation completed without model training.")
    print(json.dumps({"split_summary": split_summary, "integrity": integrity}, indent=2))


if __name__ == "__main__":
    main()
