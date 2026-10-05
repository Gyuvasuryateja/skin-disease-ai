"""Reusable, framework-independent data loading for prepared HAM10000 splits.

The loader returns float32 NumPy arrays in channels-first (N, C, H, W) layout so
it can be passed to a future PyTorch, TensorFlow, or custom training adapter.
No model training is defined in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class PreprocessingConfig:
    """Image settings compatible with common ImageNet-pretrained backbones."""

    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    padding_color: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class Batch:
    """A batch of normalized images and integer class labels."""

    images: np.ndarray
    labels: np.ndarray
    image_ids: tuple[str, ...]


def resize_and_pad(image: Image.Image, config: PreprocessingConfig) -> Image.Image:
    """Letterbox an image to a square while preserving its original aspect ratio."""
    image = image.convert("RGB")
    width, height = image.size
    scale = min(config.image_size / width, config.image_size / height)
    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (config.image_size, config.image_size), config.padding_color)
    offset = (
        (config.image_size - resized.width) // 2,
        (config.image_size - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    return canvas


def apply_training_augmentation(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply mild, dermoscopy-appropriate random augmentation to a training image only."""
    image = image.convert("RGB")
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.5:
        image = ImageOps.flip(image)

    angle = float(rng.uniform(-20.0, 20.0))
    image = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
    return image


def normalize_image(image: Image.Image, config: PreprocessingConfig) -> np.ndarray:
    """Letterbox, scale to [0, 1], normalize, and return C×H×W float32 data."""
    image = resize_and_pad(image, config)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(config.mean, dtype=np.float32)) / np.asarray(
        config.std, dtype=np.float32
    )
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1)), dtype=np.float32)


class HAM10000BatchLoader:
    """Iterate a single prepared split as repeatable NumPy mini-batches."""

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: str,
        class_to_index: dict[str, int],
        batch_size: int = 32,
        shuffle: bool = False,
        augment: bool = False,
        seed: int = 42,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if augment and split != "train":
            raise ValueError("Random augmentation is permitted only for the training split.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one.")

        manifest = pd.read_csv(manifest_path)
        self.records = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.records.empty:
            raise ValueError(f"No records found for split '{split}'.")
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.class_to_index = class_to_index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.preprocessing = preprocessing or PreprocessingConfig()
        self._epoch = 0

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        indices = np.arange(len(self.records))
        if self.shuffle:
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            selected = indices[start : start + self.batch_size]
            batch_records = self.records.iloc[selected]
            images: list[np.ndarray] = []
            labels: list[int] = []
            image_ids: list[str] = []
            for record in batch_records.itertuples(index=False):
                image_path = self.dataset_root / record.relative_path
                with Image.open(image_path) as image:
                    image.load()
                    if self.augment:
                        image = apply_training_augmentation(image, rng)
                    images.append(normalize_image(image, self.preprocessing))
                labels.append(self.class_to_index[record.dx])
                image_ids.append(record.image_id)

            yield Batch(
                images=np.stack(images).astype(np.float32, copy=False),
                labels=np.asarray(labels, dtype=np.int64),
                image_ids=tuple(image_ids),
            )


def load_class_mapping(prepared_dir: str | Path) -> dict[str, int]:
    """Read the class-to-index mapping generated during preparation."""
    path = Path(prepared_dir) / "label_to_index.json"
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {str(label): int(index) for label, index in mapping.items()}


def load_preprocessing_config(prepared_dir: str | Path) -> PreprocessingConfig:
    """Restore the exact preprocessing settings saved during dataset preparation."""
    path = Path(prepared_dir) / "preparation_config.json"
    with path.open(encoding="utf-8") as handle:
        saved_config = json.load(handle)
    input_config = saved_config["input"]
    image_size = input_config["image_size"]
    if len(image_size) != 2 or image_size[0] != image_size[1]:
        raise ValueError("Saved preprocessing configuration must define a square input size.")
    normalization = input_config["normalization"]
    return PreprocessingConfig(
        image_size=int(image_size[0]),
        mean=tuple(float(value) for value in normalization["mean"]),
        std=tuple(float(value) for value in normalization["std"]),
    )


def create_dataloaders(
    prepared_dir: str | Path,
    dataset_root: str | Path,
    batch_size: int = 32,
    seed: int = 42,
    preprocessing: PreprocessingConfig | None = None,
) -> dict[str, HAM10000BatchLoader]:
    """Create train/validation/test loaders with augmentation isolated to training."""
    prepared_dir = Path(prepared_dir)
    manifest_path = prepared_dir / "split_manifest.csv"
    class_to_index = load_class_mapping(prepared_dir)
    config = preprocessing or load_preprocessing_config(prepared_dir)
    return {
        "train": HAM10000BatchLoader(
            manifest_path,
            dataset_root,
            "train",
            class_to_index,
            batch_size=batch_size,
            shuffle=True,
            augment=True,
            seed=seed,
            preprocessing=config,
        ),
        "val": HAM10000BatchLoader(
            manifest_path,
            dataset_root,
            "val",
            class_to_index,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            seed=seed,
            preprocessing=config,
        ),
        "test": HAM10000BatchLoader(
            manifest_path,
            dataset_root,
            "test",
            class_to_index,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
            seed=seed,
            preprocessing=config,
        ),
    }


def verify_preprocessed_loaders(loaders: dict[str, HAM10000BatchLoader]) -> dict[str, dict[str, object]]:
    """Load every prepared image and validate normalized batch shape and finite values."""
    result: dict[str, dict[str, object]] = {}
    for split, loader in loaders.items():
        image_count = 0
        batch_count = 0
        observed_min = float("inf")
        observed_max = float("-inf")
        for batch in loader:
            expected_shape = (len(batch.labels), 3, loader.preprocessing.image_size, loader.preprocessing.image_size)
            if batch.images.shape != expected_shape:
                raise ValueError(
                    f"Unexpected preprocessed shape for {split}: {batch.images.shape}; expected {expected_shape}."
                )
            if not np.isfinite(batch.images).all():
                raise ValueError(f"Non-finite values encountered in {split} preprocessing.")
            image_count += len(batch.labels)
            batch_count += 1
            observed_min = min(observed_min, float(batch.images.min()))
            observed_max = max(observed_max, float(batch.images.max()))
        result[split] = {
            "images_loaded": image_count,
            "batches_loaded": batch_count,
            "shape": [3, loader.preprocessing.image_size, loader.preprocessing.image_size],
            "value_min": observed_min,
            "value_max": observed_max,
            "random_augmentation": loader.augment,
        }
    return result
