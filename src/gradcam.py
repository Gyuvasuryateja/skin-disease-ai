"""Grad-CAM generation for the supported MobileNetV2 skin-lesion classifier.

The implementation uses the same normalized model input as inference. It does
not alter model weights or train the model. Generated maps visualize model
influence only; they do not establish a medical finding or diagnosis.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Final

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models.mobilenetv2 import MobileNetV2

from ham10000_data import PreprocessingConfig, normalize_image


DEFAULT_OVERLAY_OPACITY: Final[float] = 0.45


class GradCAMError(RuntimeError):
    """Raised when a Grad-CAM visualization cannot be computed safely."""


class UnsupportedGradCAMModelError(GradCAMError):
    """Raised when the loaded architecture has no supported CAM target layer."""


@dataclass(frozen=True)
class GradCAMResult:
    """Predicted-class Grad-CAM values aligned to the square model input."""

    probabilities: np.ndarray
    predicted_index: int
    cam: np.ndarray


def resolve_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the final spatial MobileNetV2 feature block used for Grad-CAM.

    MobileNetV2 performs global pooling immediately after ``features[-1]``.
    This final Conv-BN-ReLU block therefore retains the last spatial activation
    map while being directly relevant to the classifier logits.
    """
    if not isinstance(model, MobileNetV2):
        raise UnsupportedGradCAMModelError(
            "Grad-CAM is currently supported only for the deployed MobileNetV2 architecture."
        )
    if not hasattr(model, "features") or len(model.features) == 0:
        raise UnsupportedGradCAMModelError(
            "The MobileNetV2 model has no feature block available for Grad-CAM."
        )
    target_layer = model.features[-1]
    if not any(isinstance(module, nn.Conv2d) for module in target_layer.modules()):
        raise UnsupportedGradCAMModelError(
            "The final MobileNetV2 feature block does not contain a convolutional layer."
        )
    return target_layer


def generate_predicted_class_gradcam(
    model: nn.Module,
    normalized_image: np.ndarray,
    target_layer: nn.Module | None = None,
) -> GradCAMResult:
    """Run inference and Grad-CAM for the model's own highest-logit class.

    ``normalized_image`` must be the same C×H×W float32 representation used by
    prediction. Input gradients are enabled solely to obtain the Grad-CAM
    weights; no parameter updates, optimizer steps, or training occur.
    """
    if normalized_image.ndim != 3:
        raise GradCAMError(
            f"Expected one normalized C×H×W image for Grad-CAM; received shape {normalized_image.shape}."
        )
    if not np.isfinite(normalized_image).all():
        raise GradCAMError("Cannot generate Grad-CAM from non-finite image values.")

    layer = target_layer or resolve_gradcam_target_layer(model)
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def capture_activations(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal activations, gradients
        if not isinstance(output, torch.Tensor):
            raise GradCAMError("The configured Grad-CAM target did not return a tensor.")
        activations = output

        def capture_gradients(gradient: torch.Tensor) -> None:
            nonlocal gradients
            gradients = gradient

        output.register_hook(capture_gradients)

    handle = layer.register_forward_hook(capture_activations)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        image_tensor = torch.from_numpy(np.expand_dims(normalized_image, axis=0)).to(
            dtype=torch.float32
        )
        # The trained MobileNetV2 feature extractor is frozen. Requiring an input
        # gradient keeps the feature activations in the autograd graph for CAM.
        image_tensor.requires_grad_(True)
        logits = model(image_tensor)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < 1:
            raise GradCAMError(f"Model returned invalid logits with shape {tuple(logits.shape)}.")
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.isclose(
            float(probabilities.sum()), 1.0, atol=1e-5
        ):
            raise GradCAMError("Model returned invalid class probabilities.")
        predicted_index = int(probabilities.argmax())
        logits[0, predicted_index].backward()
        if activations is None or gradients is None:
            raise GradCAMError(
                "Grad-CAM could not capture feature activations and gradients from the target layer."
            )
        if activations.ndim != 4 or gradients.shape != activations.shape:
            raise GradCAMError("Grad-CAM target activations and gradients are incompatible.")

        channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((channel_weights * activations).sum(dim=1)).squeeze(0)
        cam_min = float(cam.min().detach().item())
        cam_max = float(cam.max().detach().item())
        if not np.isfinite(cam_min) or not np.isfinite(cam_max):
            raise GradCAMError("Grad-CAM produced non-finite values.")
        # After ReLU, cam_min >= 0. The map is meaningful when ReLU keeps at
        # least one positive value and the retained values are not all equal.
        # Absolute magnitudes are model dependent, so uniformity is judged
        # relative to the map's own extrema rather than by a fixed epsilon.
        if cam_max <= 0.0 or cam_max <= cam_min:
            raise GradCAMError("Grad-CAM produced a uniform map and cannot provide a meaningful visualization.")
        normalized_cam = ((cam - cam_min) / (cam_max - cam_min)).detach().cpu().numpy()
        return GradCAMResult(
            probabilities=probabilities.astype(np.float32, copy=False),
            predicted_index=predicted_index,
            cam=normalized_cam.astype(np.float32, copy=False),
        )
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def generate_gradcam_for_image(
    model: nn.Module,
    image: Image.Image,
    preprocessing: PreprocessingConfig,
    target_layer: nn.Module | None = None,
) -> GradCAMResult:
    """Apply the established inference preprocessing and compute Grad-CAM."""
    normalized_image = normalize_image(image, preprocessing)
    return generate_predicted_class_gradcam(model, normalized_image, target_layer)


def map_cam_to_original_image(
    cam: np.ndarray,
    original_size: tuple[int, int],
    preprocessing: PreprocessingConfig,
) -> Image.Image:
    """Remove inference letterbox padding and align the CAM to the source image."""
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise GradCAMError("Grad-CAM map must be a finite two-dimensional array.")
    original_width, original_height = original_size
    if original_width < 1 or original_height < 1:
        raise GradCAMError("The original image has invalid dimensions.")

    image_size = preprocessing.image_size
    cam_image = Image.fromarray(
        np.rint(np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
    ).resize((image_size, image_size), Image.Resampling.BILINEAR)

    scale = min(image_size / original_width, image_size / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    cropped_cam = cam_image.crop((left, top, left + resized_width, top + resized_height))
    return cropped_cam.resize((original_width, original_height), Image.Resampling.BILINEAR)


def downscale_for_response(image: Image.Image, maximum_dimension: int) -> Image.Image:
    """Bound inline API image payloads without changing model preprocessing."""
    if maximum_dimension < 1:
        raise ValueError("maximum_dimension must be positive.")
    display_image = image.convert("RGB").copy()
    display_image.thumbnail(
        (maximum_dimension, maximum_dimension), Image.Resampling.LANCZOS
    )
    return display_image


def colorize_heatmap(cam_image: Image.Image) -> Image.Image:
    """Map a grayscale CAM to a perceptually clear blue-to-red heatmap."""
    values = np.asarray(cam_image.convert("L"), dtype=np.float32) / 255.0
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    colored = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.rint(colored * 255.0).astype(np.uint8), mode="RGB")


def overlay_heatmap(
    original_image: Image.Image,
    original_aligned_cam: Image.Image,
    opacity: float = DEFAULT_OVERLAY_OPACITY,
) -> Image.Image:
    """Blend the Grad-CAM heatmap over the supplied source image."""
    if not 0.0 < opacity < 1.0:
        raise ValueError("Grad-CAM overlay opacity must be between zero and one.")
    original = original_image.convert("RGB")
    cam = original_aligned_cam.resize(original.size, Image.Resampling.BILINEAR)
    return Image.blend(original, colorize_heatmap(cam), opacity)


def image_to_jpeg_data_url(image: Image.Image, quality: int = 90) -> str:
    """Encode an in-memory display image without persisting an uploaded image."""
    if not 1 <= quality <= 95:
        raise ValueError("JPEG quality must be between 1 and 95.")
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
