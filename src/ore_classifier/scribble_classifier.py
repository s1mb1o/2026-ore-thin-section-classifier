from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ScribblePixelClassifier:
    class_ids: tuple[int, ...]
    centroids: np.ndarray
    scale: np.ndarray
    feature_names: tuple[str, ...]
    scales: tuple[int, ...]

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        features, _ = extract_pixel_features(image, scales=self.scales)
        h, w, feature_count = features.shape
        flat = features.reshape(-1, feature_count)
        distances = []
        for centroid in self.centroids:
            normalized = (flat - centroid[None, :]) / self.scale[None, :]
            distances.append((normalized * normalized).sum(axis=1))
        logits = -0.5 * np.stack(distances, axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
        return probs.reshape(h, w, len(self.class_ids)).transpose(2, 0, 1).astype(np.float32)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(image)
        index = probs.argmax(axis=0)
        class_ids = np.asarray(self.class_ids, dtype=np.uint8)
        return class_ids[index]


def fit_scribble_pixel_classifier(
    image: np.ndarray,
    scribble_labels: np.ndarray,
    scales: tuple[int, ...] = (3, 7),
) -> ScribblePixelClassifier:
    features, feature_names = extract_pixel_features(image, scales=scales)
    if scribble_labels.shape != features.shape[:2]:
        raise ValueError("scribble_labels must match image spatial shape")
    labels = scribble_labels.astype(np.int32)
    class_ids = tuple(int(item) for item in sorted(np.unique(labels[labels > 0]).tolist()))
    if len(class_ids) < 2:
        raise ValueError("at least two labeled classes are required")

    flat_features = features.reshape(-1, features.shape[2])
    flat_labels = labels.reshape(-1)
    centroids = []
    labeled_values = []
    for class_id in class_ids:
        class_features = flat_features[flat_labels == class_id]
        if class_features.size == 0:
            raise ValueError(f"class {class_id} has no scribble pixels")
        centroids.append(class_features.mean(axis=0))
        labeled_values.append(class_features)
    labeled = np.concatenate(labeled_values, axis=0)
    scale = labeled.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return ScribblePixelClassifier(
        class_ids=class_ids,
        centroids=np.stack(centroids, axis=0).astype(np.float32),
        scale=scale,
        feature_names=feature_names,
        scales=scales,
    )


def extract_pixel_features(image: np.ndarray, scales: tuple[int, ...] = (3, 7)) -> tuple[np.ndarray, tuple[str, ...]]:
    array = image.astype(np.float32)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("image must be HxW or HxWx3")
    if array.max() > 1.0:
        array = array / 255.0
    rgb = array[..., :3]
    gray = rgb.mean(axis=2)
    channels: list[np.ndarray] = [rgb[..., 0], rgb[..., 1], rgb[..., 2], gray]
    names = ["r", "g", "b", "gray"]
    for scale in scales:
        kernel = max(3, int(scale) | 1)
        blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        mean_sq = cv2.GaussianBlur(gray * gray, (kernel, kernel), 0)
        local_std = np.sqrt(np.maximum(mean_sq - blur * blur, 0.0))
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        channels.extend([blur, local_std, grad])
        names.extend([f"gray_blur_{kernel}", f"gray_std_{kernel}", f"gray_grad_{kernel}"])
    features = np.stack(channels, axis=2).astype(np.float32)
    return features, tuple(names)
