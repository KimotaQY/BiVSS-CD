from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np


def binary_mask(mask: Any, shape: tuple[int, int] | None = None) -> np.ndarray:
    value = np.asarray(mask)
    value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {value.shape}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"mask shape {value.shape} does not match image shape {shape}")
    return (value > 0).astype(np.uint8)


def mask_iou(mask_a: Any, mask_b: Any) -> float:
    a = binary_mask(mask_a)
    b = binary_mask(mask_b, a.shape)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def union_mask(objects: Mapping[int, Mapping[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for item in objects.values():
        if item is not None and item.get("mask") is not None:
            result |= binary_mask(item["mask"], shape)
    return result


def bbox_from_mask(mask: Any) -> tuple[int, int, int, int]:
    value = binary_mask(mask)
    ys, xs = np.where(value)
    if not len(xs):
        return (0, 0, 1, 1)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def crop_iou(mask_a: Any, mask_b: Any) -> float:
    a, b = binary_mask(mask_a), binary_mask(mask_b)
    ax, ay, aw, ah = bbox_from_mask(a)
    bx, by, bw, bh = bbox_from_mask(b)
    ca, cb = a[ay : ay + ah, ax : ax + aw], b[by : by + bh, bx : bx + bw]
    height, width = max(ca.shape[0], cb.shape[0]), max(ca.shape[1], cb.shape[1])

    def centered(value: np.ndarray) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.uint8)
        y, x = (height - value.shape[0]) // 2, (width - value.shape[1]) // 2
        canvas[y : y + value.shape[0], x : x + value.shape[1]] = value
        return canvas

    return mask_iou(centered(ca), centered(cb))


def filter_small_components(mask: Any, minimum_area: int) -> np.ndarray:
    value = binary_mask(mask)
    if minimum_area <= 1:
        return value
    height, width = value.shape
    seen = np.zeros_like(value, dtype=bool)
    result = np.zeros_like(value)
    for y in range(height):
        for x in range(width):
            if not value[y, x] or seen[y, x]:
                continue
            queue, component = deque([(y, x)]), []
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and value[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) >= minimum_area:
                ys, xs = zip(*component)
                result[np.asarray(ys), np.asarray(xs)] = 1
    return result


def consensus_fusion(forward: Any, backward: Any, mode: str = "intersection") -> np.ndarray:
    a, b = binary_mask(forward), binary_mask(backward)
    if a.shape != b.shape:
        raise ValueError("forward and backward masks must have the same shape")
    if mode == "intersection":
        return np.logical_and(a, b).astype(np.uint8)
    if mode == "union":
        return np.logical_or(a, b).astype(np.uint8)
    raise ValueError("mode must be 'intersection' or 'union'")
