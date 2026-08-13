"""Semantic-Spatial Cross-Comparison and Elimination (SSCCE v4).

This module is a NumPy port of the `merge_masks_v4` path used by the paper's
`baseline_bi_ssccev4` experiments. Its branch order is intentionally kept close
to the research implementation because small changes alter the final masks.
"""

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from .masks import binary_mask


ObjectMap = Mapping[int, Mapping[str, Any]]


def _v4_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Match the research implementation, including 0 IoU for two empty masks."""
    a, b = mask_a > 0, mask_b > 0
    true_positive = np.logical_and(a, b).sum()
    false_positive = np.logical_and(a, ~b).sum()
    false_negative = np.logical_and(~a, b).sum()
    return float(true_positive / (true_positive + false_positive + false_negative + 1e-7))


def _union_mask(objects: ObjectMap, shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=np.uint8)
    for item in objects.values():
        if item is not None and item.get("mask") is not None:
            union = np.maximum(union, binary_mask(item["mask"], shape))
    return union


def _absolute_box(item: Mapping[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
    box = item.get("box")
    if box is None:
        mask = binary_mask(item["mask"], image_shape)
        ys, xs = np.where(mask)
        if not len(xs):
            return np.asarray([0, 0, 1, 1], dtype=float)
        return np.asarray(
            [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1],
            dtype=float,
        )
    value = np.asarray(box, dtype=float).reshape(-1)
    if value.size != 4:
        raise ValueError(f"SAM3 box must contain four xywh values, got {value}")
    # SAM3 returns relative xywh boxes. The original v4 implementation scales
    # all coordinates by image width; preserve that behavior for reproduction.
    if np.max(np.abs(value)) <= 1.0:
        value = value * image_shape[1]
    return value


def _clip_box(box: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x, y, box_width, box_height = (int(value) for value in box)
    x, y = max(0, min(x, width)), max(0, min(y, height))
    box_width = max(0, min(box_width, width - x))
    box_height = max(0, min(box_height, height - y))
    return x, y, box_width, box_height


def _roi_iou(mask_a: np.ndarray, mask_b: np.ndarray, box: np.ndarray) -> float:
    x, y, width, height = _clip_box(box, mask_a.shape)
    return _v4_iou(mask_a[y : y + height, x : x + width], mask_b[y : y + height, x : x + width])


def _centered_box_crop_iou(
    mask_a: np.ndarray,
    box_a: np.ndarray,
    mask_b: np.ndarray,
    box_b: np.ndarray,
) -> float:
    ax, ay, aw, ah = _clip_box(box_a, mask_a.shape)
    bx, by, bw, bh = _clip_box(box_b, mask_b.shape)
    crop_a, crop_b = mask_a[ay : ay + ah, ax : ax + aw], mask_b[by : by + bh, bx : bx + bw]
    max_height, max_width = max(crop_a.shape[0], crop_b.shape[0]), max(crop_a.shape[1], crop_b.shape[1])
    if max_height == 0 or max_width == 0:
        return 0.0

    def center(crop: np.ndarray) -> np.ndarray:
        canvas = np.zeros((max_height, max_width), dtype=np.uint8)
        y, x = (max_height - crop.shape[0]) // 2, (max_width - crop.shape[1]) // 2
        canvas[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
        return canvas

    return _v4_iou(center(crop_a), center(crop_b))


def semantic_spatial_changes(
    anchor: ObjectMap,
    propagated: ObjectMap,
    image_shape: tuple[int, int],
    iou_threshold: float,
) -> dict[int, np.ndarray]:
    """Return the object masks retained by the paper's SSCCE v4 rule."""
    if not propagated:
        return {
            object_id: binary_mask(item["mask"], image_shape)
            for object_id, item in anchor.items()
            if item is not None and item.get("mask") is not None
        }
    if not anchor:
        return {
            object_id: binary_mask(item["mask"], image_shape)
            for object_id, item in propagated.items()
            if item is not None and item.get("mask") is not None
        }

    anchor_union = _union_mask(anchor, image_shape)
    propagated_union = _union_mask(propagated, image_shape)
    retained: dict[int, np.ndarray] = {}

    for object_id in set(anchor) | set(propagated):
        anchor_item, propagated_item = anchor.get(object_id), propagated.get(object_id)
        if propagated_item is None:
            mask = binary_mask(anchor_item["mask"], image_shape)
            box = _absolute_box(anchor_item, image_shape)
            if _roi_iou(mask, propagated_union, box) <= iou_threshold:
                retained[object_id] = mask
            continue
        if anchor_item is None:
            mask = binary_mask(propagated_item["mask"], image_shape)
            box = _absolute_box(propagated_item, image_shape)
            if _roi_iou(mask, anchor_union, box) <= iou_threshold:
                retained[object_id] = mask
            continue

        anchor_mask = binary_mask(anchor_item["mask"], image_shape)
        propagated_mask = binary_mask(propagated_item["mask"], image_shape)
        anchor_box = _absolute_box(anchor_item, image_shape)
        propagated_box = _absolute_box(propagated_item, image_shape)
        anchor_to_global = _roi_iou(anchor_mask, propagated_union, anchor_box)
        propagated_to_global = _roi_iou(propagated_mask, anchor_union, propagated_box)
        morphology_iou = _centered_box_crop_iou(
            anchor_mask, anchor_box, propagated_mask, propagated_box
        )

        if (
            morphology_iou <= iou_threshold
            and anchor_to_global > iou_threshold
            and propagated_to_global > iou_threshold
        ):
            continue
        if morphology_iou <= iou_threshold:
            retained[object_id] = np.logical_xor(anchor_mask, propagated_mask).astype(np.uint8)
            continue
        if anchor_to_global <= iou_threshold or propagated_to_global <= iou_threshold:
            residual = np.zeros(image_shape, dtype=np.uint8)
            if anchor_to_global <= iou_threshold:
                x, y, width, height = _clip_box(anchor_box, image_shape)
                context = np.zeros(image_shape, dtype=np.uint8)
                context[y : y + height, x : x + width] = propagated_union[y : y + height, x : x + width]
                residual += np.logical_xor(anchor_mask, context).astype(np.uint8)
            if propagated_to_global <= iou_threshold:
                x, y, width, height = _clip_box(propagated_box, image_shape)
                context = np.zeros(image_shape, dtype=np.uint8)
                context[y : y + height, x : x + width] = anchor_union[y : y + height, x : x + width]
                residual += np.logical_xor(propagated_mask, context).astype(np.uint8)
            retained[object_id] = (residual > 0).astype(np.uint8)

    return retained


def combine_changes(changes: Mapping[int, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for mask in changes.values():
        result = np.maximum(result, binary_mask(mask, shape))
    return result


def _instances(mask: np.ndarray) -> list[np.ndarray]:
    """Match ``skimage.measure.label(..., connectivity=2)`` without a dependency."""
    value = binary_mask(mask)
    height, width = value.shape
    seen = np.zeros_like(value, dtype=bool)
    result: list[np.ndarray] = []
    for y in range(height):
        for x in range(width):
            if not value[y, x] or seen[y, x]:
                continue
            component = np.zeros_like(value)
            queue = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                component[cy, cx] = 1
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if value[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            result.append(component)
    return result


def instance_level_changes(
    anchor: ObjectMap,
    propagated: ObjectMap,
    image_shape: tuple[int, int],
    overlap_threshold: float = 0.30,
    minimum_object_area: int = 0,
) -> np.ndarray:
    """Port the instance-level branch enabled by default in the verified baseline."""
    first = _instances(_union_mask(anchor, image_shape))
    second = _instances(_union_mask(propagated, image_shape))
    changed = np.zeros(image_shape, dtype=np.uint8)

    def add_unmatched(source: list[np.ndarray], target: list[np.ndarray]) -> None:
        nonlocal changed
        for instance in source:
            area = int(instance.sum())
            if area == 0 or area < minimum_object_area:
                continue
            matched = any(
                int(np.logical_and(instance, candidate).sum()) / area >= overlap_threshold
                for candidate in target
                if np.logical_and(instance, candidate).any()
            )
            if not matched:
                changed = np.maximum(changed, instance)

    add_unmatched(first, second)
    add_unmatched(second, first)
    return changed


def filter_change_instances(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Filter 8-connected change components like the verified implementation."""
    value = binary_mask(mask)
    if minimum_area <= 0:
        return value
    result = np.zeros_like(value)
    for instance in _instances(value):
        if int(instance.sum()) >= minimum_area:
            result = np.maximum(result, instance)
    return result
