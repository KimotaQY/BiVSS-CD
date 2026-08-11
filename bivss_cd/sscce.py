from collections.abc import Mapping
from typing import Any

import numpy as np

from .masks import binary_mask, crop_iou, mask_iou, union_mask


def semantic_spatial_changes(
    anchor: Mapping[int, Mapping[str, Any]],
    propagated: Mapping[int, Mapping[str, Any]],
    image_shape: tuple[int, int],
    iou_threshold: float,
) -> dict[int, np.ndarray]:
    """Apply SSCCE to same-ID masks and their global semantic context.

    Same-ID morphology is compared after centered cropping, while global unions
    verify whether an apparent difference is already explained by another object.
    """
    anchor_union = union_mask(anchor, image_shape)
    propagated_union = union_mask(propagated, image_shape)
    changed: dict[int, np.ndarray] = {}
    for object_id in sorted(set(anchor) | set(propagated)):
        left, right = anchor.get(object_id), propagated.get(object_id)
        if left is None or right is None:
            item = left or right
            assert item is not None
            mask = binary_mask(item["mask"], image_shape)
            context = propagated_union if left is not None else anchor_union
            if mask_iou(mask, np.logical_and(mask, context)) <= iou_threshold:
                changed[object_id] = mask
            continue

        left_mask = binary_mask(left["mask"], image_shape)
        right_mask = binary_mask(right["mask"], image_shape)
        morphology_iou = crop_iou(left_mask, right_mask)
        left_context_iou = mask_iou(left_mask, np.logical_and(left_mask, propagated_union))
        right_context_iou = mask_iou(right_mask, np.logical_and(right_mask, anchor_union))
        if morphology_iou <= iou_threshold:
            residual = np.logical_xor(left_mask, right_mask).astype(np.uint8)
            if residual.any():
                changed[object_id] = residual
        elif left_context_iou <= iou_threshold or right_context_iou <= iou_threshold:
            residual = np.logical_or(
                np.logical_and(left_mask, np.logical_not(propagated_union)),
                np.logical_and(right_mask, np.logical_not(anchor_union)),
            ).astype(np.uint8)
            if residual.any():
                changed[object_id] = residual
    return changed


def combine_changes(changes: Mapping[int, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for mask in changes.values():
        result |= binary_mask(mask, shape)
    return result
