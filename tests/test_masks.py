import numpy as np
import pytest

from bivss_cd.masks import consensus_fusion, filter_small_components, mask_iou


def test_mask_iou_handles_overlap_and_empty_masks():
    a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    b = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    assert mask_iou(a, b) == pytest.approx(1 / 3)
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0


def test_consensus_supports_intersection_and_union():
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[1, 1], [0, 0]])
    np.testing.assert_array_equal(consensus_fusion(a, b), a)
    np.testing.assert_array_equal(consensus_fusion(a, b, "union"), b)


def test_filter_small_components():
    mask = np.array([[1, 0, 1], [0, 0, 1], [0, 0, 0]])
    expected = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(filter_small_components(mask, 2), expected)


def test_invalid_consensus_mode():
    with pytest.raises(ValueError):
        consensus_fusion(np.zeros((1, 1)), np.zeros((1, 1)), "invalid")
