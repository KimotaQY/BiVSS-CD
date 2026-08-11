import numpy as np

from bivss_cd.sscce import combine_changes, semantic_spatial_changes


def item(mask):
    return {"mask": np.asarray(mask, dtype=np.uint8), "box": None}


def test_same_id_unchanged_object_is_eliminated():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    assert semantic_spatial_changes({1: item(mask)}, {1: item(mask)}, mask.shape, 0.3) == {}


def test_new_object_is_retained():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    changes = semantic_spatial_changes({1: item(mask)}, {}, mask.shape, 0.3)
    np.testing.assert_array_equal(combine_changes(changes, mask.shape), mask)


def test_shape_change_produces_symmetric_difference():
    first = np.zeros((8, 8), dtype=np.uint8)
    second = np.zeros((8, 8), dtype=np.uint8)
    first[2:6, 2:6] = 1
    second[3:5, 3:5] = 1
    changes = semantic_spatial_changes({1: item(first)}, {1: item(second)}, first.shape, 0.5)
    expected = np.logical_xor(first, second).astype(np.uint8)
    np.testing.assert_array_equal(changes[1], expected)
