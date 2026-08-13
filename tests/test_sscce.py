import numpy as np

from bivss_cd.sscce import combine_changes, instance_level_changes, semantic_spatial_changes


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


def test_v4_global_context_eliminates_id_drift():
    left = np.zeros((8, 8), dtype=np.uint8)
    right = np.zeros((8, 8), dtype=np.uint8)
    left[1:3, 1:3] = 1
    right[5, 5] = 1
    # Other IDs explain both masks globally. v4 therefore treats the dissimilar
    # same-ID pair as identity drift instead of a real change.
    anchor = {
        1: {"mask": left, "box": np.array([1 / 8, 1 / 8, 2 / 8, 2 / 8])},
        2: {"mask": right, "box": np.array([5 / 8, 5 / 8, 2 / 8, 2 / 8])},
    }
    propagated = {
        1: {"mask": right, "box": np.array([5 / 8, 5 / 8, 2 / 8, 2 / 8])},
        2: {"mask": left, "box": np.array([1 / 8, 1 / 8, 2 / 8, 2 / 8])},
    }
    changes = semantic_spatial_changes(anchor, propagated, left.shape, 0.3)
    assert changes == {}


def test_instance_level_change_detection_restores_unmatched_objects():
    stable_t1 = np.zeros((8, 8), dtype=np.uint8)
    stable_t2 = np.zeros((8, 8), dtype=np.uint8)
    stable_t1[1:3, 1:3] = 1
    stable_t2[1:3, 1:3] = 1
    disappeared = np.zeros((8, 8), dtype=np.uint8)
    disappeared[5:7, 5:7] = 1
    result = instance_level_changes(
        {1: item(stable_t1), 2: item(disappeared)},
        {1: item(stable_t2)},
        stable_t1.shape,
        overlap_threshold=0.3,
    )
    np.testing.assert_array_equal(result, disappeared)


def test_instance_level_components_use_eight_connectivity_and_area_filter():
    diagonal = np.zeros((5, 5), dtype=np.uint8)
    diagonal[1, 1] = diagonal[2, 2] = 1
    result = instance_level_changes(
        {1: item(diagonal)}, {}, diagonal.shape, minimum_object_area=2
    )
    np.testing.assert_array_equal(result, diagonal)
