import numpy as np

from cleandiffuser.utils.remembercolor_diagnostics import (
    EpisodeBranchTracker,
    array_comparison,
    infer_primary_color_branch,
)


def test_primary_color_branch_uses_visible_cube_pixels():
    for branch in range(3):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[2:6, 2:6, branch] = 255
        assert infer_primary_color_branch(image) == branch


def test_array_comparison_reports_exact_and_error():
    reference = np.array([0, 2], dtype=np.uint8)
    assert array_comparison(reference, reference)["exact"] is True
    result = array_comparison(reference, np.array([0, 4], dtype=np.uint8))
    assert result["exact"] is False
    assert result["mae"] == 1.0
    assert result["max_abs"] == 2.0


def test_branch_tracker_excludes_cue_and_records_success_decomposition():
    tracker = EpisodeBranchTracker(np.array([1, 2]), n_branches=3,
                                   decision_start=10)
    tracker.update(
        elapsed_steps=np.array([[5], [5]]),
        branch_distances=np.array([[[0.01, 1.0, 1.0]],
                                   [[1.0, 0.01, 1.0]]]),
        is_static=np.array([[True], [True]]),
        is_placed=np.array([[False], [False]]),
        success=np.array([[False], [False]]),
    )
    tracker.update(
        elapsed_steps=np.array([[10, 11], [10, 11]]),
        branch_distances=np.array([
            [[0.4, 0.3, 0.5], [0.4, 0.04, 0.5]],
            [[0.2, 0.5, 0.4], [0.03, 0.5, 0.35]],
        ]),
        is_static=np.array([[False, True], [True, True]]),
        is_placed=np.array([[False, True], [False, False]]),
        success=np.array([[False, True], [False, False]]),
    )
    result = tracker.finalize()
    np.testing.assert_array_equal(result["branch_choice"], [1, 0])
    np.testing.assert_array_equal(result["branch_choice_correct"], [True, False])
    np.testing.assert_array_equal(result["success_time"], [11, -1])
    np.testing.assert_allclose(result["min_correct_target_distance"], [0.04, 0.35])
    np.testing.assert_array_equal(result["static_at_min_correct"], [True, True])
