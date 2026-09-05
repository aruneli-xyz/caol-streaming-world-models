import pytest

from measure_decoder_rf import (
    contiguous_spans,
    expected_first_pixel_frame,
    expected_native_pixel_support,
    parse_index_spec,
    summarize_temporal_support,
)


def test_expected_first_support_uses_gamma_four_t_minus_three():
    assert expected_first_pixel_frame(0) == 0
    assert expected_first_pixel_frame(1) == 1
    assert expected_first_pixel_frame(2) == 5
    assert expected_first_pixel_frame(24) == 93


def test_expected_native_support_has_one_bootstrap_then_four_frames():
    assert expected_native_pixel_support(0) == {
        "start": 0,
        "end": 0,
        "count": 1,
    }
    assert expected_native_pixel_support(1) == {
        "start": 1,
        "end": 4,
        "count": 4,
    }
    assert expected_native_pixel_support(24) == {
        "start": 93,
        "end": 96,
        "count": 4,
    }


def test_expected_native_support_clips_at_output_length():
    assert expected_native_pixel_support(3, total_pixel_frames=11) == {
        "start": 9,
        "end": 10,
        "count": 2,
    }


def test_contiguous_spans_deduplicates_and_sorts():
    assert contiguous_spans([7, 2, 3, 3, 8, 10]) == [
        [2, 3],
        [7, 8],
        [10, 10],
    ]
    assert contiguous_spans([]) == []


def test_support_summary_separates_lookahead_native_and_tail():
    changed = [False] * 15
    for frame in (4, 5, 9, 10, 11, 12, 14):
        changed[frame] = True

    summary = summarize_temporal_support(changed, latent_index=3)

    assert summary["expected_native_support"] == {
        "start": 9,
        "end": 12,
        "count": 4,
    }
    assert summary["earliest_changed_frame"] == 4
    assert summary["latest_changed_frame"] == 14
    assert summary["lookahead_frames"] == 5
    assert summary["early_changed_frame_count"] == 2
    assert summary["early_changed_frame_spans"] == [[4, 5]]
    assert summary["late_changed_frame_count"] == 1
    assert summary["late_changed_frame_spans"] == [[14, 14]]
    assert summary["changed_frame_spans"] == [[4, 5], [9, 12], [14, 14]]


def test_support_summary_handles_no_effect():
    summary = summarize_temporal_support(
        [False] * 13,
        latent_index=3,
    )
    assert summary["changed_frame_count"] == 0
    assert summary["earliest_changed_frame"] is None
    assert summary["lookahead_frames"] is None
    assert summary["early_changed_frame_count"] == 0
    assert summary["late_changed_frame_count"] == 0


def test_index_spec_bookkeeping():
    differing = [4, 7, 9]
    assert parse_index_spec(
        "none", latent_count=12, differing_indices=differing
    ) == []
    assert parse_index_spec(
        "first-differing", latent_count=12, differing_indices=differing
    ) == [4]
    assert parse_index_spec(
        "differing", latent_count=12, differing_indices=differing
    ) == differing
    assert parse_index_spec(
        "2,4-6,5", latent_count=12, differing_indices=differing
    ) == [2, 4, 5, 6]


def test_index_spec_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="descending"):
        parse_index_spec("5-3", latent_count=8, differing_indices=[])
    with pytest.raises(ValueError, match="outside"):
        parse_index_spec("8", latent_count=8, differing_indices=[])
