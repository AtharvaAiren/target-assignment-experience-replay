"""Focused integrity tests for Hugging Face miniImageNet conversion."""

from __future__ import annotations

import pytest

from experiments.gradient_reuse_mechanisms import miniimagenet_hf as conversion


def test_hugging_face_path_recovers_wordnet_and_numeric_source_index() -> None:
    assert conversion._parse_source_path(
        "n03954731_53652_n03954731.JPEG"
    ) == ("n03954731", 53652)


@pytest.mark.parametrize(
    "value",
    [
        "n03954731_53652_n03954732.JPEG",
        "n03954731_53652.JPEG",
        "n03954731_notanindex_n03954731.JPEG",
        "n03954731_53652_n03954731.jpg",
        None,
    ],
)
def test_hugging_face_path_rejects_ambiguous_or_changed_names(value) -> None:
    with pytest.raises(conversion.HuggingFaceMiniImageNetError):
        conversion._parse_source_path(value)


def test_canonical_filename_encodes_one_based_sorted_source_rank() -> None:
    match = conversion.CSV_PATTERN.fullmatch("n0153282900000724.jpg")
    assert match is not None
    assert match.group("label") == "n01532829"
    assert int(match.group("selection")[-4:]) == 724
