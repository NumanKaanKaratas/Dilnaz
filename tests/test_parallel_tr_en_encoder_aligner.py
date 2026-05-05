import torch

from external.parallel_tr_en_encoder_aligner import ParallelTrEnAligner, TextUnit, UnitAlignment


def test_group_alignments_keeps_adjacent_sources_for_fused_target():
    alignments = [
        UnitAlignment(0, 1, 1.0),
        UnitAlignment(1, 1, 1.0),
        UnitAlignment(2, 2, 0.9991),
        UnitAlignment(3, 3, 1.0),
    ]
    source_teacher = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    target_teacher = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    source_mask = torch.ones(4, dtype=torch.bool)
    target_mask = torch.ones(4, dtype=torch.bool)

    groups = ParallelTrEnAligner.group_alignments(
        None,
        alignments,
        [
            TextUnit("The", 0, 3, "target"),
            TextUnit("lioness's", 4, 13, "target"),
            TextUnit("tooth", 14, 19, "target"),
            TextUnit("broke", 20, 25, "target"),
        ],
        source_teacher,
        target_teacher,
        source_mask,
        target_mask,
    )

    assert [(group.source_indices, group.target_indices) for group in groups] == [
        ((0, 1), (1,)),
        ((2,), (2,)),
        ((3,), (3,)),
    ]


def test_group_alignments_does_not_merge_non_adjacent_sources():
    alignments = [
        UnitAlignment(0, 1, 1.0),
        UnitAlignment(2, 1, 1.0),
    ]
    source_teacher = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3)
    target_teacher = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
    source_mask = torch.ones(3, dtype=torch.bool)
    target_mask = torch.ones(2, dtype=torch.bool)

    groups = ParallelTrEnAligner.group_alignments(
        None,
        alignments,
        [
            TextUnit("left", 0, 4, "target"),
            TextUnit("shared", 5, 11, "target"),
        ],
        source_teacher,
        target_teacher,
        source_mask,
        target_mask,
    )

    assert [(group.source_indices, group.target_indices) for group in groups] == [
        ((0,), (1,)),
        ((2,), (1,)),
    ]


def test_group_alignments_merges_reordered_phrase_and_expands_target_span():
    alignments = [
        UnitAlignment(0, 8, 0.9246),
        UnitAlignment(1, 4, 1.0),
        UnitAlignment(2, 0, 1.0),
        UnitAlignment(3, 1, 1.0),
        UnitAlignment(4, 2, 1.0),
    ]
    target_units = [
        TextUnit("10", 0, 2, "target"),
        TextUnit("minutes", 3, 10, "target"),
        TextUnit("remained", 11, 19, "target"),
        TextUnit("until", 20, 25, "target"),
        TextUnit("the", 26, 29, "target"),
        TextUnit("end", 30, 33, "target"),
        TextUnit("of", 34, 36, "target"),
        TextUnit("the", 37, 40, "target"),
        TextUnit("lesson", 41, 47, "target"),
        TextUnit(".", 47, 48, "target"),
    ]
    source_teacher = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    target_teacher = torch.arange(10 * 2 * 3, dtype=torch.float32).reshape(10, 2, 3)
    source_mask = torch.ones(5, dtype=torch.bool)
    target_mask = torch.ones(10, dtype=torch.bool)

    groups = ParallelTrEnAligner.group_alignments(
        None,
        alignments,
        target_units,
        source_teacher,
        target_teacher,
        source_mask,
        target_mask,
    )

    assert [(group.source_indices, group.target_indices) for group in groups] == [
        ((0, 1), (3, 4, 5, 6, 7, 8)),
        ((2,), (0,)),
        ((3,), (1,)),
        ((4,), (2,)),
    ]
