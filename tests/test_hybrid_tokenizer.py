from pathlib import Path

from dilnaz.surface import pack_token_units
from dilnaz.tokenization import HybridTokenizer
from dilnaz.train.data.dil_data import stream_teacher_text_items_with_eos, teacher_distill_segment, trainable_segments
from dilnaz.train.data.naz_data import build_token_cache, stream_token_pieces


def tiny_tokenizer() -> HybridTokenizer:
    return HybridTokenizer(
        char_tokens=[],
        surface_tokens=["araba", "kitaplarımızdan", ",", "car"],
        numeric_tokens=[],
        common_word_tokens=[],
        contextual_tokens={},
    )


def test_trainable_segments_uses_surface_piece_limit():
    tokenizer = tiny_tokenizer()
    segments = trainable_segments(tokenizer, "araba, car", max_surface_pieces_per_unit=8, add_eos=False)
    assert [segment.text for segment in segments if not segment.text.isspace()] == ["araba", ",", "car"]


def test_trainable_segments_can_append_global_eos():
    tokenizer = tiny_tokenizer()
    segments = trainable_segments(tokenizer, "araba", max_surface_pieces_per_unit=8, add_eos=True)
    assert segments[-1].kind == "eos"
    assert segments[-1].token_ids == [tokenizer.eos_token_id]
    assert not teacher_distill_segment(segments[-1])


def test_pack_token_units_keeps_one_semantic_unit_per_token():
    tokenizer = tiny_tokenizer()
    segments = trainable_segments(tokenizer, "araba car", max_surface_pieces_per_unit=8, add_eos=False)
    surface = pack_token_units(
        [[segment.token_ids for segment in segments]],
        pad_token_id=tokenizer.pad_token_id,
        bucket_sizes=(8, 16),
        max_pieces_per_unit=8,
    )
    assert surface.unit_count == 2
    assert surface.unit_mask.tolist() == [[True, True]]


def test_naz_token_cache_is_flat_ids_plus_offsets(tmp_path: Path):
    tokenizer = tiny_tokenizer()
    train_file = tmp_path / "train.txt"
    train_file.write_text("araba car\n", encoding="utf-8")
    ids_path, offsets_path, token_count = build_token_cache(
        train_file,
        tokenizer,
        max_surface_pieces_per_unit=8,
        pad_token_id=tokenizer.pad_token_id,
        read_chars=64,
        cache_dir=tmp_path,
    )
    assert ids_path.name.endswith(".surface_ids.npy")
    assert offsets_path.name.endswith(".surface_offsets.npy")
    assert token_count >= 3


def test_stream_token_pieces_appends_eos(tmp_path: Path):
    tokenizer = tiny_tokenizer()
    train_file = tmp_path / "train.txt"
    train_file.write_text("araba\n", encoding="utf-8")
    pieces = list(stream_token_pieces(train_file, tokenizer, max_surface_pieces_per_unit=8, read_chars=64))
    assert pieces[-1] == [tokenizer.eos_token_id]


def test_dil_and_naz_text_loaders_share_global_eos_units(tmp_path: Path):
    tokenizer = tiny_tokenizer()
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"text": "araba"}\n{"text": "car"}\n', encoding="utf-8")
    dil_eos_count = 0
    for _, text, add_eos in stream_teacher_text_items_with_eos(train_file, read_chars=64):
        segments = trainable_segments(tokenizer, text, max_surface_pieces_per_unit=8, add_eos=add_eos)
        dil_eos_count += sum(segment.token_ids == [tokenizer.eos_token_id] for segment in segments)
    naz_eos_count = sum(
        pieces == [tokenizer.eos_token_id]
        for pieces in stream_token_pieces(train_file, tokenizer, max_surface_pieces_per_unit=8, read_chars=64)
    )
    assert dil_eos_count == 2
    assert naz_eos_count == dil_eos_count


def test_pack_token_units_uses_variable_width_bucket():
    packed = pack_token_units(
        [[[1], [2, 3, 4, 5, 6]]],
        pad_token_id=0,
        bucket_sizes=(8, 16),
        max_pieces_per_unit=8,
    )
    assert packed.surface_width == 8
    assert packed.unit_lengths.tolist() == [[1, 5]]
