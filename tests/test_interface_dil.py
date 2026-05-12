from types import SimpleNamespace

import torch

from dilnaz.train.interface.interface_dil import build_auto_mapping, decode_tokens


def test_auto_mapping_allows_single_word_probe():
    assert build_auto_mapping(["gerçekleştirebileceğimizi"], [[1.0]]) == {}


def test_auto_mapping_swaps_two_words():
    assert build_auto_mapping(["araba", "kitap"], [[1.0, 0.3], [0.3, 1.0]]) == {0: 1, 1: 0}


def test_decode_tokens_uses_exact_writer_query_lengths():
    class FakeTokenizer:
        def decode(self, ids):
            return " ".join(str(token_id) for token_id in ids)

    class FakeWriter:
        def __init__(self):
            self.seen_query_lengths = None

        def transition(self, latents, query_surface, surface_state):
            self.seen_query_lengths = query_surface.unit_lengths.detach().cpu().tolist()
            logits = latents.new_zeros((2, query_surface.surface_width, 11))
            logits[0, 0, 7] = 1.0
            logits[0, 1, 8] = 1.0
            logits[0, 2, 10] = 1.0
            logits[1, 0, 9] = 1.0
            logits[1, 1, 10] = 1.0
            return SimpleNamespace(token_logits=logits)

    writer = FakeWriter()
    model = SimpleNamespace(
        config=SimpleNamespace(
            pad_token_id=0,
            surface_bucket_sizes=(4, 8),
            writer_empty_token_id=12,
            writer_stop_token_id=10,
        ),
        writer=writer,
    )

    decoded = decode_tokens(model, FakeTokenizer(), torch.zeros((2, 3)), [2, 1])

    assert writer.seen_query_lengths == [[3], [2]]
    assert decoded == ["7 8", "9"]
