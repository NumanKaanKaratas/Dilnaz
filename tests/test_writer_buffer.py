import torch

from dilnaz.train.interface.writer_buffer import SlidingWriterBuffer


def make_buffer():
    config = type(
        "Config",
        (),
        {
            "writer_commit_threshold": 0.5,
            "writer_sliding_window_size": 4,
            "writer_left_frozen": 1,
            "writer_active_size": 2,
            "writer_right_guard": 1,
            "latent_size": 4,
            "writer_max_position_age": 3,
        },
    )()
    return SlidingWriterBuffer(model=object(), config=config, tokenizer=object())


def test_ready_values_force_commits_all_slots():
    buffer = make_buffer()
    buffer.pending_ages = [0, 0]
    ready = torch.tensor([False, False])

    assert buffer._ready_values(ready, commit_limit=2, force=True) == [True, True]


def test_ready_values_commits_stale_oldest_slot():
    buffer = make_buffer()
    buffer.pending_ages = [3, 0]
    ready = torch.tensor([False, False])

    assert buffer._ready_values(ready, commit_limit=2, force=False) == [True, False]
