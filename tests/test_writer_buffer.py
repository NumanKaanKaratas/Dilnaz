import torch

from dilnaz.train.interface.writer_buffer import SlidingWriterBuffer


def make_buffer():
    config = type(
        "Config",
        (),
        {
            "writer_sliding_window_size": 4,
            "writer_left_frozen": 1,
            "writer_active_size": 2,
            "writer_right_guard": 1,
            "latent_size": 4,
            "writer_max_position_age": 3,
        },
    )()
    return SlidingWriterBuffer(model=object(), config=config, tokenizer=object())


def test_emission_limit_waits_for_right_guard():
    buffer = make_buffer()
    latent = torch.zeros(1, 4)
    buffer.append(latent, None, False)
    buffer.append(latent, None, False)

    assert buffer._emission_limit(force=False) == 0


def test_emission_limit_releases_active_prefix_when_guard_exists():
    buffer = make_buffer()
    latent = torch.zeros(1, 4)
    for _ in range(3):
        buffer.append(latent, None, False)

    assert buffer._emission_limit(force=False) == 2
    assert buffer._emission_limit(force=True) == 2
