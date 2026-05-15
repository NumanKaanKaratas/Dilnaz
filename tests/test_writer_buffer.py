import torch

from dilnaz.train.interface.writer_buffer import UnitWriterBuffer


def make_buffer():
    config = type(
        "Config",
        (),
        {
            "latent_size": 4,
        },
    )()
    return UnitWriterBuffer(model=object(), config=config, tokenizer=object(), microbatch_size=2)


def test_emission_limit_emits_immediately():
    buffer = make_buffer()
    latent = torch.zeros(1, 4)
    buffer.append(latent, False)

    assert buffer._emission_limit(force=False) == 1


def test_emission_limit_caps_microbatch_size():
    buffer = make_buffer()
    latent = torch.zeros(1, 4)
    for _ in range(3):
        buffer.append(latent, False)

    assert buffer._emission_limit(force=False) == 2
    assert buffer._emission_limit(force=True) == 2
