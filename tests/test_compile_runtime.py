import torch

from dilnaz.train.common import runtime


def test_compile_forward_uses_static_inductor_contract(monkeypatch):
    captured = {}

    def fake_compile(forward, **kwargs):
        captured.update(kwargs)
        return forward

    monkeypatch.setattr(runtime.torch, "compile", fake_compile)
    monkeypatch.delenv(runtime.COMPILE_FULLGRAPH_ENV, raising=False)

    def forward(x):
        return x

    compiled = runtime.compile_forward(forward, "max-autotune-no-cudagraphs", "Probe")

    assert compiled is forward
    assert captured == {
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": False,
        "mode": "max-autotune-no-cudagraphs",
    }
    assert torch._dynamo.config.capture_scalar_outputs


def test_compile_forward_fullgraph_probe_env(monkeypatch):
    captured = {}

    def fake_compile(forward, **kwargs):
        captured.update(kwargs)
        return forward

    monkeypatch.setattr(runtime.torch, "compile", fake_compile)
    monkeypatch.setenv(runtime.COMPILE_FULLGRAPH_ENV, "1")

    runtime.compile_forward(lambda x: x, "default", "Probe")

    assert captured["fullgraph"] is True
    assert "mode" not in captured
