from __future__ import annotations

from npu_model.core.registry import Registry


def test_registry_loads() -> None:
    reg = Registry.load()
    assert "qnn" in reg.backends
    assert "ort-genai-folder" in reg.runtime_formats
    assert "passthrough" in reg.quantizers
    assert "phi3" in reg.adapters
    assert "llama" in reg.adapters
