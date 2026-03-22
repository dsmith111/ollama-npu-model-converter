from __future__ import annotations

from npu_model.backends.qnn import QnnBackend
from npu_model.core.types import BackendCapabilities


def test_qnn_detect_environment() -> None:
    backend = QnnBackend()
    caps = backend.detect_environment()
    assert isinstance(caps, BackendCapabilities)
    assert caps.backend_id == "qnn"
    # We're in a dev env without QNN, so compile should not be available
    assert isinstance(caps.compile_available, bool)
    assert isinstance(caps.diagnostics, list)
    assert len(caps.diagnostics) > 0


def test_qnn_resolve_target() -> None:
    backend = QnnBackend()
    ts = backend.resolve_target("auto", env={})
    assert ts.backend_id == "qnn"
    assert ts.name == "auto"
    assert "backend_type" in ts.params


def test_qnn_resolve_target_from_env() -> None:
    backend = QnnBackend()
    ts = backend.resolve_target("auto", env={
        "NPU_QNN_BACKEND_TYPE": "cpu",
        "NPU_QNN_BACKEND_PATH": "QnnCpu.dll",
    })
    assert ts.params["backend_type"] == "cpu"
    assert ts.params["backend_path"] == "QnnCpu.dll"


def test_qnn_compile_passthrough_falls_back(tmp_path) -> None:
    """compile with strategy=passthrough should behave like prepare."""
    from npu_model.core.types import GraphBundle

    backend = QnnBackend()
    ts = backend.resolve_target("auto", env={})

    # Create a minimal graph bundle
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    onnx = graph_dir / "model.onnx"
    onnx.write_bytes(b"\x00")

    bundle = GraphBundle(
        graphs={"model": onnx},
        tokenizer_dir=tmp_path,
        extra_files=[],
        metadata={},
    )

    out = tmp_path / "compiled"
    result = backend.compile(bundle, out, target=ts, compile_config={"strategy": "passthrough"})
    assert result.backend_metadata["compile_strategy"] == "passthrough"
    assert result.artifacts_dir.exists()
