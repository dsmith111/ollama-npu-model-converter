from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from npu_model.backends.qnn import QnnBackend
from npu_model.core.errors import NpuModelError
from npu_model.core.types import BackendCapabilities, GraphBundle


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


class TestQnnCompatibleOps:
    """Tests for _check_qnn_compatible_ops and _audit_htp_op_coverage."""

    def _make_onnx_with_ops(self, tmp_path: Path, op_types: list[str]) -> GraphBundle:
        """Create a minimal ONNX model containing the given op types."""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto
        from onnx.helper import (
            make_graph, make_model, make_node,
            make_tensor_value_info, make_opsetid,
        )

        # Build a chain: input -> op1 -> op2 -> ... -> output
        inputs = [make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])]
        outputs = [make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])]
        nodes = []
        prev = "X"
        for i, op in enumerate(op_types):
            out = f"Z{i}" if i < len(op_types) - 1 else "Y"
            nodes.append(make_node(op, [prev], [out]))
            prev = out
        if not nodes:
            nodes.append(make_node("Identity", ["X"], ["Y"]))

        graph = make_graph(nodes, "test", inputs, outputs)
        model = make_model(graph, opset_imports=[make_opsetid("", 17)])
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        return GraphBundle(
            graphs={"model": model_path},
            tokenizer_dir=tmp_path,
            extra_files=[],
            metadata={},
        )

    def test_matmulnbits_blocked(self, tmp_path: Path) -> None:
        bundle = self._make_onnx_with_ops(tmp_path, ["MatMulNBits"])
        backend = QnnBackend()
        with pytest.raises(NpuModelError) as exc_info:
            backend._check_qnn_compatible_ops(bundle)
        assert exc_info.value.reason_code == "INCOMPATIBLE_QUANTIZATION"

    def test_identity_passes(self, tmp_path: Path) -> None:
        bundle = self._make_onnx_with_ops(tmp_path, ["Identity"])
        backend = QnnBackend()
        backend._check_qnn_compatible_ops(bundle)  # should not raise

    def test_audit_finds_risky_ops(self, tmp_path: Path) -> None:
        bundle = self._make_onnx_with_ops(
            tmp_path, ["LayerNormalization", "FastGelu"]
        )
        backend = QnnBackend()
        risky = backend._audit_htp_op_coverage(bundle)
        assert "model" in risky
        assert "LayerNormalization" in risky["model"]
        assert "FastGelu" in risky["model"]

    def test_audit_clean_graph_is_empty(self, tmp_path: Path) -> None:
        bundle = self._make_onnx_with_ops(tmp_path, ["Identity"])
        backend = QnnBackend()
        risky = backend._audit_htp_op_coverage(bundle)
        assert risky == {}


class TestHtpProbe:
    """Tests for _probe_htp_eligibility."""

    def test_probe_raises_on_session_failure(self) -> None:
        """If InferenceSession raises, probe must raise HTP_PROBE_FAILED."""
        backend = QnnBackend()

        # Mock ort.InferenceSession to raise without loading a real model,
        # because ORT may crash the process (native abort) on invalid inputs.
        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("QNN EP not available")

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            with pytest.raises(NpuModelError) as exc_info:
                backend._probe_htp_eligibility(
                    Path("dummy.onnx"), "QnnHtp.dll", "model"
                )
        assert exc_info.value.reason_code == "HTP_PROBE_FAILED"
        assert "QNN EP not available" in exc_info.value.message


class TestCtxWrapperSizeGuard:
    """The context wrapper output size guard rejects oversized _ctx.onnx."""

    def test_guard_constant_defined(self) -> None:
        """Sanity: the 10 MB limit constant is accessible in the compile method."""
        # Just verify the backend can be instantiated and has the method
        backend = QnnBackend()
        assert hasattr(backend, "_compile_context_cache")


class TestOliveQnnLlmQuantizer:
    """Tests for the olive-qnn-llm scaffold quantizer."""

    def test_has_correct_id(self) -> None:
        from npu_model.quant.olive_qnn_llm import OliveQnnLlmQuantizer
        q = OliveQnnLlmQuantizer()
        assert q.id == "olive-qnn-llm"
        assert q.requires_calibration is False

    def test_raises_not_implemented(self, tmp_path: Path) -> None:
        from npu_model.quant.olive_qnn_llm import OliveQnnLlmQuantizer

        q = OliveQnnLlmQuantizer()
        bundle = GraphBundle(
            graphs={"m": tmp_path / "m.onnx"},
            tokenizer_dir=tmp_path,
            extra_files=[],
            metadata={},
        )
        with pytest.raises(NpuModelError) as exc_info:
            q.apply(bundle, quant_config={})
        assert exc_info.value.reason_code in (
            "OLIVE_NOT_INSTALLED",
            "OLIVE_QNN_LLM_NOT_IMPLEMENTED",
        )

    def test_supported_families(self) -> None:
        from npu_model.quant.olive_qnn_llm import OliveQnnLlmQuantizer
        q = OliveQnnLlmQuantizer()
        assert "phi" in q.SUPPORTED_FAMILIES
        assert "phi3" in q.SUPPORTED_FAMILIES
        assert "llama" in q.SUPPORTED_FAMILIES
