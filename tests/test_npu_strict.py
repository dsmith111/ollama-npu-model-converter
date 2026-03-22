from __future__ import annotations

import json
from pathlib import Path

from npu_model.validate.npu_strict import validate_npu_strict, NpuValidationResult


class TestNpuStrictValidator:

    def _make_bundle(self, tmp_path: Path, *, with_onnx: bool = True,
                     with_ctx: bool = False, with_bins: bool = False,
                     with_data: bool = False, with_genai: bool = True) -> Path:
        d = tmp_path / "bundle"
        d.mkdir()
        if with_onnx:
            (d / "model.onnx").write_bytes(b"\x00" * 16)
        if with_ctx:
            (d / "model_ctx.onnx").write_bytes(b"\x00" * 16)
        if with_bins:
            (d / "model_qnn.bin").write_bytes(b"\x00" * 32)
        if with_data:
            (d / "model.onnx.data").write_bytes(b"\x00" * 100)
        if with_genai:
            cfg = {"model": {"decoder": {"session_options": {}}}, "search": {}}
            (d / "genai_config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
        return d

    def test_no_onnx_files_fails(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_onnx=False)
        result = validate_npu_strict(d)
        assert not result.passed

    def test_basic_onnx_present(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path)
        result = validate_npu_strict(d)
        # Should have some checks
        assert len(result.checks) > 0
        # Onnx present check should pass
        onnx_check = next(c for c in result.checks if c.name == "onnx_present")
        assert onnx_check.status == "OK"

    def test_warns_no_ctx_graphs(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_bins=True)
        result = validate_npu_strict(d)
        ctx_check = next(c for c in result.checks if c.name == "context_cache")
        assert ctx_check.status == "FAIL"

    def test_ctx_graphs_ok(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_ctx=True, with_bins=True)
        result = validate_npu_strict(d)
        ctx_check = next(c for c in result.checks if c.name == "context_cache")
        assert ctx_check.status == "OK"

    def test_warns_data_files_present(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_data=True)
        result = validate_npu_strict(d)
        data_check = next(c for c in result.checks if c.name == "external_data")
        assert data_check.status == "FAIL"

    def test_no_data_files_ok(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path)
        result = validate_npu_strict(d)
        data_check = next(c for c in result.checks if c.name == "external_data")
        assert data_check.status == "OK"

    def test_result_properties(self, tmp_path: Path) -> None:
        d = self._make_bundle(tmp_path, with_data=True)
        result = validate_npu_strict(d)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
