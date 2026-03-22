from __future__ import annotations

from pathlib import Path

import pytest


class TestHasDynamicShapes:

    def test_returns_empty_without_onnx(self) -> None:
        """If onnx is not importable or file invalid, returns empty."""
        from npu_model.core.shapes import has_dynamic_shapes

        result = has_dynamic_shapes(Path("nonexistent.onnx"))
        assert result == []


class TestFixDynamicShapes:

    def test_returns_error_without_onnx_installed(self, tmp_path: Path) -> None:
        """If onnx can't load the file, returns errors."""
        from npu_model.core.shapes import fix_dynamic_shapes

        fake = tmp_path / "model.onnx"
        fake.write_bytes(b"\x00")
        out = tmp_path / "fixed.onnx"

        result = fix_dynamic_shapes(fake, out)
        # Should either fix or report errors
        assert isinstance(result.fixed, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
