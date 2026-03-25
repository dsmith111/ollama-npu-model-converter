from __future__ import annotations

import json
from pathlib import Path

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle
from npu_model.olive import config_builder


def test_render_template_preserves_windows_backslashes(tmp_path: Path) -> None:
    tpl = tmp_path / "tpl.j2"
    tpl.write_text('{"model_path": {{ input_model_path }}}', encoding="utf-8")

    windows_path = r"C:\Users\dsmith111\models\phi2\model.onnx"
    rendered = config_builder._render_template(
        tpl,
        {"input_model_path": json.dumps(windows_path)},
    )
    parsed = json.loads(rendered)
    assert parsed["model_path"] == windows_path


def test_build_olive_config_writes_invalid_render_for_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"\x00" * 16)
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()
    bundle = GraphBundle(
        graphs={"decoder": graph},
        tokenizer_dir=tok_dir,
        extra_files=[],
        metadata={},
    )

    monkeypatch.setattr(config_builder, "_render_template", lambda *_args, **_kwargs: "{bad json")

    with pytest.raises(NpuModelError) as exc_info:
        config_builder.build_olive_config(
            graphs=bundle,
            quant_config={"model_family": "phi"},
            work_dir=tmp_path / "work",
        )
    assert exc_info.value.reason_code == "OLIVE_TEMPLATE_RENDER_FAILED"
    assert exc_info.value.hint is not None
    assert "Rendered config was written to:" in exc_info.value.hint

    bad_file = tmp_path / "work" / "olive_phi_qnn_config.rendered.invalid.json"
    assert bad_file.exists()
    assert bad_file.read_text(encoding="utf-8") == "{bad json"

