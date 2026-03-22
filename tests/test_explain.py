from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from npu_model.core.registry import Registry
from npu_model.core.pipeline import explain_plan
from npu_model.core.errors import NpuModelError


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def phi3_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "phi3_model"
    d.mkdir()
    shutil.copy2(FIXTURES / "hf_phi3_config.json", d / "config.json")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "model.onnx").write_bytes(b"\x00")
    return d


@pytest.fixture()
def registry() -> Registry:
    return Registry.load()


def test_explain_phi3(phi3_model_dir: Path, registry: Registry) -> None:
    plan = explain_plan(
        input_spec=str(phi3_model_dir),
        backend_id="qnn",
        target="auto",
        runtime_format_id="ort-genai-folder",
        cache_dir=None,
        registry=registry,
        mode="prebuilt-ort-genai",
    )
    assert plan.adapter_id == "phi3"
    assert plan.backend_id == "qnn"
    assert plan.target_name == "auto"
    assert plan.runtime_format_id == "ort-genai-folder"
    assert plan.convert_mode == "prebuilt-ort-genai"


def test_explain_unknown_backend(phi3_model_dir: Path, registry: Registry) -> None:
    with pytest.raises(NpuModelError) as exc_info:
        explain_plan(
            input_spec=str(phi3_model_dir),
            backend_id="nonexistent",
            target="auto",
            runtime_format_id="ort-genai-folder",
            cache_dir=None,
            registry=registry,
        )
    assert exc_info.value.reason_code == "UNKNOWN_BACKEND"
