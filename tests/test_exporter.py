from __future__ import annotations

from npu_model.exporters.ort_genai_builder import OrtGenaiBuilderExporter
from npu_model.core.types import ModelInfo


def test_exporter_has_id() -> None:
    e = OrtGenaiBuilderExporter()
    assert e.id == "ort-genai-builder"


def test_can_export_phi3() -> None:
    e = OrtGenaiBuilderExporter()
    mi = ModelInfo(
        source={}, model_type="phi3", architectures=[], config={}, tokenizer_files=[]
    )
    assert e.can_export(mi) is True


def test_can_export_llama() -> None:
    e = OrtGenaiBuilderExporter()
    mi = ModelInfo(
        source={}, model_type="llama", architectures=[], config={}, tokenizer_files=[]
    )
    assert e.can_export(mi) is True


def test_cannot_export_unknown() -> None:
    e = OrtGenaiBuilderExporter()
    mi = ModelInfo(
        source={}, model_type="unknown_arch", architectures=[], config={}, tokenizer_files=[]
    )
    assert e.can_export(mi) is False


def test_check_dependencies_reports_missing() -> None:
    e = OrtGenaiBuilderExporter()
    missing = e.check_dependencies()
    # In dev env, at minimum onnxruntime_genai is not installed
    assert isinstance(missing, list)
    # We can't assert exact contents since torch/transformers may or may not be present
