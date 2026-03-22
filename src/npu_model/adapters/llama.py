from __future__ import annotations

from pathlib import Path
from typing import Any

from npu_model.adapters.base import ModelAdapter
from npu_model.core.types import GraphBundle, ModelInfo


class LlamaAdapter(ModelAdapter):
    id = "llama"

    def can_handle(self, model: ModelInfo) -> bool:
        mt = (model.model_type or "").lower()
        if "llama" in mt:
            return True
        arch = " ".join(model.architectures).lower()
        return "llama" in arch

    def export(self, model_dir: Path, out_dir: Path, *, export_config: dict[str, Any]) -> GraphBundle:
        mode = export_config.get("mode", "export")

        if mode == "prebuilt-ort-genai":
            return self.import_prebuilt(model_dir, out_dir)

        # Export mode: delegate to ORT GenAI builder
        from npu_model.exporters.ort_genai_builder import OrtGenaiBuilderExporter

        exporter = OrtGenaiBuilderExporter()
        model = export_config.get("model_info") or ModelInfo(
            source={"type": "local", "path": str(model_dir)},
            model_type="llama",
            architectures=[],
            config={},
            tokenizer_files=[],
        )
        return exporter.export(model_dir, out_dir, model, export_config=export_config)
