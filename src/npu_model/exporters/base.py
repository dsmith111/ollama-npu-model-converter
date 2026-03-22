from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from npu_model.core.types import GraphBundle, ModelInfo


class Exporter(ABC):
    """Base class for model exporters (HF → ONNX + genai_config + tokenizer)."""

    id: str

    @abstractmethod
    def can_export(self, model: ModelInfo) -> bool:
        """Return True if this exporter supports the given model."""
        raise NotImplementedError

    @abstractmethod
    def export(
        self,
        model_dir: Path,
        out_dir: Path,
        model: ModelInfo,
        *,
        export_config: dict[str, Any],
    ) -> GraphBundle:
        """Export a model from HF/local weights to ONNX + tokenizer + config.

        Returns a GraphBundle pointing at the exported artifacts.
        """
        raise NotImplementedError

    def check_dependencies(self) -> list[str]:
        """Return a list of missing dependencies, or empty if all present."""
        return []
