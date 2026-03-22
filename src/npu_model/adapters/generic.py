from __future__ import annotations

from pathlib import Path
from typing import Any

from npu_model.adapters.base import ModelAdapter
from npu_model.core.types import GraphBundle, ModelInfo


class GenericAdapter(ModelAdapter):
    """Best-effort fallback adapter. Uses prebuilt import for any model."""

    id = "generic"

    def can_handle(self, model: ModelInfo) -> bool:
        # Generic fallback: always returns False by default.
        # Enable via explicit configuration or extend this logic.
        return False

    def export(self, model_dir: Path, out_dir: Path, *, export_config: dict[str, Any]) -> GraphBundle:
        # Generic adapter always uses prebuilt import
        return self.import_prebuilt(model_dir, out_dir)
