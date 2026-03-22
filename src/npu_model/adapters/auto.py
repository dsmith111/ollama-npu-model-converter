from __future__ import annotations

from npu_model.core.errors import NpuModelError
from npu_model.core.types import ModelInfo
from npu_model.core.registry import Registry


def select_adapter(registry: Registry, model: ModelInfo) -> str:
    # Prefer adapters that can_handle(model) == True
    candidates = []
    for aid, adapter in registry.adapters.items():
        try:
            if adapter.can_handle(model):
                candidates.append(aid)
        except Exception:
            continue

    if not candidates:
        raise NpuModelError(
            stage="adapter",
            reason_code="NO_ADAPTER",
            message="No adapter found for this model.",
            hint=(
                f"model_type={model.model_type}, architectures={model.architectures}. "
                "Add a new adapter plugin or extend can_handle() rules."
            ),
        )

    # deterministic selection: lexical by id (you can improve later with scoring)
    return sorted(candidates)[0]
