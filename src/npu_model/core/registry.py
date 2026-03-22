from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from npu_model.core.errors import NpuModelError


def _instantiate(obj: Any) -> Any:
    # entry point can be a class or a factory or an instance
    if isinstance(obj, type):
        return obj()
    if callable(obj):
        try:
            return obj()
        except TypeError:
            return obj
    return obj


@dataclass(frozen=True)
class Registry:
    adapters: dict[str, Any]
    backends: dict[str, Any]
    runtime_formats: dict[str, Any]
    quantizers: dict[str, Any]

    @staticmethod
    def load() -> "Registry":
        return Registry(
            adapters=_load_group("npu_model.adapters"),
            backends=_load_group("npu_model.backends"),
            runtime_formats=_load_group("npu_model.runtime_formats"),
            quantizers=_load_group("npu_model.quantizers"),
        )


def _load_group(group: str) -> dict[str, Any]:
    eps = entry_points().select(group=group)
    out: dict[str, Any] = {}
    for ep in eps:
        try:
            obj = ep.load()
            inst = _instantiate(obj)
            pid = getattr(inst, "id", None)
            if not pid:
                raise NpuModelError(
                    stage="registry",
                    reason_code="PLUGIN_NO_ID",
                    message=f"Plugin '{ep.name}' in group '{group}' has no .id",
                    hint="Ensure the plugin class defines id: str",
                )
            out[str(pid)] = inst
        except NpuModelError:
            raise
        except Exception as e:
            raise NpuModelError(
                stage="registry",
                reason_code="PLUGIN_LOAD_FAILED",
                message=f"Failed loading plugin '{ep.name}' from '{group}'",
                hint="Fix plugin import errors or entry points.",
                cause=e,
            ) from e
    return out
