from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from npu_model.core.types import BackendCapabilities, GraphBundle, BackendPreparedBundle, TargetSpec


class Backend(ABC):
    id: str

    @abstractmethod
    def resolve_target(self, target: str, env: dict[str, str]) -> TargetSpec:
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        backend_config: dict[str, Any],
    ) -> BackendPreparedBundle:
        raise NotImplementedError

    def compile(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        compile_config: dict[str, Any],
    ) -> BackendPreparedBundle:
        """Compile graphs into backend-specific artifacts.

        Default implementation falls back to prepare() (copy-based).
        Override in backends that support real compilation.
        """
        return self.prepare(graphs, out_dir, target=target, backend_config=compile_config)

    def detect_environment(self) -> BackendCapabilities:
        """Probe the current environment for backend toolchain / runtime availability."""
        return BackendCapabilities(
            backend_id=self.id,
            compile_available=False,
            runtime_available=False,
            diagnostics=["detect_environment() not implemented for this backend"],
        )
        raise NotImplementedError
