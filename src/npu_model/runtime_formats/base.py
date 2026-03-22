from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from npu_model.core.types import BackendPreparedBundle


class RuntimeFormat(ABC):
    id: str

    @abstractmethod
    def assemble(
        self,
        prepared: BackendPreparedBundle,
        tokenizer_dir: Path,
        out_dir: Path,
        *,
        format_config: dict[str, Any],
    ) -> Path:
        raise NotImplementedError

    @abstractmethod
    def validate_layout(self, bundle_dir: Path) -> None:
        raise NotImplementedError
