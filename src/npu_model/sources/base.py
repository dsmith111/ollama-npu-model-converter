from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelSource(ABC):
    """Base class for model sources."""

    @abstractmethod
    def materialize(self) -> Path:
        """Return a local directory containing the model files."""
        raise NotImplementedError

    @abstractmethod
    def source_info(self) -> dict[str, Any]:
        """Return metadata about the source for manifest."""
        raise NotImplementedError
