from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from npu_model.core.types import GraphBundle


class QuantizationStrategy(ABC):
    id: str

    @abstractmethod
    def apply(self, graphs: GraphBundle, *, quant_config: dict[str, Any]) -> GraphBundle:
        raise NotImplementedError
