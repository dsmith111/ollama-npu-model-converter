from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from npu_model.core.types import GraphBundle


@dataclass
class PassthroughQuantizer:
    """No-op quantizer — model is used as-is.

    If the exporter already produced a quantized model (e.g. INT4), that
    quantization is preserved but no additional transformation is applied.
    For QNN HTP context-cache compilation, use --quant qnn-qdq instead.
    """
    id: str = "passthrough"

    def apply(self, graphs: GraphBundle, *, quant_config: dict[str, Any]) -> GraphBundle:
        return graphs
