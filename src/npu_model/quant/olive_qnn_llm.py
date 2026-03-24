"""Olive-backed QNN LLM quantizer — production route for LLMs on QNN HTP.

Unlike the generic ``qnn-qdq`` quantizer, this path uses Olive's LLM-specific
passes for splitting, static shaping, KV-cache handling, and model-family-aware
optimization.  It is the recommended quantizer for Phi, Phi-3, Llama, and
similar decoder-only transformer families.

**Status: scaffold / not-yet-implemented.**  The quantizer currently raises a
clear error directing users to track the implementation progress.  Once Olive
integration is complete, this will be the default for supported LLM families.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle


@dataclass
class OliveQnnLlmQuantizer:
    """Olive-backed LLM quantizer for QNN HTP deployment.

    Production route for decoder-only LLMs (Phi, Phi-3, Llama, Mistral, …).
    Uses Olive passes: CaptureSplitInfo, StaticLLM, LLMAugmentedDataLoader,
    EPContextBinaryGenerator, etc.

    Requires: ``pip install olive-ai[qnn]``
    """
    id: str = "olive-qnn-llm"
    requires_calibration: bool = False  # Olive handles its own calibration

    # Model families this quantizer is designed for.
    SUPPORTED_FAMILIES = frozenset({
        "phi", "phi3", "llama", "llama2", "llama3", "mistral", "gemma",
    })

    def apply(self, graphs: GraphBundle, *, quant_config: dict[str, Any]) -> GraphBundle:
        # Check Olive availability
        try:
            import olive  # noqa: F401
            _has_olive = True
        except ImportError:
            _has_olive = False

        if not _has_olive:
            raise NpuModelError(
                stage="quant",
                reason_code="OLIVE_NOT_INSTALLED",
                message=(
                    "The olive-qnn-llm quantizer requires Olive (olive-ai) to be installed."
                ),
                hint=(
                    "Install Olive with QNN support:\n"
                    "  pip install olive-ai[qnn]\n\n"
                    "Then re-run with: --quant olive-qnn-llm\n\n"
                    "Alternatively, use the generic (experimental) path:\n"
                    "  --quant qnn-qdq --calib-prompts <prompts.txt>"
                ),
            )

        # Scaffold: Olive is installed but integration is not yet complete.
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_QNN_LLM_NOT_IMPLEMENTED",
            message=(
                "The olive-qnn-llm quantizer is not yet implemented. "
                "This is the planned production path for LLMs on QNN HTP."
            ),
            hint=(
                "Track implementation progress in the project repository.\n\n"
                "In the meantime, the generic qnn-qdq path may work for some "
                "models, but it is experimental for LLMs:\n"
                "  npu-model convert --quant qnn-qdq --calib-prompts <prompts.txt> ...\n\n"
                "Olive passes needed: CaptureSplitInfo, StaticLLM, "
                "LLMAugmentedDataLoader, EPContextBinaryGenerator."
            ),
        )
