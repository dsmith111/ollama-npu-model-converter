"""Olive-backed QNN LLM quantizer.

This is the supported route for decoder-only LLM families on QNN HTP.
It generates a family-specific Olive config, runs Olive externally, then maps
produced artifacts back into the internal GraphBundle contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle
from npu_model.olive.artifacts import collect_olive_outputs
from npu_model.olive.config_builder import (
    SUPPORTED_OLIVE_LLM_FAMILIES,
    build_olive_config,
)
from npu_model.olive.runner import run_olive_cli


@dataclass
class OliveQnnLlmQuantizer:
    """Olive-backed LLM quantizer for QNN HTP deployment.

    Production route for decoder-only LLMs (Phi / Phi-3 / Llama).
    Requires: ``pip install olive-ai[auto-opt]``
    """

    id: str = "olive-qnn-llm"
    requires_calibration: bool = False

    SUPPORTED_FAMILIES = SUPPORTED_OLIVE_LLM_FAMILIES

    def apply(self, graphs: GraphBundle, *, quant_config: dict[str, Any]) -> GraphBundle:
        try:
            import olive  # noqa: F401
            _has_olive = True
        except ImportError:
            _has_olive = False

        if not _has_olive:
            raise NpuModelError(
                stage="quant",
                reason_code="OLIVE_NOT_INSTALLED",
                message="The olive-qnn-llm quantizer requires Olive (olive-ai) to be installed.",
                hint=(
                    "Install Olive with QNN support:\n"
                    "  pip install olive-ai[auto-opt]\n\n"
                    "Then re-run with: --quant olive-qnn-llm\n\n"
                    "Alternatively, use the generic (experimental) path:\n"
                    "  --quant qnn-qdq --calib-prompts <prompts.txt>"
                ),
            )

        if not graphs.graphs:
            raise NpuModelError(
                stage="quant",
                reason_code="OLIVE_INPUT_EMPTY",
                message="No input ONNX graphs were provided for olive-qnn-llm.",
            )

        first_graph = next(iter(graphs.graphs.values()))
        work_dir = Path(first_graph).parent / "_olive_qnn_llm"

        plan = build_olive_config(graphs=graphs, quant_config=quant_config, work_dir=work_dir)
        run_olive_cli(
            config_path=plan.config_path,
            work_dir=work_dir,
            timeout_s=int(quant_config.get("olive_timeout_s", 14_400)),
        )
        return collect_olive_outputs(
            olive_output_dir=plan.output_dir,
            fallback_tokenizer_dir=graphs.tokenizer_dir,
            fallback_extra_files=graphs.extra_files,
            family=plan.family,
        )
