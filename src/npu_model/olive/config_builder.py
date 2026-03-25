from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle


SUPPORTED_OLIVE_LLM_FAMILIES: frozenset[str] = frozenset({"phi", "phi3", "llama"})

_TEMPLATE_BY_FAMILY: dict[str, str] = {
    "phi": "phi2_qnn.json.j2",
    "phi3": "phi_qnn.json.j2",
    "llama": "llama_qnn.json.j2",
}


@dataclass(frozen=True)
class OliveConfigPlan:
    family: str
    template_path: Path
    config_path: Path
    output_dir: Path
    primary_graph: Path


def _normalize_family(raw: str | None) -> str:
    text = (raw or "").strip().lower()
    if text.startswith("phi3"):
        return "phi3"
    if text.startswith("phi"):
        return "phi"
    if "llama" in text:
        return "llama"
    return text


def detect_supported_family(
    *,
    graphs: GraphBundle,
    quant_config: dict[str, Any],
) -> str:
    """Resolve model family and enforce initial Olive support scope."""
    hints = [
        quant_config.get("model_family"),
        graphs.metadata.get("model_family"),
        quant_config.get("model_type"),
        graphs.metadata.get("model_type"),
        quant_config.get("input_spec"),
    ]
    family = ""
    for hint in hints:
        family = _normalize_family(str(hint) if hint is not None else "")
        if family:
            break

    if family not in SUPPORTED_OLIVE_LLM_FAMILIES:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_MODEL_FAMILY_UNSUPPORTED",
            message=(
                "olive-qnn-llm currently supports only model families "
                f"{sorted(SUPPORTED_OLIVE_LLM_FAMILIES)}; got '{family or 'unknown'}'."
            ),
            hint=(
                "For unsupported families, use the experimental path:\n"
                "  --quant qnn-qdq --calib-prompts <prompts.txt>"
            ),
        )
    return family


def _render_template(template_path: Path, values: dict[str, Any]) -> str:
    text = template_path.read_text(encoding="utf-8")
    for key, value in values.items():
        token_pattern = re.compile(r"{{\s*" + re.escape(key) + r"\s*}}")
        replacement = str(value)
        text = token_pattern.sub(lambda _m, r=replacement: r, text)
    return text


def build_olive_config(
    *,
    graphs: GraphBundle,
    quant_config: dict[str, Any],
    work_dir: Path,
) -> OliveConfigPlan:
    """Create a family-specific Olive config file from templates."""
    family = detect_supported_family(graphs=graphs, quant_config=quant_config)

    if not graphs.graphs:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_INPUT_EMPTY",
            message="No ONNX graphs were provided to olive-qnn-llm.",
        )

    primary_graph = next(iter(graphs.graphs.values()))
    template_name = _TEMPLATE_BY_FAMILY[family]
    template_path = Path(__file__).resolve().parent / "templates" / template_name
    if not template_path.exists():
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_TEMPLATE_MISSING",
            message=f"Olive template not found: {template_path}",
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "olive_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "olive_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / f"olive_{family}_qnn_config.json"

    model_source = quant_config.get("olive_model_source") or str(primary_graph)
    template_values = {
        "family": family,
        "input_model_path": json.dumps(str(model_source)),
        "output_dir": json.dumps(str(output_dir)),
        "cache_dir": json.dumps(str(cache_dir)),
        "target_system": json.dumps("qnn_system"),
    }
    config_text = _render_template(template_path, template_values)
    try:
        parsed = json.loads(config_text)
    except Exception as e:
        bad_path = work_dir / f"olive_{family}_qnn_config.rendered.invalid.json"
        bad_path.write_text(config_text, encoding="utf-8")
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_TEMPLATE_RENDER_FAILED",
            message=f"Failed to render valid Olive config from template: {template_path}",
            hint=(
                f"Rendered config was written to: {bad_path}\n"
                f"JSON parse error: {type(e).__name__}: {e}"
            ),
            cause=e,
        ) from e
    config_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    return OliveConfigPlan(
        family=family,
        template_path=template_path,
        config_path=config_path,
        output_dir=output_dir,
        primary_graph=primary_graph,
    )
