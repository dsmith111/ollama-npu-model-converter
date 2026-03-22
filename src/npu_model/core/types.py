from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ConvertMode(str, Enum):
    """How the adapter should produce graphs."""
    EXPORT = "export"                      # HF → ONNX via exporter
    PREBUILT = "prebuilt-ort-genai"        # copy existing ORT GenAI folder


@dataclass(frozen=True)
class ModelInfo:
    source: dict[str, Any]
    model_type: str | None
    architectures: list[str]
    config: dict[str, Any]
    tokenizer_files: list[str]


@dataclass
class GraphBundle:
    graphs: dict[str, Path]          # name -> onnx path
    tokenizer_dir: Path              # folder containing tokenizer assets
    extra_files: list[Path]          # context binaries, configs, etc.
    metadata: dict[str, Any]         # dims, vocab, etc (adapter-provided)


@dataclass(frozen=True)
class TargetSpec:
    backend_id: str
    name: str
    params: dict[str, Any]           # backend-opaque
    schema_version: int = 1

    def normalized_repr(self) -> str:
        """Human-readable target representation for explain / diagnostics."""
        parts = [f"{self.backend_id}:{self.name}"]
        for k, v in sorted(self.params.items()):
            parts.append(f"  {k}={v}")
        return "\n".join(parts)


@dataclass
class BackendPreparedBundle:
    graphs: dict[str, Path]
    artifacts_dir: Path
    backend_metadata: dict[str, Any]


@dataclass
class BackendCapabilities:
    """Reported by Backend.detect_environment()."""
    backend_id: str
    compile_available: bool
    runtime_available: bool
    toolchain_info: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExplainPlan:
    input_spec: str
    materialized_dir: Path
    model_type: str | None
    architectures: list[str]
    adapter_id: str
    backend_id: str
    target_name: str
    runtime_format_id: str
    quantizer_id: str
    convert_mode: str = "export"

    def to_rich_text(self) -> str:
        arch = ", ".join(self.architectures) if self.architectures else "(none)"
        return (
            "[bold]Plan[/bold]\n"
            f"  input: {self.input_spec}\n"
            f"  materialized: {self.materialized_dir}\n"
            f"  model_type: {self.model_type}\n"
            f"  architectures: {arch}\n"
            f"  adapter: {self.adapter_id}\n"
            f"  backend: {self.backend_id}\n"
            f"  target: {self.target_name}\n"
            f"  runtime: {self.runtime_format_id}\n"
            f"  quantizer: {self.quantizer_id}\n"
            f"  mode: {self.convert_mode}\n"
        )


@dataclass(frozen=True)
class ConvertResult:
    bundle_dir: Path
    manifest_path: Path
    pack_dir: Path | None = None
