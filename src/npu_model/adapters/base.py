from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from npu_model.core.types import GraphBundle, ModelInfo


class ModelAdapter(ABC):
    id: str

    @abstractmethod
    def can_handle(self, model: ModelInfo) -> bool:
        raise NotImplementedError

    @abstractmethod
    def export(self, model_dir: Path, out_dir: Path, *, export_config: dict[str, Any]) -> GraphBundle:
        """Export a model to ONNX graphs + tokenizer + extras.

        In 'export' mode: calls an exporter to produce ONNX from HF weights.
        In 'prebuilt' mode: imports an already-converted ORT GenAI directory.
        """
        raise NotImplementedError

    def import_prebuilt(self, model_dir: Path, out_dir: Path) -> GraphBundle:
        """Import a prebuilt ORT GenAI directory (copy onnx + tokenizer + extras).

        Default implementation provides the copy-based import that all adapters
        shared previously. Override only if your adapter needs special handling.
        """
        return _import_prebuilt_ort_genai(model_dir, out_dir, adapter_id=self.id)


def _import_prebuilt_ort_genai(model_dir: Path, out_dir: Path, *, adapter_id: str) -> GraphBundle:
    """Shared implementation: copy a prebuilt ORT GenAI directory into a GraphBundle."""
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)

    # ONNX graphs
    graph_dir = out_dir / "graphs"
    graph_dir.mkdir(exist_ok=True)
    graphs: dict[str, Path] = {}
    for p in sorted(model_dir.rglob("*.onnx")):
        dst = graph_dir / p.name
        shutil.copy2(p, dst)
        graphs[p.stem] = dst

    # Tokenizer assets
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    _TOKENIZER_FILES = [
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    ]
    for fname in _TOKENIZER_FILES:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, tok_dir / src.name)

    # Extra files (configs, context binaries, etc.)
    extra_dir = out_dir / "extra"
    extra_dir.mkdir(exist_ok=True)
    extra_files: list[Path] = []
    _EXTRA_PATTERNS = [
        "genai_config.json", "generation_config.json",
        "inference_model.json", "hash_record_sha256.json",
        "context_*", "*.bin", "*.md",
    ]
    for pat in _EXTRA_PATTERNS:
        for src in model_dir.glob(pat):
            if src.is_file():
                dst = extra_dir / src.name
                shutil.copy2(src, dst)
                extra_files.append(dst)

    return GraphBundle(
        graphs=graphs,
        tokenizer_dir=tok_dir,
        extra_files=extra_files,
        metadata={"adapter": adapter_id, "mode": "prebuilt"},
    )
