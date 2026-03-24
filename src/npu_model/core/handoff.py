"""Handoff bundle — portable intermediate artifact for staged pipelines.

When the full export→quantize→compile pipeline can't run on a single machine
(e.g., quantization requires more RAM than available), the pipeline can stop
after a stage and produce a "handoff bundle" that can be transferred to
another machine to continue.

Layout::

    handoff_bundle/
        handoff_manifest.json   # stage info, model metadata
        graphs/
            model.qdq.onnx      # (or model.onnx if stopped after export)
            model.qdq.onnx.data # (if external data)
        tokenizer/
            tokenizer.json
            tokenizer.model
            tokenizer_config.json
            ...
        extras/
            genai_config.json
            chat_template.jinja
            ...

Manifest metadata contract (enriched fields)::

    metadata.model_type          — HF model_type string (e.g. "phi3")
    metadata.model_family        — lowercased family key (e.g. "phi3", "llama")
    metadata.adapter_id          — adapter that produced the graphs
    metadata.quantizer_id        — quantizer used (e.g. "qnn-qdq")
    metadata.quantization_format — "qdq", "passthrough", or None
    metadata.split_count         — number of graph splits
    metadata.layout              — "monolith" or "split"
    metadata.graph_metadata      — adapter-provided graph metadata dict
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.core.errors import NpuModelError
from npu_model.core.manifest import collect_files, write_manifest
from npu_model.core.types import GraphBundle

_log = logging.getLogger("npu_model.handoff")


@dataclass(frozen=True)
class HandoffBundle:
    bundle_dir: Path
    manifest_path: Path
    stopped_after: str


def create_handoff_bundle(
    *,
    graphs: GraphBundle,
    out_dir: Path,
    stopped_after: str,
    metadata: dict[str, Any],
) -> HandoffBundle:
    """Package current pipeline state into a portable handoff bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Graphs
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    for name, p in graphs.graphs.items():
        dst = graphs_dir / p.name
        shutil.copy2(p, dst)
        # Co-located .data files
        data = p.parent / f"{p.name}.data"
        if data.exists():
            shutil.copy2(data, graphs_dir / data.name)

    # Tokenizer
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    if graphs.tokenizer_dir.exists():
        for p in sorted(graphs.tokenizer_dir.glob("*")):
            if p.is_file():
                shutil.copy2(p, tok_dir / p.name)

    # Extra files
    extras_dir = out_dir / "extras"
    extras_dir.mkdir(exist_ok=True)
    for p in graphs.extra_files:
        if p.exists() and p.is_file():
            dst = extras_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

    # Manifest
    manifest = {
        "handoff": True,
        "stopped_after": stopped_after,
        "metadata": metadata,
        "files": collect_files(out_dir),
    }
    manifest_path = out_dir / "handoff_manifest.json"
    write_manifest(manifest_path, manifest)

    return HandoffBundle(
        bundle_dir=out_dir,
        manifest_path=manifest_path,
        stopped_after=stopped_after,
    )


def load_handoff_bundle(bundle_dir: Path) -> tuple[GraphBundle, dict[str, Any]]:
    """Load a handoff bundle back into a GraphBundle + metadata.

    Validates that enriched metadata fields are present when a manifest
    exists, and logs warnings for missing optional fields.
    """
    bundle_dir = bundle_dir.expanduser().resolve()

    manifest_path = bundle_dir / "handoff_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("metadata", {})

        # Validate enriched metadata — warn on missing fields so callers
        # know the bundle is from an older tool version.
        _EXPECTED_FIELDS = [
            "model_family", "quantization_format", "split_count", "layout",
        ]
        missing = [f for f in _EXPECTED_FIELDS if f not in metadata]
        if missing:
            _log.warning(
                "Handoff manifest is missing enriched metadata fields: %s. "
                "Bundle may have been created by an older tool version.",
                ", ".join(missing),
            )
    else:
        metadata = {}

    # Collect graphs
    graphs_dir = bundle_dir / "graphs"
    graphs: dict[str, Path] = {}
    if graphs_dir.exists():
        for p in sorted(graphs_dir.glob("*.onnx")):
            graphs[p.stem] = p
    else:
        # Flat layout fallback (might be a raw dir)
        for p in sorted(bundle_dir.rglob("*.onnx")):
            graphs[p.stem] = p

    # Tokenizer dir
    tok_dir = bundle_dir / "tokenizer"
    if not tok_dir.exists():
        tok_dir = bundle_dir  # fallback

    # Extra files
    extras_dir = bundle_dir / "extras"
    extra_files: list[Path] = []
    if extras_dir.exists():
        for p in sorted(extras_dir.glob("*")):
            if p.is_file():
                extra_files.append(p)

    bundle = GraphBundle(
        graphs=graphs,
        tokenizer_dir=tok_dir,
        extra_files=extra_files,
        metadata=metadata.get("graph_metadata", {}),
    )

    return bundle, metadata


# LLM families for which the generic monolithic QDQ path is experimental.
_LLM_FAMILIES = frozenset({
    "phi", "phi3", "llama", "llama2", "llama3", "mistral", "gemma",
})


def validate_handoff_for_compile(
    metadata: dict[str, Any],
    *,
    compile_strategy: str,
    allow_experimental: bool = False,
) -> None:
    """Validate that a handoff bundle is compatible with the requested
    compile strategy.  Raises ``NpuModelError`` if the bundle should not
    be compiled through the requested path.

    Parameters
    ----------
    metadata:
        The ``metadata`` dict from ``load_handoff_bundle()``.
    compile_strategy:
        The compile strategy that will be used (e.g. ``"context-cache"``).
    allow_experimental:
        If ``True``, skip the monolithic-LLM rejection gate
        (user explicitly opted in).
    """
    stopped_after = metadata.get("stopped_after") or metadata.get("handoff_stage")
    model_family = (metadata.get("model_family") or "").lower()
    layout = (metadata.get("layout") or "").lower()
    quant_format = metadata.get("quantization_format")

    # Gate 1: context-cache requires a quantized handoff
    if compile_strategy in ("context-cache", "ort-ep-context"):
        if stopped_after == "export":
            raise NpuModelError(
                stage="backend",
                reason_code="HANDOFF_NOT_QUANTIZED",
                message=(
                    "This handoff bundle stopped after 'export' — it has not been "
                    "quantized.  Context-cache compilation requires a QDQ-quantized bundle."
                ),
                hint=(
                    "Re-run quantization first:\n"
                    "  npu-model convert --input <bundle> --stop-after quantize "
                    "--quant qnn-qdq --calib-prompts <prompts.txt>"
                ),
            )

        # Gate 2: monolithic LLM + generic QDQ → experimental, warn or block
        if (
            model_family in _LLM_FAMILIES
            and layout == "monolith"
            and quant_format == "qdq"
            and not allow_experimental
        ):
            raise NpuModelError(
                stage="backend",
                reason_code="MONOLITHIC_LLM_QDQ_EXPERIMENTAL",
                message=(
                    f"Handoff bundle is a monolithic QDQ graph for model family "
                    f"'{model_family}'.  This compile path is experimental for LLMs "
                    f"and is likely to produce oversized or invalid context artifacts."
                ),
                hint=(
                    "Options:\n"
                    "  1. Use the Olive-backed LLM pipeline (recommended):\n"
                    "       npu-model convert --quant olive-qnn-llm ...\n"
                    "  2. Use a prebuilt model:\n"
                    "       --mode prebuilt-ort-genai\n"
                    "  3. Opt into experimental generic path (at your own risk):\n"
                    "       --compile-opt allow_experimental=true"
                ),
            )
