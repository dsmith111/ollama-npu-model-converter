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
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.core.manifest import collect_files, write_manifest
from npu_model.core.types import GraphBundle


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
    """Load a handoff bundle back into a GraphBundle + metadata."""
    bundle_dir = bundle_dir.expanduser().resolve()

    manifest_path = bundle_dir / "handoff_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("metadata", {})
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
