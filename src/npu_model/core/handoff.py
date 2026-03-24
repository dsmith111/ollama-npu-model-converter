"""Portable handoff bundle helpers for staged conversion workflows."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
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


_SUPPORTED_OLIVE_LLM_FAMILIES = frozenset({"phi", "phi3", "llama"})
_KNOWN_LLM_FAMILIES = frozenset({"phi", "phi3", "llama", "llama2", "llama3", "mistral", "gemma"})


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
    for _name, p in graphs.graphs.items():
        dst = graphs_dir / p.name
        shutil.copy2(p, dst)
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

    # Extras
    extras_dir = out_dir / "extras"
    extras_dir.mkdir(exist_ok=True)
    for p in graphs.extra_files:
        if p.exists() and p.is_file():
            dst = extras_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

    meta = dict(metadata)
    meta.setdefault("stopped_after", stopped_after)

    manifest = {
        "handoff": True,
        "stopped_after": stopped_after,
        "metadata": meta,
        "files": collect_files(out_dir),
    }
    manifest_path = out_dir / "handoff_manifest.json"
    write_manifest(manifest_path, manifest)

    return HandoffBundle(bundle_dir=out_dir, manifest_path=manifest_path, stopped_after=stopped_after)


def export_handoff_zip(bundle_dir: Path, zip_path: Path) -> Path:
    """Export a handoff directory as a .zip archive."""
    bundle_dir = bundle_dir.expanduser().resolve()
    zip_path = zip_path.expanduser().resolve()

    manifest = bundle_dir / "handoff_manifest.json"
    if not manifest.exists():
        raise NpuModelError(
            stage="handoff",
            reason_code="HANDOFF_MANIFEST_MISSING",
            message=f"Not a handoff bundle (missing handoff_manifest.json): {bundle_dir}",
        )

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(bundle_dir.rglob("*")):
            if src.is_file():
                arcname = src.relative_to(bundle_dir)
                zf.write(src, arcname=str(arcname))
    return zip_path


def _resolve_extracted_bundle_root(extracted_root: Path) -> Path:
    if (extracted_root / "handoff_manifest.json").exists():
        return extracted_root
    children = [p for p in extracted_root.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "handoff_manifest.json").exists():
        return children[0]
    raise NpuModelError(
        stage="handoff",
        reason_code="HANDOFF_MANIFEST_MISSING",
        message=(
            "Zip archive does not contain a valid handoff bundle root "
            "with handoff_manifest.json."
        ),
    )


def load_handoff_input(path: Path) -> tuple[GraphBundle, dict[str, Any]]:
    """Load handoff from either a directory or a .zip archive."""
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp_dir = Path(tempfile.mkdtemp(prefix="npu_handoff_"))
        try:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmp_dir)
        except zipfile.BadZipFile as e:
            raise NpuModelError(
                stage="handoff",
                reason_code="HANDOFF_ZIP_INVALID",
                message=f"Invalid handoff zip: {path}",
                cause=e,
            ) from e
        bundle_root = _resolve_extracted_bundle_root(tmp_dir)
        graphs, metadata = load_handoff_bundle(bundle_root)
        metadata = dict(metadata)
        metadata["_handoff_temp_dir"] = str(tmp_dir)
        metadata["_handoff_source"] = str(path)
        return graphs, metadata

    if not path.exists() or not path.is_dir():
        raise NpuModelError(
            stage="handoff",
            reason_code="HANDOFF_INPUT_NOT_FOUND",
            message=f"Handoff input not found: {path}",
        )
    graphs, metadata = load_handoff_bundle(path)
    metadata = dict(metadata)
    metadata["_handoff_source"] = str(path)
    return graphs, metadata


def load_handoff_bundle(bundle_dir: Path) -> tuple[GraphBundle, dict[str, Any]]:
    """Load a handoff directory into GraphBundle + metadata."""
    bundle_dir = bundle_dir.expanduser().resolve()

    metadata: dict[str, Any] = {}
    manifest_path = bundle_dir / "handoff_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = dict(manifest.get("metadata") or {})
        metadata.setdefault("stopped_after", manifest.get("stopped_after"))

        expected_fields = ["model_family", "quantization_format", "split_count", "layout"]
        missing = [f for f in expected_fields if f not in metadata]
        if missing:
            _log.warning(
                "Handoff manifest is missing enriched metadata fields: %s. "
                "Bundle may have been created by an older tool version.",
                ", ".join(missing),
            )

    graphs_dir = bundle_dir / "graphs"
    graphs: dict[str, Path] = {}
    if graphs_dir.exists():
        for p in sorted(graphs_dir.glob("*.onnx")):
            graphs[p.stem] = p
    else:
        for p in sorted(bundle_dir.rglob("*.onnx")):
            graphs[p.stem] = p

    tok_dir = bundle_dir / "tokenizer"
    if not tok_dir.exists():
        tok_dir = bundle_dir

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


def validate_handoff_for_compile(
    metadata: dict[str, Any],
    *,
    compile_strategy: str,
    allow_experimental: bool = False,
) -> None:
    """Validate handoff compatibility with compile strategy."""
    if compile_strategy not in ("context-cache", "ort-ep-context"):
        return

    stopped_after = (metadata.get("stopped_after") or metadata.get("handoff_stage") or "").lower()
    model_family = (metadata.get("model_family") or "").lower()
    layout = (metadata.get("layout") or "").lower()
    quant_format = (metadata.get("quantization_format") or "").lower()
    quantizer_id = (metadata.get("quantizer_id") or "").lower()

    if stopped_after != "quantize":
        if stopped_after == "export":
            raise NpuModelError(
                stage="backend",
                reason_code="HANDOFF_NOT_QUANTIZED",
                message=(
                    "This handoff bundle stopped after 'export' and is not quantized. "
                    "Context-cache compilation requires a quantized QDQ handoff."
                ),
                hint="Re-run convert with --stop-after quantize.",
            )
        raise NpuModelError(
            stage="backend",
            reason_code="HANDOFF_STAGE_INVALID",
            message=(
                f"Handoff stage must be 'quantize' for context-cache compile; got '{stopped_after or 'unknown'}'."
            ),
            hint="Re-run convert with --stop-after quantize.",
        )

    if quant_format != "qdq":
        raise NpuModelError(
            stage="backend",
            reason_code="HANDOFF_NOT_QDQ",
            message=(
                f"Handoff quantization_format must be 'qdq' for context-cache compile; got '{quant_format or 'unknown'}'."
            ),
            hint="Use qdq quantization before compile-context.",
        )

    if model_family in _SUPPORTED_OLIVE_LLM_FAMILIES:
        split_count = metadata.get("split_count")
        if not layout or split_count is None:
            raise NpuModelError(
                stage="backend",
                reason_code="HANDOFF_SPLIT_METADATA_REQUIRED",
                message=(
                    "Supported LLM handoff is missing split metadata (layout/split_count)."
                ),
                hint="Regenerate the handoff using the current toolchain.",
            )

        try:
            split_count_i = int(split_count)
        except Exception:
            split_count_i = 0

        if (layout != "split" or split_count_i <= 1) and not allow_experimental:
            raise NpuModelError(
                stage="backend",
                reason_code="HANDOFF_LAYOUT_UNSUPPORTED",
                message=(
                    f"Supported LLM family '{model_family}' requires split layout for compile-context. "
                    f"Got layout='{layout or 'unknown'}', split_count={split_count!r}."
                ),
                hint="Use --quant olive-qnn-llm to generate split/static QDQ handoff artifacts.",
            )

        if quantizer_id != "olive-qnn-llm" and not allow_experimental:
            raise NpuModelError(
                stage="backend",
                reason_code="SUPPORTED_LLM_REQUIRES_OLIVE",
                message=(
                    f"Model family '{model_family}' must use quantizer 'olive-qnn-llm' for the supported route. "
                    f"Got quantizer '{quantizer_id}'."
                ),
                hint=(
                    "Re-run convert with --quant olive-qnn-llm, or pass --allow-experimental "
                    "to compile-context to force the generic path."
                ),
            )

    if model_family in _KNOWN_LLM_FAMILIES and layout == "monolith" and not allow_experimental:
        raise NpuModelError(
            stage="backend",
            reason_code="MONOLITHIC_LLM_QDQ_EXPERIMENTAL",
            message=(
                f"Handoff bundle is monolithic QDQ for model family '{model_family}'. "
                "This compile path is experimental for LLMs."
            ),
            hint="Use olive-qnn-llm for split/static handoff, or pass --allow-experimental.",
        )
