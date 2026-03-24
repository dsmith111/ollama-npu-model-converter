from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from npu_model.core.errors import NpuModelError
from npu_model.core.types import GraphBundle


def _looks_like_onnx_data(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".onnx.data") or name.endswith(".data")


def _copy_unique(files: Iterable[Path], dst_dir: Path) -> list[Path]:
    copied: list[Path] = []
    seen: set[str] = set()
    for src in files:
        if not src.exists() or not src.is_file():
            continue
        if src.name in seen:
            continue
        seen.add(src.name)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def collect_olive_outputs(
    *,
    olive_output_dir: Path,
    fallback_tokenizer_dir: Path,
    fallback_extra_files: list[Path],
    family: str,
) -> GraphBundle:
    """Collect Olive-produced graphs into the GraphBundle contract."""
    if not olive_output_dir.exists():
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_OUTPUT_DIR_MISSING",
            message=f"Olive output directory not found: {olive_output_dir}",
        )

    stage_dir = olive_output_dir / "_npu_model_bundle"
    graphs_dir = stage_dir / "graphs"
    tokenizer_dir = stage_dir / "tokenizer"
    extras_dir = stage_dir / "extras"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    extras_dir.mkdir(parents=True, exist_ok=True)

    graph_candidates = sorted(
        p for p in olive_output_dir.rglob("*.onnx")
        if "_ctx" not in p.stem.lower() and "_npu_model_bundle" not in str(p)
    )
    if not graph_candidates:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_NO_ONNX_OUTPUT",
            message=f"Olive run completed but no ONNX outputs were found under {olive_output_dir}.",
        )

    new_graphs: dict[str, Path] = {}
    seen_names: set[str] = set()
    for src in graph_candidates:
        base = src.stem
        name = base
        idx = 1
        while name in seen_names:
            idx += 1
            name = f"{base}_{idx}"
        seen_names.add(name)
        dst = graphs_dir / src.name
        shutil.copy2(src, dst)
        data_file = src.parent / f"{src.name}.data"
        if data_file.exists():
            shutil.copy2(data_file, graphs_dir / data_file.name)
        new_graphs[name] = dst

    tokenizer_candidates = sorted(
        p for p in olive_output_dir.rglob("*")
        if p.is_file()
        and p.name in {
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
        }
        and "_npu_model_bundle" not in str(p)
    )
    if tokenizer_candidates:
        _copy_unique(tokenizer_candidates, tokenizer_dir)
    elif fallback_tokenizer_dir.exists():
        _copy_unique(fallback_tokenizer_dir.glob("*"), tokenizer_dir)

    extra_candidates = sorted(
        p for p in olive_output_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".json", ".bin", ".jinja"}
        and p.name != "config.json"
        and "_npu_model_bundle" not in str(p)
    )
    copied_extras = _copy_unique(extra_candidates, extras_dir)
    copied_extras.extend(_copy_unique(fallback_extra_files, extras_dir))

    split_count = len(new_graphs)
    layout = "split" if split_count > 1 else "monolith"
    if family in {"phi", "phi3", "llama"} and layout != "split":
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_OUTPUT_NOT_SPLIT",
            message=(
                f"Olive output for family '{family}' produced {split_count} graph(s). "
                "Supported LLM path requires split layout."
            ),
            hint="Check Olive recipe and ensure split passes (CaptureSplitInfo/StaticLLM) are enabled.",
        )

    if any(_looks_like_onnx_data(p) for p in copied_extras):
        # keep data files with graphs only; never as extra sidecar files
        copied_extras = [p for p in copied_extras if not _looks_like_onnx_data(p)]

    return GraphBundle(
        graphs=new_graphs,
        tokenizer_dir=tokenizer_dir,
        extra_files=copied_extras,
        metadata={
            "quantizer": "olive-qnn-llm",
            "model_family": family,
            "layout": layout,
            "split_count": split_count,
        },
    )

