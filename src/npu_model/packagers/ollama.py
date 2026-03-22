from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from npu_model.core.errors import NpuModelError
from npu_model.core.manifest import collect_files, write_manifest
from npu_model.runtime_formats.ort_genai_folder import (
    collect_ollama_files,
    validate_ollama_ortgenai_dir,
    write_modelfile,
)

import npu_model


@dataclass(frozen=True)
class PackResult:
    pack_dir: Path
    file_count: int


def pack_for_ollama(
    bundle_dir: Path,
    model_name: str,
    out_dir: Path,
    *,
    num_ctx: int = 512,
    num_predict: int = 128,
) -> PackResult:
    """Build an Ollama-publishable directory from an ORT GenAI bundle.

    Uses atomic staging: builds into a temp directory, validates, then
    replaces the destination to avoid leaving an empty output on failure.
    """
    bundle_dir = bundle_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not bundle_dir.exists():
        raise NpuModelError(
            stage="pack",
            reason_code="BUNDLE_NOT_FOUND",
            message=f"Bundle not found: {bundle_dir}",
        )

    # --- collect allowlisted files ---
    files = collect_ollama_files(bundle_dir)
    if not files:
        raise NpuModelError(
            stage="pack",
            reason_code="NO_FILES_COLLECTED",
            message="No allowlisted files found in bundle directory.",
            hint="Ensure the input directory contains *.onnx, genai_config.json, and tokenizer files.",
        )

    # --- NPU-only guard: refuse .onnx.data without context-cache artifacts ---
    collected_names = {f.name for f in files}
    has_data_files = any(n.endswith(".onnx.data") for n in collected_names)
    has_ctx_graphs = any("_ctx" in n and n.endswith(".onnx") for n in collected_names)
    has_qnn_bins = any(n.endswith(".bin") for n in collected_names)

    if has_data_files and not has_ctx_graphs:
        raise NpuModelError(
            stage="pack",
            reason_code="ONNX_DATA_WITHOUT_CTX",
            message=(
                "Bundle contains .onnx.data (raw float weights) but no *_ctx.onnx "
                "context-cache artifacts. This model will NOT run on the NPU and will "
                "cause disk/memory pressure at runtime."
            ),
            hint=(
                "Use --compile-strategy context-cache to generate QNN context-cache artifacts.\n"
                "For prebuilt models, use --mode prebuilt-ort-genai with a properly prepared bundle."
            ),
        )

    # --- atomic staging ---
    staging = out_dir.parent / f"{out_dir.name}.staging.{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        # Copy collected files flat into staging root
        for src in files:
            dst = staging / src.name
            # Handle name collisions from nested dirs by preferring shallower files
            if not dst.exists():
                shutil.copy2(src, dst)

        # Write Modelfile
        write_modelfile(staging, num_ctx=num_ctx, num_predict=num_predict)

        # Write publish manifest
        manifest_payload = {
            "tool": {"name": "npu-model", "version": npu_model.__version__},
            "pack_type": "ollama-ortgenai",
            "model_name": model_name,
            "params": {"num_ctx": num_ctx, "num_predict": num_predict},
            "files": collect_files(staging),
        }
        write_manifest(staging / "_publish_manifest.json", manifest_payload)

        # Validate staging before promoting
        validation = validate_ollama_ortgenai_dir(staging)
        if validation.errors:
            msg = "Staged output failed validation:\n  " + "\n  ".join(validation.errors)
            raise NpuModelError(
                stage="pack",
                reason_code="STAGING_VALIDATION_FAILED",
                message=msg,
            )

        # Count files (excluding Modelfile and manifest for the "copied" count)
        copied_count = len([p for p in staging.iterdir() if p.is_file()])

        # Atomic swap: remove existing destination, rename staging into place
        if out_dir.exists():
            shutil.rmtree(out_dir)
        staging.rename(out_dir)

    except NpuModelError:
        # Clean up staging on our errors
        if staging.exists():
            shutil.rmtree(staging)
        raise
    except Exception as e:
        if staging.exists():
            shutil.rmtree(staging)
        raise NpuModelError(
            stage="pack",
            reason_code="PACK_FAILED",
            message=f"Failed to build Ollama pack: {e}",
            cause=e,
        ) from e

    return PackResult(pack_dir=out_dir, file_count=copied_count)
