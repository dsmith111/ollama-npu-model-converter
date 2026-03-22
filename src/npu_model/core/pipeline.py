from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from npu_model.core.cache import ConversionCache, compute_cache_key
from npu_model.core.errors import NpuModelError
from npu_model.core.manifest import collect_files, write_manifest
from npu_model.core.registry import Registry
from npu_model.core.types import ConvertMode, ConvertResult, ExplainPlan, GraphBundle
from npu_model.inspect.hf_inspector import inspect_hf_style_dir
from npu_model.sources.local import materialize_local
from npu_model.sources.hf import parse_hf_spec, materialize_hf
from npu_model.adapters.auto import select_adapter


# ---------------------------------------------------------------------------
# Source materialization (single call, reused across explain + convert)
# ---------------------------------------------------------------------------

def _materialize(input_spec: str, cache_dir: Optional[Path]) -> tuple[Path, dict[str, Any]]:
    if input_spec.startswith("hf:"):
        spec = parse_hf_spec(input_spec)
        model_dir = materialize_hf(spec, cache_dir)
        return model_dir, {"type": "hf", "repo_id": spec.repo_id, "revision": spec.revision}
    else:
        model_dir = materialize_local(Path(input_spec))
        return model_dir, {"type": "local", "path": str(model_dir)}


def _resolve_mode(mode: str) -> ConvertMode:
    try:
        return ConvertMode(mode)
    except ValueError:
        raise NpuModelError(
            stage="pipeline",
            reason_code="UNKNOWN_MODE",
            message=f"Unknown convert mode: {mode}",
            hint=f"Supported modes: {[m.value for m in ConvertMode]}",
        )


# ---------------------------------------------------------------------------
# explain_plan
# ---------------------------------------------------------------------------

def explain_plan(
    input_spec: str,
    backend_id: str,
    target: str,
    runtime_format_id: str,
    cache_dir: Optional[Path],
    registry: Registry,
    quantizer_id: str = "passthrough",
    mode: str = "export",
) -> ExplainPlan:
    convert_mode = _resolve_mode(mode)
    model_dir, source = _materialize(input_spec, cache_dir)
    mi = inspect_hf_style_dir(model_dir, source=source)

    adapter_id = select_adapter(registry, mi)

    backend = registry.backends.get(backend_id)
    if backend is None:
        raise NpuModelError(
            stage="backend",
            reason_code="UNKNOWN_BACKEND",
            message=f"Unknown backend: {backend_id}",
            hint=f"Available: {sorted(registry.backends.keys())}",
        )

    runtime = registry.runtime_formats.get(runtime_format_id)
    if runtime is None:
        raise NpuModelError(
            stage="runtime",
            reason_code="UNKNOWN_RUNTIME_FORMAT",
            message=f"Unknown runtime format: {runtime_format_id}",
            hint=f"Available: {sorted(registry.runtime_formats.keys())}",
        )

    if quantizer_id not in registry.quantizers:
        raise NpuModelError(
            stage="quant",
            reason_code="UNKNOWN_QUANTIZER",
            message=f"Unknown quantizer: {quantizer_id}",
            hint=f"Available: {sorted(registry.quantizers.keys())}",
        )

    ts = backend.resolve_target(target, env=dict(os.environ))

    return ExplainPlan(
        input_spec=input_spec,
        materialized_dir=model_dir,
        model_type=mi.model_type,
        architectures=mi.architectures,
        adapter_id=adapter_id,
        backend_id=backend_id,
        target_name=ts.name,
        runtime_format_id=runtime_format_id,
        quantizer_id=quantizer_id,
        convert_mode=convert_mode.value,
    )


# ---------------------------------------------------------------------------
# convert_model
# ---------------------------------------------------------------------------

def convert_model(
    input_spec: str,
    out_dir: Path,
    backend_id: str,
    target: str,
    runtime_format_id: str,
    quantizer_id: str,
    cache_dir: Optional[Path],
    registry: Registry,
    mode: str = "export",
    compile_config: Optional[dict[str, Any]] = None,
    export_options: Optional[dict[str, Any]] = None,
    pack_ollama_name: Optional[str] = None,
    pack_ollama_opts: Optional[dict[str, Any]] = None,
    use_cache: bool = True,
    keep_work: bool = False,
    stop_after: Optional[str] = None,
) -> ConvertResult:
    convert_mode = _resolve_mode(mode)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Materialize (once) ----
    model_dir, source = _materialize(input_spec, cache_dir)
    mi = inspect_hf_style_dir(model_dir, source=source)

    # ---- 2. Select plugins ----
    adapter_id = select_adapter(registry, mi)
    adapter = registry.adapters[adapter_id]

    backend = registry.backends.get(backend_id)
    if backend is None:
        raise NpuModelError(
            stage="backend", reason_code="UNKNOWN_BACKEND",
            message=f"Unknown backend: {backend_id}",
            hint=f"Available: {sorted(registry.backends.keys())}",
        )
    runtime = registry.runtime_formats.get(runtime_format_id)
    if runtime is None:
        raise NpuModelError(
            stage="runtime", reason_code="UNKNOWN_RUNTIME_FORMAT",
            message=f"Unknown runtime format: {runtime_format_id}",
            hint=f"Available: {sorted(registry.runtime_formats.keys())}",
        )
    quantizer = registry.quantizers.get(quantizer_id)
    if quantizer is None:
        raise NpuModelError(
            stage="quant", reason_code="UNKNOWN_QUANTIZER",
            message=f"Unknown quantizer: {quantizer_id}",
            hint=f"Available: {sorted(registry.quantizers.keys())}",
        )

    # Resolve backend target early (needed for cache key)
    ts = backend.resolve_target(target, env=dict(os.environ))
    cc = compile_config or {}
    strategy = cc.get("strategy", "passthrough")

    # ---- 2b. Check cache ----
    import npu_model
    cache = ConversionCache(out_dir)
    cache_key = compute_cache_key(
        input_spec=input_spec,
        input_revision=source.get("revision"),
        adapter_id=adapter_id,
        mode=convert_mode.value,
        backend_id=backend_id,
        target_name=ts.name,
        target_params=ts.params,
        compile_strategy=strategy,
        compile_config=cc,
        quantizer_id=quantizer_id,
        runtime_format_id=runtime_format_id,
        tool_version=npu_model.__version__,
    )

    if use_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            bundle_dir, manifest_path = cache.restore(cached, out_dir)
            pack_dir = _maybe_pack_ollama(
                bundle_dir, pack_ollama_name, pack_ollama_opts, out_dir,
            )
            return ConvertResult(
                bundle_dir=bundle_dir,
                manifest_path=manifest_path,
                pack_dir=pack_dir,
            )

    # ---- 3. Adapter export / import ----
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    adapter_out = work / f"adapter_{adapter.id}"
    export_config: dict[str, Any] = {"mode": convert_mode.value, "model_info": mi}
    if export_options:
        export_config.update(export_options)

    # Auto-fix: when using a real quantizer (e.g. qnn-qdq), export in fp32 or fp16
    # so the quantizer handles quantization — don't let the builder try int4.
    if quantizer_id != "passthrough":
        current_precision = export_config.get("precision", "int4")
        if current_precision == "int4":
            export_config["precision"] = "fp16"
            import logging
            logging.getLogger("npu_model").info(
                "Auto-adjusted export precision from int4 to fp16 "
                "(quantizer '%s' will handle quantization)", quantizer_id,
            )

    try:
        graphs = adapter.export(model_dir, adapter_out, export_config=export_config)
    except NpuModelError:
        raise
    except Exception as e:
        raise NpuModelError(
            stage="export",
            reason_code="ADAPTER_EXPORT_FAILED",
            message=f"Adapter '{adapter_id}' export failed: {e}",
            hint=(
                f"Mode: {convert_mode.value}\n"
                f"Input: {model_dir}\n"
                "If using export mode, ensure deps are installed: pip install npu-model[export]\n"
                "For prebuilt import: --mode prebuilt-ort-genai"
            ),
            cause=e,
        ) from e

    # ---- 4. Bridge + normalize tokenizer for runtime compatibility ----
    import logging
    _log = logging.getLogger("npu_model")

    # 4a. Bridge: copy missing tokenizer files from HF source into export output
    from npu_model.core.tokenizer_bridge import bridge_tokenizer_files

    bridged = bridge_tokenizer_files(
        src_model_dir=model_dir, dst_tokenizer_dir=graphs.tokenizer_dir,
    )
    if bridged:
        _log.info("Tokenizer files bridged from source: %s", ", ".join(bridged))

    # 4b. Normalize: rewrite unsupported tokenizer_class if possible
    from npu_model.core.tokenizer_norm import normalize_tokenizer_config

    tok_norm = normalize_tokenizer_config(
        graphs.tokenizer_dir, model_type=mi.model_type,
    )
    if tok_norm.changed:
        _log.info(
            "Tokenizer normalized: %s -> %s", tok_norm.original_class, tok_norm.new_class,
        )
    for w in tok_norm.warnings:
        _log.warning("Tokenizer: %s", w)

    # 4c. If packing for Ollama, fail hard on unsupported tokenizer
    if pack_ollama_name and not tok_norm.changed and tok_norm.original_class:
        from npu_model.core.tokenizer_norm import _is_unsupported
        if _is_unsupported(tok_norm.original_class):
            raise NpuModelError(
                stage="tokenizer",
                reason_code="TOKENIZER_UNSUPPORTED_FOR_OLLAMA",
                message=(
                    f"tokenizer_class='{tok_norm.original_class}' is not supported "
                    f"by ORT GenAI on this platform, and normalization could not fix it."
                ),
                hint=(
                    "Ensure tokenizer.model (SentencePiece) is present in the source model.\n"
                    "The tokenizer bridge attempted to copy it from the HF snapshot but "
                    "it was not found. Check that the source model includes tokenizer.model."
                ),
            )

    # ---- 4d. Stop after export (for staged workflows) ----
    if stop_after == "export":
        from npu_model.core.handoff import create_handoff_bundle
        handoff = create_handoff_bundle(
            graphs=graphs,
            out_dir=out_dir / "handoff_bundle",
            stopped_after="export",
            metadata={
                "input_spec": input_spec,
                "adapter_id": adapter_id,
                "model_type": mi.model_type,
                "graph_metadata": graphs.metadata,
            },
        )
        _log.info("Stopped after export. Handoff bundle: %s", handoff.bundle_dir)
        return ConvertResult(
            bundle_dir=handoff.bundle_dir,
            manifest_path=handoff.manifest_path,
        )

    # ---- 4e. Policy: QNN/HTP + context-cache requires proper quantization ----
    is_npu_target = (
        backend_id == "qnn"
        and ts.params.get("backend_type", "").lower() in ("htp", "")
    )
    if is_npu_target and strategy == "context-cache" and quantizer_id == "passthrough":
        if pack_ollama_name:
            raise NpuModelError(
                stage="pipeline",
                reason_code="NPU_REQUIRES_QUANTIZATION",
                message=(
                    "QNN HTP context-cache compilation requires QDQ quantization. "
                    "Passthrough quantizer cannot produce NPU-ready artifacts."
                ),
                hint=(
                    "Use --quant qnn-qdq with calibration data:\n"
                    "  npu-model convert --quant qnn-qdq --calib-prompts prompts.txt --compile-strategy context-cache ..."
                ),
            )
        else:
            _log.warning(
                "QNN HTP context-cache compilation with passthrough quantizer. "
                "The model may not be quantized — HTP requires QDQ quantization."
            )

    # ---- 5. Shape fixing (required for QNN HTP) ----
    num_ctx_val = (pack_ollama_opts or {}).get("num_ctx", 512)

    if is_npu_target and strategy == "context-cache":
        from npu_model.core.shapes import has_dynamic_shapes, fix_dynamic_shapes

        num_ctx = num_ctx_val
        shape_fix_dir = work / "shape_fixed"
        shape_fix_dir.mkdir(exist_ok=True)
        new_graphs: dict[str, Path] = {}
        for gname, gpath in graphs.graphs.items():
            dynamic = has_dynamic_shapes(gpath)
            if dynamic:
                _log.info("Fixing dynamic shapes in %s: %s", gpath.name, dynamic)
                result = fix_dynamic_shapes(
                    gpath, shape_fix_dir / gpath.name,
                    batch_size=1, sequence_length=num_ctx,
                )
                if result.fixed and result.output_path:
                    new_graphs[gname] = result.output_path
                    for w in result.warnings:
                        _log.warning("Shape fix: %s", w)
                else:
                    for e in result.errors:
                        _log.error("Shape fix error: %s", e)
                    new_graphs[gname] = gpath  # keep original
            else:
                new_graphs[gname] = gpath
        graphs = GraphBundle(
            graphs=new_graphs,
            tokenizer_dir=graphs.tokenizer_dir,
            extra_files=graphs.extra_files,
            metadata=graphs.metadata,
        )

    # ---- 6. Auto-calibration + Quantize ----
    quant_config: dict[str, Any] = {}
    if quantizer_id != "passthrough":
        # Check model size — quantization loads the entire model into RAM.
        # On memory-constrained devices, this silently crashes (SIGSEGV / OOM).
        total_model_bytes = sum(
            p.stat().st_size for p in graphs.graphs.values() if p.exists()
        )
        for gpath in graphs.graphs.values():
            data_file = gpath.parent / f"{gpath.name}.data"
            if data_file.exists():
                total_model_bytes += data_file.stat().st_size

        import os as _os
        try:
            mem_info = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError):
            mem_info = None  # Windows — no sysconf

        model_gb = total_model_bytes / (1024**3)

        # Heuristic: quantization needs ~2-3x model size in RAM
        if model_gb > 4.0:
            _log.warning(
                "Model size is %.1f GB. Quantization requires loading the entire model "
                "into RAM (typically 2-3x model size). On devices with limited memory, "
                "this may crash silently.", model_gb,
            )
            if model_gb > 6.0:
                # Build actionable staged workflow commands
                _input_escaped = input_spec.replace("'", "\\'")
                raise NpuModelError(
                    stage="quant",
                    reason_code="MODEL_TOO_LARGE_FOR_QUANTIZATION",
                    message=(
                        f"Model is {model_gb:.1f} GB — too large to quantize in-memory "
                        f"on this device. Quantization requires ~{model_gb * 2.5:.0f} GB RAM."
                    ),
                    hint=(
                        "Use a staged workflow across two machines:\n"
                        "\n"
                        "Step 1 — On a machine with enough RAM, export only:\n"
                        f"  npu-model convert --input '{_input_escaped}' "
                        f"--out .\\handoff --stop-after export --keep-work\n"
                        "\n"
                        "Step 2 — Copy the handoff_bundle/ folder to the target device.\n"
                        "\n"
                        "Step 3 — On the target device (with QNN EP), compile + pack:\n"
                        f"  npu-model compile-context --input .\\handoff\\handoff_bundle "
                        f"--out .\\compiled\n"
                        f"  npu-model pack-ollama --input .\\compiled "
                        f"--name <ollama-tag> --out .\\publish\n"
                        "\n"
                        "Or use a prebuilt model: --mode prebuilt-ort-genai"
                    ),
                )

        # Check if the quantizer needs calibration data
        requires_calib = getattr(quantizer, "requires_calibration", True)
        if requires_calib:
            from npu_model.calib.prompt_source import get_prompt_source
            from npu_model.calib.data_reader import build_calibration_reader

            calib_source = (export_options or {}).get("calib_source", "builtin:mixed_small")
            calib_prompts_file = (export_options or {}).get("calib_prompts_file")
            calib_samples = int((export_options or {}).get("calib_samples", 64))
            calib_maxlen = int((export_options or {}).get("calib_maxlen", min(num_ctx_val, 256)))

            prompt_src = get_prompt_source(calib_source, calib_prompts_file)
            prompts = prompt_src.load()

            # Pick the first graph to inspect input names/dtypes
            first_graph_path = next(iter(graphs.graphs.values()))

            _log.info(
                "Building calibration data: %d samples, maxlen=%d, source=%s",
                calib_samples, calib_maxlen,
                calib_prompts_file or calib_source,
            )
            reader = build_calibration_reader(
                prompts=prompts,
                tokenizer_dir=graphs.tokenizer_dir,
                onnx_path=first_graph_path,
                num_samples=calib_samples,
                max_seq_len=calib_maxlen,
            )
            quant_config["calibration_data_reader"] = reader

    graphs = quantizer.apply(graphs, quant_config=quant_config)

    # ---- 6b. Stop after quantize (for staged workflows) ----
    if stop_after == "quantize":
        from npu_model.core.handoff import create_handoff_bundle
        handoff = create_handoff_bundle(
            graphs=graphs,
            out_dir=out_dir / "handoff_bundle",
            stopped_after="quantize",
            metadata={
                "input_spec": input_spec,
                "adapter_id": adapter_id,
                "model_type": mi.model_type,
                "quantizer_id": quantizer_id,
                "graph_metadata": graphs.metadata,
            },
        )
        _log.info("Stopped after quantize. Handoff bundle: %s", handoff.bundle_dir)
        _log.info(
            "Next: transfer handoff_bundle/ to target device and run:\n"
            "  npu-model compile-context --input %s --out <compiled_dir>",
            handoff.bundle_dir,
        )
        return ConvertResult(
            bundle_dir=handoff.bundle_dir,
            manifest_path=handoff.manifest_path,
        )

    # ---- 7. Backend resolve + compile/prepare ----
    backend_out = work / f"backend_{backend_id}"
    try:
        if strategy != "passthrough" and hasattr(backend, "compile"):
            prepared = backend.compile(graphs, backend_out, target=ts, compile_config=cc)
        else:
            prepared = backend.prepare(graphs, backend_out, target=ts, backend_config=cc)
    except NpuModelError:
        raise
    except Exception as e:
        # Enhanced error with diagnostics
        diag_lines = [
            f"Backend: {backend_id}",
            f"Target: {ts.normalized_repr()}",
            f"Strategy: {strategy}",
        ]
        if hasattr(backend, "detect_environment"):
            caps = backend.detect_environment()
            diag_lines.append(f"Compile available: {caps.compile_available}")
            for d in caps.diagnostics:
                diag_lines.append(f"  {d}")
        raise NpuModelError(
            stage="backend",
            reason_code="BACKEND_PREPARE_FAILED",
            message=f"Backend '{backend_id}' preparation failed: {e}",
            hint="\n".join(diag_lines),
            cause=e,
        ) from e

    # ---- 6. Assemble runtime bundle ----
    bundle_dir = runtime.assemble(prepared, graphs.tokenizer_dir, out_dir, format_config={})

    # ---- 7. Manifest ----
    plan = ExplainPlan(
        input_spec=input_spec,
        materialized_dir=model_dir,
        model_type=mi.model_type,
        architectures=mi.architectures,
        adapter_id=adapter_id,
        backend_id=backend_id,
        target_name=ts.name,
        runtime_format_id=runtime_format_id,
        quantizer_id=quantizer_id,
        convert_mode=convert_mode.value,
    )
    manifest = {
        "tool": {"name": "npu-model", "version": npu_model.__version__},
        "input": source,
        "plan": asdict(plan),
        "backend_metadata": prepared.backend_metadata,
        "bundle": {"format": runtime_format_id, "path": str(bundle_dir)},
        "files": collect_files(bundle_dir),
    }
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    # ---- 7b. Store in cache ----
    if use_cache:
        cache.put(cache_key, bundle_dir, manifest_path, meta={
            "input_spec": input_spec,
            "adapter_id": adapter_id,
            "backend_id": backend_id,
            "target": ts.name,
            "strategy": strategy,
        })

    # ---- 8. Strict NPU validation (when packing for NPU target) ----
    if pack_ollama_name and is_npu_target and strategy == "context-cache":
        from npu_model.validate.npu_strict import validate_npu_strict

        npu_result = validate_npu_strict(bundle_dir)
        for check in npu_result.checks:
            if check.status == "FAIL":
                _log.error("NPU validation: %s — %s", check.name, check.detail)
            elif check.status == "WARN":
                _log.warning("NPU validation: %s — %s", check.name, check.detail)

        if not npu_result.passed:
            raise NpuModelError(
                stage="validate",
                reason_code="NPU_VALIDATION_FAILED",
                message="Strict NPU validation failed. Bundle would not run on HTP.",
                hint="\n".join(
                    f"  [{c.status}] {c.name}: {c.detail}"
                    for c in npu_result.checks if c.status == "FAIL"
                ),
            )
        _log.info("STRICT NPU VALIDATION PASSED")

    # ---- 9. Optional: pack for Ollama ----
    pack_dir = _maybe_pack_ollama(bundle_dir, pack_ollama_name, pack_ollama_opts, out_dir)

    # ---- 9. Clean up work dir (unless --keep-work) ----
    if not keep_work and work.exists():
        shutil.rmtree(work)

    return ConvertResult(bundle_dir=bundle_dir, manifest_path=manifest_path, pack_dir=pack_dir)


def _maybe_pack_ollama(
    bundle_dir: Path,
    name: Optional[str],
    opts: Optional[dict[str, Any]],
    out_dir: Path,
) -> Path | None:
    if not name:
        return None
    from npu_model.packagers.ollama import pack_for_ollama
    opts = opts or {}
    result = pack_for_ollama(
        bundle_dir=bundle_dir,
        model_name=name,
        out_dir=out_dir / "ollama_publish",
        num_ctx=opts.get("num_ctx", 512),
        num_predict=opts.get("num_predict", 128),
    )
    return result.pack_dir
