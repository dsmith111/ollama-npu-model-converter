from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from npu_model.core.registry import Registry
from npu_model.core.pipeline import explain_plan, convert_model
from npu_model.core.errors import NpuModelError


app = typer.Typer(no_args_is_help=True, add_completion=False)
handoff_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(handoff_app, name="handoff")
registry = Registry.load()


def _handle_err(e: Exception) -> None:
    if isinstance(e, NpuModelError):
        rprint(f"[bold red]Error[/bold red] [{e.stage}/{e.reason_code}]: {e}")
        if e.hint:
            rprint(f"[yellow]Hint:[/yellow] {e.hint}")
        raise typer.Exit(code=2)
    raise e


# ---------------------------------------------------------------------------
# handoff
# ---------------------------------------------------------------------------

@handoff_app.command("export")
def handoff_export(
    input: Path = typer.Option(..., "--input", help="Path to handoff bundle directory"),
    out: Path = typer.Option(..., "--out", help="Output .zip path"),
) -> None:
    """Export a handoff bundle directory into a .zip for transfer."""
    try:
        from npu_model.core.handoff import export_handoff_zip

        zip_path = export_handoff_zip(input, out)
        rprint(f"[green]OK[/green] handoff zip: {zip_path}")
    except Exception as e:
        _handle_err(e)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@app.command("doctor")
def doctor() -> None:
    """Run preflight environment checks and show actionable remediation."""
    from npu_model.cli.doctor import run_doctor, print_doctor_report

    checks = run_doctor()
    all_ok = print_doctor_report(checks)
    if not all_ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# list-* commands
# ---------------------------------------------------------------------------

@app.command("list-backends")
def list_backends() -> None:
    """List available backend plugins."""
    for b in registry.backends.keys():
        rprint(b)


@app.command("list-adapters")
def list_adapters() -> None:
    """List available model adapter plugins."""
    for a in registry.adapters.keys():
        rprint(a)


@app.command("list-runtime-formats")
def list_runtime_formats() -> None:
    """List available runtime format plugins."""
    for rf in registry.runtime_formats.keys():
        rprint(rf)


@app.command("list-quantizers")
def list_quantizers() -> None:
    """List available quantization strategy plugins."""
    for q in registry.quantizers.keys():
        rprint(q)


@app.command("list-targets")
def list_targets(
    backend: str = typer.Option("qnn", "--backend", help="Backend id to list targets for"),
) -> None:
    """List available targets for a backend."""
    b = registry.backends.get(backend)
    if b is None:
        rprint(f"[bold red]Error[/bold red]: Unknown backend: {backend}")
        raise typer.Exit(code=2)
    ts = b.resolve_target("auto", env={})
    rprint(f"[bold]{ts.name}[/bold]")
    for k, v in sorted(ts.params.items()):
        rprint(f"  {k} = {v}")


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

@app.command("explain")
def explain(
    input: str = typer.Option(..., "--input", help="Model source: local path or hf:<repo>[@rev]"),
    backend: str = typer.Option("qnn", "--backend", help="Backend id (e.g. qnn)"),
    target: str = typer.Option("auto", "--target", help="Target selection: auto or backend-specific"),
    runtime: str = typer.Option("ort-genai-folder", "--runtime", help="Runtime format id"),
    mode: str = typer.Option("export", "--mode", help="Convert mode: export or prebuilt-ort-genai"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Cache directory for HF downloads"),
) -> None:
    """Show what the converter will do (adapter/backend/runtime selection) without converting."""
    try:
        plan = explain_plan(
            input_spec=input,
            backend_id=backend,
            target=target,
            runtime_format_id=runtime,
            cache_dir=cache_dir,
            registry=registry,
            mode=mode,
        )
        rprint(plan.to_rich_text())
    except Exception as e:
        _handle_err(e)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

@app.command("convert")
def convert(
    input: str = typer.Option(..., "--input", help="Model source: local path or hf:<repo>[@rev]"),
    out: Path = typer.Option(..., "--out", help="Output directory"),
    backend: str = typer.Option("qnn", "--backend", help="Backend id (e.g. qnn)"),
    target: str = typer.Option("auto", "--target", help="Target selection: auto or backend-specific"),
    runtime: str = typer.Option("ort-genai-folder", "--runtime", help="Runtime format id"),
    quant: Optional[str] = typer.Option(None, "--quant", help="Quantization strategy (auto-selected for NPU)"),
    mode: str = typer.Option("export", "--mode", help="Convert mode: export or prebuilt-ort-genai"),
    precision: str = typer.Option("int4", "--precision", help="Export precision: int4, fp16, fp32, bf16"),
    execution_provider: str = typer.Option("cpu", "--execution-provider", help="Export EP: cpu, cuda, dml"),
    compile_strategy: Optional[str] = typer.Option(
        None, "--compile-strategy",
        help="Backend compile strategy (auto-selected for NPU)",
    ),
    ollama_name: Optional[str] = typer.Option(
        None, "--pack-ollama",
        help="Pack for Ollama with this model name. Auto-selects NPU defaults.",
    ),
    num_ctx: int = typer.Option(512, "--num-ctx", help="Context length for Ollama Modelfile / shape fixing"),
    num_predict: int = typer.Option(128, "--num-predict", help="num_predict for Ollama Modelfile"),
    calib_source: str = typer.Option(
        "builtin:mixed_small", "--calib-source",
        help="Calibration prompt source (builtin:mixed_small, builtin:instruct_small, etc.)",
    ),
    calib_prompts: Optional[Path] = typer.Option(
        None, "--calib-prompts", help="Path to calibration prompts file (overrides --calib-source)",
    ),
    calib_samples: int = typer.Option(64, "--calib-samples", help="Number of calibration samples"),
    calib_maxlen: int = typer.Option(256, "--calib-maxlen", help="Max sequence length for calibration"),
    stop_after: Optional[str] = typer.Option(
        None, "--stop-after",
        help="Stop after stage: export or quantize (produces handoff bundle for staged workflows)",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force rebuild, ignore conversion cache"),
    keep_work: bool = typer.Option(False, "--keep-work", help="Keep intermediate _work directory after conversion"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Cache directory for HF downloads"),
    olive_python: Optional[Path] = typer.Option(
        None,
        "--olive-python",
        help="External Python interpreter used for Olive runs (fallback: NPU_MODEL_OLIVE_PYTHON).",
    ),
) -> None:
    """Convert a model into a runtime bundle.

    When --pack-ollama is provided, auto-selects NPU-optimized defaults:
      backend=qnn, quant=auto-family (olive-qnn-llm for supported LLMs), compile-strategy=context-cache
    """
    # ---- NPU preset: auto-select defaults when --pack-ollama is provided ----
    effective_quant = quant
    effective_compile = compile_strategy
    quantizer_was_auto = False

    if ollama_name and mode != "prebuilt-ort-genai":
        if effective_quant is None:
            quantizer_was_auto = True
            rprint(
                "[dim]Auto-selected:[/dim] --quant auto "
                "(olive-qnn-llm for supported LLM families; qnn-qdq otherwise)"
            )
        if effective_compile is None:
            effective_compile = "context-cache"
            rprint("[dim]Auto-selected:[/dim] --compile-strategy context-cache (NPU requires compiled artifacts)")

    # Defaults for non-NPU paths
    if effective_quant is None and not quantizer_was_auto:
        effective_quant = "passthrough"
    if effective_compile is None:
        effective_compile = "passthrough"

    try:
        result = convert_model(
            input_spec=input,
            out_dir=out,
            backend_id=backend,
            target=target,
            runtime_format_id=runtime,
            quantizer_id=effective_quant,
            quantizer_was_auto=quantizer_was_auto,
            cache_dir=cache_dir,
            registry=registry,
            mode=mode,
            compile_config={"strategy": effective_compile},
            export_options={
                "precision": precision,
                "execution_provider": execution_provider,
                "calib_source": calib_source,
                "calib_prompts_file": calib_prompts,
                "calib_samples": calib_samples,
                "calib_maxlen": calib_maxlen,
                "olive_python": str(olive_python) if olive_python else None,
            },
            pack_ollama_name=ollama_name if not stop_after else None,
            pack_ollama_opts={"num_ctx": num_ctx, "num_predict": num_predict} if ollama_name and not stop_after else None,
            use_cache=not no_cache,
            keep_work=keep_work or bool(stop_after),
            stop_after=stop_after,
        )
        rprint(f"[green]OK[/green] bundle: {result.bundle_dir}")
        rprint(f"[green]OK[/green] manifest: {result.manifest_path}")
        if stop_after:
            rprint()
            rprint(f"[bold]Stopped after:[/bold] {stop_after}")
            rprint(f"[bold]Handoff bundle:[/bold] {result.bundle_dir}")
            rprint()
            rprint("[bold]Next steps (on target device):[/bold]")
            rprint(f"  npu-model compile-context --input {result.bundle_dir} --out .\\compiled")
            if ollama_name:
                rprint(f"  npu-model pack-ollama --input .\\compiled --name {ollama_name} --out .\\publish")
        elif result.pack_dir:
            rprint(f"[green]OK[/green] ollama publish: {result.pack_dir}")
            rprint()
            rprint("[bold]Ready to publish:[/bold]")
            rprint(f"  cd {result.pack_dir}")
            rprint(f"  ollama create {ollama_name} -f Modelfile")
            rprint(f"  ollama push {ollama_name}")
    except Exception as e:
        _handle_err(e)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@app.command("validate")
def validate(
    input: Path = typer.Option(
        ..., "--input", help="Path to bundle or publish directory to validate",
    ),
    as_mode: Optional[str] = typer.Option(
        None, "--as",
        help="Validation mode: ollama-ortgenai, genai-config, runtime-load, or omit for basic layout",
    ),
    bundle: Optional[Path] = typer.Option(
        None, "--bundle", help="(deprecated, use --input) Path to runtime bundle directory",
    ),
    backend: str = typer.Option("qnn", "--backend", help="Backend id"),
    target: str = typer.Option("auto", "--target", help="Target selection"),
    prompt: str = typer.Option("Hello", "--prompt", help="Test prompt"),
    max_tokens: int = typer.Option(8, "--max-tokens", help="Max tokens to generate"),
) -> None:
    """Validate a runtime bundle or publish directory."""
    check_dir = input or bundle
    if check_dir is None:
        rprint("[bold red]Error[/bold red]: --input is required.")
        raise typer.Exit(code=2)

    try:
        if as_mode == "ollama-ortgenai":
            from npu_model.runtime_formats.ort_genai_folder import validate_ollama_ortgenai_dir

            result = validate_ollama_ortgenai_dir(check_dir)
            if result.errors:
                for err in result.errors:
                    rprint(f"[bold red]ERROR[/bold red]: {err}")
                for w in result.warnings:
                    rprint(f"[yellow]WARN[/yellow]: {w}")
                raise typer.Exit(code=2)
            for w in result.warnings:
                rprint(f"[yellow]WARN[/yellow]: {w}")
            rprint("[green]OK[/green] ollama-ortgenai validation passed.")

        elif as_mode == "genai-config":
            from npu_model.runtime_formats.ort_genai_folder import validate_genai_config

            result = validate_genai_config(check_dir)
            if result.errors:
                for err in result.errors:
                    rprint(f"[bold red]ERROR[/bold red]: {err}")
                for w in result.warnings:
                    rprint(f"[yellow]WARN[/yellow]: {w}")
                raise typer.Exit(code=2)
            for w in result.warnings:
                rprint(f"[yellow]WARN[/yellow]: {w}")
            rprint("[green]OK[/green] genai-config validation passed.")

        elif as_mode == "runtime-load":
            _validate_runtime_load(check_dir, prompt, max_tokens)

        elif as_mode in ("strict-npu", "npu-ready"):
            from npu_model.validate.npu_strict import validate_npu_strict

            npu_result = validate_npu_strict(check_dir)
            for check in npu_result.checks:
                if check.status == "FAIL":
                    rprint(f"[bold red]FAIL[/bold red]: {check.name} — {check.detail}")
                elif check.status == "WARN":
                    rprint(f"[yellow]WARN[/yellow]: {check.name} — {check.detail}")
                elif check.status == "SKIP":
                    rprint(f"[dim]SKIP[/dim]: {check.name} — {check.detail}")
                else:
                    rprint(f"[green]OK[/green]: {check.name} — {check.detail}")
            if not npu_result.passed:
                raise typer.Exit(code=2)
            rprint("[green]STRICT NPU VALIDATION PASSED[/green]")

        else:
            rt = registry.runtime_formats.get("ort-genai-folder")
            if rt is not None:
                rt.validate_layout(check_dir)
                rprint("[green]OK[/green] layout validation passed.")
            else:
                rprint("[yellow]Warning:[/yellow] No runtime format plugin to validate against.")
    except typer.Exit:
        raise
    except Exception as e:
        _handle_err(e)


def _validate_runtime_load(check_dir: Path, prompt: str, max_tokens: int) -> None:
    """Optional runtime load validation — requires onnxruntime-genai."""
    try:
        import onnxruntime_genai as og
    except ImportError:
        rprint(
            "[yellow]WARN[/yellow]: onnxruntime-genai not installed. "
            "Skipping runtime load validation.\n"
            "  Install with: pip install onnxruntime-genai"
        )
        return

    try:
        model = og.Model(str(check_dir))
        tokenizer = og.Tokenizer(model)
        tokens = tokenizer.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_tokens)
        params.input_ids = tokens
        output = model.generate(params)
        decoded = tokenizer.decode(output[0])
        rprint(f"[green]OK[/green] runtime load + generate succeeded")
        rprint(f"  prompt: {prompt!r}")
        rprint(f"  output: {decoded!r}")
    except Exception as e:
        rprint(f"[bold red]ERROR[/bold red]: Runtime load failed: {e}")
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# compile-context (staged workflow: takes handoff bundle → produces ctx artifacts)
# ---------------------------------------------------------------------------

@app.command("compile-context")
def compile_context(
    input: Path = typer.Option(
        ..., "--input", help="Path to handoff bundle directory or .zip",
    ),
    out: Path = typer.Option(..., "--out", help="Output directory for compiled artifacts"),
    backend: str = typer.Option("qnn", "--backend", help="Backend id"),
    target: str = typer.Option("auto", "--target", help="Target selection"),
    allow_experimental: bool = typer.Option(
        False, "--allow-experimental", help="Allow experimental generic LLM compile path",
    ),
) -> None:
    """Generate QNN context-cache artifacts from a handoff bundle.

    Use this when quantization was done on a separate (big-RAM) machine.
    Input should be a handoff bundle produced by: convert --stop-after quantize
    """
    try:
        from npu_model.core.handoff import load_handoff_input, validate_handoff_for_compile

        graphs, metadata = load_handoff_input(input)

        if not graphs.graphs:
            rprint("[bold red]Error[/bold red]: No ONNX graphs found in handoff bundle.")
            raise typer.Exit(code=2)

        # Validate the handoff bundle is compatible with context-cache
        validate_handoff_for_compile(
            metadata,
            compile_strategy="context-cache",
            allow_experimental=allow_experimental,
        )

        rprint(f"[dim]Loaded handoff bundle:[/dim] {len(graphs.graphs)} graph(s)")
        for name, p in graphs.graphs.items():
            rprint(f"  {name}: {p.name}")

        backend_plugin = registry.backends.get(backend)
        if backend_plugin is None:
            rprint(f"[bold red]Error[/bold red]: Unknown backend: {backend}")
            raise typer.Exit(code=2)

        import os
        ts = backend_plugin.resolve_target(target, env=dict(os.environ))

        rprint(f"[dim]Compiling with:[/dim] {ts.normalized_repr()}")

        # Show model size before compilation
        total_bytes = 0
        for gname, gpath in graphs.graphs.items():
            if gpath.exists():
                total_bytes += gpath.stat().st_size
                data = gpath.parent / f"{gpath.name}.data"
                if data.exists():
                    total_bytes += data.stat().st_size
        rprint(f"[dim]Model size:[/dim] {total_bytes / (1024**3):.1f} GB")

        out = out.expanduser().resolve()
        prepared = backend_plugin.compile(
            graphs, out,
            target=ts,
            compile_config={"strategy": "context-cache"},
        )

        # Assemble output dir with ctx graphs + tokenizer + extras
        import shutil
        final_dir = out / "compiled_bundle"
        final_dir.mkdir(parents=True, exist_ok=True)

        for name, p in prepared.graphs.items():
            shutil.copy2(p, final_dir / p.name)
        for p in sorted(prepared.artifacts_dir.glob("*.bin")):
            shutil.copy2(p, final_dir / p.name)
        if graphs.tokenizer_dir.exists():
            for p in sorted(graphs.tokenizer_dir.glob("*")):
                if p.is_file():
                    shutil.copy2(p, final_dir / p.name)
        for p in graphs.extra_files:
            if p.exists() and p.is_file():
                if p.name.endswith(".onnx.data"):
                    continue
                dst = final_dir / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)

        if list(final_dir.rglob("*.onnx.data")):
            rprint("[bold red]Error[/bold red]: Compiled bundle still contains .onnx.data files.")
            raise typer.Exit(code=2)

        from npu_model.validate.npu_strict import validate_npu_strict
        npu_result = validate_npu_strict(final_dir)
        if not npu_result.passed:
            fail_details = [
                f"  - {c.name}: {c.detail}"
                for c in npu_result.checks
                if c.status == "FAIL"
            ]
            rprint("[bold red]Error[/bold red]: Strict NPU validation failed after compile.")
            for d in fail_details:
                rprint(d)
            raise typer.Exit(code=2)

        rprint(f"[green]OK[/green] compiled bundle: {final_dir}")
        rprint()
        rprint("[bold]Next:[/bold]")
        rprint(f"  npu-model pack-ollama --input {final_dir} --name <tag> --out <publish_dir>")
    except typer.Exit:
        raise
    except Exception as e:
        _handle_err(e)


# ---------------------------------------------------------------------------
# pack-ollama (standalone)
# ---------------------------------------------------------------------------

@app.command("pack-ollama")
def pack_ollama(
    input: Path = typer.Option(
        ..., "--input", help="Path to ORT GenAI bundle directory",
    ),
    name: str = typer.Option(
        ..., "--name", help="Ollama model name, e.g. dsmith111/phi3:mini-qnn",
    ),
    out: Path = typer.Option(
        ..., "--out", help="Output directory for Ollama publish",
    ),
    num_ctx: int = typer.Option(512, "--num-ctx", help="num_ctx parameter for Modelfile"),
    num_predict: int = typer.Option(128, "--num-predict", help="num_predict parameter for Modelfile"),
) -> None:
    """Create an Ollama-publishable directory (Modelfile + allowlisted files)."""
    try:
        from npu_model.packagers.ollama import pack_for_ollama

        result = pack_for_ollama(
            bundle_dir=input,
            model_name=name,
            out_dir=out,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        rprint(f"[green]OK[/green] {result.file_count} files -> {result.pack_dir}")
        rprint()
        rprint("[bold]Ready to publish:[/bold]")
        rprint(f"  cd {result.pack_dir}")
        rprint(f"  ollama create {name} -f Modelfile")
        rprint(f"  ollama push {name}")
    except Exception as e:
        _handle_err(e)


# ---------------------------------------------------------------------------
# publish (thin wrapper: convert + pack with NPU defaults)
# ---------------------------------------------------------------------------

@app.command("publish")
def publish(
    input: str = typer.Option(..., "--input", help="Model source: local path or hf:<repo>[@rev]"),
    name: str = typer.Option(..., "--name", help="Ollama model name, e.g. dsmith111/phi3:mini-qnn"),
    out: Path = typer.Option(..., "--out", help="Output directory"),
    mode: str = typer.Option("export", "--mode", help="Convert mode: export or prebuilt-ort-genai"),
    num_ctx: int = typer.Option(512, "--num-ctx", help="Context length"),
    num_predict: int = typer.Option(128, "--num-predict", help="Max predict tokens"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force rebuild"),
    keep_work: bool = typer.Option(False, "--keep-work", help="Keep intermediate files"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Cache directory for HF downloads"),
    olive_python: Optional[Path] = typer.Option(
        None,
        "--olive-python",
        help="External Python interpreter used for Olive runs (fallback: NPU_MODEL_OLIVE_PYTHON).",
    ),
) -> None:
    """One-command HF -> NPU -> Ollama publish.

    Equivalent to: convert --pack-ollama <name> with NPU defaults auto-applied.
    """
    try:
        rprint("[bold]npu-model publish[/bold]")
        rprint(f"  input: {input}")
        rprint(f"  name:  {name}")
        rprint(f"  mode:  {mode}")
        rprint()

        effective_quant = "passthrough" if mode == "prebuilt-ort-genai" else None
        effective_compile = "passthrough" if mode == "prebuilt-ort-genai" else "context-cache"
        quantizer_was_auto = mode != "prebuilt-ort-genai"

        result = convert_model(
            input_spec=input,
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id=effective_quant,
            quantizer_was_auto=quantizer_was_auto,
            cache_dir=cache_dir,
            registry=registry,
            mode=mode,
            compile_config={"strategy": effective_compile},
            export_options={"olive_python": str(olive_python) if olive_python else None},
            pack_ollama_name=name,
            pack_ollama_opts={"num_ctx": num_ctx, "num_predict": num_predict},
            use_cache=not no_cache,
            keep_work=keep_work,
        )
        rprint(f"[green]OK[/green] bundle: {result.bundle_dir}")
        rprint(f"[green]OK[/green] manifest: {result.manifest_path}")
        if result.pack_dir:
            rprint(f"[green]OK[/green] ollama publish: {result.pack_dir}")
            rprint()
            rprint("[bold]Next steps:[/bold]")
            rprint(f"  cd {result.pack_dir}")
            rprint(f"  ollama create {name} -f Modelfile")
            rprint(f"  ollama push {name}")
    except Exception as e:
        _handle_err(e)
