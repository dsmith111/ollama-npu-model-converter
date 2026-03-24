"""npu-model doctor - preflight environment checker."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from rich import print as rprint
from rich.table import Table


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    remediation: str = ""
    warn: bool = False  # True = advisory only, not a hard failure


def _check_python() -> CheckResult:
    v = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 10)
    return CheckResult(
        name="Python",
        ok=ok,
        detail=f"{v} ({sys.executable})",
        remediation="" if ok else "Python >= 3.10 required.",
    )


def _check_os() -> CheckResult:
    info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    return CheckResult(name="OS / Arch", ok=True, detail=info)


def _check_package(
    import_name: str,
    pip_name: str | None = None,
    dist_name: str | None = None,
) -> CheckResult:
    pip_name = pip_name or import_name
    dist_name = dist_name or pip_name
    try:
        ver = version(dist_name)
        return CheckResult(name=import_name, ok=True, detail=f"v{ver}")
    except PackageNotFoundError:
        return CheckResult(
            name=import_name,
            ok=False,
            detail="not installed",
            remediation=f"pip install {pip_name}",
        )


def _probe_module_import(
    module: str,
    label: str,
    remediation: str,
    timeout: int = 15,
) -> CheckResult:
    code = f"import {module}; print(getattr({module}, '__version__', 'unknown'))"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as e:
        return CheckResult(
            name=label,
            ok=False,
            detail=f"probe failed ({type(e).__name__})",
            remediation=remediation,
        )

    if result.returncode == 0:
        ver = (result.stdout or "").strip() or "unknown"
        return CheckResult(name=label, ok=True, detail=f"import ok (v{ver})")

    err = (result.stderr or result.stdout or "").strip().splitlines()
    detail = err[-1] if err else f"exit code {result.returncode}"
    return CheckResult(
        name=label,
        ok=False,
        detail=f"import failed: {detail}",
        remediation=remediation,
    )


def _check_ort_providers() -> CheckResult:
    code = (
        "import json, onnxruntime as ort; "
        "print(json.dumps(ort.get_available_providers()))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        return CheckResult(
            name="ORT QNN EP",
            ok=False,
            detail=f"provider probe failed ({type(e).__name__})",
            remediation="pip install onnxruntime-qnn  (or ensure QNN libs on PATH)",
        )

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip().splitlines()
        detail = err[-1] if err else f"exit code {result.returncode}"
        return CheckResult(
            name="ORT QNN EP",
            ok=False,
            detail=f"provider probe failed: {detail}",
            remediation="pip install onnxruntime-qnn  (or ensure QNN libs on PATH)",
        )

    try:
        providers = json.loads((result.stdout or "").strip())
    except Exception:
        providers = []

    has_qnn = "QNNExecutionProvider" in providers
    return CheckResult(
        name="ORT QNN EP",
        ok=has_qnn,
        detail=f"providers: {', '.join(providers)}" if providers else "(none)",
        remediation="" if has_qnn else "pip install onnxruntime-qnn  (or ensure QNN libs on PATH)",
    )


def _check_olive() -> CheckResult:
    pkg = _check_package("olive-ai", "olive-ai[auto-opt]", dist_name="olive-ai")
    if not pkg.ok:
        return pkg
    probe = _probe_module_import(
        "olive",
        "olive import",
        "Install a compatible olive-ai build for this Python + architecture.",
    )
    if not probe.ok:
        return CheckResult(
            name="olive-ai",
            ok=False,
            detail=probe.detail,
            remediation=probe.remediation,
        )
    return CheckResult(name="olive-ai", ok=True, detail=pkg.detail)


def _check_machine_role() -> CheckResult:
    machine = (platform.machine() or "").lower()
    if machine in ("amd64", "x86_64"):
        return CheckResult(
            name="Machine Role",
            ok=True,
            detail="x64 host (recommended for export/quantize with Olive)",
            warn=True,
        )
    if "arm64" in machine or "aarch64" in machine:
        return CheckResult(
            name="Machine Role",
            ok=True,
            detail="ARM64 host (recommended for compile/runtime on device)",
            warn=True,
        )
    return CheckResult(
        name="Machine Role",
        ok=True,
        detail=f"{machine or 'unknown'} (unable to classify host role)",
        warn=True,
    )


def _check_genai_builder() -> CheckResult:
    try:
        ver = version("onnxruntime-genai")
    except PackageNotFoundError:
        return CheckResult(
            name="ORT GenAI Builder",
            ok=False,
            detail="not installed",
            remediation="pip install onnxruntime-genai",
        )

    try:
        result = subprocess.run(
            [sys.executable, "-m", "onnxruntime_genai.models.builder", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        return CheckResult(
            name="ORT GenAI Builder",
            ok=False,
            detail=f"probe failed ({type(e).__name__})",
            remediation="Install a compatible onnxruntime-genai build for this Python + architecture.",
        )

    if result.returncode == 0:
        return CheckResult(name="ORT GenAI Builder", ok=True, detail=f"available (v{ver})")

    err = (result.stderr or result.stdout or "").strip().splitlines()
    detail = err[-1] if err else f"exit code {result.returncode}"
    return CheckResult(
        name="ORT GenAI Builder",
        ok=False,
        detail=f"unavailable: {detail}",
        remediation="Install a compatible onnxruntime-genai build for this Python + architecture.",
    )


def _check_env_var(name: str, hint: str, *, required: bool = True) -> CheckResult:
    val = os.environ.get(name)
    if val:
        from pathlib import Path

        exists = Path(val).is_dir()
        return CheckResult(
            name=name,
            ok=exists,
            detail=val if exists else f"{val} (NOT A DIRECTORY)",
            remediation="" if exists else f"Set {name} to a valid directory.",
        )
    if required:
        return CheckResult(
            name=name,
            ok=False,
            detail="not set",
            remediation=hint,
        )
    return CheckResult(
        name=name,
        ok=True,
        warn=True,
        detail="not set (optional)",
        remediation=hint,
    )


def _check_registry() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        from npu_model.core.registry import Registry

        reg = Registry.load()
        for label, group in [
            ("Adapters", reg.adapters),
            ("Backends", reg.backends),
            ("Runtime Formats", reg.runtime_formats),
            ("Quantizers", reg.quantizers),
        ]:
            names = sorted(group.keys())
            results.append(CheckResult(
                name=f"Registry: {label}",
                ok=len(names) > 0,
                detail=", ".join(names) if names else "(none)",
            ))
    except Exception as e:
        results.append(CheckResult(
            name="Registry",
            ok=False,
            detail=f"Failed to load: {e}",
            remediation="pip install -e '.[dev]'  (reinstall the package)",
        ))
    return results


def run_doctor() -> list[CheckResult]:
    """Run all preflight checks and return results."""
    checks: list[CheckResult] = []

    # System
    checks.append(_check_os())
    checks.append(_check_python())

    # Core deps
    checks.append(_check_package("typer"))
    checks.append(_check_package("rich"))
    checks.append(_check_package("huggingface_hub", "huggingface_hub", dist_name="huggingface_hub"))

    # Export deps
    checks.append(_check_package("torch"))
    checks.append(_check_package("transformers"))
    checks.append(_check_package("onnx", "onnx"))
    checks.append(_check_package("onnx_ir", "onnx-ir", dist_name="onnx-ir"))
    checks.append(_check_package("onnxruntime_genai", "onnxruntime-genai", dist_name="onnxruntime-genai"))
    checks.append(_probe_module_import(
        "onnxruntime_genai",
        "onnxruntime_genai import",
        "Reinstall matching onnxruntime-genai / onnxruntime wheels for this Python + architecture.",
    ))
    checks.append(_check_genai_builder())
    checks.append(_check_olive())

    # Backend deps
    checks.append(_check_package("onnxruntime", "onnxruntime", dist_name="onnxruntime"))
    checks.append(_probe_module_import(
        "onnxruntime",
        "onnxruntime import",
        "Reinstall matching onnxruntime / onnxruntime-qnn wheels for this Python + architecture.",
    ))
    checks.append(_check_ort_providers())
    checks.append(_check_machine_role())
    checks.append(_check_env_var(
        "QNN_SDK_ROOT",
        "Set QNN_SDK_ROOT to your Qualcomm AI Engine SDK directory.",
        required=False,
    ))

    # Registry
    checks.extend(_check_registry())

    return checks


def print_doctor_report(checks: list[CheckResult]) -> bool:
    """Print a formatted doctor report. Returns True if all checks pass."""
    import npu_model

    rprint(f"[bold]npu-model doctor[/bold]  v{npu_model.__version__}\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", min_width=25)
    table.add_column("Status", min_width=6)
    table.add_column("Detail")
    table.add_column("Fix")

    all_ok = True
    for c in checks:
        if c.warn:
            status = "[yellow]WARN[/yellow]"
        elif c.ok:
            status = "[green]OK[/green]"
        else:
            status = "[red]FAIL[/red]"
        if not c.ok and not c.warn:
            all_ok = False
        table.add_row(c.name, status, c.detail, c.remediation)

    rprint(table)
    rprint()

    if all_ok:
        rprint("[green]All checks passed.[/green]")
    else:
        failed = [c for c in checks if not c.ok]
        rprint(f"[yellow]{len(failed)} check(s) need attention.[/yellow]")
        needs_export = any(
            c.name in (
                "torch",
                "transformers",
                "onnxruntime_genai",
                "onnxruntime_genai import",
                "ORT GenAI Builder",
            )
            and not c.ok
            for c in checks
        )
        needs_qnn = any(c.name in ("onnxruntime import", "ORT QNN EP") and not c.ok for c in checks)
        if needs_export:
            rprint("  [bold]For export support:[/bold]  pip install npu-model\\[export]")
        if needs_qnn:
            rprint("  [bold]For QNN backend:[/bold]   pip install npu-model\\[qnn]")

    return all_ok

