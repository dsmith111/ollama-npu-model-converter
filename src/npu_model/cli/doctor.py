"""npu-model doctor — preflight environment checker."""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

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


def _check_package(pkg: str, pip_name: str | None = None) -> CheckResult:
    pip_name = pip_name or pkg
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        return CheckResult(name=pkg, ok=True, detail=f"v{ver}")
    except ImportError:
        return CheckResult(
            name=pkg,
            ok=False,
            detail="not installed",
            remediation=f"pip install {pip_name}",
        )


def _check_ort_providers() -> CheckResult:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_qnn = "QNNExecutionProvider" in providers
        return CheckResult(
            name="ORT QNN EP",
            ok=has_qnn,
            detail=f"providers: {', '.join(providers)}",
            remediation="" if has_qnn else "pip install onnxruntime-qnn  (or ensure QNN libs on PATH)",
        )
    except ImportError:
        return CheckResult(
            name="ORT QNN EP",
            ok=False,
            detail="onnxruntime not installed",
            remediation="pip install onnxruntime  (or onnxruntime-qnn for QNN support)",
        )


def _check_genai_builder() -> CheckResult:
    try:
        import onnxruntime_genai  # noqa: F401
    except ImportError:
        return CheckResult(
            name="ORT GenAI Builder",
            ok=False,
            detail="not installed",
            remediation="pip install onnxruntime-genai",
        )

    # Try importing the builder — catch both "module not found" (builder absent)
    # and transitive import errors (builder exists but a dependency like onnx_ir is missing).
    builder_found = False
    missing_dep: str | None = None
    for import_path in (
        "onnxruntime_genai.models.builder",
        "onnxruntime_genai.models",
    ):
        try:
            __import__(import_path)
            builder_found = True
            break
        except ModuleNotFoundError as e:
            # Distinguish "builder module missing" from "builder found but needs onnx_ir"
            missing_name = getattr(e, "name", None) or ""
            if missing_name and missing_name != import_path and not import_path.startswith(missing_name):
                # A transitive dep is missing (e.g. onnx_ir)
                missing_dep = missing_name
                builder_found = True  # builder exists, just can't load
                break
        except ImportError:
            continue

    if missing_dep:
        pip_name = missing_dep.replace("_", "-")
        return CheckResult(
            name="ORT GenAI Builder",
            ok=False,
            detail=f"builder found but missing dependency: {missing_dep}",
            remediation=f"pip install {pip_name}  (or: pip install npu-model\\[export])",
        )

    if not builder_found:
        # Also check if the CLI entrypoint works
        import subprocess, sys
        try:
            result = subprocess.run(
                [sys.executable, "-m", "onnxruntime_genai.models.builder", "--help"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                builder_found = True
        except Exception:
            pass

    if builder_found:
        return CheckResult(name="ORT GenAI Builder", ok=True, detail="available")

    ver = getattr(onnxruntime_genai, "__version__", "unknown")
    return CheckResult(
        name="ORT GenAI Builder",
        ok=False,
        detail=f"onnxruntime_genai v{ver} installed but builder module not found",
        remediation=(
            "Builder may not be included in this build/version. "
            "Try: pip install -U onnxruntime-genai  or check "
            "https://github.com/microsoft/onnxruntime-genai for ARM64 builder support."
        ),
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
    checks.append(_check_package("huggingface_hub", "huggingface_hub"))

    # Export deps
    checks.append(_check_package("torch"))
    checks.append(_check_package("transformers"))
    checks.append(_check_package("onnx", "onnx"))
    checks.append(_check_package("onnx_ir", "onnx-ir"))
    checks.append(_check_package("onnxruntime_genai", "onnxruntime-genai"))
    checks.append(_check_genai_builder())

    # Backend deps
    checks.append(_check_package("onnxruntime", "onnxruntime"))
    checks.append(_check_ort_providers())
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
        # Print install extras summary
        needs_export = any(c.name in ("torch", "transformers", "onnxruntime_genai", "ORT GenAI Builder")
                          and not c.ok for c in checks)
        needs_qnn = any(c.name in ("ORT QNN EP",) and not c.ok for c in checks)
        if needs_export:
            rprint("  [bold]For export support:[/bold]  pip install npu-model\\[export]")
        if needs_qnn:
            rprint("  [bold]For QNN backend:[/bold]   pip install npu-model\\[qnn]")

    return all_ok
