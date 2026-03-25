from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from npu_model.core.errors import NpuModelError


@dataclass(frozen=True)
class OliveEnvReport:
    python_exe: Path
    version: tuple[int, int, int]
    machine: str
    olive_installed: bool
    olive_version: str | None
    runtime_import_checked: bool = False
    runtime_import_ok: bool | None = None
    runtime_import_error: str | None = None

    @property
    def version_str(self) -> str:
        major, minor, patch = self.version
        return f"{major}.{minor}.{patch}"

    @property
    def is_x64(self) -> bool:
        m = self.machine.lower()
        return m in {"x86_64", "amd64"}


def probe_olive_python(
    python_exe: Path,
    *,
    deep_import_probe: bool = False,
    deep_probe_timeout: int = 60,
) -> OliveEnvReport:
    """Fast probe for Olive host interpreter compatibility.

    This probe is intentionally lightweight: it checks interpreter arch/version
    and whether the ``olive-ai`` distribution is installed via metadata.
    It does NOT import olive or onnxruntime. Runtime validation is deferred to
    the actual runner invocation.
    """
    python_exe = python_exe.expanduser().resolve()
    if not python_exe.exists():
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_PYTHON_NOT_FOUND",
            message=f"Olive Python interpreter not found: {python_exe}",
            hint="Pass a valid interpreter path via --olive-python or NPU_MODEL_OLIVE_PYTHON.",
        )

    probe_script = (
        "from importlib.metadata import PackageNotFoundError, version\n"
        "import json, platform, sys\n"
        "result = {\n"
        "  'version': [sys.version_info[0], sys.version_info[1], sys.version_info[2]],\n"
        "  'machine': platform.machine(),\n"
        "  'olive_installed': False,\n"
        "  'olive_version': None,\n"
        "}\n"
        "try:\n"
        "  result['olive_version'] = version('olive-ai')\n"
        "  result['olive_installed'] = True\n"
        "except PackageNotFoundError:\n"
        "  pass\n"
        "print(json.dumps(result))\n"
    )

    try:
        proc = subprocess.run(
            [str(python_exe), "-c", probe_script],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as e:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_ENV_PROBE_TIMEOUT",
            message=f"Timed out probing Olive interpreter: {python_exe}",
            cause=e,
        ) from e
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "")[-1000:]
        stdout_tail = (e.stdout or "")[-1000:]
        detail = stderr_tail or stdout_tail or "No output captured."
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_ENV_PROBE_FAILED",
            message=f"Failed probing Olive interpreter: {python_exe}",
            hint=detail,
            cause=e,
        ) from e

    try:
        payload = json.loads((proc.stdout or "").strip())
    except Exception as e:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_ENV_PROBE_INVALID_OUTPUT",
            message=f"Unexpected probe output from Olive interpreter: {python_exe}",
            hint=(proc.stdout or proc.stderr or "").strip()[:1200],
            cause=e,
        ) from e

    version_list = payload.get("version") or [0, 0, 0]
    version = (int(version_list[0]), int(version_list[1]), int(version_list[2]))
    report = OliveEnvReport(
        python_exe=python_exe,
        version=version,
        machine=str(payload.get("machine") or "unknown"),
        olive_installed=bool(payload.get("olive_installed")),
        olive_version=payload.get("olive_version"),
    )

    if report.version >= (3, 14, 0):
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_UNSUPPORTED_PYTHON",
            message=(
                "Olive must run under Python 3.13 or lower. "
                f"Selected interpreter is Python {report.version_str}: {report.python_exe}"
            ),
            hint=(
                "Create a Python 3.13 x64 venv and pass it explicitly:\n"
                "  npu-model convert ... --quant olive-qnn-llm --olive-python <path-to-python-3.13>"
            ),
        )

    if not report.olive_installed:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_NOT_INSTALLED",
            message=f"'olive-ai' is not installed in external interpreter: {report.python_exe}",
            hint=(
                "Install Olive in that interpreter:\n"
                f"  {report.python_exe} -m pip install olive-ai[auto-opt]"
            ),
        )

    if deep_import_probe:
        deep_code = (
            "import olive\n"
            "print('ok')\n"
        )
        try:
            deep = subprocess.run(
                [str(python_exe), "-c", deep_code],
                capture_output=True,
                text=True,
                timeout=deep_probe_timeout,
            )
        except Exception as e:
            return OliveEnvReport(
                python_exe=report.python_exe,
                version=report.version,
                machine=report.machine,
                olive_installed=report.olive_installed,
                olive_version=report.olive_version,
                runtime_import_checked=True,
                runtime_import_ok=False,
                runtime_import_error=f"{type(e).__name__}: {e}",
            )

        if deep.returncode == 0:
            return OliveEnvReport(
                python_exe=report.python_exe,
                version=report.version,
                machine=report.machine,
                olive_installed=report.olive_installed,
                olive_version=report.olive_version,
                runtime_import_checked=True,
                runtime_import_ok=True,
                runtime_import_error=None,
            )

        err = (deep.stderr or deep.stdout or "").strip().splitlines()
        detail = err[-1] if err else f"exit code {deep.returncode}"
        return OliveEnvReport(
            python_exe=report.python_exe,
            version=report.version,
            machine=report.machine,
            olive_installed=report.olive_installed,
            olive_version=report.olive_version,
            runtime_import_checked=True,
            runtime_import_ok=False,
            runtime_import_error=detail,
        )

    return report
