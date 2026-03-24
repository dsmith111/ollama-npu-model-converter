from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from npu_model.core.errors import NpuModelError


def _olive_command(config_path: Path) -> list[str]:
    # Prefer module execution for virtual-env consistency.
    return [sys.executable, "-m", "olive", "run", "--config", str(config_path)]


def run_olive_cli(*, config_path: Path, work_dir: Path, timeout_s: int = 14_400) -> None:
    """Run Olive as an external CLI process."""
    if shutil.which(sys.executable) is None:
        raise NpuModelError(
            stage="quant",
            reason_code="PYTHON_NOT_FOUND",
            message="Python executable not available for Olive CLI invocation.",
        )

    cmd = _olive_command(config_path)
    try:
        subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_CLI_NOT_FOUND",
            message="Unable to invoke Olive CLI.",
            hint="Install Olive: pip install olive-ai[auto-opt]",
            cause=e,
        ) from e
    except subprocess.TimeoutExpired as e:
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_RUN_TIMEOUT",
            message=f"Olive run timed out after {timeout_s} seconds.",
            hint=f"Config: {config_path}",
            cause=e,
        ) from e
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "")[-3000:]
        stdout_tail = (e.stdout or "")[-1500:]
        detail = stderr_tail or stdout_tail or "No output captured."
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_RUN_FAILED",
            message=f"Olive run failed (exit {e.returncode}).",
            hint=f"Config: {config_path}\nOutput tail:\n{detail}",
            cause=e,
        ) from e

