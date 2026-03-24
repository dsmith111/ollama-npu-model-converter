from __future__ import annotations

import subprocess
from pathlib import Path

from npu_model.core.errors import NpuModelError


def _wrapper_command(python_exe: Path, config_path: Path) -> list[str]:
    wrapper_path = Path(__file__).resolve().parent / "run_wrapper.py"
    return [str(python_exe), str(wrapper_path), "--config", str(config_path)]

def _cli_command(python_exe: Path, config_path: Path) -> list[str]:
    return [str(python_exe), "-m", "olive", "run", "--config", str(config_path)]

def _run_command(cmd: list[str], *, work_dir: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as e:
        raise e
    except subprocess.TimeoutExpired as e:
        raise e
    except subprocess.CalledProcessError as e:
        raise e


def run_olive_cli(
    *,
    python_exe: Path,
    config_path: Path,
    work_dir: Path,
    timeout_s: int = 14_400,
) -> None:
    """Run Olive in an external interpreter.

    Preferred mode runs a wrapper that calls ``olive.run(config_dict)``.
    Falls back to ``python -m olive run --config``.
    """
    python_exe = python_exe.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()

    commands = [
        _wrapper_command(python_exe, config_path),
        _cli_command(python_exe, config_path),
    ]
    last_error: BaseException | None = None

    for cmd in commands:
        try:
            _run_command(cmd, work_dir=work_dir, timeout_s=timeout_s)
            return
        except FileNotFoundError as e:
            last_error = e
            continue
        except subprocess.TimeoutExpired as e:
            raise NpuModelError(
                stage="quant",
                reason_code="OLIVE_RUN_TIMEOUT",
                message=f"Olive run timed out after {timeout_s} seconds.",
                hint=f"Interpreter: {python_exe}\nConfig: {config_path}",
                cause=e,
            ) from e
        except subprocess.CalledProcessError as e:
            last_error = e
            continue

    if isinstance(last_error, FileNotFoundError):
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_CLI_NOT_FOUND",
            message=f"Unable to invoke Olive runner with interpreter: {python_exe}",
            hint=(
                "Ensure --olive-python points to a valid Python executable with Olive installed."
            ),
            cause=last_error,
        ) from last_error

    if isinstance(last_error, subprocess.CalledProcessError):
        stderr_tail = (last_error.stderr or "")[-3500:]
        stdout_tail = (last_error.stdout or "")[-2000:]
        detail = stderr_tail or stdout_tail or "No output captured."
        raise NpuModelError(
            stage="quant",
            reason_code="OLIVE_RUN_FAILED",
            message=(
                "Olive run failed in external interpreter "
                f"(exit {last_error.returncode}): {python_exe}"
            ),
            hint=f"Config: {config_path}\nOutput tail:\n{detail}",
            cause=last_error,
        ) from last_error

    raise NpuModelError(
        stage="quant",
        reason_code="OLIVE_RUN_FAILED",
        message=f"Olive run failed in external interpreter: {python_exe}",
    )
