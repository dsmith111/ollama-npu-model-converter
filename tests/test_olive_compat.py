from __future__ import annotations

import sys
from pathlib import Path

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.olive.compat import probe_olive_python


def test_probe_missing_python_raises(tmp_path: Path) -> None:
    bad = tmp_path / "does_not_exist_python.exe"
    with pytest.raises(NpuModelError) as exc_info:
        probe_olive_python(bad)
    assert exc_info.value.reason_code == "OLIVE_PYTHON_NOT_FOUND"


def test_probe_current_python_reports_supported_or_clean_error() -> None:
    exe = Path(sys.executable)
    if sys.version_info >= (3, 14):
        with pytest.raises(NpuModelError) as exc_info:
            probe_olive_python(exe)
        assert exc_info.value.reason_code == "OLIVE_UNSUPPORTED_PYTHON"
        return

    try:
        report = probe_olive_python(exe)
        assert report.python_exe.exists()
    except NpuModelError as e:
        # Non-fatal in test envs without Olive installed.
        assert e.reason_code in ("OLIVE_NOT_INSTALLED",)

