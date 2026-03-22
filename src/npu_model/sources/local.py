from __future__ import annotations

from pathlib import Path

from npu_model.core.errors import NpuModelError


def materialize_local(path: Path) -> Path:
    p = path.expanduser().resolve()
    if not p.exists():
        raise NpuModelError(
            stage="source",
            reason_code="LOCAL_NOT_FOUND",
            message=f"Local path does not exist: {p}",
        )
    cfg = p / "config.json"
    if not cfg.exists():
        raise NpuModelError(
            stage="source",
            reason_code="LOCAL_MISSING_CONFIG",
            message=f"Local model directory missing config.json: {p}",
            hint="Point --input at a directory containing config.json (HF-style) or extend the inspector.",
        )
    return p
