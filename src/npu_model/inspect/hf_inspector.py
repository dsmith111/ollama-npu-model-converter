from __future__ import annotations

import json
from pathlib import Path

from npu_model.core.errors import NpuModelError
from npu_model.core.types import ModelInfo


_TOKENIZER_CANDIDATES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]


def inspect_hf_style_dir(model_dir: Path, source: dict) -> ModelInfo:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise NpuModelError(
            stage="inspect",
            reason_code="MISSING_CONFIG",
            message=f"config.json not found in: {model_dir}",
        )

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise NpuModelError(
            stage="inspect",
            reason_code="CONFIG_PARSE_FAILED",
            message=f"Failed parsing config.json: {cfg_path}",
            cause=e,
        ) from e

    model_type = cfg.get("model_type")
    architectures = cfg.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = []

    tokenizer_files = [f for f in _TOKENIZER_CANDIDATES if (model_dir / f).exists()]

    return ModelInfo(
        source=source,
        model_type=model_type,
        architectures=[str(x) for x in architectures],
        config=cfg,
        tokenizer_files=tokenizer_files,
    )
