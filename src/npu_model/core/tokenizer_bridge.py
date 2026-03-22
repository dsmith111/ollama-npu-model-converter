"""Bridge tokenizer files from the original HF snapshot into the exported bundle.

Exporters (e.g. ORT GenAI model builder) sometimes omit tokenizer assets
that are present in the source HF directory — most critically ``tokenizer.model``
(SentencePiece).  This module copies missing tokenizer files into the exported
tokenizer directory so that downstream normalization and packaging steps have
the full set of assets available.
"""
from __future__ import annotations

import shutil
from pathlib import Path


TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def bridge_tokenizer_files(*, src_model_dir: Path, dst_tokenizer_dir: Path) -> list[str]:
    """Copy tokenizer files from the original HF snapshot into the exported tokenizer dir.

    Only copies files that are **missing** from *dst_tokenizer_dir*.
    Searches the source root and common nested tokenizer directories.

    Returns a list of filenames that were copied.
    """
    copied: list[str] = []
    dst_tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Search common locations (root and any nested tokenizer dirs)
    candidates: list[Path] = [src_model_dir]
    for p in src_model_dir.iterdir():
        if p.is_dir() and p.name in ("tokenizer", "tokenizers"):
            candidates.append(p)

    for name in TOKENIZER_FILES:
        if (dst_tokenizer_dir / name).exists():
            continue

        for base in candidates:
            src = base / name
            if src.exists() and src.is_file():
                shutil.copy2(src, dst_tokenizer_dir / name)
                copied.append(name)
                break

    return copied
