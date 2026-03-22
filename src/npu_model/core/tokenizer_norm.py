"""Normalize tokenizer_config.json for ORT GenAI runtime compatibility.

Some HF exports (via onnxruntime_genai.models.builder) produce a
``tokenizer_class`` value that the ORT GenAI runtime on certain platforms
(especially Windows ARM64 QNN) does not support.  This module rewrites
``tokenizer_config.json`` to use a known-working tokenizer class.

Unsupported classes include:
  - ``TokenizersBackend``  (generic HF tokenizers backend reference)
  - Any class ending in ``Fast`` (e.g. ``LlamaTokenizerFast``)

The fallback logic is:
  - If ``tokenizer.model`` exists → use SentencePiece-based class
  - Otherwise → keep the original and warn
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# Tokenizer classes that are NOT supported by ORT GenAI on win-arm64-qnn
_UNSUPPORTED_CLASSES: set[str] = {
    "TokenizersBackend",
}

# Model-family → preferred SentencePiece tokenizer class
_SENTENCEPIECE_FALLBACKS: dict[str, str] = {
    "phi3": "LlamaTokenizer",
    "phi": "LlamaTokenizer",
    "llama": "LlamaTokenizer",
    "mistral": "LlamaTokenizer",
    "gemma": "GemmaTokenizer",
    "qwen2": "QWen2Tokenizer",
}

_DEFAULT_SENTENCEPIECE_CLASS = "LlamaTokenizer"

# Model-family → preferred BPE/JSON tokenizer class (no tokenizer.model)
_BPE_FALLBACKS: dict[str, str] = {
    "phi": "CodeGenTokenizer",
    "phi3": "LlamaTokenizer",
    "gpt2": "GPT2Tokenizer",
    "codegen": "CodeGenTokenizer",
    "starcoder": "GPT2Tokenizer",
}

_DEFAULT_BPE_CLASS = "GPT2Tokenizer"


@dataclass
class TokenizerNormResult:
    changed: bool
    original_class: str | None
    new_class: str | None
    warnings: list[str]


def _is_unsupported(tokenizer_class: str) -> bool:
    if tokenizer_class in _UNSUPPORTED_CLASSES:
        return True
    if tokenizer_class.endswith("Fast"):
        return True
    return False


def _pick_fallback(model_type: str | None) -> str:
    if model_type:
        key = model_type.lower()
        if key in _SENTENCEPIECE_FALLBACKS:
            return _SENTENCEPIECE_FALLBACKS[key]
    return _DEFAULT_SENTENCEPIECE_CLASS


def _pick_bpe_fallback(model_type: str | None) -> str:
    if model_type:
        key = model_type.lower()
        if key in _BPE_FALLBACKS:
            return _BPE_FALLBACKS[key]
    return _DEFAULT_BPE_CLASS


def normalize_tokenizer_config(
    tokenizer_dir: Path,
    *,
    model_type: str | None = None,
) -> TokenizerNormResult:
    """Rewrite tokenizer_config.json if it contains an unsupported tokenizer_class.

    Parameters
    ----------
    tokenizer_dir:
        Directory containing tokenizer files (tokenizer_config.json, tokenizer.model, etc.)
    model_type:
        HF model_type (e.g. "phi3", "llama") used to select the correct fallback class.

    Returns
    -------
    TokenizerNormResult with details of what was changed (if anything).
    """
    warnings: list[str] = []
    config_path = tokenizer_dir / "tokenizer_config.json"

    if not config_path.exists():
        return TokenizerNormResult(changed=False, original_class=None, new_class=None, warnings=[])

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        warnings.append("Could not parse tokenizer_config.json; skipping normalization.")
        return TokenizerNormResult(changed=False, original_class=None, new_class=None, warnings=warnings)

    original_class = cfg.get("tokenizer_class")
    if not original_class or not _is_unsupported(original_class):
        return TokenizerNormResult(
            changed=False, original_class=original_class, new_class=None, warnings=[],
        )

    # Check if we have tokenizer.model (SentencePiece) available
    has_sp = (tokenizer_dir / "tokenizer.model").exists()

    if has_sp:
        new_class = _pick_fallback(model_type)
        cfg["tokenizer_class"] = new_class
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return TokenizerNormResult(
            changed=True,
            original_class=original_class,
            new_class=new_class,
            warnings=[],
        )

    # No tokenizer.model — check if tokenizer.json exists (BPE/JSON tokenizer)
    has_json = (tokenizer_dir / "tokenizer.json").exists()

    if has_json:
        # BPE tokenizer path: rewrite to a compatible BPE class.
        # ORT GenAI loads the tokenizer from tokenizer.json directly;
        # the class name just needs to be one it recognizes.
        new_class = _pick_bpe_fallback(model_type)
        cfg["tokenizer_class"] = new_class
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return TokenizerNormResult(
            changed=True,
            original_class=original_class,
            new_class=new_class,
            warnings=[],
        )

    # Neither tokenizer.model nor tokenizer.json found
    warnings.append(
        f"tokenizer_class='{original_class}' is unsupported by ORT GenAI, "
        f"and neither tokenizer.model nor tokenizer.json found for fallback."
    )
    return TokenizerNormResult(
        changed=False, original_class=original_class, new_class=None, warnings=warnings,
    )
