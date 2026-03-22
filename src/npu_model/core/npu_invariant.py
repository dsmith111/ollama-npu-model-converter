"""NPU-only invariants — enforced during compilation and validation.

When targeting QNN HTP (NPU-only), these invariants must hold:
  - CPU EP fallback is disabled
  - All graphs run entirely on QNNExecutionProvider
  - Context-cache artifacts exist (no raw float weights at runtime)
"""
from __future__ import annotations

from typing import Any


def apply_npu_only_session_options(sess_options: Any) -> None:
    """Configure ORT SessionOptions for strict NPU-only execution.

    Sets session config entries that prevent silent CPU fallback,
    ensuring the session creation fails if any op can't run on QNN HTP.

    Parameters
    ----------
    sess_options:
        An ``onnxruntime.SessionOptions`` instance.
    """
    sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")


def apply_context_cache_session_options(
    sess_options: Any,
    *,
    context_file_path: str,
    embed_mode: str = "0",
) -> None:
    """Configure ORT SessionOptions for QNN EP context-cache generation.

    Uses the documented ``ep.context_*`` session config entries (NOT provider options).

    Parameters
    ----------
    sess_options:
        An ``onnxruntime.SessionOptions`` instance.
    context_file_path:
        Path where the context output files should be written.
    embed_mode:
        "0" = separate .bin file (recommended), "1" = embedded in ONNX.
    """
    sess_options.add_session_config_entry("ep.context_enable", "1")
    sess_options.add_session_config_entry("ep.context_embed_mode", embed_mode)
    sess_options.add_session_config_entry("ep.context_file_path", context_file_path)
