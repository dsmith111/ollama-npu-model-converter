"""Build an ORT CalibrationDataReader from prompts + tokenizer + ONNX graph.

Tokenizes prompts into fixed-shape batches matching the ONNX model's input
signature, then yields them as feeds for ORT static quantization.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from npu_model.core.errors import NpuModelError

_log = logging.getLogger("npu_model.calib")


class OnnxCalibrationDataReader:
    """CalibrationDataReader that yields tokenized prompt batches.

    Implements the ``onnxruntime.quantization.CalibrationDataReader`` protocol
    (a ``get_next()`` iterator).
    """

    def __init__(
        self,
        feeds: list[dict[str, np.ndarray]],
    ) -> None:
        self._feeds = feeds
        self._iter: Iterator[dict[str, np.ndarray]] = iter(feeds)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return next(self._iter)
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iter = iter(self._feeds)


def _load_tokenizer(tokenizer_dir: Path) -> Any:
    """Load a tokenizer from a directory, trying transformers first."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        # Many models (GPT-2, Phi-2, CodeGen) don't define a pad token.
        # For calibration padding we need one — use eos_token as fallback.
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        return tok
    except ImportError:
        pass

    try:
        from tokenizers import Tokenizer
        tok_json = tokenizer_dir / "tokenizer.json"
        if tok_json.exists():
            return Tokenizer.from_file(str(tok_json))
    except ImportError:
        pass

    raise NpuModelError(
        stage="calib",
        reason_code="NO_TOKENIZER_LIB",
        message="Cannot load tokenizer: neither transformers nor tokenizers is installed.",
        hint="pip install transformers  (or: pip install npu-model[export])",
    )


def _get_onnx_input_info(onnx_path: Path) -> dict[str, dict[str, Any]]:
    """Inspect ONNX graph inputs to determine names and dtypes."""
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        # Fallback: assume standard LLM input names
        return {
            "input_ids": {"dtype": "int64"},
            "attention_mask": {"dtype": "int64"},
        }

    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception:
        return {
            "input_ids": {"dtype": "int64"},
            "attention_mask": {"dtype": "int64"},
        }

    _ONNX_DTYPE_MAP = {
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
    }

    # Only return inputs that look like tokenizer outputs
    _CALIB_INPUT_NAMES = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    info: dict[str, dict[str, Any]] = {}
    for inp in model.graph.input:
        if inp.name in _CALIB_INPUT_NAMES and inp.type.HasField("tensor_type"):
            dtype_int = inp.type.tensor_type.elem_type
            dtype_str = _ONNX_DTYPE_MAP.get(dtype_int, "int64")
            info[inp.name] = {"dtype": dtype_str}

    if not info:
        # No recognized inputs — use defaults
        info = {"input_ids": {"dtype": "int64"}, "attention_mask": {"dtype": "int64"}}

    return info


def build_calibration_reader(
    *,
    prompts: list[str],
    tokenizer_dir: Path,
    onnx_path: Path,
    num_samples: int = 64,
    max_seq_len: int = 256,
    batch_size: int = 1,
) -> OnnxCalibrationDataReader:
    """Build a CalibrationDataReader from prompts.

    Parameters
    ----------
    prompts:
        Text prompts to tokenize for calibration.
    tokenizer_dir:
        Directory containing tokenizer files.
    onnx_path:
        Path to the ONNX model (used to determine input names/dtypes).
    num_samples:
        Maximum number of calibration samples to generate.
    max_seq_len:
        Maximum sequence length for tokenized inputs.
    batch_size:
        Batch size (typically 1 for calibration).
    """
    tokenizer = _load_tokenizer(tokenizer_dir)
    input_info = _get_onnx_input_info(onnx_path)

    # Take up to num_samples prompts
    prompts = prompts[:num_samples]

    feeds: list[dict[str, np.ndarray]] = []
    for prompt in prompts:
        # Tokenize
        if hasattr(tokenizer, "__call__"):
            # transformers AutoTokenizer
            encoded = tokenizer(
                prompt,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            feed: dict[str, np.ndarray] = {}
            for name, info in input_info.items():
                if name in encoded:
                    arr = encoded[name]
                elif name == "position_ids" and "input_ids" in encoded:
                    # Auto-generate position_ids
                    seq_len = encoded["input_ids"].shape[-1]
                    arr = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
                else:
                    continue

                # Cast to expected dtype
                target_dtype = getattr(np, info["dtype"])
                if arr.dtype != target_dtype:
                    arr = arr.astype(target_dtype)

                # Ensure batch dimension
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                feed[name] = arr

            if feed:
                feeds.append(feed)

        elif hasattr(tokenizer, "encode"):
            # tokenizers.Tokenizer (Rust)
            encoding = tokenizer.encode(prompt)
            ids = encoding.ids[:max_seq_len]
            # Pad
            pad_len = max_seq_len - len(ids)
            ids = ids + [0] * pad_len
            mask = [1] * (max_seq_len - pad_len) + [0] * pad_len

            feed = {}
            if "input_ids" in input_info:
                dtype = getattr(np, input_info["input_ids"]["dtype"])
                feed["input_ids"] = np.array([ids], dtype=dtype)
            if "attention_mask" in input_info:
                dtype = getattr(np, input_info["attention_mask"]["dtype"])
                feed["attention_mask"] = np.array([mask], dtype=dtype)
            if "position_ids" in input_info:
                dtype = getattr(np, input_info["position_ids"]["dtype"])
                feed["position_ids"] = np.arange(max_seq_len, dtype=dtype).reshape(1, -1)

            if feed:
                feeds.append(feed)

    if not feeds:
        raise NpuModelError(
            stage="calib",
            reason_code="NO_CALIB_FEEDS",
            message="Failed to generate any calibration feeds from prompts.",
            hint="Check that the tokenizer can process the calibration prompts.",
        )

    _log.info("Built %d calibration samples (seq_len=%d)", len(feeds), max_seq_len)
    return OnnxCalibrationDataReader(feeds)
