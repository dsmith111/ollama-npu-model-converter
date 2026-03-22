from __future__ import annotations

from pathlib import Path

import pytest

from npu_model.core.types import GraphBundle
from npu_model.quant.passthrough import PassthroughQuantizer


def test_passthrough_is_noop(tmp_path: Path) -> None:
    q = PassthroughQuantizer()
    bundle = GraphBundle(
        graphs={"m": tmp_path / "m.onnx"},
        tokenizer_dir=tmp_path,
        extra_files=[],
        metadata={},
    )
    assert q.apply(bundle, quant_config={}) is bundle


def test_passthrough_has_id() -> None:
    q = PassthroughQuantizer()
    assert q.id == "passthrough"


def test_qnn_qdq_has_id() -> None:
    from npu_model.quant.qnn_qdq import QnnQdqQuantizer
    q = QnnQdqQuantizer()
    assert q.id == "qnn-qdq"
