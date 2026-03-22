from __future__ import annotations

from npu_model.core.types import TargetSpec


def test_target_spec_schema_version() -> None:
    ts = TargetSpec(backend_id="qnn", name="auto", params={"backend_type": "htp"})
    assert ts.schema_version == 1


def test_target_spec_normalized_repr() -> None:
    ts = TargetSpec(
        backend_id="qnn",
        name="auto",
        params={"backend_type": "htp", "backend_path": "QnnHtp.dll"},
    )
    r = ts.normalized_repr()
    assert "qnn:auto" in r
    assert "backend_type=htp" in r
    assert "backend_path=QnnHtp.dll" in r


def test_target_spec_custom_version() -> None:
    ts = TargetSpec(backend_id="test", name="x", params={}, schema_version=2)
    assert ts.schema_version == 2
