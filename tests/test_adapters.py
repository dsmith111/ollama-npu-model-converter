from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from npu_model.adapters.phi3 import Phi3Adapter
from npu_model.adapters.llama import LlamaAdapter
from npu_model.adapters.base import _import_prebuilt_ort_genai


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def prebuilt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prebuilt"
    d.mkdir()
    shutil.copy2(FIXTURES / "hf_phi3_config.json", d / "config.json")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "model.onnx").write_bytes(b"\x00" * 16)
    (d / "genai_config.json").write_text("{}", encoding="utf-8")
    (d / "weights.bin").write_bytes(b"\x00" * 8)
    return d


class TestPrebuiltImport:

    def test_import_prebuilt_collects_onnx(self, prebuilt_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        bundle = _import_prebuilt_ort_genai(prebuilt_dir, out, adapter_id="test")
        assert "model" in bundle.graphs
        assert bundle.graphs["model"].exists()

    def test_import_prebuilt_collects_tokenizer(self, prebuilt_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        bundle = _import_prebuilt_ort_genai(prebuilt_dir, out, adapter_id="test")
        assert (bundle.tokenizer_dir / "tokenizer.json").exists()

    def test_import_prebuilt_collects_extra(self, prebuilt_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        bundle = _import_prebuilt_ort_genai(prebuilt_dir, out, adapter_id="test")
        extra_names = {p.name for p in bundle.extra_files}
        assert "genai_config.json" in extra_names
        assert "weights.bin" in extra_names

    def test_import_prebuilt_metadata(self, prebuilt_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        bundle = _import_prebuilt_ort_genai(prebuilt_dir, out, adapter_id="phi3")
        assert bundle.metadata["mode"] == "prebuilt"
        assert bundle.metadata["adapter"] == "phi3"


class TestPhi3AdapterPrebuilt:

    def test_prebuilt_mode(self, prebuilt_dir: Path, tmp_path: Path) -> None:
        adapter = Phi3Adapter()
        out = tmp_path / "out"
        bundle = adapter.export(prebuilt_dir, out, export_config={"mode": "prebuilt-ort-genai"})
        assert "model" in bundle.graphs
        assert bundle.metadata["mode"] == "prebuilt"


class TestLlamaAdapterPrebuilt:

    def test_prebuilt_mode(self, tmp_path: Path) -> None:
        d = tmp_path / "llama_prebuilt"
        d.mkdir()
        shutil.copy2(FIXTURES / "hf_llama_config.json", d / "config.json")
        (d / "tokenizer.model").write_bytes(b"\x00")
        (d / "decoder.onnx").write_bytes(b"\x00" * 16)
        (d / "genai_config.json").write_text("{}", encoding="utf-8")

        adapter = LlamaAdapter()
        out = tmp_path / "out"
        bundle = adapter.export(d, out, export_config={"mode": "prebuilt-ort-genai"})
        assert "decoder" in bundle.graphs
