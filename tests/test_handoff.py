from __future__ import annotations

import json
from pathlib import Path

import pytest

from npu_model.core.handoff import create_handoff_bundle, load_handoff_bundle
from npu_model.core.types import GraphBundle


@pytest.fixture()
def sample_graphs(tmp_path: Path) -> GraphBundle:
    g_dir = tmp_path / "graphs_src"
    g_dir.mkdir()
    (g_dir / "model.onnx").write_bytes(b"\x00" * 16)
    (g_dir / "model.onnx.data").write_bytes(b"\x00" * 64)

    tok_dir = tmp_path / "tokenizer_src"
    tok_dir.mkdir()
    (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tok_dir / "tokenizer.model").write_bytes(b"\x00")

    extras_dir = tmp_path / "extras_src"
    extras_dir.mkdir()
    (extras_dir / "genai_config.json").write_text("{}", encoding="utf-8")

    return GraphBundle(
        graphs={"model": g_dir / "model.onnx"},
        tokenizer_dir=tok_dir,
        extra_files=[extras_dir / "genai_config.json"],
        metadata={"adapter": "phi3"},
    )


class TestCreateHandoffBundle:

    def test_creates_bundle_dir(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        hb = create_handoff_bundle(
            graphs=sample_graphs,
            out_dir=out,
            stopped_after="export",
            metadata={"test": True},
        )
        assert hb.bundle_dir.exists()
        assert hb.manifest_path.exists()
        assert hb.stopped_after == "export"

    def test_contains_graphs(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        create_handoff_bundle(
            graphs=sample_graphs, out_dir=out, stopped_after="export", metadata={},
        )
        assert (out / "graphs" / "model.onnx").exists()

    def test_copies_data_file(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        create_handoff_bundle(
            graphs=sample_graphs, out_dir=out, stopped_after="export", metadata={},
        )
        assert (out / "graphs" / "model.onnx.data").exists()

    def test_copies_tokenizer(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        create_handoff_bundle(
            graphs=sample_graphs, out_dir=out, stopped_after="export", metadata={},
        )
        assert (out / "tokenizer" / "tokenizer.json").exists()
        assert (out / "tokenizer" / "tokenizer.model").exists()

    def test_copies_extras(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        create_handoff_bundle(
            graphs=sample_graphs, out_dir=out, stopped_after="export", metadata={},
        )
        assert (out / "extras" / "genai_config.json").exists()

    def test_manifest_has_stage(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        hb = create_handoff_bundle(
            graphs=sample_graphs, out_dir=out, stopped_after="quantize", metadata={},
        )
        manifest = json.loads(hb.manifest_path.read_text(encoding="utf-8"))
        assert manifest["stopped_after"] == "quantize"
        assert manifest["handoff"] is True


class TestLoadHandoffBundle:

    def test_roundtrip(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        create_handoff_bundle(
            graphs=sample_graphs, out_dir=out, stopped_after="export",
            metadata={"model_type": "phi3", "graph_metadata": {"adapter": "phi3"}},
        )

        loaded_graphs, meta = load_handoff_bundle(out)
        assert "model" in loaded_graphs.graphs
        assert loaded_graphs.graphs["model"].exists()
        assert loaded_graphs.tokenizer_dir.exists()
        assert meta.get("model_type") == "phi3"

    def test_enriched_manifest_metadata(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        """Handoff manifest should include model_family, layout, split_count, etc."""
        out = tmp_path / "handoff"
        hb = create_handoff_bundle(
            graphs=sample_graphs,
            out_dir=out,
            stopped_after="quantize",
            metadata={
                "model_type": "phi3",
                "model_family": "phi3",
                "quantizer_id": "qnn-qdq",
                "quantization_format": "qdq",
                "split_count": 1,
                "layout": "monolith",
            },
        )
        manifest = json.loads(hb.manifest_path.read_text(encoding="utf-8"))
        meta = manifest["metadata"]
        assert meta["model_family"] == "phi3"
        assert meta["quantization_format"] == "qdq"
        assert meta["split_count"] == 1
        assert meta["layout"] == "monolith"

    def test_detects_external_data(self, tmp_path: Path, sample_graphs: GraphBundle) -> None:
        out = tmp_path / "handoff"
        create_handoff_bundle(
            graphs=sample_graphs, out_dir=out, stopped_after="export", metadata={},
        )
        loaded_graphs, _ = load_handoff_bundle(out)
        model_path = loaded_graphs.graphs["model"]
        data_path = model_path.parent / f"{model_path.name}.data"
        assert data_path.exists()

    def test_loads_flat_dir(self, tmp_path: Path) -> None:
        """Can load a flat directory (no structured layout)."""
        d = tmp_path / "flat"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"\x00" * 16)
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")

        loaded, meta = load_handoff_bundle(d)
        assert "model" in loaded.graphs
