from __future__ import annotations

from pathlib import Path

import pytest

from npu_model.calib.prompts_builtin import MIXED_SMALL, BUILTIN_CORPORA, DEFAULT_CORPUS_ID
from npu_model.calib.prompt_source import (
    BuiltinPromptSource,
    FilePromptSource,
    get_prompt_source,
)
from npu_model.core.errors import NpuModelError


class TestBuiltinPrompts:

    def test_mixed_small_has_prompts(self) -> None:
        assert len(MIXED_SMALL) >= 32

    def test_all_corpora_non_empty(self) -> None:
        for name, corpus in BUILTIN_CORPORA.items():
            assert len(corpus) > 0, f"Corpus '{name}' is empty"

    def test_default_corpus_exists(self) -> None:
        assert DEFAULT_CORPUS_ID in BUILTIN_CORPORA


class TestPromptSources:

    def test_builtin_source_loads(self) -> None:
        src = BuiltinPromptSource("builtin:mixed_small")
        prompts = src.load()
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    def test_builtin_source_unknown_fails(self) -> None:
        src = BuiltinPromptSource("builtin:nonexistent")
        with pytest.raises(NpuModelError, match="Unknown builtin corpus"):
            src.load()

    def test_file_source_loads(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.txt"
        f.write_text("Hello world\n# comment\n\nSecond prompt\n", encoding="utf-8")
        src = FilePromptSource(f)
        prompts = src.load()
        assert prompts == ["Hello world", "Second prompt"]

    def test_file_source_missing_fails(self, tmp_path: Path) -> None:
        src = FilePromptSource(tmp_path / "missing.txt")
        with pytest.raises(NpuModelError, match="not found"):
            src.load()

    def test_file_source_empty_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("# only comments\n\n", encoding="utf-8")
        src = FilePromptSource(f)
        with pytest.raises(NpuModelError, match="No prompts found"):
            src.load()

    def test_get_prompt_source_default(self) -> None:
        src = get_prompt_source()
        assert isinstance(src, BuiltinPromptSource)

    def test_get_prompt_source_file_override(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.txt"
        f.write_text("test\n", encoding="utf-8")
        src = get_prompt_source(calib_prompts_file=f)
        assert isinstance(src, FilePromptSource)


class TestCalibDataReader:

    def test_reader_protocol(self) -> None:
        """OnnxCalibrationDataReader implements get_next/rewind."""
        np = pytest.importorskip("numpy")
        from npu_model.calib.data_reader import OnnxCalibrationDataReader

        feeds = [
            {"input_ids": np.array([[1, 2, 3]], dtype=np.int64)},
            {"input_ids": np.array([[4, 5, 6]], dtype=np.int64)},
        ]
        reader = OnnxCalibrationDataReader(feeds)
        first = reader.get_next()
        assert first is not None
        second = reader.get_next()
        assert second is not None
        third = reader.get_next()
        assert third is None  # exhausted

        reader.rewind()
        again = reader.get_next()
        assert again is not None


class TestGetOnnxInputInfo:
    """Tests for _get_onnx_input_info returning all graph inputs."""

    def test_returns_all_inputs_including_past_kv(self, tmp_path: Path) -> None:
        """A decoder-with-past ONNX model must expose past_key_values inputs."""
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto
        from onnx.helper import make_graph, make_model, make_tensor_value_info, make_node
        from npu_model.calib.data_reader import _get_onnx_input_info

        inputs = [
            make_tensor_value_info("input_ids", TensorProto.INT64, [1, 128]),
            make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 128]),
            make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, [1, 32, 10, 96]),
            make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, [1, 32, 10, 96]),
            make_tensor_value_info("past_key_values.1.key", TensorProto.FLOAT16, [1, 32, 10, 96]),
            make_tensor_value_info("past_key_values.1.value", TensorProto.FLOAT16, [1, 32, 10, 96]),
        ]
        outputs = [make_tensor_value_info("logits", TensorProto.FLOAT, [1, 128, 51200])]
        node = make_node("Identity", ["input_ids"], ["logits"])
        graph = make_graph([node], "test_graph", inputs, outputs)
        model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])

        model_path = tmp_path / "decoder_with_past.onnx"
        onnx.save(model, str(model_path))

        info = _get_onnx_input_info(model_path)

        # All six inputs must be present
        assert "input_ids" in info
        assert "attention_mask" in info
        assert "past_key_values.0.key" in info
        assert "past_key_values.0.value" in info
        assert "past_key_values.1.key" in info
        assert "past_key_values.1.value" in info

        # Verify shapes and dtypes
        assert info["past_key_values.0.key"]["shape"] == [1, 32, 10, 96]
        assert info["past_key_values.0.key"]["dtype"] == "float32"
        assert info["past_key_values.1.key"]["dtype"] == "float16"


class TestDecoderWithPastCalibration:
    """Regression: calibration reader must produce feeds for past_key_values inputs."""

    def _make_decoder_with_past_onnx(self, path: Path) -> Path:
        """Create a minimal ONNX model with tokenizer + past_key_values inputs."""
        import onnx
        from onnx import TensorProto
        from onnx.helper import make_graph, make_model, make_tensor_value_info, make_node

        inputs = [
            make_tensor_value_info("input_ids", TensorProto.INT64, [1, 32]),
            make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 32]),
            make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, [1, 4, 8, 16]),
            make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, [1, 4, 8, 16]),
        ]
        outputs = [make_tensor_value_info("logits", TensorProto.FLOAT, [1, 32, 100])]
        node = make_node("Identity", ["input_ids"], ["logits"])
        graph = make_graph([node], "test_graph", inputs, outputs)
        model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])

        model_path = path / "model_with_past.onnx"
        onnx.save(model, str(model_path))
        return model_path

    def _make_tokenizer(self, path: Path) -> Path:
        """Create a minimal tokenizer_config.json so _load_tokenizer can try (and fail)."""
        # We'll mock the tokenizer instead of creating real files
        return path

    def test_feeds_include_past_key_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Feeds produced by build_calibration_reader must include past_key_values entries."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("onnx")
        from npu_model.calib import data_reader
        from npu_model.calib.data_reader import build_calibration_reader

        model_path = self._make_decoder_with_past_onnx(tmp_path)
        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()

        # Mock the tokenizer loader to return a fake transformers-style tokenizer
        class FakeTokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def __call__(self, prompt, max_length=32, padding="max_length",
                         truncation=True, return_tensors="np"):
                ids = np.ones((1, max_length), dtype=np.int64)
                mask = np.ones((1, max_length), dtype=np.int64)
                return {"input_ids": ids, "attention_mask": mask}

        monkeypatch.setattr(data_reader, "_load_tokenizer", lambda _: FakeTokenizer())

        reader = build_calibration_reader(
            prompts=["Hello world"],
            tokenizer_dir=tok_dir,
            onnx_path=model_path,
            num_samples=1,
            max_seq_len=32,
        )

        feed = reader.get_next()
        assert feed is not None

        # Must contain tokenizer-derived inputs
        assert "input_ids" in feed
        assert "attention_mask" in feed

        # Must contain past_key_values zero-filled tensors
        assert "past_key_values.0.key" in feed
        assert "past_key_values.0.value" in feed

        # Verify shapes match the ONNX model
        assert feed["past_key_values.0.key"].shape == (1, 4, 8, 16)
        assert feed["past_key_values.0.value"].shape == (1, 4, 8, 16)

        # Verify they are zero-filled
        assert np.all(feed["past_key_values.0.key"] == 0)
        assert np.all(feed["past_key_values.0.value"] == 0)

        # Verify dtype
        assert feed["past_key_values.0.key"].dtype == np.float32

    def test_tokenizer_only_model_still_works(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model with only tokenizer inputs must still produce valid feeds (no regression)."""
        np = pytest.importorskip("numpy")
        import onnx
        from onnx import TensorProto
        from onnx.helper import make_graph, make_model, make_tensor_value_info, make_node
        from npu_model.calib import data_reader
        from npu_model.calib.data_reader import build_calibration_reader

        inputs = [
            make_tensor_value_info("input_ids", TensorProto.INT64, [1, 16]),
            make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 16]),
        ]
        outputs = [make_tensor_value_info("logits", TensorProto.FLOAT, [1, 16, 100])]
        node = make_node("Identity", ["input_ids"], ["logits"])
        graph = make_graph([node], "test_graph", inputs, outputs)
        model = make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        model_path = tmp_path / "tokenizer_only.onnx"
        onnx.save(model, str(model_path))

        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()

        class FakeTokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def __call__(self, prompt, max_length=16, padding="max_length",
                         truncation=True, return_tensors="np"):
                return {
                    "input_ids": np.ones((1, max_length), dtype=np.int64),
                    "attention_mask": np.ones((1, max_length), dtype=np.int64),
                }

        monkeypatch.setattr(data_reader, "_load_tokenizer", lambda _: FakeTokenizer())

        reader = build_calibration_reader(
            prompts=["test"],
            tokenizer_dir=tok_dir,
            onnx_path=model_path,
            num_samples=1,
            max_seq_len=16,
        )

        feed = reader.get_next()
        assert feed is not None
        assert set(feed.keys()) == {"input_ids", "attention_mask"}
        assert "past_key_values" not in str(feed.keys())
