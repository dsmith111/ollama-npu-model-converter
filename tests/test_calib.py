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
