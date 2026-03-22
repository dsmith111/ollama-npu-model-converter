"""Prompt sources for calibration data generation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from npu_model.core.errors import NpuModelError


class PromptSource(ABC):
    @abstractmethod
    def load(self) -> list[str]:
        raise NotImplementedError


class BuiltinPromptSource(PromptSource):
    """Load prompts from the built-in corpus."""

    def __init__(self, corpus_id: str = "builtin:mixed_small") -> None:
        self.corpus_id = corpus_id

    def load(self) -> list[str]:
        from npu_model.calib.prompts_builtin import BUILTIN_CORPORA

        if self.corpus_id not in BUILTIN_CORPORA:
            raise NpuModelError(
                stage="calib",
                reason_code="UNKNOWN_CORPUS",
                message=f"Unknown builtin corpus: {self.corpus_id}",
                hint=f"Available: {sorted(BUILTIN_CORPORA.keys())}",
            )
        return BUILTIN_CORPORA[self.corpus_id]


class FilePromptSource(PromptSource):
    """Load prompts from a text file (one prompt per line)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[str]:
        if not self.path.exists():
            raise NpuModelError(
                stage="calib",
                reason_code="CALIB_FILE_NOT_FOUND",
                message=f"Calibration prompts file not found: {self.path}",
            )
        lines = self.path.read_text(encoding="utf-8").splitlines()
        # Strip whitespace, ignore blank lines and comments
        prompts = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        if not prompts:
            raise NpuModelError(
                stage="calib",
                reason_code="EMPTY_CALIB_FILE",
                message=f"No prompts found in calibration file: {self.path}",
            )
        return prompts


def get_prompt_source(
    calib_source: str = "builtin:mixed_small",
    calib_prompts_file: Path | None = None,
) -> PromptSource:
    """Get the appropriate prompt source based on CLI args."""
    if calib_prompts_file is not None:
        return FilePromptSource(calib_prompts_file)
    return BuiltinPromptSource(calib_source)
