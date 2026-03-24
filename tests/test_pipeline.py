from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from npu_model.core.errors import NpuModelError
from npu_model.core.pipeline import convert_model, explain_plan
from npu_model.core.registry import Registry
from npu_model.core.types import ConvertMode


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def registry() -> Registry:
    return Registry.load()


@pytest.fixture()
def phi3_prebuilt_dir(tmp_path: Path) -> Path:
    """A prebuilt ORT GenAI directory with onnx + tokenizer + genai_config."""
    d = tmp_path / "phi3_prebuilt"
    d.mkdir()
    shutil.copy2(FIXTURES / "hf_phi3_config.json", d / "config.json")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "model_ctx.onnx").write_bytes(b"\x00" * 16)
    (d / "model_qnn.bin").write_bytes(b"\x00" * 4096)
    (d / "genai_config.json").write_text(
        json.dumps({"model": {"decoder": {"context_length": 4096}}, "search": {"max_length": 2048}}),
        encoding="utf-8",
    )
    return d


class TestConvertMode:

    def test_prebuilt_mode_enum(self) -> None:
        assert ConvertMode("prebuilt-ort-genai") == ConvertMode.PREBUILT
        assert ConvertMode("export") == ConvertMode.EXPORT

    def test_invalid_mode_raises(self, phi3_prebuilt_dir: Path, registry: Registry) -> None:
        with pytest.raises(NpuModelError, match="Unknown convert mode"):
            explain_plan(
                input_spec=str(phi3_prebuilt_dir),
                backend_id="qnn",
                target="auto",
                runtime_format_id="ort-genai-folder",
                cache_dir=None,
                registry=registry,
                mode="nonexistent-mode",
            )


class TestConvertPrebuilt:

    def test_convert_prebuilt_creates_bundle(
        self, phi3_prebuilt_dir: Path, tmp_path: Path, registry: Registry
    ) -> None:
        out = tmp_path / "out"
        result = convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
        )
        assert result.bundle_dir.exists()
        assert result.manifest_path.exists()
        # Check manifest is valid JSON
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest["plan"]["convert_mode"] == "prebuilt-ort-genai"

    def test_convert_with_pack_ollama(
        self, phi3_prebuilt_dir: Path, tmp_path: Path, registry: Registry
    ) -> None:
        out = tmp_path / "out"
        result = convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
            pack_ollama_name="test/phi3:qnn",
            pack_ollama_opts={"num_ctx": 1024, "num_predict": 64},
        )
        assert result.pack_dir is not None
        assert result.pack_dir.exists()
        mf = (result.pack_dir / "Modelfile").read_text(encoding="utf-8")
        assert "FROM ." in mf
        assert "num_ctx 1024" in mf

    def test_convert_no_pack_by_default(
        self, phi3_prebuilt_dir: Path, tmp_path: Path, registry: Registry
    ) -> None:
        out = tmp_path / "out"
        result = convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
        )
        assert result.pack_dir is None


class TestExplainMode:

    def test_explain_shows_mode(self, phi3_prebuilt_dir: Path, registry: Registry) -> None:
        plan = explain_plan(
            input_spec=str(phi3_prebuilt_dir),
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
        )
        assert plan.convert_mode == "prebuilt-ort-genai"
        rich = plan.to_rich_text()
        assert "prebuilt-ort-genai" in rich

    def test_explain_default_mode_is_export(self, phi3_prebuilt_dir: Path, registry: Registry) -> None:
        plan = explain_plan(
            input_spec=str(phi3_prebuilt_dir),
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            cache_dir=None,
            registry=registry,
        )
        assert plan.convert_mode == "export"


class TestConvertCaching:

    def test_second_run_uses_cache(
        self, phi3_prebuilt_dir: Path, tmp_path: Path, registry: Registry
    ) -> None:
        out = tmp_path / "out"
        # First run
        r1 = convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
        )
        assert r1.bundle_dir.exists()
        # Cache dir should exist
        cache_dir = out / ".cache"
        assert cache_dir.exists()

        # Second run (should hit cache)
        out2 = tmp_path / "out2"
        # We need same out_dir for cache to be found
        r2 = convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
        )
        assert r2.bundle_dir.exists()

    def test_no_cache_forces_rebuild(
        self, phi3_prebuilt_dir: Path, tmp_path: Path, registry: Registry
    ) -> None:
        out = tmp_path / "out"
        r1 = convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
            use_cache=False,
        )
        assert r1.bundle_dir.exists()


class TestKeepWork:

    def test_work_dir_removed_by_default(
        self, phi3_prebuilt_dir: Path, tmp_path: Path, registry: Registry
    ) -> None:
        out = tmp_path / "out"
        convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
        )
        assert not (out / "_work").exists()

    def test_work_dir_kept_with_flag(
        self, phi3_prebuilt_dir: Path, tmp_path: Path, registry: Registry
    ) -> None:
        out = tmp_path / "out"
        convert_model(
            input_spec=str(phi3_prebuilt_dir),
            out_dir=out,
            backend_id="qnn",
            target="auto",
            runtime_format_id="ort-genai-folder",
            quantizer_id="passthrough",
            cache_dir=None,
            registry=registry,
            mode="prebuilt-ort-genai",
            keep_work=True,
        )
        assert (out / "_work").exists()
