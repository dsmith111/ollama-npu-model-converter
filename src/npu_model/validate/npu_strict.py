"""Strict NPU validation — verifies a bundle can actually run on QNN HTP.

Creates ORT sessions with QNNExecutionProvider and cpu_ep_fallback disabled
to confirm graphs are fully accelerated.  Also checks for context-cache
artifacts and proper genai_config session options.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NpuValidationResult:
    passed: bool
    checks: list[NpuCheck] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [c.detail for c in self.checks if c.status == "FAIL"]

    @property
    def warnings(self) -> list[str]:
        return [c.detail for c in self.checks if c.status == "WARN"]


@dataclass
class NpuCheck:
    name: str
    status: str  # "OK", "FAIL", "WARN", "SKIP"
    detail: str


def validate_npu_strict(bundle_dir: Path) -> NpuValidationResult:
    """Run strict NPU validation on an ORT GenAI bundle directory.

    Checks that the bundle will actually execute on QNN HTP without
    CPU fallback.
    """
    checks: list[NpuCheck] = []

    # 1. Check for ONNX graphs
    onnx_files = list(bundle_dir.rglob("*.onnx"))
    if not onnx_files:
        checks.append(NpuCheck("onnx_present", "FAIL", "No .onnx files found in bundle"))
        return NpuValidationResult(passed=False, checks=checks)
    checks.append(NpuCheck("onnx_present", "OK", f"{len(onnx_files)} .onnx file(s) found"))

    # 2. Check for context-cache artifacts
    ctx_onnx = [f for f in onnx_files if "_ctx" in f.stem.lower()]
    qnn_bins = list(bundle_dir.rglob("*_qnn.bin")) + list(bundle_dir.rglob("*.bin"))
    if ctx_onnx:
        checks.append(NpuCheck(
            "context_cache",
            "OK",
            f"Context-cache graphs found: {[f.name for f in ctx_onnx]}",
        ))
    else:
        checks.append(NpuCheck(
            "context_cache",
            "FAIL",
            "No *_ctx.onnx context-cache graphs found. "
            "QNN HTP requires pre-compiled context-cache artifacts.",
        ))

    if qnn_bins:
        checks.append(NpuCheck(
            "qnn_binaries",
            "OK",
            f"QNN binary artifacts found: {[f.name for f in qnn_bins]}",
        ))
    else:
        checks.append(NpuCheck(
            "qnn_binaries",
            "WARN",
            "No QNN binary artifacts (.bin) found.",
        ))

    # 3. Check for .onnx.data files (must NOT be in NPU-only bundles)
    data_files = list(bundle_dir.rglob("*.onnx.data"))
    if data_files:
        checks.append(NpuCheck(
            "external_data",
            "FAIL",
            f"ONNX external data files present: {[f.name for f in data_files]}. "
            "NPU-only bundles must use context-cache artifacts, not raw float weights. "
            "The publish bundle must not contain .onnx.data.",
        ))
    else:
        checks.append(NpuCheck("external_data", "OK", "No .onnx.data files (good)"))

    # 4. Check genai_config.json for QNN session options
    genai_cfg_path = bundle_dir / "genai_config.json"
    if not genai_cfg_path.exists():
        # Try nested
        candidates = list(bundle_dir.rglob("genai_config.json"))
        if candidates:
            genai_cfg_path = candidates[0]

    if genai_cfg_path.exists():
        try:
            cfg = json.loads(genai_cfg_path.read_text(encoding="utf-8"))
            model_cfg = cfg.get("model", {})
            decoder = model_cfg.get("decoder", {})
            session = decoder.get("session_options", {})

            # Check provider
            provider_options = session.get("provider_options", [])
            has_qnn = False
            for po in (provider_options if isinstance(provider_options, list) else [provider_options]):
                if isinstance(po, dict) and "qnn" in str(po).lower():
                    has_qnn = True
                    break
            # Also check top-level provider specification
            if not has_qnn:
                ep_list = session.get("execution_provider", [])
                if isinstance(ep_list, list):
                    has_qnn = any("qnn" in str(ep).lower() for ep in ep_list)

            if has_qnn:
                checks.append(NpuCheck("genai_config_qnn", "OK", "genai_config references QNN EP"))
            else:
                checks.append(NpuCheck(
                    "genai_config_qnn",
                    "WARN",
                    "genai_config.json does not appear to reference QNN EP in session_options. "
                    "The Ollama runtime may set this automatically, but verify.",
                ))

            # Check disable_cpu_ep_fallback
            disable_fb = session.get("disable_cpu_ep_fallback")
            if disable_fb and str(disable_fb) == "1":
                checks.append(NpuCheck(
                    "cpu_fallback_disabled",
                    "OK",
                    "CPU EP fallback is disabled (strict NPU execution)",
                ))
            else:
                checks.append(NpuCheck(
                    "cpu_fallback_disabled",
                    "WARN",
                    "disable_cpu_ep_fallback not set to '1' in genai_config. "
                    "Model may silently fall back to CPU for some ops.",
                ))
        except Exception as e:
            checks.append(NpuCheck("genai_config_parse", "WARN", f"Could not parse genai_config: {e}"))
    else:
        checks.append(NpuCheck("genai_config", "WARN", "genai_config.json not found"))

    # 5. Check for dynamic shapes (if onnx is importable)
    try:
        from npu_model.core.shapes import has_dynamic_shapes

        for onnx_file in onnx_files:
            dynamic = has_dynamic_shapes(onnx_file)
            if dynamic:
                checks.append(NpuCheck(
                    f"shapes_{onnx_file.name}",
                    "FAIL",
                    f"{onnx_file.name} has dynamic dimensions: {dynamic}. "
                    f"QNN HTP requires all static shapes.",
                ))
            else:
                checks.append(NpuCheck(
                    f"shapes_{onnx_file.name}",
                    "OK",
                    f"{onnx_file.name}: all shapes static",
                ))
    except ImportError:
        checks.append(NpuCheck("shapes", "SKIP", "onnx not installed, skipping shape check"))

    # 6. Attempt QNN EP session load (if ORT is available)
    _check_qnn_session_load(checks, onnx_files)

    passed = all(c.status != "FAIL" for c in checks)
    return NpuValidationResult(passed=passed, checks=checks)


def _check_qnn_session_load(checks: list[NpuCheck], onnx_files: list[Path]) -> None:
    """Try to create ORT sessions with QNN EP and CPU fallback disabled."""
    try:
        import onnxruntime as ort
    except ImportError:
        checks.append(NpuCheck(
            "qnn_session_load",
            "SKIP",
            "onnxruntime not installed, skipping session load test",
        ))
        return

    providers = ort.get_available_providers()
    if "QNNExecutionProvider" not in providers:
        checks.append(NpuCheck(
            "qnn_session_load",
            "SKIP",
            "QNNExecutionProvider not available, skipping session load test",
        ))
        return

    for onnx_file in onnx_files:
        try:
            sess_opts = ort.SessionOptions()
            # Use session config entry (NOT provider option) per ORT docs
            sess_opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            qnn_provider_opts = {
                "backend_path": "QnnHtp.dll",
            }
            _sess = ort.InferenceSession(
                str(onnx_file),
                sess_options=sess_opts,
                providers=["QNNExecutionProvider"],
                provider_options=[qnn_provider_opts],
            )
            del _sess
            checks.append(NpuCheck(
                f"qnn_load_{onnx_file.name}",
                "OK",
                f"{onnx_file.name}: QNN EP session created (no CPU fallback)",
            ))
        except Exception as e:
            err_str = str(e)
            detail = f"{onnx_file.name}: QNN EP session failed: "
            if "dynamic" in err_str.lower() or "shape" in err_str.lower():
                detail += "dynamic shapes not supported by QNN HTP"
            elif "unsupported" in err_str.lower() or "op" in err_str.lower():
                detail += f"unsupported operation: {err_str[:200]}"
            else:
                detail += err_str[:300]
            checks.append(NpuCheck(f"qnn_load_{onnx_file.name}", "FAIL", detail))
