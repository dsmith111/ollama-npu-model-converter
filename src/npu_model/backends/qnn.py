from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npu_model.backends.base import Backend
from npu_model.core.errors import NpuModelError
from npu_model.core.types import BackendCapabilities, GraphBundle, BackendPreparedBundle, TargetSpec


@dataclass
class QnnBackend(Backend):
    """
    Backend plugin: QNN.

    All QNN/HTP/Snapdragon-specific knobs live here behind TargetSpec.params.
    Core pipeline treats params as opaque.

    Compilation strategies (selected via compile_config["strategy"]):
      - "ort-ep-context": use ORT QNN EP context caching to generate compiled artifacts
      - "passthrough":    copy graphs as-is (default / fallback)
    """
    id: str = "qnn"

    # -------------------------------------------------------------------
    # resolve_target
    # -------------------------------------------------------------------

    def resolve_target(self, target: str, env: dict[str, str]) -> TargetSpec:
        name = target or "auto"
        params = {
            "backend_type": env.get("NPU_QNN_BACKEND_TYPE", "htp"),
            "backend_path": env.get("NPU_QNN_BACKEND_PATH", "QnnHtp.dll"),
        }
        return TargetSpec(backend_id=self.id, name=name, params=params)

    # -------------------------------------------------------------------
    # detect_environment
    # -------------------------------------------------------------------

    def detect_environment(self) -> BackendCapabilities:
        diag: list[str] = []
        compile_ok = False
        runtime_ok = False

        # Check if onnxruntime-qnn is importable
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "QNNExecutionProvider" in providers:
                compile_ok = True
                runtime_ok = True
                diag.append("QNNExecutionProvider available via onnxruntime")
            else:
                diag.append(
                    f"onnxruntime installed but QNNExecutionProvider not in providers: {providers}"
                )
        except ImportError:
            diag.append("onnxruntime not installed")

        # Check for QNN SDK env
        qnn_sdk = os.environ.get("QNN_SDK_ROOT")
        if qnn_sdk and Path(qnn_sdk).is_dir():
            diag.append(f"QNN_SDK_ROOT set: {qnn_sdk}")
        else:
            diag.append("QNN_SDK_ROOT not set or not a directory")

        return BackendCapabilities(
            backend_id=self.id,
            compile_available=compile_ok,
            runtime_available=runtime_ok,
            toolchain_info={
                "qnn_sdk_root": qnn_sdk,
            },
            diagnostics=diag,
        )

    # -------------------------------------------------------------------
    # prepare (copy-based passthrough, always works)
    # -------------------------------------------------------------------

    def prepare(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        backend_config: dict[str, Any],
    ) -> BackendPreparedBundle:
        out_dir.mkdir(parents=True, exist_ok=True)

        graphs_out = out_dir / "graphs"
        graphs_out.mkdir(exist_ok=True)
        prepared_graphs: dict[str, Path] = {}
        for name, p in graphs.graphs.items():
            dst = graphs_out / p.name
            shutil.copy2(p, dst)
            prepared_graphs[name] = dst
            # Copy co-located ONNX external data file if present
            data_file = p.parent / f"{p.name}.data"
            if data_file.exists():
                shutil.copy2(data_file, graphs_out / data_file.name)

        artifacts_dir = out_dir / "backend_artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for p in graphs.extra_files:
            if p.exists() and p.is_file():
                shutil.copy2(p, artifacts_dir / p.name)

        backend_metadata = {
            "backend": self.id,
            "target": target.name,
            "params": dict(target.params),
            "compile_strategy": "passthrough",
        }

        return BackendPreparedBundle(
            graphs=prepared_graphs,
            artifacts_dir=artifacts_dir,
            backend_metadata=backend_metadata,
        )

    # -------------------------------------------------------------------
    # compile (real compilation via ORT QNN EP context caching)
    # -------------------------------------------------------------------

    def compile(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        compile_config: dict[str, Any],
    ) -> BackendPreparedBundle:
        strategy = compile_config.get("strategy", "passthrough")

        if strategy == "passthrough":
            return self.prepare(graphs, out_dir, target=target, backend_config=compile_config)

        if strategy in ("ort-ep-context", "context-cache"):
            return self._compile_context_cache(graphs, out_dir, target=target, config=compile_config)

        raise NpuModelError(
            stage="backend",
            reason_code="UNKNOWN_COMPILE_STRATEGY",
            message=f"QNN backend: unknown compile strategy '{strategy}'",
            hint="Supported strategies: passthrough, context-cache",
        )

    def _compile_context_cache(
        self,
        graphs: GraphBundle,
        out_dir: Path,
        *,
        target: TargetSpec,
        config: dict[str, Any],
    ) -> BackendPreparedBundle:
        """Generate QNN context-cache artifacts via ORT QNN EP.

        Uses the documented session config entries:
          - session.disable_cpu_ep_fallback = 1  (NPU-only, no silent CPU fallback)
          - ep.context_enable = 1
          - ep.context_embed_mode = 0  (separate .bin file)
          - ep.context_file_path = <output path>

        Produces:
          - <model>_ctx.onnx   (lightweight wrapper graph)
          - <model>_qnn.bin    (compiled QNN context binary)

        The _ctx.onnx + _qnn.bin pair is what the HTP backend uses at runtime.
        The original model.onnx (+ .data) is NOT needed after this step.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise NpuModelError(
                stage="backend",
                reason_code="MISSING_ORT",
                message="onnxruntime is required for context-cache compilation",
                hint="pip install onnxruntime-qnn",
            )

        providers = ort.get_available_providers()
        if "QNNExecutionProvider" not in providers:
            raise NpuModelError(
                stage="backend",
                reason_code="NO_QNN_EP",
                message="QNNExecutionProvider not available in this onnxruntime build",
                hint="Install onnxruntime-qnn or ensure QNN libraries are on PATH.",
            )

        from npu_model.core.npu_invariant import (
            apply_npu_only_session_options,
            apply_context_cache_session_options,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        compile_dir = out_dir / "context_cache_compile"
        compile_dir.mkdir(exist_ok=True)

        backend_path = target.params.get("backend_path", "QnnHtp.dll")
        embed_mode = config.get("ep_context_embed_mode", "0")

        # Copy graphs + co-located .data files into compile dir
        for name, src_path in graphs.graphs.items():
            dst = compile_dir / src_path.name
            if not dst.exists():
                shutil.copy2(src_path, dst)
            data = src_path.parent / f"{src_path.name}.data"
            if data.exists():
                dst_data = compile_dir / data.name
                if not dst_data.exists():
                    shutil.copy2(data, dst_data)

        # Build session options using documented session config entries
        # (NOT provider_options — those are for QNN EP-specific knobs only)
        context_errors: list[str] = []
        for name, src_path in graphs.graphs.items():
            graph_in_compile = compile_dir / src_path.name
            ctx_output_path = compile_dir / f"{graph_in_compile.stem}_ctx.onnx"

            sess_options = ort.SessionOptions()
            apply_npu_only_session_options(sess_options)
            apply_context_cache_session_options(
                sess_options,
                context_file_path=str(ctx_output_path),
                embed_mode=embed_mode,
            )

            # provider_options contains ONLY QNN EP knobs (backend_path, perf settings)
            qnn_provider_opts: dict[str, str] = {
                "backend_path": backend_path,
            }
            # Pass through any extra QNN EP options from compile config
            for k, v in config.items():
                if k.startswith("qnn_"):
                    qnn_provider_opts[k[4:]] = str(v)

            try:
                _sess = ort.InferenceSession(
                    str(graph_in_compile),
                    sess_options=sess_options,
                    providers=["QNNExecutionProvider"],
                    provider_options=[qnn_provider_opts],
                )
                del _sess
            except Exception as e:
                context_errors.append(f"{name} ({src_path.name}): {e}")

        if context_errors:
            raise NpuModelError(
                stage="backend",
                reason_code="CONTEXT_CACHE_FAILED",
                message="QNN context-cache generation failed:\n  "
                + "\n  ".join(context_errors),
                hint=(
                    "Common causes:\n"
                    "  - Model not properly QDQ-quantized (use --quant qnn-qdq with calibration data)\n"
                    "  - Dynamic shapes present (should be fixed automatically)\n"
                    "  - Unsupported ops for QNN HTP\n"
                    "  - QNN SDK / runtime version mismatch"
                ),
            )

        # Collect generated artifacts
        artifacts_dir = out_dir / "backend_artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        ctx_graphs: dict[str, Path] = {}
        for p in sorted(compile_dir.glob("*")):
            if not p.is_file():
                continue
            if "_ctx" in p.stem and p.suffix == ".onnx":
                dst = artifacts_dir / p.name
                shutil.copy2(p, dst)
                key = p.stem.replace("_ctx", "")
                ctx_graphs[key] = dst
            elif p.suffix == ".bin":
                shutil.copy2(p, artifacts_dir / p.name)

        # Copy extra files (genai_config, etc.) from adapter
        for p in graphs.extra_files:
            if p.exists() and p.is_file():
                dst = artifacts_dir / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)

        # HARD REQUIREMENT: context-cache must produce ctx artifacts
        if not ctx_graphs:
            raise NpuModelError(
                stage="backend",
                reason_code="NO_CTX_ARTIFACTS",
                message="Context-cache compilation completed but no *_ctx.onnx artifacts were generated.",
                hint=(
                    "ORT session creation did not produce context-cache output files. "
                    "This usually means the model was not accepted by QNN HTP. "
                    "Ensure the model is properly QDQ-quantized with calibration data."
                ),
            )

        qnn_bins = list(artifacts_dir.glob("*.bin"))
        if not qnn_bins:
            raise NpuModelError(
                stage="backend",
                reason_code="NO_QNN_BINS",
                message="Context-cache compilation produced _ctx.onnx but no .bin artifacts.",
                hint="Expected QNN compiled binary alongside context ONNX wrapper.",
            )

        backend_metadata = {
            "backend": self.id,
            "target": target.name,
            "params": dict(target.params),
            "compile_strategy": "context-cache",
            "ep_context_embed_mode": embed_mode,
            "ctx_graphs": [p.name for p in ctx_graphs.values()],
            "qnn_bins": [p.name for p in qnn_bins],
        }

        return BackendPreparedBundle(
            graphs=ctx_graphs,
            artifacts_dir=artifacts_dir,
            backend_metadata=backend_metadata,
        )
