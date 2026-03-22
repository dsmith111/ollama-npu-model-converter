# Plugin Development Guide

This guide shows how to add a new **backend** or **model adapter** plugin.

## Adding a new Backend

1. Create a new file, e.g. `src/npu_model/backends/my_backend.py`:

```python
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npu_model.backends.base import Backend
from npu_model.core.types import GraphBundle, BackendPreparedBundle, TargetSpec


@dataclass
class MyBackend(Backend):
    id: str = "my-backend"

    def resolve_target(self, target: str, env: dict[str, str]) -> TargetSpec:
        name = target or "auto"
        params = {"device": env.get("MY_DEVICE", "default")}
        return TargetSpec(backend_id=self.id, name=name, params=params)

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
        prepared_graphs = {}
        for name, p in graphs.graphs.items():
            dst = graphs_out / p.name
            shutil.copy2(p, dst)
            prepared_graphs[name] = dst

        artifacts_dir = out_dir / "backend_artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        return BackendPreparedBundle(
            graphs=prepared_graphs,
            artifacts_dir=artifacts_dir,
            backend_metadata={"backend": self.id, "target": target.name},
        )
```

2. Register it in `pyproject.toml`:

```toml
[project.entry-points."npu_model.backends"]
my-backend = "npu_model.backends.my_backend:MyBackend"
```

3. Reinstall: `pip install -e ".[dev]"`

4. Verify: `npu-model list-backends` should show `my-backend`.

## Adding a new Model Adapter

1. Create a new file, e.g. `src/npu_model/adapters/my_model.py`:

```python
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from npu_model.adapters.base import ModelAdapter
from npu_model.core.types import GraphBundle, ModelInfo


class MyModelAdapter(ModelAdapter):
    id = "my-model"

    def can_handle(self, model: ModelInfo) -> bool:
        # Return True when this adapter should handle the model
        return (model.model_type or "").lower() == "my-model"

    def export(self, model_dir: Path, out_dir: Path, *, export_config: dict[str, Any]) -> GraphBundle:
        out_dir.mkdir(parents=True, exist_ok=True)

        graph_dir = out_dir / "graphs"
        graph_dir.mkdir(exist_ok=True)
        graphs = {}
        for p in sorted(model_dir.glob("*.onnx")):
            dst = graph_dir / p.name
            shutil.copy2(p, dst)
            graphs[p.stem] = dst

        tok_dir = out_dir / "tokenizer"
        tok_dir.mkdir(exist_ok=True)

        return GraphBundle(
            graphs=graphs,
            tokenizer_dir=tok_dir,
            extra_files=[],
            metadata={"adapter": self.id},
        )
```

2. Register it in `pyproject.toml`:

```toml
[project.entry-points."npu_model.adapters"]
my-model = "npu_model.adapters.my_model:MyModelAdapter"
```

3. Reinstall and verify with `npu-model list-adapters`.

## Key rules

- **Adapters** must implement `can_handle(model: ModelInfo) -> bool` — the auto-selection system calls this.
- **Backends** must keep all hardware-specific details behind `TargetSpec.params` (opaque to the core).
- Plugin classes must have an `id: str` attribute.
- Heavy dependencies should be imported inside methods, not at module level, to keep startup fast.

## Adding a Runtime Format

Same pattern: subclass `RuntimeFormat` from `runtime_formats/base.py`, register via entry points under `npu_model.runtime_formats`.

## Adding a Quantization Strategy

Subclass or implement the interface from `quant/base.py`, register under `npu_model.quantizers`.
