# npu-model (converter tool)

This repo is a plugin-based model conversion pipeline.

## What it does (MVP)
- Takes an input model (local directory or HuggingFace repo)
- Auto-selects a model adapter (Phi-3 / Llama to start)
- Runs a backend plugin (QNN to start)
- Assembles an ORT GenAI folder bundle (copy-based in MVP)
- Writes a manifest.json with hashes + provenance

## Install (dev)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Quick commands

```powershell
npu-model list-backends
npu-model list-adapters
npu-model list-runtime-formats

npu-model explain --input C:\models\phi3-mini-onnx-qnn --backend qnn --target auto --runtime ort-genai-folder

npu-model convert --input C:\models\phi3-mini-onnx-qnn --backend qnn --target auto --runtime ort-genai-folder --out .\out\phi3-mini-qnn
```

## Notes

* MVP primarily copies an already-prepared directory into a standardized bundle.
* Real ONNX export and backend compilation can be added inside adapters/backends without changing the core pipeline.
