# Architecture

## Pipeline stages

The `npu-model convert` command runs through these stages in order:

1. **Materialize source** — resolve `hf:<repo>` or local path to a local directory
2. **Inspect** — read `config.json` to produce a `ModelInfo`
3. **Select adapter** — auto-detect the right `ModelAdapter` based on `model_type` and `architectures`
4. **Adapter export** — adapter copies/exports ONNX graphs + tokenizer into a `GraphBundle`
5. **Quantization** — apply a `QuantizationStrategy` (MVP: passthrough no-op)
6. **Backend resolve target** — backend resolves `auto` or named target into a `TargetSpec`
7. **Backend prepare** — backend processes graphs into a `BackendPreparedBundle`
8. **Runtime assemble** — `RuntimeFormat` lays out the final bundle directory
9. **Manifest** — write `manifest.json` with file hashes + provenance

## Data flow

```
Input (HF or local)
  -> ModelInfo
  -> ModelAdapter.export() -> GraphBundle
  -> QuantizationStrategy.apply() -> GraphBundle (possibly modified)
  -> Backend.resolve_target() -> TargetSpec
  -> Backend.prepare() -> BackendPreparedBundle
  -> RuntimeFormat.assemble() -> bundle directory
  -> manifest.json
```

## Plugin system

Plugins are discovered via Python entry points (defined in `pyproject.toml`):

| Group                      | Base class       | Purpose                          |
|----------------------------|------------------|----------------------------------|
| `npu_model.adapters`       | `ModelAdapter`   | Model family export logic        |
| `npu_model.backends`       | `Backend`        | Hardware-specific preparation    |
| `npu_model.runtime_formats`| `RuntimeFormat`  | Output bundle layout             |
| `npu_model.quantizers`     | `QuantizationStrategy` | Quantization (optional)    |

## Error codes

Errors are structured as `NpuModelError(stage, reason_code, message, hint)`.

| Stage      | Code                  | Meaning                                      |
|------------|-----------------------|----------------------------------------------|
| source     | LOCAL_NOT_FOUND       | Local path doesn't exist                     |
| source     | LOCAL_MISSING_CONFIG  | No config.json in local directory             |
| source     | HF_BAD_SPEC          | Bad `hf:` spec format                        |
| source     | HF_DOWNLOAD_FAILED   | HF download error                            |
| inspect    | MISSING_CONFIG        | config.json not found in model dir            |
| inspect    | CONFIG_PARSE_FAILED   | config.json can't be parsed                  |
| adapter    | NO_ADAPTER            | No adapter matches the model                 |
| backend    | UNKNOWN_BACKEND       | Backend id not found                         |
| runtime    | UNKNOWN_RUNTIME_FORMAT| Runtime format id not found                  |
| runtime    | MISSING_ONNX_DIR     | Bundle missing onnx directory                |
| runtime    | NO_ONNX_FILES         | No .onnx files in bundle                    |
| quant      | UNKNOWN_QUANTIZER     | Quantizer id not found                       |
| registry   | PLUGIN_NO_ID          | Plugin class has no `id` attribute           |
| registry   | PLUGIN_LOAD_FAILED    | Plugin failed to load                        |
| pack       | BUNDLE_NOT_FOUND      | Bundle directory for packing not found       |

## Manifest format

`manifest.json` includes:

- `tool`: name + version
- `input`: source info (HF repo or local path)
- `plan`: adapter, backend, target, runtime, quantizer selections
- `backend_metadata`: backend-specific metadata
- `bundle`: format id + path
- `files`: list of `{path, size, sha256}` for all files in the bundle
