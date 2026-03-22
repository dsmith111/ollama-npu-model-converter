# Runtime Format: ORT GenAI Folder

The `ort-genai-folder` runtime format produces a self-contained directory suitable for ORT GenAI inference.

## Bundle layout

```
ort_genai/
  genai_config.json          # (if provided by adapter)
  backend_metadata.json      # backend info for debugging
  onnx/
    *.onnx                   # ONNX graph files
  tokenizer/
    tokenizer.json           # tokenizer assets
    tokenizer_config.json
    special_tokens_map.json
    ...
  backend_artifacts/
    *.bin                    # context binaries, compiled artifacts
    *.json                   # backend-specific configs
```

## Validation

`validate_layout()` checks:

1. `onnx/` directory exists
2. At least one `.onnx` file is present in `onnx/`

Future validation can check:
- `genai_config.json` references files that exist
- tokenizer files are present and valid
- backend artifacts match expected names

## Assembly process

1. Copy ONNX graphs from `BackendPreparedBundle.graphs` into `onnx/`
2. Copy tokenizer assets from `GraphBundle.tokenizer_dir` into `tokenizer/`
3. Copy backend artifacts from `BackendPreparedBundle.artifacts_dir` into `backend_artifacts/`
4. If `genai_config.json` exists in artifacts, promote it to bundle root
5. Write `backend_metadata.json` at bundle root
6. Run `validate_layout()` to confirm the bundle is well-formed

---

## Ollama Packaging Contract v1

### Overview

`npu-model pack-ollama` produces a flat directory that can be used directly with
`ollama create <name> -f Modelfile` followed by `ollama push <name>`.

### Hard requirements (fail if missing)

| Requirement | Rule |
|---|---|
| ONNX model files | At least 1 `*.onnx` file |
| GenAI config | `genai_config.json` present |
| Tokenizer | `tokenizer.model` **or** `tokenizer.json` present |

### File pickup rules (allowlist)

Only these files are copied into the publish directory:

| Pattern | Reason |
|---|---|
| `*.onnx` | Primary model graphs |
| `*.bin` | Context binaries / weights |
| `*.json` | Configs (genai_config, tokenizer_config, etc.) |
| `tokenizer.model` | SentencePiece tokenizer (exact name match) |
| `*.jinja` | Chat templates |

### Deny list (never included)

These extensions cause Ollama to mis-detect the model format and are **excluded**:

- `*.gguf`
- `*.safetensors`
- `*.lib`
- `*.pdb`

### Modelfile

```
FROM .

# Safe defaults for ORT GenAI KV cache
PARAMETER num_ctx 512
PARAMETER num_predict 128
```

`num_ctx` and `num_predict` are configurable via `--num-ctx` and `--num-predict`.

### Output structure

```
<publish_dir>/
  Modelfile
  _publish_manifest.json     # file hashes, tool version, params
  embeddings.onnx            # (example)
  decoder.onnx               # (example)
  genai_config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  weights.bin                # (example)
  chat_template.jinja        # (if present)
```

All files are placed **flat at the root** — no subdirectories. This matches
Ollama's expectation when using `FROM .`.

### Atomic staging

The packer uses atomic staging to prevent empty output on failure:

1. Build into `<out>.staging.<pid>`
2. Validate the staged directory passes hard requirements
3. Remove existing destination (if any)
4. Rename staging → destination

### Validation

```powershell
npu-model validate --input <publish_dir> --as ollama-ortgenai
```

- **Errors** (exit code 2): missing ONNX, missing genai_config, missing tokenizer, presence of `.gguf`/`.safetensors`
- **Warnings** (exit code 0): missing `*.bin`, missing `chat_template.jinja`, missing optional configs

### Warnings (soft, non-blocking)

| Warning | Reason |
|---|---|
| No `*.bin` files | Context binaries may be needed at runtime |
| No `chat_template.jinja` | Chat mode may not work |
| Missing `config.json` | Optional HF config |
| Missing `generation_config.json` | Optional generation config |
| Missing `special_tokens_map.json` | Optional tokenizer metadata |
