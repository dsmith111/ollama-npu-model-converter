# Known-Good Conversions

This document tracks verified end-to-end model conversions — from input to a working `ollama run`.

## Known Limitations

### ONNX External Data Files (`.onnx.data`)
FP16/FP32 exports produce models >2GB that require ONNX external data files (`model.onnx.data`).
**Ollama does not currently materialize `.onnx.data` files** when serving ORT GenAI models,
so these exported models will fail at runtime with `file_size: The system cannot find the file specified`.

**Workarounds:**
- Use `--mode prebuilt-ort-genai` with a pre-built model that has already been properly structured
- Use `--precision int4` (requires onnx-ir fix for serialization — track upstream onnx-ir releases)
- Wait for Ollama to add `.onnx.data` support in ORT GenAI model materialization

### INT4 Export
INT4 export via ORT GenAI builder v0.12.2 + onnx-ir v0.2.0 fails with a serialization error
(`producer_version=None`). This is an upstream onnx-ir bug. Track onnx-ir releases for a fix.

---

## Certification Matrix

| Model Family | HF Repo / Input | Precision | Ollama Tag | Status |
|---|---|---|---|---|
| Phi-3 (prebuilt) | local prebuilt ORT GenAI folder | prebuilt | `dsmith111/phi3:mini-qnn` | **Works** |
| Phi-3 (export FP16) | `hf:microsoft/Phi-3-mini-4k-instruct` | fp16 | `dsmith111/phi3:mini-qnn-exported` | Export works, Ollama run fails (.data) |
| Phi-3 (export INT4) | `hf:microsoft/Phi-3-mini-4k-instruct` | int4 | — | Blocked by onnx-ir bug |
| Llama | local prebuilt | prebuilt | `dsmith111/llama3.2:1b-qnn` | Pending |

---

## Recipe: Phi-3 Mini (FP16 export) — VALIDATED

### Command
```powershell
npu-model convert `
  --input hf:microsoft/Phi-3-mini-4k-instruct `
  --out .\out\phi3-fp16 `
  --mode export `
  --precision fp16 `
  --pack-ollama dsmith111/phi3:mini-qnn-exported
```

### Result
- [x] Export succeeded (builder ran to completion)
- [x] `ollama create` succeeded
- [x] `ollama push` succeeded — https://ollama.com/dsmith111/phi3:mini-qnn-exported
- [ ] `ollama run` — **FAILS** (missing `model.onnx.data` — Ollama doesn't materialize external data files)

### Output files
- `model.onnx` (249 KB — graph only, weights are external)
- `model.onnx.data` (7.6 GB — weights)
- `genai_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `Modelfile`
- `_publish_manifest.json`

### Versions
- npu-model: 0.1.0
- onnxruntime-genai: 0.12.2
- onnxruntime: 1.24.4
- torch: 2.10.0+cpu
- transformers: 5.3.0
- onnx-ir: 0.2.0
- Python: 3.14.3 (ARM64)
- OS: Windows 11 ARM64

---

## Recipe: Phi-3 Mini (prebuilt import)

### Prerequisites
- Pre-exported ORT GenAI folder with `.onnx` + `genai_config.json` + tokenizer

### Command
```powershell
npu-model convert `
  --input C:\models\phi3-mini-onnx-qnn `
  --out .\out\phi3-mini-qnn `
  --backend qnn `
  --target auto `
  --mode prebuilt-ort-genai `
  --pack-ollama dsmith111/phi3:mini-qnn `
  --num-ctx 512 `
  --num-predict 128
```

### Publish
```powershell
cd .\out\phi3-mini-qnn\ollama_publish
ollama create dsmith111/phi3:mini-qnn -f Modelfile
ollama push dsmith111/phi3:mini-qnn
```

### Expected output files
- [ ] `*.onnx` (at least 1)
- [ ] `genai_config.json`
- [ ] `tokenizer.json` or `tokenizer.model`
- [ ] `Modelfile`
- [ ] `_publish_manifest.json`

### Versions observed
- npu-model: `_____`
- onnxruntime: `_____`
- onnxruntime-genai: `_____`
- Python: `_____`

### Result
- [ ] `ollama create` succeeded
- [ ] `ollama push` succeeded
- [ ] `ollama run` produces output
- TTFT: `_____` ms
- tok/s: `_____`

---

## Recipe: Llama 3.2 1B (prebuilt import)

### Command
```powershell
npu-model convert `
  --input C:\models\llama-3.2-1b-onnx-qnn `
  --out .\out\llama-3.2-1b-qnn `
  --backend qnn `
  --target auto `
  --mode prebuilt-ort-genai `
  --pack-ollama dsmith111/llama3.2:1b-qnn
```

### Versions observed
- npu-model: `_____`
- onnxruntime: `_____`

### Result
- [ ] `ollama run` produces output
- TTFT: `_____` ms
- tok/s: `_____`

---

## Recipe: Phi-3 Mini (full export)

### Prerequisites
- `pip install npu-model[export]`

### Command
```powershell
npu-model convert `
  --input hf:microsoft/Phi-3-mini-4k-instruct `
  --out .\out\phi3-mini-export `
  --backend qnn `
  --target auto `
  --mode export `
  --pack-ollama dsmith111/phi3:mini-qnn-exported
```

### Versions observed
- npu-model: `_____`
- onnxruntime-genai: `_____`
- torch: `_____`
- transformers: `_____`

### Result
- [ ] Export succeeded
- [ ] `ollama run` produces output

---

## Using the publish script

For repeatable publishing, use the included script:

```powershell
.\scripts\publish_model.ps1 `
  -Input "C:\models\phi3-mini-onnx-qnn" `
  -Name "dsmith111/phi3:mini-qnn" `
  -Mode "prebuilt-ort-genai" `
  -OutDir ".\out\phi3-mini-qnn"
```

Add `-NoPush` to skip the push step. Add `-KeepWork` to preserve intermediate files.

---

## How to add a new entry

1. Run `npu-model doctor` to verify your environment
2. Run the conversion command
3. Fill in the template above with:
   - Exact command used
   - Tool + dependency versions (`npu-model doctor` output)
   - Output file list
   - Whether `ollama run` succeeded
   - Performance numbers (TTFT, tok/s) if available
4. Submit as a PR
