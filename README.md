<div align="center">
  <img alt="ollama" width="240" src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7">
</div>

# Ollama - Legacy GPU + Think Parameter Fork

A patched build of [Ollama](https://github.com/ollama/ollama) (v0.15.4) with three additions:

1. **`PARAMETER think` support** - Persist thinking mode defaults in Modelfiles
2. **CUDA compute 3.5/3.7** - Tesla K80, K40, K20 support
3. **ROCm gfx803** - AMD RX 470/480/570/580 (Polaris) support

> The `PARAMETER think` feature has been submitted as [PR #14108](https://github.com/ollama/ollama/pull/14108) to upstream Ollama. The legacy GPU patches are maintained here for the community.

---

## What Problem Does This Solve?

### The Think Parameter Gap

Thinking-capable models (Nemotron, DeepSeek-R1, QwQ, etc.) auto-enable thinking in Ollama. There's no way to create a model variant that defaults to thinking **off** — you must pass `--think=false` every CLI invocation or `"think": false` in every API request. Apps like Open WebUI that don't expose the think parameter have no workaround at all.

This fork adds `PARAMETER think` to Modelfiles, so you can persist the default:

```
FROM nemotron-3-nano:30b
PARAMETER think false
```

```bash
ollama create nemotron-3-nano:30b-nothink -f Modelfile
ollama run nemotron-3-nano:30b-nothink   # No thinking — instant responses
ollama run nemotron-3-nano:30b           # Still thinks as before
```

### Legacy GPU Support

NVIDIA dropped CUDA compute 3.5/3.7 from default builds (Tesla K80, K40, K20), and AMD's gfx803 (Polaris RX 580/570/480/470) is excluded from ROCm builds. These GPUs still work for inference and are commonly found in budget multi-GPU setups, research labs, and secondhand server hardware.

This fork adds them back to the build configuration so they're detected and used by Ollama.

---

## PARAMETER think — Usage Guide

### Creating a No-Think Model Variant

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM nemotron-3-nano:30b
PARAMETER think false
EOF

# Create the model
ollama create nemotron-3-nano:30b-nothink -f Modelfile
```

Both models share the same weights — no extra disk space.

### Supported Values

| Value | Effect |
|-------|--------|
| `true`, `on`, `yes`, `1` | Enable thinking by default |
| `false`, `off`, `no`, `0` | Disable thinking by default |
| `high`, `medium`, `low` | Set thinking budget level |

### Priority Chain

Request-level always overrides model-level:

| Model Setting | API Request | Result |
|--------------|-------------|--------|
| `PARAMETER think false` | *(none)* | Thinking **OFF** |
| `PARAMETER think false` | `"think": true` | Thinking **ON** (override) |
| *(none)* | *(none)* | Thinking **ON** (auto-enable, original behavior) |
| *(none)* | `"think": false` | Thinking **OFF** |

### API Example

```bash
# Uses model default (no think parameter needed)
curl http://localhost:11434/api/chat -d '{
  "model": "nemotron-3-nano:30b-nothink",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "stream": false
}'

# Override model default for this request
curl http://localhost:11434/api/chat -d '{
  "model": "nemotron-3-nano:30b-nothink",
  "messages": [{"role": "user", "content": "Explain quantum computing"}],
  "stream": false,
  "think": true
}'
```

### Works With All Thinking Models

Any model with thinking capability can use this parameter:
- Nemotron 3 Nano
- DeepSeek-R1
- QwQ
- Any future thinking-capable model

---

## Legacy GPU Support

### NVIDIA CUDA 3.5/3.7

Adds `35-virtual` and `37-virtual` compute architectures to the CUDA 11 build preset.

**Supported GPUs:**
| GPU | Compute | VRAM |
|-----|---------|------|
| Tesla K80 | 3.7 | 2x 12GB |
| Tesla K40 | 3.5 | 12GB |
| Tesla K20 | 3.5 | 5GB |

### AMD ROCm gfx803

Adds `gfx803` to the ROCm 6 build targets and the CMakeLists.txt filter regex.

**Supported GPUs:**
| GPU | Architecture | VRAM |
|-----|-------------|------|
| RX 580 | gfx803 | 8GB |
| RX 570 | gfx803 | 4/8GB |
| RX 480 | gfx803 | 4/8GB |
| RX 470 | gfx803 | 4/8GB |

---

## Building From Source

### Prerequisites

- Go 1.24.1+
- CUDA Toolkit 11.x (for CUDA 3.5/3.7 support)
- cmake 3.21+

### Build the Go Binary

```bash
git clone https://github.com/sanchez314c/ollama.git
cd ollama
git checkout legacy-gpu-think-support

# Build with all CPU cores
CGO_ENABLED=1 go build -o ollama .
```

### Build CUDA Runner Libraries

```bash
# For CUDA 11 (required for compute 3.5/3.7)
cmake -B build --preset "CUDA 11"
cmake --build build --parallel

# Libraries output to build/lib/ollama/
```

### Build ROCm Runner Libraries

```bash
cmake -B build --preset "ROCm 6"
cmake --build build --parallel
```

### Quick Start (Go Binary Only)

If you already have working Ollama runner libraries (from a previous install), you only need to rebuild the Go binary for the `PARAMETER think` feature. The existing `.so` files in `/usr/local/lib/ollama/` are compatible:

```bash
# Build just the Go binary
CGO_ENABLED=1 go build -o ollama .

# Replace the installed binary
sudo systemctl stop ollama
sudo cp ollama /usr/local/bin/ollama
sudo systemctl start ollama
```

---

## Changes From Upstream

All changes are in the `legacy-gpu-think-support` branch. The `feature/parameter-think` branch contains only the think parameter changes (submitted as [PR #14108](https://github.com/ollama/ollama/pull/14108)).

### Files Modified

| File | Change |
|------|--------|
| `parser/parser.go` | Intercept `think` parameter before `FormatParams()` |
| `server/routes.go` | Check model options for stored think default (both handlers) |
| `cmd/cmd.go` | Check model parameters in `inferThinkingOption()` |
| `CMakePresets.json` | Add CUDA 35/37-virtual, ROCm gfx803 |
| `CMakeLists.txt` | Add gfx803 to AMDGPU_TARGETS regex |
| `ml/.../ggml-cuda/CMakeLists.txt` | Add 35/37-virtual to default architectures |

### Upstream Ollama

This fork is based on [Ollama v0.15.4](https://github.com/ollama/ollama/releases/tag/v0.15.4). For the full Ollama documentation, model library, API reference, and community integrations, see the [upstream repository](https://github.com/ollama/ollama).
