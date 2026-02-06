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

## Multi-GPU Setup: Modern + Legacy GPUs

If you're mixing modern GPUs (RTX 3090, 4090, etc.) with legacy GPUs (K80, K40), the key insight is: **use the modern GPU for compute, legacy GPUs for weight storage only.** Without this, you'll hit `CUBLAS_STATUS_ARCH_MISMATCH` errors and model stalls.

### The Problem

When a model splits across GPUs with different compute capabilities (e.g., 8.6 + 3.7), CUDA tries to run compute kernels on all GPUs. The legacy GPU's slow/incompatible kernels cause crashes or stalls.

### The Solution

```
┌──────────────────────────────────────────────────────────────┐
│                    MULTI-GPU ARCHITECTURE                     │
│                                                              │
│  CUDA_VISIBLE_DEVICES=2,0,1  (reorder: modern GPU = index 0) │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │   K80 #1    │  │   K80 #2    │  │     RTX 3090         │ │
│  │   11 GB     │  │   11 GB     │  │      24 GB           │ │
│  │  (index 1)  │  │  (index 2)  │  │    (index 0)         │ │
│  ├─────────────┤  ├─────────────┤  ├──────────────────────┤ │
│  │   WEIGHTS   │  │   WEIGHTS   │  │  WEIGHTS + COMPUTE   │ │
│  │    ONLY     │  │    ONLY     │  │  (KV cache, scratch)  │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│                                                              │
│  main_gpu=0 → All compute goes to the modern GPU (index 0)   │
│  num_gpu=999 → All GPUs used for tensor weight storage        │
└──────────────────────────────────────────────────────────────┘
```

### Step 1: Reorder GPUs in systemd Service

Edit `/etc/systemd/system/ollama.service` to put the modern GPU first:

```ini
[Service]
# GPU reorder: modern GPU's PCI index first, then legacy GPUs
# Check nvidia-smi to find your GPU indices
Environment="CUDA_VISIBLE_DEVICES=2,0,1"
Environment="OLLAMA_SCHED_SPREAD=1"
Environment="OLLAMA_NUM_GPU=999"
Environment="OLLAMA_FLASH_ATTENTION=true"
```

Find your GPU indices with `nvidia-smi` — the number in the left column. Put the modern GPU's index first in `CUDA_VISIBLE_DEVICES`.

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Step 2: Set main_gpu=0 on Every Model

This tells Ollama to send all compute (KV cache, scratch buffers, attention) to GPU index 0 (which is now your modern GPU after reordering):

```bash
# For a single model
cat > /tmp/Modelfile << 'EOF'
FROM your-model:tag
PARAMETER main_gpu 0
PARAMETER num_gpu 999
EOF
ollama create your-model:tag -f /tmp/Modelfile
```

Or batch-apply to all models:

```bash
for model in $(ollama list | tail -n +2 | awk '{print $1}'); do
    MODELFILE=$(ollama show "$model" --modelfile 2>/dev/null)
    if ! echo "$MODELFILE" | grep -q "main_gpu"; then
        { echo "$MODELFILE"; echo "PARAMETER main_gpu 0"; echo "PARAMETER num_gpu 999"; } > /tmp/Modelfile.update
        ollama create "$model" -f /tmp/Modelfile.update
        echo "Updated: $model"
    fi
done
```

### Step 3: Context Window Guidelines

KV cache lives on the main GPU. Larger context = more VRAM consumed on the modern GPU, leaving less for weights:

| Model Size | Recommended num_ctx | Notes |
|------------|-------------------|-------|
| 1-3B | 65536 | Full context, fits on modern GPU alone |
| 7-8B | 32768 | Good balance |
| 12-16B | 24576-32768 | Moderate context |
| 27-32B | 8192-16384 | KV cache is large, must share VRAM |

### Why This Works

1. `CUDA_VISIBLE_DEVICES=2,0,1` makes the RTX 3090 appear as GPU 0
2. `main_gpu=0` routes ALL compute operations to GPU 0 (the 3090)
3. `num_gpu=999` spreads tensor weights across all GPUs
4. K80s only hold weight data — no CUBLAS operations, no stalls
5. Result: ~46GB usable VRAM, modern GPU handles all the heavy lifting

### Tested Configuration

| GPU | Role | VRAM | Compute |
|-----|------|------|---------|
| RTX 3090 | Compute + Weights | 24 GB | 8.6 |
| Tesla K80 #1 | Weights only | 11 GB | 3.7 |
| Tesla K80 #2 | Weights only | 11 GB | 3.7 |
| **Total** | | **46 GB** | |

This setup runs 30B+ parameter models that wouldn't fit on the 3090 alone, with the K80s providing the extra VRAM needed for weight storage.

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
