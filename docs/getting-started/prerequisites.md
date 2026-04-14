# Prerequisites

What you need before building and running LARQL.

---

## Required

### Rust 1.82+

LARQL is written in Rust and requires a recent toolchain.

**Verify:**
```bash
rustc --version
# rustc 1.82.0 or higher
```

**Install:**
- macOS/Linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Windows: Download from [rustup.rs](https://rustup.rs)

### Git

For cloning the repository.

**Verify:**
```bash
git --version
```

---

## Optional

### Model Weights (for vindex extraction)

To extract a model into a vindex, you need either:

1. **HuggingFace model** — Downloaded automatically via `huggingface-cli` or cached in `~/.cache/huggingface/hub/`
2. **GGUF file** — llama.cpp quantized format (dequantized during extraction)
3. **Pre-built vindex** — Download from HuggingFace (no model weights needed)

For a quick start without downloading model weights, use a pre-built vindex:

```bash
larql hf download chrishayuk/gemma-3-4b-it-vindex
```

### HuggingFace CLI (for model download)

If extracting from HuggingFace models:

**Install:**
```bash
pip install huggingface_hub
huggingface-cli login
```

### Metal GPU (macOS only)

LARQL automatically uses Metal GPU acceleration on Apple Silicon.

**Verify:**
```bash
system_profiler SPDisplaysDataType | grep Metal
# Metal Support: Metal 3
```

Build with Metal support:
```bash
cargo build --release --features metal
```

---

## System Requirements

### Memory

| Operation | RAM Required |
|-----------|-------------|
| Build LARQL | ~4 GB |
| Extract Gemma 3 4B (streaming) | ~2 GB peak |
| Load browse-only vindex | ~100 MB + mmap |
| Load inference vindex | ~1.3 GB (walk mode) |
| Full dense inference | ~7 GB |

The vindex uses mmap, so only accessed data consumes RAM. A 3 GB vindex might use only 200 MB RSS during typical queries.

### Disk Space

| Item | Size |
|------|------|
| LARQL build (release) | ~500 MB |
| Gemma 3 4B model (safetensors) | ~8 GB |
| Gemma 3 4B vindex (f16, browse) | ~3 GB |
| Gemma 3 4B vindex (f16, inference) | ~6 GB |

### Operating Systems

| OS | Status |
|----|--------|
| macOS (Apple Silicon) | Full support, Metal GPU |
| macOS (Intel) | Full support, CPU only |
| Linux (x86_64) | Full support, CPU only |
| Windows | Builds, limited testing |

---

## Recommended Models for Getting Started

| Model | Size | Why |
|-------|------|-----|
| Gemma 3 4B | 8 GB | Best tested, probe labels available |
| Gemma 2 2B | 5 GB | Smaller, good for testing |
| Qwen 2.5 3B | 6 GB | Alternative architecture |

For the quickstart, Gemma 3 4B is recommended because it has the best probe label coverage.
