# Quickstart: Windows

A tested walkthrough for building and running LARQL on Windows.

> **Test Evidence:** See [quickstart-windows-testlog.md](quickstart-windows-testlog.md) for complete terminal output from the test run.

**Prerequisites:**
- Windows 10/11
- [Rust 1.82+](https://rustup.rs/)
- [Git for Windows](https://gitforwindows.org/) (includes Git Bash)
- Visual Studio Build Tools with C++ workload
- Python + `huggingface_hub`: `pip install huggingface_hub`

---

## 1. Clone and Build

```bash
git clone https://github.com/az9713/larql-tutorial.git
cd larql-tutorial
```

### Build Core Components

On Windows, build the CLI and REPL packages explicitly — this skips `larql-server` which has a known protobuf linker issue on Windows:

```bash
cargo build --release -p larql-cli -p larql-lql
```

**Expected output:**
```
   Compiling libc v0.2.172
   Compiling ndarray v0.16.1
   Compiling intel-mkl-src v0.8.1
   Compiling larql-models v0.1.0
   Compiling larql-compute v0.1.0
   Compiling larql-vindex v0.1.0
   Compiling larql-inference v0.1.0
   Compiling larql-lql v0.1.0
   Compiling larql-cli v0.1.0
    Finished `release` profile [optimized] target(s) in 2m 15s
```

The binary is at `target/release/larql.exe`.

### Verify

```bash
./target/release/larql --version
# larql 0.1.0
```

---

## 2. Start the REPL

```bash
./target/release/larql repl
```

**Output:**
```
   ╦   ╔═╗ ╦═╗ ╔═╗ ╦
   ║   ╠═╣ ╠╦╝ ║═╬╗║
   ╩═╝ ╩ ╩ ╩╚═ ╚═╝╚╩═╝
   Lazarus Query Language v0.1

larql>
```

Type `help` to see all LQL commands. Type `quit` or press Ctrl+D to exit.

---

## 3. Get a Vindex

A **vindex** is a decompiled model — gate vectors, embeddings, and token metadata — stored as a queryable directory. No GPU needed to query it.

**Supported model architectures:** Gemma2, Gemma3, Gemma4, LLaMA, Mistral

### Option A: TinyLlama (fastest, ~2.4 GB download, no auth required)

```bash
# Step 1: Download model files
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --include "*.json" "*.safetensors" "*.txt" "*.model"

# Step 2: Extract vindex (~6 min, produces ~620 MB)
./target/release/larql extract-index TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    -o tinyllama.vindex --f16
```

**Expected extraction output:**
```
Extracting: ...TinyLlama-1.1B-Chat-v1.0/snapshots/... → ./tinyllama.vindex (level=browse, dtype=f16)

── loading ──
  Streaming mode: 1 safetensors shards (mmap'd, not loaded)
  loading: 0.0s

── gate_vectors ──
  gate L 0: 0.1s
  ...
  gate L21: 0.1s
  gate_vectors: 1.3s

── embeddings ──
  embeddings: 0.4s

── down_meta ──
  Whole-word vocab: 22294 tokens (of 32000)
  down L 0: 21.9s
  ...
  down L21: 9.2s
  down_meta: 384.2s

── tokenizer ──
  tokenizer: 0.0s

── Summary ──
  Output: ./tinyllama.vindex
  Build time: 6.4min
  gate_vectors.bin: 484.0 MB
  embeddings.bin: 125.0 MB
  down_meta.bin: 10.4 MB
  tokenizer.json: 3.5 MB
  Total: 0.61 GB
```

### Option B: Gemma 3 4B (requires HuggingFace login, ~8 GB download)

```bash
# Login first (one-time)
huggingface-cli login

# Download
huggingface-cli download google/gemma-3-4b-it \
    --include "*.json" "*.safetensors" "*.txt" "*.model"

# Extract (~3 GB vindex)
./target/release/larql extract-index google/gemma-3-4b-it \
    -o gemma3-4b.vindex --f16
```

---

## 4. Query the Vindex

Load it in the REPL:

```bash
./target/release/larql repl
```

```sql
larql> USE "./tinyllama.vindex";
```

**Output:**
```
Using: ./tinyllama.vindex (22 layers, 123.9K features, model: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
```

### STATS: Vindex overview

```sql
larql> STATS;
```

**Actual output (TinyLlama):**
```
Model:           TinyLlama/TinyLlama-1.1B-Chat-v1.0

Features:        123.9K (5.6K x 22 layers)

Knowledge Graph:
  (no relation clusters found)

  By layer band:
    Syntax (L0-13):     78.8K features
    Knowledge (L14-27): 45.1K features
    Output (L28-33):    0 features

Index size:      622.9 MB
Path:            ./tinyllama.vindex
```

### WALK: Which features fire for a prompt?

```sql
larql> WALK "The capital of France is" TOP 5;
```

**Actual output (TinyLlama):**
```
Feature scan for "The capital of France is" (token "is", 22 layers, mode=hybrid (default))

  L 0: F578   gate=-0.2  top="net"            down=[net, imag, moth]
  L 0: F1061  gate=-0.1  top="was"            down=[was, Was, were]
  L 1: F784   gate=-0.1  top="Marx"           down=[Marx, witz, Felix]
  ...
  L21: F522   gate=-0.0  top="infty"          down=[infty, Hitler, typo]

888.0ms

Note: pure vindex scan (no attention). For inference use INFER.
```

> **Note:** WALK on TinyLlama shows raw feature activations. The clean knowledge-graph output shown in the main docs (`capital → Paris`) is from larger Gemma models with labeled probe features.

### DESCRIBE: Entity knowledge lookup

```sql
larql> DESCRIBE "France";
```

**Output (TinyLlama — small model):**
```
France
  (no edges found)
```

> **Note:** DESCRIBE requires labeled probe edges extracted from larger models. TinyLlama produces `(no edges found)` — this is expected. Use Gemma 3 4B for rich DESCRIBE output.

---

## 5. What Just Happened

You queried a model's internal feature representations without running inference:

1. `extract-index` streamed the safetensors file via mmap, extracted gate vectors (the routing weights of each FFN feature), embeddings, and down-projection token metadata.
2. `WALK "The capital of France is"` tokenized the prompt, computed a residual-stream proxy, and scanned all 22 layers for matching features by KNN.
3. No GPU. No forward pass. Just vector lookup.

---

## Problems Encountered and Solutions

### Problem 1: macOS-only BLAS dependency

**Error:**
```
error: library kind 'framework' is only supported on Apple targets
```

**Fix:** Platform-conditional BLAS in `crates/larql-compute/Cargo.toml` and `crates/larql-inference/Cargo.toml`:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
blas-src = { version = "0.10", features = ["accelerate"] }

[target.'cfg(target_os = "windows")'.dependencies]
blas-src = { version = "0.10", default-features = false, features = ["intel-mkl"] }
intel-mkl-src = { version = "0.8", features = ["mkl-static-lp64-seq"] }

[target.'cfg(all(unix, not(target_os = "macos")))'.dependencies]
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

### Problem 2: Missing hf-hub / reqwest / tokenizers

**Error:** `failed to resolve: use of undeclared crate or module 'hf_hub'`

**Fix:** Explicit deps added to `crates/larql-vindex/Cargo.toml` and `crates/larql-inference/Cargo.toml`.

### Problem 3: larql-server won't compile (protobuf-src)

**Error:** `error LNK2019: unresolved external symbol ceilf` (11 unresolved externals)

**Fix:** Skip the server crate: `cargo build --release -p larql-cli -p larql-lql`

### Problem 4: `extract-index google/gemma-3-4b-it` errors with "not a directory"

**Cause:** Model not in HuggingFace cache. Must download first.

**Fix:**
```bash
huggingface-cli download google/gemma-3-4b-it --include "*.json" "*.safetensors" "*.txt" "*.model"
```

---

## Next Steps

- [Query Knowledge](../guides/query-knowledge.md) — More LQL examples
- [Edit Knowledge](../guides/edit-knowledge.md) — INSERT, DELETE, patches
- [Troubleshooting](../troubleshooting/windows-build.md) — More Windows-specific issues
