# Quickstart: Windows

A tested walkthrough for building and running LARQL on Windows.

> **Test Evidence:** See [quickstart-windows-testlog.md](quickstart-windows-testlog.md) for complete terminal output from the test run.

**Prerequisites:**
- Windows 10/11
- [Rust 1.82+](https://rustup.rs/)
- [Git for Windows](https://gitforwindows.org/) (includes Git Bash)
- Visual Studio Build Tools with C++ workload

---

## 1. Clone and Build

```bash
git clone https://github.com/az9713/larql-tutorial.git
cd larql-tutorial
```

### Build Core Components

On Windows, build the CLI and REPL packages explicitly (avoids server compilation issues):

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
   Compiling larql-core v0.1.0
   Compiling larql-vindex v0.1.0
   Compiling larql-inference v0.1.0
   Compiling larql-lql v0.1.0
   Compiling larql-cli v0.1.0
    Finished `release` profile [optimized] target(s) in 2m 15s
```

The binary is at `target/release/larql.exe`.

### Verify Installation

```bash
./target/release/larql --version
```

**Output:**
```
larql 0.1.0
```

---

## 2. Test the REPL

Start the interactive LQL environment:

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

Type `help` to see available commands:

```
LQL Commands:

  Lifecycle:
    EXTRACT MODEL <id> INTO <path>;     Decompile model → vindex
    COMPILE <vindex> INTO MODEL <path>;  Recompile vindex → weights
    USE <path>;                          Set active vindex
    USE MODEL <id>;                      Set active model (live weights)

  Query (pure vindex, no model needed):
    WALK <prompt> [TOP n];               Feature scan for a token
    SELECT ... FROM EDGES WHERE ...;     Query edges
    DESCRIBE <entity>;                   Knowledge about an entity
    EXPLAIN WALK <prompt>;               Feature trace (no attention)
    EXPLAIN INFER <prompt>;              Feature trace (with attention)

  Inference (requires model weights):
    INFER <prompt> [TOP n] [COMPARE];    Full prediction with attention

  Mutation:
    INSERT INTO EDGES (...) VALUES (...); Add edge
    DELETE FROM EDGES WHERE ...;          Remove edges
    UPDATE EDGES SET ... WHERE ...;       Modify edges

  Introspection:
    SHOW RELATIONS;                      List relation types
    SHOW LAYERS;                         Layer summary
    SHOW FEATURES <layer>;               Feature details
    SHOW MODELS;                         List vindexes
    STATS;                               Summary stats

  Meta:
    clear                                Clear the screen
    help, \h, \?                         Show this help
    exit, quit, \q                       Exit REPL
```

Exit with `quit` or Ctrl+D.

---

## 3. Get a Vindex

A **vindex** (vector index) is a decompiled model you can query without running inference.

**Supported models:** Gemma2, Gemma3, Gemma4, LLaMA, Mistral (GPT-2 not supported)

### Option A: Download Pre-built

```bash
./target/release/larql hf download chrishayuk/gemma-3-4b-it-vindex
```

> **Note:** Pre-built vindexes may not be available yet. If you get a 404 error, use Option B.

### Option B: Extract from HuggingFace Model (Recommended)

First download the model, then extract:

```bash
# Download model (~8 GB for Gemma 3 4B)
huggingface-cli download google/gemma-3-4b-it

# Extract vindex (~3 GB output)
./target/release/larql extract-index google/gemma-3-4b-it \
    -o gemma3-4b.vindex --f16
```

For a smaller test, try Gemma 2B (~4 GB download).

---

## 4. Query the Model

Load the vindex:

```sql
larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
```

Or for a local vindex:

```sql
larql> USE "gemma3-4b.vindex";
```

### DESCRIBE: What does the model know?

```sql
larql> DESCRIBE "France";
```

**Output:**
```
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27  (probe)
    language    → French               35.2  L24  (probe)
    continent   → Europe               14.4  L25  (probe)
    borders     → Spain                13.3  L18  (probe)
```

### WALK: Which features fire for a prompt?

```sql
larql> WALK "The capital of France is" TOP 5;
```

**Output:**
```
Walk: "The capital of France is" (top 5)
  L27  F9515   Paris        1436.9
  L24  F4532   French         26.1
  L25  F8891   Europe         14.4
  L26  F2201   the            12.3
  L23  F7721   country        11.8
```

### STATS: Vindex overview

```sql
larql> STATS;
```

**Output:**
```
Model:    google/gemma-3-4b-it
Family:   gemma3
Layers:   34
Features: 348,160
Bands:    syntax [0-13], knowledge [14-27], output [28-33]
Labels:   1,967 probe-confirmed relations
Size:     3.2 GB
```

---

## Problems Encountered and Solutions

### Problem 1: macOS-only BLAS dependency

**Error:**
```
error: library kind 'framework' is only supported on Apple targets
```

**Cause:** The original code used Apple's Accelerate framework unconditionally.

**Solution:** Platform-conditional BLAS dependencies in `Cargo.toml`:

**Files changed:**
- `crates/larql-compute/Cargo.toml`
- `crates/larql-inference/Cargo.toml`

```toml
# macOS: Accelerate framework
[target.'cfg(target_os = "macos")'.dependencies]
blas-src = { version = "0.10", features = ["accelerate"] }

# Windows: Intel MKL (pre-built)
[target.'cfg(target_os = "windows")'.dependencies]
blas-src = { version = "0.10", default-features = false, features = ["intel-mkl"] }
intel-mkl-src = { version = "0.8", features = ["mkl-static-lp64-seq"] }

# Linux: OpenBLAS
[target.'cfg(all(unix, not(target_os = "macos")))'.dependencies]
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

### Problem 2: Unresolved crate dependencies

**Error:**
```
error[E0433]: failed to resolve: use of undeclared crate or module `hf_hub`
error[E0433]: failed to resolve: use of undeclared crate or module `reqwest`
```

**Cause:** Dependencies declared but not properly linked in `larql-vindex`.

**Solution:** Explicitly add dependencies:
```bash
cd crates/larql-vindex
cargo add hf-hub@0.5
cargo add reqwest@0.12 --features blocking,json
```

**File changed:** `crates/larql-vindex/Cargo.toml`

### Problem 3: Missing tokenizers crate

**Error:**
```
error[E0433]: failed to resolve: use of undeclared crate or module `tokenizers`
```

**Solution:**
```bash
cd crates/larql-inference
cargo add tokenizers@0.21
```

**File changed:** `crates/larql-inference/Cargo.toml`

### Problem 4: larql-server won't compile (protobuf)

**Error:**
```
error LNK2019: unresolved external symbol ceilf
error LNK2019: unresolved external symbol nanf
error LNK1120: 11 unresolved externals
```

**Cause:** `protobuf-src` has linker issues on Windows.

**Workaround:** Don't build the server component:
```bash
# Instead of: cargo build --release
cargo build --release -p larql-cli -p larql-lql
```

The CLI and REPL work fully without gRPC.

---

## Summary of Code Changes

| File | Change |
|------|--------|
| `crates/larql-compute/Cargo.toml` | Platform-conditional BLAS (Intel MKL for Windows) |
| `crates/larql-inference/Cargo.toml` | Platform-conditional BLAS + tokenizers |
| `crates/larql-vindex/Cargo.toml` | Explicit hf-hub and reqwest dependencies |

---

## Next Steps

- [Query Knowledge](../guides/query-knowledge.md) — More LQL examples
- [Edit Knowledge](../guides/edit-knowledge.md) — INSERT, DELETE, patches
- [Troubleshooting](../troubleshooting/windows-build.md) — More Windows-specific issues
