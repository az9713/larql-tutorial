# Crate Architecture

LARQL is organized into 10 Rust crates with clean dependency boundaries. This document explains what each crate does and how they interact.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         larql-cli                               │
│         (extract-index, build, serve, repl, hf, convert)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────┐
│                         larql-lql                               │
│              (parser, executor, REPL, USE REMOTE)               │
└───────┬─────────────────────┬─────────────────────┬─────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ larql-vindex  │     │larql-inference│     │ larql-server  │
│ (KNN, patch,  │     │(forward pass, │     │ (HTTP, gRPC,  │
│  extract)     │     │ attention)    │     │  WebSocket)   │
└───────┬───────┘     └───────┬───────┘     └───────────────┘
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐     ┌───────────────┐
│ larql-compute │     │  larql-core   │
│ (BLAS, Metal, │     │ (Graph, Edge, │
│  Q4 shaders)  │     │  algorithms)  │
└───────┬───────┘     └───────────────┘
        │
        ▼
┌───────────────┐
│ larql-models  │
│(architectures,│
│weight loading)│
└───────────────┘

┌───────────────┐     ┌────────────────────┐
│larql-python   │     │ kv-cache-benchmark │
│(PyO3 bindings)│     │  (5-way strategy   │
└───────────────┘     │   comparison)      │
                      └────────────────────┘
```

---

## Crate Details

### larql-models

**Purpose:** Describes what a model IS — no computation.

**Key types:**
- `ModelArchitecture` trait (82 methods)
- `ModelConfig` struct
- `ModelWeights` struct

**Responsibilities:**
- Parse config.json to detect architecture
- Map tensor key patterns (layer.N.mlp.gate_proj → index)
- Load weights from safetensors or GGUF
- Quantization format definitions (Q4_0, Q4_K, f16, bf16)

**Dependencies:** None (foundation crate)

**Tests:** 130 (all 12 architectures, quant formats)

```rust
use larql_models::{detect_architecture, load_model_dir};

let arch = detect_architecture(Path::new("model/"))?;
let weights = load_model_dir("model/")?;
```

---

### larql-compute

**Purpose:** Hardware abstraction for matrix operations.

**Key types:**
- `ComputeBackend` trait
- `CpuBackend`, `MetalBackend` structs
- `QuantFormat`, `QuantWeight` enums

**Responsibilities:**
- BLAS matmul via Apple Accelerate (AMX)
- Metal GPU shaders (48 kernels)
- Q4/Q6/Q8 quantized matvec
- KV cache management
- Auto-calibration (CPU vs GPU routing)

**Dependencies:** larql-models (for quant format definitions)

**Tests:** 83 (cross-backend correctness, Metal vs CPU)

```rust
use larql_compute::{default_backend, ComputeBackend};

let backend = default_backend();
let output = backend.q4k_matvec(&weights, &input, rows, cols);
```

---

### larql-core

**Purpose:** Knowledge graph engine.

**Key types:**
- `Graph` struct
- `Edge` struct
- `Schema`, `Node` types

**Responsibilities:**
- Triple-based edge storage
- Graph algorithms (shortest path, PageRank, BFS/DFS)
- LLM integration (ModelProvider trait)
- Serialization (JSON, MessagePack, packed binary)

**Dependencies:** None (independent from ML stack)

**Tests:** 167 (algorithms, serialization, BFS extraction)

```rust
use larql_core::{Graph, Edge, shortest_path};

let mut graph = Graph::new();
graph.add_edge(Edge::new("France", "capital", "Paris"));
let path = shortest_path(&graph, "France", "Paris");
```

---

### larql-vindex

**Purpose:** Vindex lifecycle — extract, load, query, mutate, patch.

**Key types:**
- `VectorIndex` struct
- `PatchedVindex` struct
- `VindexPatch`, `PatchOp` types

**Responsibilities:**
- Streaming extraction from safetensors/GGUF
- Gate KNN (BLAS matmul, Q4 shader, HNSW)
- Walk (multi-layer feature scan)
- Patch overlay (readonly base + in-memory mutations)
- Vindexfile parsing and execution
- HuggingFace Hub download/publish

**Dependencies:** larql-models, larql-compute

**Tests:** 146 (KNN, walk, patch, MoE, streaming)

```rust
use larql_vindex::{VectorIndex, PatchedVindex};

let index = VectorIndex::load_vindex(&path)?;
let mut patched = PatchedVindex::new(index);
let hits = patched.gate_knn(layer, &query, 10);
```

---

### larql-inference

**Purpose:** Forward pass execution.

**Key types:**
- `InferenceModel` struct
- `WalkFfn` struct
- `FullPipelineLayer` struct

**Responsibilities:**
- BLAS-fused attention (online softmax, no [seq,seq] matrix)
- WalkFfn (mmap'd down projection)
- Dense FFN (full matmul)
- Residual stream tracing
- Per-layer architecture parameterization

**Dependencies:** larql-models, larql-compute, larql-vindex

**Tests:** 96 (attention, FFN, trace, walkers)

```rust
use larql_inference::InferenceModel;

let model = InferenceModel::load("google/gemma-3-4b-it")?;
let result = model.predict(&tokens, 5);
```

---

### larql-lql

**Purpose:** LQL parser, executor, and REPL.

**Key types:**
- `Statement` enum (AST)
- `Session` struct
- `LqlError` enum

**Responsibilities:**
- Lexer and parser for LQL
- Execute statements against vindex/weight/remote backends
- REPL with history and completion
- USE REMOTE HTTP forwarding

**Dependencies:** larql-vindex, larql-inference

**Tests:** 272 (parser, executor, mutations, COMPILE)

```rust
use larql_lql::{Session, parse};

let mut session = Session::new();
session.execute(&parse("USE \"vindex\";")?)?;
let output = session.execute(&parse("DESCRIBE \"France\";")?)?;
```

---

### larql-server

**Purpose:** HTTP/gRPC server for vindex queries.

**Key types:**
- `AppState` struct
- Route handlers

**Responsibilities:**
- REST endpoints (DESCRIBE, WALK, SELECT, INFER, etc.)
- gRPC service (same functionality)
- WebSocket streaming (layer-by-layer DESCRIBE)
- Session-scoped patches
- Auth, rate limiting, TLS

**Dependencies:** larql-vindex, larql-inference, axum, tonic

**Tests:** 107 (endpoints, auth, rate limiting)

```bash
larql serve output/vindex --port 8080
```

---

### larql-cli

**Purpose:** Command-line interface.

**Commands:**
- `extract-index` — Model to vindex
- `build` — Vindexfile to vindex
- `serve` — Start HTTP server
- `repl` — Interactive LQL
- `lql` — Batch LQL execution
- `hf` — HuggingFace operations
- `convert` — Format conversion
- `verify` — Vindex integrity check

**Dependencies:** All other crates

```bash
larql extract-index google/gemma-3-4b-it -o vindex --f16
larql repl
larql serve vindex --port 8080
```

---

### larql-python

**Purpose:** Python bindings via PyO3.

**Key classes:**
- `WalkModel` — Load vindex, query, infer
- `Trace` — Residual stream decomposition

**Dependencies:** larql-vindex, larql-inference, pyo3

```python
import larql

model = larql.WalkModel("gemma3-4b.vindex")
edges = model.describe("France")
trace = model.trace("The capital of France is")
```

---

### kv-cache-benchmark

**Purpose:** KV cache strategy comparison (separate from main LARQL).

**Strategies:**
1. Standard KV (FP16)
2. TurboQuant (3-4 bit)
3. Markov Residual Stream
4. Hybrid RS + Cracked Attention
5. RS Graph Walk

**Dependencies:** Optional feature flag links to larql-inference

**Tests:** 66 (synthetic benchmarks, real model integration)

---

## Dependency Rules

1. **larql-models** has no dependencies (foundation)
2. **larql-compute** depends only on larql-models
3. **larql-core** is independent (graph algorithms, no ML)
4. **larql-vindex** and **larql-inference** depend on compute
5. **larql-lql** depends on vindex and inference
6. **larql-server** and **larql-cli** depend on everything
7. **kv-cache-benchmark** is optional, feature-gated

No circular dependencies. The graph is a DAG.

---

## Why This Structure

### Separation of Concerns

Each crate has one job:
- models: describe
- compute: calculate
- vindex: store and query
- inference: run forward pass
- lql: parse and execute
- server: network
- cli: user interface

### Testability

Each crate can be tested independently:
```bash
cargo test -p larql-models      # Architecture tests
cargo test -p larql-vindex      # KNN, patch tests
cargo test -p larql-lql         # Parser, executor tests
```

### Compile Times

Changes to larql-lql don't recompile larql-models. The dependency graph is shallow.

### Flexibility

Use just what you need:
```rust
// Just vindex queries, no inference
use larql_vindex::VectorIndex;

// Just inference, no LQL
use larql_inference::InferenceModel;
```

---

## Feature Flags

| Crate | Flag | Effect |
|-------|------|--------|
| larql-compute | `metal` | Enable Metal GPU backend |
| larql-core | `http` | Enable HTTP model provider |
| larql-core | `msgpack` | Enable MessagePack serialization |
| kv-cache-benchmark | `real-model` | Link to larql-inference |

---

## Adding a New Architecture

1. **larql-models:** Add `architectures/newarch.rs` implementing `ModelArchitecture`
2. **larql-models:** Update `detect.rs` to recognize the model_type
3. **larql-vindex:** Layer bands in `config/types.rs` (if different)
4. **Tests:** Add integration test in `tests/test_architectures.rs`

No changes needed in other crates — the trait abstracts everything.

---

## Related Docs

- Individual crate READMEs have detailed API documentation
- [System Design](../architecture/system-design.md) — High-level architecture
- [larql-models README](../../crates/larql-models/README.md)
- [larql-vindex README](../../crates/larql-vindex/README.md)
- [larql-inference README](../../crates/larql-inference/README.md)
