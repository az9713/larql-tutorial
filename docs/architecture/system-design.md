# System Design

High-level architecture of LARQL, data flows, and key design decisions.

---

## Overview

LARQL is a system for querying and editing neural network knowledge. It consists of 10 Rust crates organized in a layered architecture.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Layer                                │
│                                                                     │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │
│   │  larql-cli  │   │larql-server │   │     larql-python        │   │
│   │   (REPL,    │   │  (HTTP,     │   │     (PyO3 bindings)     │   │
│   │   commands) │   │   gRPC)     │   │                         │   │
│   └──────┬──────┘   └──────┬──────┘   └───────────┬─────────────┘   │
│          │                 │                      │                 │
└──────────┼─────────────────┼──────────────────────┼─────────────────┘
           │                 │                      │
           └────────────────┬┴──────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│                           │  Query Layer                            │
│                           ▼                                         │
│                    ┌─────────────┐                                  │
│                    │  larql-lql  │                                  │
│                    │  (parser,   │                                  │
│                    │  executor)  │                                  │
│                    └──────┬──────┘                                  │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
┌──────────────────┐ ┌─────────────┐ ┌─────────────────┐
│   larql-vindex   │ │larql-infer  │ │   larql-core    │
│   (KNN, patch,   │ │(forward     │ │   (Graph,       │
│    extract)      │ │ pass)       │ │    algorithms)  │
└────────┬─────────┘ └──────┬──────┘ └─────────────────┘
         │                  │
         └────────┬─────────┘
                  │
                  ▼
         ┌───────────────┐
         │ larql-compute │
         │ (BLAS, Metal, │
         │  quantization)│
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ larql-models  │
         │(architectures,│
         │ weight load)  │
         └───────────────┘
```

---

## Crate Responsibilities

| Layer | Crate | Responsibility |
|-------|-------|----------------|
| User | larql-cli | CLI commands, REPL |
| User | larql-server | HTTP/gRPC/WebSocket server |
| User | larql-python | Python bindings |
| Query | larql-lql | LQL parser and executor |
| Engine | larql-vindex | Vindex lifecycle, KNN, patches |
| Engine | larql-inference | Forward pass, attention |
| Engine | larql-core | Knowledge graph algorithms |
| Compute | larql-compute | Hardware abstraction |
| Foundation | larql-models | Architecture definitions |

---

## Data Flows

### Query: DESCRIBE

```
User: DESCRIBE "France"
         │
         ▼
    ┌─────────┐
    │ Parser  │  LQL → AST
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │Executor │  Session.execute()
    └────┬────┘
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
┌───────────┐              ┌─────────────┐
│ Embeddings│              │  Gate KNN   │
│  Lookup   │              │  per layer  │
└─────┬─────┘              └──────┬──────┘
      │                           │
      │  query vector             │  top-K features
      └──────────┬────────────────┘
                 │
                 ▼
          ┌─────────────┐
          │  Down Meta  │  Feature → output token
          │   Lookup    │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │Probe Labels │  Feature → relation label
          └──────┬──────┘
                 │
                 ▼
            Response JSON
```

### Inference: INFER (Walk Mode)

```
User: INFER "The capital of France is"
         │
         ▼
    ┌─────────────┐
    │  Tokenize   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Embedding  │  token_ids → vectors
    │   Lookup    │
    └──────┬──────┘
           │
           │ For each layer (0..34):
           │
    ┌──────┴──────────────────────────────┐
    │                                     │
    │      ┌─────────────┐                │
    │      │  Attention  │  BLAS-fused    │
    │      │   (GQA)     │  online softmax│
    │      └──────┬──────┘                │
    │             │                       │
    │             ▼                       │
    │      ┌─────────────┐                │
    │      │  RMS Norm   │                │
    │      └──────┬──────┘                │
    │             │                       │
    │             ▼                       │
    │      ┌─────────────┐                │
    │      │   WalkFfn   │                │
    │      │  gate KNN + │                │
    │      │  mmap down  │                │
    │      └──────┬──────┘                │
    │             │                       │
    │             ▼                       │
    │      ┌─────────────┐                │
    │      │ Residual Add│                │
    │      └──────┬──────┘                │
    │             │                       │
    └─────────────┼───────────────────────┘
                  │
                  ▼
           ┌─────────────┐
           │ Final Norm  │
           └──────┬──────┘
                  │
                  ▼
           ┌─────────────┐
           │   LM Head   │  → logits → top-K
           └──────┬──────┘
                  │
                  ▼
             Response JSON
```

### Mutation: INSERT

```
User: INSERT ("Atlantis", "capital", "Poseidon")
         │
         ▼
    ┌─────────────┐
    │  Embed      │  entity → vector
    │  Target     │  target → vector
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Find Free   │  Low-activation features
    │   Slots     │  across 8 layers
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Install    │  gate = entity_embed
    │Constellation│  down = target_embed × alpha
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Refine    │  Gram-Schmidt
    │    Pass     │  vs other facts + decoys
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Patch     │  Store in overlay
    │  Overlay    │  (base unchanged)
    └─────────────┘
```

---

## Key Design Decisions

### 1. Readonly Base + Patch Overlay

**Decision:** Base vindex files are never modified after extraction. All mutations go through an in-memory overlay.

**Rationale:**
- Safety: Can't corrupt the original extraction
- Shareability: Patches are small, vindexes are large
- Reversibility: Remove a patch to revert
- Composition: Stack multiple patches

**ADR:** [005-patch-overlay](../../crates/larql-vindex/docs/adr/005-patch-overlay.md)

### 2. WalkFfn Instead of Dense FFN

**Decision:** Replace the down projection matmul with gate KNN + mmap reads from feature-major layout.

**Rationale:**
- Faster: 517ms vs 535ms (better cache locality)
- Same accuracy: Proven bit-perfect across all 34 layers
- Lower memory: 1.3 GB vs 7 GB (no FFN weights in RAM)

**ADR:** [002-walk-ffn](../../crates/larql-inference/docs/adr/002-walk-ffn.md)

### 3. BLAS-Fused Attention

**Decision:** Use BLAS `gemv` in an online-softmax loop instead of materializing the [seq, seq] attention matrix.

**Rationale:**
- Memory: O(seq) instead of O(seq²)
- Speed: 1.6x faster at head_dim=256
- Flexibility: Supports GQA, softcap, attention capture

**ADR:** [001-fused-attention](../../crates/larql-inference/docs/adr/001-fused-attention.md)

### 4. Trait-Based Architecture Dispatch

**Decision:** All model architectures implement `ModelArchitecture` trait (82 methods). No branching in compute paths.

**Rationale:**
- Extensibility: Add new models without touching inference code
- Testability: Each architecture tested independently
- Clarity: Architecture-specific behavior is explicit

**ADR:** [001-weights-as-database](../../crates/larql-vindex/docs/adr/001-weights-as-database.md)

### 5. Multi-Layer Constellation for INSERT

**Decision:** Install facts across 8 layers at low alpha (0.25) instead of single layer at high alpha.

**Rationale:**
- Single-layer at alpha=0.25: too weak
- Single-layer at alpha=1.0: breaks neighbors
- Multi-layer at alpha=0.25: accumulates to strong signal without interference

**Documentation:** [training-free-insert.md](../training-free-insert.md)

### 6. LQL Instead of Library-Only API

**Decision:** Provide a SQL-like query language with REPL, not just Rust/Python APIs.

**Rationale:**
- Exploration: Interactive discovery of model knowledge
- Composability: Pipe statements together
- Remote transparency: Same syntax works locally and over HTTP
- Familiarity: SQL-like syntax is widely known

---

## Scaling Characteristics

### Memory

| Component | Scaling |
|-----------|---------|
| Gate KNN | O(layers × features × hidden) |
| Walk (mmap) | O(1) RSS, O(access) page faults |
| Attention | O(seq) per position |
| Patches | O(operations) |

### Latency

| Operation | Complexity |
|-----------|------------|
| DESCRIBE | O(layers × features) for KNN |
| WALK | O(layers × features) |
| INFER | O(layers × seq × hidden²) |
| INSERT | O(layers) + O(refine facts²) |

### Throughput

| Mode | Gemma 3 4B, M3 Max |
|------|-------------------|
| Browse queries (DESCRIBE) | ~3,000/sec |
| Walk queries | ~3,000/sec |
| Inference (walk) | ~2/sec |
| Inference (dense) | ~2/sec |

---

## External Dependencies

### Required

| Dependency | Purpose |
|------------|---------|
| Rust 1.82+ | Language |
| BLAS (Accelerate) | Matrix operations |

### Optional

| Dependency | Purpose | Flag |
|------------|---------|------|
| Metal | GPU acceleration | `--features metal` |
| HuggingFace Hub | Model download | default |
| PyO3 | Python bindings | larql-python |

---

## Failure Modes

### Recoverable

| Failure | Recovery |
|---------|----------|
| Parse error | Return error, session intact |
| KNN timeout | Return partial results |
| Network error (USE REMOTE) | Retry or switch to local |
| Patch conflict | Report, don't apply |

### Non-Recoverable

| Failure | Prevention |
|---------|------------|
| Checksum mismatch | Re-extract vindex |
| OOM during extraction | Increase memory or stream |
| Corrupted weights | Re-download model |

---

## Security Considerations

### Input Validation

- LQL parser rejects malformed input
- Path traversal prevented in file operations
- Entity names sanitized for file/URL use

### Authentication

- API key authentication for server
- Rate limiting per IP
- Session isolation for patches

### Data Integrity

- SHA256 checksums for all binary files
- Checksums verified on load
- Patches signed with model ID

---

## Future Directions

See [ROADMAP.md](../../ROADMAP.md) for planned features:

- CUDA backend for NVIDIA GPUs
- Streaming token generation
- Distributed vindex sharding
- Knowledge graph export
- Model distillation from vindexes
