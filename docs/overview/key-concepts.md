# Key Concepts

A glossary of LARQL terminology. Terms are grouped by domain.

---

## Core Concepts

**Vindex** — A directory containing a transformer model's weights reorganized for queryability. Short for "vector index." The gate vectors become a KNN index, embeddings become a token lookup, and down projections become edge metadata. A vindex enables querying model knowledge without forward passes.

**LQL** — Lazarus Query Language. A SQL-like language for querying and editing vindexes. Supports `DESCRIBE`, `WALK`, `INFER`, `INSERT`, `DELETE`, `SELECT`, and 15+ other statement types.

**Feature** — A gate-output pair in an FFN layer. Each feature has a gate vector (what triggers it) and a down vector (what it outputs). Features are the fundamental unit of stored knowledge in a vindex.

**Gate Vector** — The row of W_gate that determines when a feature activates. High dot product with the residual stream means the feature fires. Gate vectors form the KNN index in a vindex.

**Down Vector** — The column of W_down that determines what a feature outputs. When a feature fires, its down vector is added to the residual stream, pushing logits toward certain tokens.

**Residual Stream** — The hidden state vector that flows through all layers of a transformer. Each layer reads from and writes to this stream. The residual stream at layer L contains all information the model has computed so far.

---

## Vindex Structure

**Extraction Level** — How much of the model to include in a vindex:

| Level | Size (f16) | Enables |
|-------|-----------|---------|
| Browse | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | ~6 GB | + INFER |
| All | ~10 GB | + COMPILE to safetensors |

**Layer Band** — A range of layers with similar function:

| Band | Gemma 3 4B | Function |
|------|-----------|----------|
| Syntax | L0-13 | Grammar, morphology, function words |
| Knowledge | L14-27 | Factual relations (capital, language, etc.) |
| Output | L28-33 | Answer formatting, token selection |

**Gate KNN** — K-nearest-neighbor search against gate vectors. Given a query embedding, returns the features with highest gate scores. This is how LARQL finds which features encode knowledge about an entity.

**Down Meta** — Per-feature metadata stored in `down_meta.bin`. Contains the top-1 output token, confidence score, and source information for each feature.

**Probe Label** — A relation label (like "capital" or "language") confirmed by probing experiments. Stored in `feature_labels.json`. Appears in DESCRIBE output when available.

---

## Queries and Operations

**DESCRIBE** — Query all knowledge edges for an entity. Returns features that fire when the entity appears in the residual stream, annotated with relation labels and output tokens.

**WALK** — Trace which features activate for a given prompt across all layers. Returns a layer-by-layer breakdown of feature activations with gate scores.

**INFER** — Run full forward-pass inference and return top-K predictions. Requires model weights in the vindex (extraction level `inference` or `all`).

**SELECT** — SQL-style query over feature metadata. Supports filtering by layer, relation, confidence, and ordering.

**TRACE** — Decompose the forward pass into per-layer attribution. Shows how attention and FFN contributions change the answer probability across layers.

---

## Knowledge Editing

**Patch** — A lightweight JSON file (.vlp) capturing INSERT/DELETE/UPDATE operations. Patches overlay an immutable base vindex without modifying it.

**Patch Overlay** — The in-memory layer that applies patches to a readonly base vindex. All mutations go through the overlay; base files are never modified.

**Constellation** — A multi-layer insert pattern where a fact is installed across 8 layers at low alpha (0.25). This validated regime achieves reliable retrieval without breaking neighboring facts.

**Alpha** — The scale factor for inserted down vectors. Higher alpha means stronger effect but more interference with existing knowledge. Validated range: 0.10-0.50.

**Refine Pass** — Gram-Schmidt orthogonalization of inserted gates against existing gates and decoy residuals. Prevents cross-fact interference at install time.

**COMPILE** — Bake patches into a standalone vindex or model. `COMPILE INTO VINDEX` produces a new vindex with facts embedded in the weights. `COMPILE INTO MODEL` exports to safetensors for use with HuggingFace Transformers.

---

## Inference Engine

**WalkFfn** — A zero-copy FFN implementation that reads down projections from mmap'd vindex files instead of computing them. Faster than dense inference (517ms vs 535ms on Gemma 3 4B) because the feature-major layout has better cache behavior.

**Walk Mode** — Inference using WalkFfn. Gate KNN selects active features, then outputs are read from the vindex. No down projection matmul.

**Dense Mode** — Standard inference with full matrix multiplications. More compute, same accuracy, slightly slower on Apple Silicon.

**BLAS-Fused Attention** — An attention implementation using BLAS `gemv` in an online-softmax loop. Never materializes the [seq, seq] attention matrix. 1.6x faster than the naive path at Gemma's head_dim.

**Compute Backend** — The hardware abstraction for matrix operations. CPU uses Apple Accelerate (AMX). Metal GPU uses custom Q4/Q8 shaders. Both produce identical results.

---

## Model Architectures

**ModelArchitecture** — A 82-method trait describing a transformer architecture. Covers tensor key patterns, norm types, activation functions, RoPE configuration, MoE routing, and MLA compression. Every supported model family implements this trait.

**Gated FFN** — An FFN with a separate gate projection: `output = down(up(x) * gate(x))`. Used by Gemma, Llama, Mistral, Qwen.

**MoE** — Mixture of Experts. An FFN with multiple expert sub-networks and a router that selects which experts to use. Supported formats: PerExpert (Mixtral) and PackedMxfp4 (GPT-OSS).

**MLA** — Multi-head Latent Attention. A compressed attention mechanism where keys and values share a low-rank latent. Used by DeepSeek.

---

## Serialization and Storage

**Safetensors** — The standard HuggingFace weight format. LARQL reads safetensors for extraction and writes them for COMPILE INTO MODEL.

**GGUF** — The llama.cpp quantized format. LARQL can dequantize GGUF files during extraction.

**f16 Storage** — Half-precision storage for vindex files. Halves file sizes with negligible accuracy loss. Enabled with `--f16` during extraction.

**mmap** — Memory-mapped file access. Gate vectors and weight files are mmap'd, meaning the OS pages them on demand. Only accessed data consumes RAM.

**Zero-Copy** — Data accessed directly from mmap'd buffers without heap allocation. Gate vectors are sliced from disk, not loaded to heap.

---

## Server and Ecosystem

**Vindexfile** — A declarative build specification for vindexes. Like a Dockerfile but for model knowledge. Supports `FROM`, `PATCH`, `INSERT`, `DELETE`, `LABELS`, `EXPOSE`.

**HuggingFace Hub** — Vindexes can be downloaded from and published to HuggingFace using the `hf://` URI scheme.

**USE REMOTE** — LQL command to connect to a vindex server. Queries are forwarded over HTTP; the server runs gate KNN and returns results.

**Session Isolation** — Per-session patch overlays on the server. Each session gets its own PatchedVindex; patches in one session don't affect others.

---

## Benchmarking

**kv-cache-benchmark** — A separate crate comparing KV cache strategies: Standard (FP16), TurboQuant (3-4 bit), Markov Residual Stream (bounded window), Hybrid RS+CA (cached static attention), and RS Graph Walk (no matmul).

**Walk Boundary Sweep** — A correctness proof that walks produce identical logits to dense inference across all layer boundaries.

**Criterion** — The benchmarking framework used for micro-benchmarks. Reports go to `target/criterion/`.
