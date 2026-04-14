# Inference Engine

LARQL includes a full inference engine for running forward passes. This document explains the architecture, modes, and optimizations.

---

## Overview

The inference engine supports two modes:

| Mode | FFN Path | Speed (Gemma 4B) | Memory |
|------|----------|------------------|--------|
| **Walk** | Gate KNN + mmap'd down | 517ms | 1.3 GB |
| **Dense** | Full matmul | 535ms | 7 GB |

Walk mode is the default. It's faster and uses less memory.

---

## Architecture

```
                    Input Tokens
                         │
                         ▼
                ┌─────────────────┐
                │   Embeddings    │  (lookup from vindex)
                └────────┬────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               │
┌─────────────────┐                      │
│  Attention      │ ◀── RoPE, GQA,       │
│  (BLAS-fused)   │     softcap          │
└────────┬────────┘                      │
         │                               │
         ▼                               │
┌─────────────────┐      ┌───────────────┴───────────────┐
│  RMS Norm       │      │                               │
└────────┬────────┘      │                               │
         │               │                               │
         ▼               ▼                               │
┌─────────────────┐    ┌─────────────────┐               │
│  FFN (Walk)     │ OR │  FFN (Dense)    │               │
│  - gate KNN     │    │  - full matmul  │               │
│  - mmap'd down  │    │                 │               │
└────────┬────────┘    └────────┬────────┘               │
         │                      │                        │
         └──────────┬───────────┘                        │
                    │                                    │
                    ▼                                    │
              Residual Add  ◀────────────────────────────┘
                    │
                    │ (repeat for 34 layers)
                    │
                    ▼
           ┌─────────────────┐
           │   Final Norm    │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   LM Head       │  (project to vocab)
           └────────┬────────┘
                    │
                    ▼
               Top-K Logits
```

---

## BLAS-Fused Attention

Standard attention materializes a `[seq, seq]` matrix. For long sequences, this dominates memory.

LARQL uses an online-softmax loop with BLAS `gemv`:

```rust
for qi in 0..seq_len {
    // 1. Compute scores for all keys up to qi
    let scores = blas_gemv(&K[..=qi], &Q[qi]);  // AMX-accelerated
    
    // 2. Scale + optional softcap
    let scaled = scores.scale(1.0 / sqrt(head_dim));
    let capped = softcap.map_or(scaled, |c| c * tanh(scaled / c));
    
    // 3. Online softmax (numerically stable)
    let probs = online_softmax(&capped);
    
    // 4. Weighted sum of values
    let output = blas_gemv(&V[..=qi].T(), &probs);
    
    outputs[qi] = output;
}
```

Benefits:
- Never allocates `[seq, seq]` matrix
- 1.6x faster at head_dim=256 (Gemma 3)
- Supports GQA, softcap, attention weight capture

---

## WalkFfn: The Walk Mode FFN

The standard FFN:
```
output = W_down @ activation(W_gate @ x * W_up @ x)
```

WalkFfn replaces the down projection:

```rust
// 1. Gate and up projections (same as dense)
let gate_out = blas_gemv(&W_gate, &x);
let up_out = blas_gemv(&W_up, &x);
let activated = geglu(&gate_out, &up_out);

// 2. Gate KNN to find active features
let top_k = vindex.gate_knn(layer, &x, 8192);

// 3. Read down vectors from mmap (zero-copy)
let mut output = vec![0.0; hidden_size];
for (feature, score) in top_k {
    let down_vec = vindex.down_feature(layer, feature);  // mmap slice
    output.add_scaled(down_vec, activated[feature]);
}
```

### Why Walk is Faster

| Aspect | Dense | Walk |
|--------|-------|------|
| Down projection | 10240 × 2560 matmul | 8192 vector reads |
| Memory layout | Row-major (scattered) | Feature-major (sequential) |
| Cache behavior | Random access | Sequential mmap |

The feature-major layout in `down_features.bin` has better cache locality than the safetensors row-major layout.

### Building Walk Files

Walk mode requires feature-major weight files:

```bash
# Convert gate vectors from f16 to f32
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex

# Build feature-major down vectors
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex

# Build feature-major up vectors
cargo run --release -p larql-vindex --example build_up_features -- path/to/vindex
```

---

## Compute Backend

All GPU operations go through `ComputeBackend`:

```rust
pub trait ComputeBackend {
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;
    fn q4k_matvec(&self, weights: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32>;
    fn decode_token(&self, layers: &[FullPipelineLayer], x: &[f32], ...) -> Vec<f32>;
    fn prefill_q4(&self, layers: &[FullPipelineLayer], x: &[f32], seq_len: usize, ...) -> Vec<f32>;
    // ... 15 methods total
}
```

### CPU Backend

- BLAS via Apple Accelerate (AMX)
- NEON kernels for Q4 matvec
- Always available

### Metal Backend

- Custom shaders for Q4_K, Q6_K, Q8
- Fused gate+up FFN kernels
- KV cache management
- Cooperative SIMD reductions

Enable with `--features metal`.

### Auto-Calibration

At startup, the backend benchmarks CPU vs Metal for various sizes:

```rust
let backend = default_backend();  // Auto-selects and calibrates
```

Small operations use CPU (less overhead). Large operations use Metal.

---

## Per-Layer Architecture

Transformers aren't uniform. Each layer can have different:

- head_dim (Gemma 4)
- num_kv_heads (GQA ratio varies)
- RoPE base (dual bases in Gemma 3)
- Norm epsilon
- Activation function

LARQL captures this in `FullPipelineLayer`:

```rust
struct FullPipelineLayer {
    // Weights
    q_weights: QuantWeight,
    k_weights: QuantWeight,
    v_weights: QuantWeight,
    o_weights: QuantWeight,
    gate_weights: QuantWeight,
    up_weights: QuantWeight,
    down_weights: QuantWeight,
    norms: Vec<f32>,
    
    // Per-layer params
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    rope_base: f32,
    norm_eps: f32,
    activation: Activation,
    is_sliding: bool,
    softcap: Option<f32>,
}
```

No model-type branching in the compute path. Everything is parameterized.

---

## Quantization Support

| Format | Bits | Use Case |
|--------|------|----------|
| f32 | 32 | Maximum precision |
| f16 | 16 | Default storage |
| Q8_0 | 8 | Attention weights |
| Q4_K | 4 | FFN weights, Metal |
| Q4_KF | 4 | llama.cpp-exact kernel |
| Q6_K | 6 | Higher precision Q4 |

The pipeline auto-detects weight format and routes to appropriate kernels.

---

## Performance

### Component Breakdown (Gemma 3 4B, M3 Max)

| Component | Time | % of Total |
|-----------|------|------------|
| Logits (262K vocab gemv) | 221ms | 41% |
| FFN × 34 layers (walk) | 194ms | 36% |
| Attention × 34 layers | 84ms | 16% |
| Other (norms, residuals) | 18ms | 3% |
| **Total (walk)** | **517ms** | |
| **Total (dense)** | **535ms** | |

### Comparison with Ollama

```
LARQL Q4_KF:  8.5ms/token = 117 tok/s
Ollama:       10.3ms/token = 98 tok/s
Speedup:      17%
```

---

## Inference Modes

### `predict_walk`

Default mode. Uses WalkFfn for FFN layers.

```rust
let result = predict_walk(&model, &vindex, &tokens, top_k);
```

### `predict_dense`

Uses full matmul for FFN. Useful for comparison.

```rust
let result = predict_dense(&model, &tokens, top_k);
```

### `predict_honest`

Production pipeline with KV cache and GPU acceleration.

```rust
let result = predict_honest(&backend, &layers, &tokens, top_k);
```

### `generate`

Auto-regressive generation:

```rust
let tokens = generate(&model, &vindex, &prompt, max_tokens, temp);
```

---

## Tracing

The trace module captures per-layer attribution:

```rust
let trace = trace_forward(&model, &vindex, &tokens);

// Per-layer logit contributions
for layer in 0..34 {
    let attn_contrib = trace.attention_contribution(layer, "Paris");
    let ffn_contrib = trace.ffn_contribution(layer, "Paris");
    println!("L{}: attn={:.1}, ffn={:.1}", layer, attn_contrib, ffn_contrib);
}
```

Output shows where the model "decides" on an answer — the phase transition.

---

## Tiered Context

For long contexts, LARQL supports tiered storage:

| Tier | Storage | Per Window | 370K Tokens | vs KV Cache |
|------|---------|------------|-------------|-------------|
| Boundary residual | f32 | 10 KB | 18.9 MB | 3,100x |
| Tier 4 int8 | i8 | 58 KB | 110 MB | 511x |
| Full KV cache | f16 | ~30 MB | 56,000 MB | 1x |

Boundary residuals capture the state at window boundaries. The Markov property of the residual stream means this is sufficient to continue generation.

---

## Walk-Only Mode

Drop FFN weights to save memory:

```rust
let model = InferenceModel::load_walk_only("google/gemma-3-4b-it")?;
// RAM: 7 GB → 1.3 GB
// Requires: down_features.bin, up_features.bin in vindex
```

Attention and embedding weights stay in memory. FFN weights are never loaded — WalkFfn reads from the vindex.

---

## Related Docs

- [Inference Engine Doc](../inference-engine.md) — Detailed component breakdown
- [FFN Graph Layer](../ffn-graph-layer.md) — WalkFfn architecture
- [Walk Boundary Sweep](../walk-boundary-sweep.md) — Correctness proof
- [larql-inference README](../../crates/larql-inference/README.md) — Crate documentation
- [larql-compute README](../../crates/larql-compute/README.md) — Compute backend documentation
