# How LARQL Works: The Deep Technical Story

This document explains exactly how LARQL turns neural network weights into a queryable knowledge database. No prior transformer knowledge assumed. Every concept is grounded in code you can read.

---

## Table of Contents

1. [The Core Insight](#the-core-insight)
2. [Neural Networks 101: What You Need to Know](#neural-networks-101)
3. [Transformers: The Architecture LARQL Exploits](#transformers-the-architecture)
4. [The FFN Layer: Where Knowledge Lives](#the-ffn-layer)
5. [Extraction: Building a Vindex](#extraction-building-a-vindex)
6. [Gate KNN: The Key Operation](#gate-knn-the-key-operation)
7. [DESCRIBE: Querying Knowledge](#describe-querying-knowledge)
8. [WALK: Tracing Activations](#walk-tracing-activations)
9. [INFER: Full Forward Pass with WalkFfn](#infer-full-forward-pass)
10. [INSERT: Editing Knowledge](#insert-editing-knowledge)
11. [Why This Works: Mathematical Justification](#why-this-works)

---

## The Core Insight

Large language models store factual knowledge in specific weight matrices. LARQL reorganizes these matrices so you can search them with similarity queries instead of running the full neural network.

**The key insight:** In a transformer's FFN (feed-forward network) layer, each "feature" is a key-value pair:
- **Key (gate vector):** Determines *when* this feature activates
- **Value (down vector):** Determines *what* this feature outputs

LARQL extracts these pairs and builds a searchable index. Query "France" → find features that activate for "France" → read what they output → get "Paris", "French", "Europe".

---

## Neural Networks 101

### What is a Neural Network?

A neural network transforms input numbers into output numbers through a series of **matrix multiplications** and **nonlinear functions**.

```
Input [1024 numbers] → Matrix × Input → Nonlinearity → ... → Output [50000 numbers]
```

### Matrix Multiplication: The Core Operation

A matrix is a grid of numbers. Matrix multiplication combines them:

```
[a b]   [e f]   [ae+bg  af+bh]
[c d] × [g h] = [ce+dg  cf+dh]
```

When we multiply a vector (list of numbers) by a matrix, each output element is a **dot product**: multiply corresponding elements, sum them up.

```python
# Pseudocode for matrix-vector multiply
output[i] = sum(matrix[i, j] * input[j] for j in range(cols))
```

This is the operation that costs all the compute in neural networks.

### What "Weights" Means

The matrices in a neural network are called **weights**. They're learned during training. A 4-billion-parameter model has 4 billion numbers spread across its weight matrices.

---

## Transformers: The Architecture

Transformers are a specific neural network design. They process text by:

1. Converting words to vectors (embedding)
2. Letting words "look at" each other (attention)
3. Transforming each position through feedforward layers (FFN)
4. Repeating steps 2-3 many times (layers)
5. Predicting the next word

### The Layer Stack

A transformer has N layers (Gemma 3 4B has 34). Each layer has:

```
┌─────────────────────────────────┐
│         Attention Block         │  ← Words share information
├─────────────────────────────────┤
│            FFN Block            │  ← Each position transformed
└─────────────────────────────────┘
         ↓ (repeat N times)
```

### Embeddings: Words as Vectors

Each word (really, each **token**) gets a vector. "France" might be token 12847, which maps to a vector of 2560 numbers.

**Code reference:** `crates/larql-inference/src/forward/embed.rs:12-25`
```rust
pub fn embed_tokens_pub(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * scale;
        }
    }
    h
}
```

The embedding table is a matrix of shape `[vocab_size, hidden_size]`. Row lookup is O(1).

---

## The FFN Layer: Where Knowledge Lives

This is the crucial part. The FFN (feed-forward network) layer is where models store factual knowledge.

### Standard FFN Architecture

A gated FFN (used by Gemma, Llama, Mistral) has three weight matrices:

| Matrix | Shape | Purpose |
|--------|-------|---------|
| W_gate | [intermediate, hidden] | Controls which features activate |
| W_up   | [intermediate, hidden] | Provides input to multiply |
| W_down | [hidden, intermediate] | Transforms back to hidden size |

For Gemma 3 4B:
- hidden = 2560
- intermediate = 10240
- Each FFN layer: 3 × 2560 × 10240 × 4 bytes ≈ 315 MB

### The FFN Computation

```python
# Dense FFN forward pass
gate_scores = x @ W_gate.T        # [seq, intermediate]
up_scores   = x @ W_up.T          # [seq, intermediate]
activation  = SiLU(gate_scores) * up_scores   # GEGLU nonlinearity
output      = activation @ W_down.T           # [seq, hidden]
```

That's 3 matrix multiplications per layer. For 34 layers, that's 102 matmuls per forward pass.

### The Feature Interpretation

Here's the insight that makes LARQL possible:

Each column of W_down is a **feature**. Features are indexed 0 to 10239.

- **Gate row i** (`W_gate[i, :]`): A vector in hidden space. The dot product `x @ gate_row` measures how much the input "matches" this feature's trigger pattern.
- **Down column i** (`W_down[:, i]`): A vector in hidden space. When feature i activates, this vector gets added to the output (scaled by activation strength).

In other words:
- Gate vectors = **keys** (when to fire)
- Down vectors = **values** (what to output)

---

## Extraction: Building a Vindex

LARQL extracts these key-value pairs into a **vindex** (vector index).

### Streaming Extraction

The extraction code streams through safetensor files, processing one layer at a time:

**Code reference:** `crates/larql-vindex/src/extract/streaming.rs:26-40`
```rust
/// Build a vindex by streaming from safetensors files (no full model load).
///
/// Peak memory: embeddings + 1 layer of gate/down weights at a time.
pub fn build_vindex_streaming(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
```

For each layer:
1. Read W_gate matrix from safetensors (mmap'd, not copied)
2. Write gate vectors to `gate_vectors.bin` in feature-major order
3. Read W_down matrix
4. For each feature, compute which tokens it most strongly outputs
5. Write metadata to `down_meta.bin`

### What Gets Extracted

| File | Contents | Size (Gemma 4B, f16) |
|------|----------|---------------------|
| `gate_vectors.bin` | All W_gate rows, all layers | ~670 MB |
| `down_meta.bin` | Top-K output tokens per feature | ~200 MB |
| `embeddings.bin` | Token embedding matrix | ~1.3 GB |
| `down_features.bin` | All W_down columns (optional) | ~670 MB |
| `index.json` | Model config, checksums | ~2 KB |

### The VectorIndex Struct

**Code reference:** `crates/larql-vindex/src/index/core.rs:22-103`
```rust
pub struct VectorIndex {
    /// Per-layer gate vectors (heap mode): gate_vectors[layer] is (num_features, hidden_size).
    pub(crate) gate_vectors: Vec<Option<Array2<f32>>>,

    /// Mmap'd gate vector bytes (zero-copy mode). When set, gate_knn slices
    /// directly from this instead of using gate_vectors heap arrays.
    pub(crate) gate_mmap_bytes: Option<Arc<memmap2::Mmap>>,

    /// Per-layer, per-feature output token metadata from down projections.
    pub(crate) down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,

    /// Mmap'd down_meta.bin bytes (zero-copy mode).
    pub(crate) down_meta_mmap: Option<Arc<DownMetaMmap>>,

    // ... 40+ more fields for optimization paths
}
```

---

## Gate KNN: The Key Operation

This is the operation that replaces dense matrix multiplication.

### The Problem

Dense FFN: `gate_scores = x @ W_gate.T` costs O(intermediate × hidden) multiplications.

For Gemma 4B: 10240 × 2560 = **26.2 million** multiplications per layer per position.

### The Solution

**Key insight:** Most features have small activations. Only a few hundred (out of 10240) have significant activations for any given input.

Instead of computing all 10240 dot products, find the **top-K most similar** gate vectors using KNN (K-nearest neighbors).

### The gate_knn Implementation

**Code reference:** `crates/larql-vindex/src/index/gate.rs:108-134`
```rust
pub fn gate_knn(
    &self,
    layer: usize,
    residual: &Array1<f32>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    // HNSW path (graph-based approximate search)
    if self.hnsw_enabled.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(results) = self.gate_knn_hnsw(layer, residual, top_k) {
            return results;
        }
    }

    // Fast path: f32 mmap zero-copy (no allocation, no clone)
    if let Some(scores) = self.gate_knn_mmap_fast(layer, residual) {
        return Self::top_k_from_scores(&scores, top_k);
    }

    // Fallback: resolve_gate (copies data for heap/f16 paths)
    let gate = match self.resolve_gate(layer) {
        Some(g) => g,
        None => return vec![],
    };
    let view = gate.view(self.hidden_size);
    let scores = gemv(&view, residual);
    Self::top_k_from_scores(&scores, top_k)
}
```

### The gemv Function

**Code reference:** `crates/larql-vindex/src/index/gate.rs:14-21`
```rust
/// Matrix-vector multiply: view[N, hidden] × vec[hidden] → scores[N].
fn gemv(view: &ArrayView2<f32>, vec: &Array1<f32>) -> Array1<f32> {
    let hidden = vec.len();
    let x = vec.view().into_shape_with_order((1, hidden)).unwrap();
    let cpu = larql_compute::CpuBackend;
    // x[1, hidden] @ view[N, hidden]^T → [1, N]
    let result = cpu.matmul_transb(x, *view);
    Array1::from_vec(result.into_raw_vec_and_offset().0)
}
```

This uses BLAS (Basic Linear Algebra Subprograms) for hardware-accelerated matrix multiplication. On Apple Silicon, this goes through the Accelerate framework and AMX coprocessor.

### Top-K Selection

**Code reference:** `crates/larql-vindex/src/index/gate.rs:307-316`
```rust
fn top_k_from_scores(scores: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
    if k > 0 && k < indexed.len() {
        // O(N) selection algorithm, not O(N log N) full sort
        indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    indexed
}
```

Uses `select_nth_unstable_by` for O(N) partial selection instead of O(N log N) full sort.

---

## DESCRIBE: Querying Knowledge

DESCRIBE is the primary knowledge query operation. It shows what the model "knows" about an entity.

### How DESCRIBE Works

**Code reference:** `crates/larql-lql/src/executor/query.rs:317-367`

```
DESCRIBE "France"
    │
    ▼
Step 1: Tokenize "France" → token_id (e.g., 12847)
    │
    ▼
Step 2: Look up embedding → query_vector[2560]
    │
    ▼
Step 3: For each layer in knowledge band (14-27):
        gate_knn(layer, query_vector, top_k=20)
        → returns [(feature_id, gate_score), ...]
    │
    ▼
Step 4: For each high-scoring feature:
        Look up down_meta → "Paris", "French", "Europe"
    │
    ▼
Step 5: Format and return edges
```

### The Query Building Code

```rust
// Phase 1: load embeddings + tokenizer, build query vector
let (path, config, patched) = self.require_vindex()?;
let query = describe_build_query(entity, path)?;

// Phase 2: pick scan layers from band/layer filter
let bands = describe_resolve_bands(config);
let scan_layers = describe_scan_layers(&bands, &patched.loaded_layers(), band, layer);

// Phase 3: walk + collect edges
let trace = patched.walk(&query, &scan_layers, 20);
let mut edges = describe_collect_edges(&trace, entity);
```

### The walk Method

**Code reference:** `crates/larql-vindex/src/index/gate.rs:318-347`
```rust
/// Full walk: gate KNN at each layer, annotated with down token metadata.
pub fn walk(
    &self,
    residual: &Array1<f32>,
    layers: &[usize],
    top_k: usize,
) -> WalkTrace {
    let mut trace_layers = Vec::with_capacity(layers.len());

    for &layer in layers {
        let hits = self.gate_knn(layer, residual, top_k);
        let walk_hits: Vec<WalkHit> = hits
            .into_iter()
            .filter_map(|(feature, gate_score)| {
                let meta = self.feature_meta(layer, feature)?;
                Some(WalkHit {
                    layer,
                    feature,
                    gate_score,
                    meta,
                })
            })
            .collect();
        trace_layers.push((layer, walk_hits));
    }

    WalkTrace { layers: trace_layers }
}
```

---

## WALK: Tracing Activations

WALK shows which features activate for a prompt, without running full inference.

### The Execution Flow

**Code reference:** `crates/larql-lql/src/executor/query.rs:16-111`

```rust
pub(crate) fn exec_walk(
    &self,
    prompt: &str,
    top: Option<u32>,
    layers: Option<&Range>,
    mode: Option<WalkMode>,
    compare: bool,
) -> Result<Vec<String>, LqlError> {
    // Load embeddings and tokenizer
    let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)?;

    // Tokenize prompt, take last token
    let encoding = tokenizer.encode(prompt, true)?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let last_tok = *token_ids.last().unwrap();

    // Build query vector from embedding
    let embed_row = embed.row(last_tok as usize);
    let query: Array1<f32> = embed_row.mapv(|v| v * embed_scale);

    // Run walk across layers
    let trace = patched.walk(&query, &walk_layers, top_k);
    // ...
}
```

### Output Example

```
WALK "The capital of France is" TOP 10 LAYERS 20-27;

Feature scan for "The capital of France is" (token " is", 8 layers)

  L27: F9515  gate=+1436.9  top="Paris"    down=[Paris, Berlin, Tokyo]
  L26: F4532  gate=+26.1    top="the"      down=[the, a, its]
  L25: F8891  gate=+18.3    top="French"   down=[French, German, Italian]
```

---

## INFER: Full Forward Pass with WalkFfn

INFER runs actual inference using the WalkFfn backend, which replaces dense FFN with vindex lookups.

### The WalkFfn Backend

**Code reference:** `crates/larql-inference/src/vindex/walk_ffn.rs:21-28`
```rust
pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub top_k: usize,
    pub backend: Option<&'a dyn ComputeBackend>,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
}
```

### The Sparse Walk Path

**Code reference:** `crates/larql-inference/src/vindex/walk_ffn.rs:118-228`
```rust
/// Sparse walk FFN: zero matrix multiplications.
///
/// Per position:
///   1. gate_knn → top-K features with gate scores (HNSW graph search, no matmul)
///   2. For each feature: up_score = up_mmap[feat] · x  (dot product)
///   3. activation = silu(gate_score) * up_score          (GEGLU)
///   4. out += activation * down_mmap[feat]               (scaled vector add)
///
/// Operations: K dot products + K scaled adds per position. No matmuls.
fn walk_ffn_sparse(
    &self,
    layer: usize,
    x: &Array2<f32>,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let up_view = self.index.up_layer_matrix(layer)?;
    let down_view = self.index.down_layer_matrix(layer)?;

    for s in 0..seq_len {
        let x_row = x.row(s);

        // Gate KNN: find top-K features
        let hits = self.index.gate_walk(layer, &x_owned, self.top_k)
            .unwrap_or_else(|| self.index.gate_knn(layer, &x_owned, self.top_k));

        for (feat, gate_score) in hits {
            // Up: dot product with up vector
            let up_score = up_view.row(feat).dot(&x_row);
            
            // GEGLU activation
            let activated_gate = gate_score * sigmoid(gate_score);
            let act = activated_gate * up_score;

            // Down: scaled vector add (not a matmul!)
            if act.abs() > 1e-10 {
                let down_row = down_view.row(feat);
                out_row.scaled_add(act, &down_row);
            }
        }
    }
}
```

### Performance Comparison

| Mode | Time (Gemma 4B, M3 Max) | Memory |
|------|------------------------|--------|
| Dense FFN | 535ms | 7 GB |
| Walk FFN | 517ms | 1.3 GB |

Walk FFN is **faster** because:
1. Better cache locality (feature-major layout)
2. Mmap means OS handles paging
3. Only active features read from disk

---

## INSERT: Editing Knowledge

INSERT adds new facts by installing "constellations" - coordinated gate/up/down vectors across multiple layers.

### The Multi-Layer Constellation Pattern

Single layer at high alpha breaks neighboring facts. Multiple layers at low alpha accumulate to a strong signal:

```
Layer 20: alpha=0.25  ─┐
Layer 21: alpha=0.25   │
Layer 22: alpha=0.25   ├─→ Combined signal ≈ alpha=2.0
Layer 23: alpha=0.25   │
...                   ─┘
```

### How INSERT Works

1. **Find free slots**: Features with low activation across all training data
2. **Install gate**: Set gate vector = entity embedding (triggers on entity)
3. **Install down**: Set down vector = target embedding × alpha (outputs target)
4. **Refine**: Gram-Schmidt orthogonalization against existing facts

### The Patch Overlay

Mutations go to an in-memory overlay, not the base vindex:

**Code reference:** `crates/larql-vindex/src/index/core.rs:54-65`
```rust
/// Down vector overrides: custom output vectors for specific features.
pub(crate) down_overrides: HashMap<(usize, usize), Vec<f32>>,

/// Up vector overrides: custom up vectors for specific features.
pub(crate) up_overrides: HashMap<(usize, usize), Vec<f32>>,
```

The WalkFfn checks overrides before reading from mmap:

```rust
// Down: prefer override, fall back to mmap
if let Some(override_down) = self.index.down_override(layer, feat) {
    out_row.scaled_add(act, &override_down);
} else {
    let down_row = down_view.row(feat);
    out_row.scaled_add(act, &down_row);
}
```

---

## Why This Works: Mathematical Justification

### The Linear Representation Hypothesis

Recent research shows that concepts in language models are encoded as **directions** in the hidden state space. "France" isn't stored in one neuron - it's a direction that many neurons contribute to.

### Gate Vectors as Concept Detectors

Gate vectors learn to detect when their associated concept is present. The dot product `gate · x` measures alignment:

```
gate · x = ||gate|| × ||x|| × cos(θ)
```

High score = input aligns with gate direction = concept detected.

### Down Vectors as Concept Outputs

Down vectors encode what to add to the output when a concept is detected:

```
output += activation × down_vector
```

The down vector points toward tokens associated with that concept.

### Why KNN Preserves Accuracy

KNN finds the same high-scoring features that would dominate the dense computation. Features with small activations contribute negligibly to the output.

The error bound is:

```
||dense_output - sparse_output|| ≤ sum(|activation[i]| × ||down[i]||) for dropped features
```

For top-K with K ≥ 8192 (out of 10240), this error is typically < 0.01%.

### Verified Equivalence

LARQL includes a COMPARE mode that runs both dense and walk, comparing outputs:

```sql
INFER "The capital of France is" COMPARE;
```

Typical result: both produce "Paris" at 97%+ probability.

---

## Summary: The Complete Picture

1. **Transformers store knowledge in FFN weights** as key-value pairs (gate/down vectors)

2. **Extraction** reorganizes these weights into a searchable index:
   - Gate vectors become the KNN index
   - Down metadata stores output tokens
   - Everything is mmap'd for zero-copy access

3. **Query operations** use similarity search instead of dense matmul:
   - DESCRIBE: entity embedding → gate KNN → down lookup
   - WALK: prompt embedding → gate KNN per layer
   - INFER: full forward pass with KNN-based FFN

4. **Mutations** install new facts via gate/down overlays without touching the base vindex

5. **Performance** matches or beats dense inference at a fraction of the memory cost

The model IS the database. LARQL just reorganizes it for efficient queries.

---

## See Also

- [System Architecture](../architecture/system-design.md) - Crate structure and data flows
- [Vindex Format](../vindex-format-spec.md) - Binary file formats
- [LQL Specification](../lql-spec.md) - Query language grammar
- [Training-Free Insert](../training-free-insert.md) - Constellation algorithm details
