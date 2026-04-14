# The Vindex Format

A vindex (vector index) is a transformer model's weights reorganized for queryability. This document explains what's inside, how it's built, and how it's queried.

---

## What Problem It Solves

Standard model weights are optimized for forward passes:
- Matrices stored row-major for matmul
- All layers in one file
- No indexing structure

LARQL needs the opposite:
- Gate vectors indexed for KNN search
- Down projections annotated with output metadata
- Layer-wise access for targeted queries

A vindex bridges this gap. Same information, different organization.

---

## Directory Layout

```
model.vindex/
├── gate_vectors.bin        # KNN index (W_gate per layer)
├── gate_vectors_q4.bin     # Q4 quantized gates (7x smaller)
├── embeddings.bin          # Token embeddings
├── down_meta.bin           # Feature metadata (binary)
├── index.json              # Config, provenance, checksums
├── tokenizer.json          # Tokenizer
├── feature_labels.json     # Probe-confirmed relation labels
├── relation_clusters.json  # Discovered relation types
│
│ # Inference weights (extraction level: inference/all)
├── attn_weights.bin        # Q, K, V, O projections
├── up_weights.bin          # FFN up projections
├── down_weights.bin        # FFN down projections
├── norms.bin               # Layer norms
├── lm_head.bin             # Output projection
│
│ # Optimized layouts (built separately)
├── interleaved.bin         # gate|up|down packed per layer
├── interleaved_q4.bin      # Q4 version (7x smaller)
├── down_features.bin       # Feature-major down vectors (for WalkFfn)
└── up_features.bin         # Feature-major up vectors
```

---

## Extraction Levels

| Level | Flag | LQL | Size (f16) | Enables |
|-------|------|-----|-----------|---------|
| Browse | `--level browse` | `EXTRACT ... INTO ...` | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | `--level inference` | `... WITH INFERENCE` | ~6 GB | + INFER |
| All | `--level all` | `... WITH ALL` | ~10 GB | + COMPILE to safetensors |

**Browse** includes: gate vectors, embeddings, down metadata, tokenizer, config.

**Inference** adds: attention weights, FFN weights, norms, lm_head.

**All** adds: additional weight copies for COMPILE INTO MODEL.

---

## Key Files

### gate_vectors.bin

The KNN index. Contains W_gate for all layers, contiguous in memory.

```
[Layer 0 gates: features × hidden_size]
[Layer 1 gates: features × hidden_size]
...
[Layer N gates: features × hidden_size]
```

For Gemma 3 4B: 34 layers × 10,240 features × 2,560 dims × 2 bytes (f16) = 1.8 GB.

Accessed via mmap. Only touched layers consume RSS.

### down_meta.bin

Per-feature metadata in binary format:

```rust
struct FeatureMeta {
    top_token: u32,      // Most likely output token
    c_score: f32,        // Confidence (from probing)
    source: u8,          // Label source: probe, cluster, none
    reserved: [u8; 3],
}
```

~12 bytes per feature. For 348K features: ~4 MB.

### index.json

Configuration and provenance:

```json
{
  "version": "0.3",
  "model_id": "google/gemma-3-4b-it",
  "family": "gemma3",
  "created": "2026-04-10T14:32:00Z",
  "config": {
    "num_layers": 34,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "vocab_size": 262144,
    "head_dim": 256,
    "num_heads": 16,
    "num_kv_heads": 8
  },
  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  },
  "extract_level": "inference",
  "storage_dtype": "f16",
  "checksums": {
    "gate_vectors.bin": "sha256:abc123...",
    "embeddings.bin": "sha256:def456..."
  }
}
```

### feature_labels.json

Probe-confirmed relation labels:

```json
{
  "L27_F9515": "capital",
  "L24_F4532": "language",
  "L25_F8891": "continent"
}
```

These appear in DESCRIBE output with `source: "probe"`.

---

## Supported Architectures

| Family | Models | Features |
|--------|--------|----------|
| Gemma 4 | 31B, E2B | Per-layer head_dim, PLE, KV sharing |
| Gemma 3 | 4B-27B | QK-norm, sliding window |
| Gemma 2 | 2B-27B | Softcapping, QK-norm |
| Llama | 7B-405B | GQA, RoPE scaling |
| Mistral | 7B | Sliding window |
| Mixtral | 8x7B, 8x22B | MoE (8 experts) |
| Qwen | 0.5B-72B | Attention bias |
| Phi | 2.7B-14B | Gated FFN |
| DeepSeek | V2, V3 | MoE + MLA |
| GPT-OSS | 120B | MoE (MXFP4) |
| GPT-2 | 117M-1.5B | Dense GELU |

---

## MoE Handling

For Mixture of Experts models (Mixtral, DeepSeek, GPT-OSS):

- Each expert's FFN is extracted as a contiguous range of features
- Expert index stored in feature metadata
- Gate KNN can be scoped to a specific expert
- Router weights included for full inference

Example: Mixtral 8x7B has 8 experts × 14,336 features = 114,688 features per layer.

---

## Quantization

| Format | Bits | Compression | Use |
|--------|------|-------------|-----|
| f32 | 32 | 1x | Maximum precision |
| f16 | 16 | 2x | Default extraction (`--f16`) |
| Q8_0 | 8 | 4x | Attention weights |
| Q4_K | 4 | 8x | FFN weights, gate KNN |
| Q4_0 | 4 | 8x | Legacy format |

Gate KNN supports both f32/f16 (BLAS matmul) and Q4 (Metal shader) paths. The compute backend auto-selects based on calibration.

---

## mmap and Zero-Copy

Vindex files are memory-mapped, not loaded to heap:

```rust
let mmap = unsafe { Mmap::map(&file)? };
let gates: &[f32] = bytemuck::cast_slice(&mmap[offset..]);
// gates points directly into the file
```

Benefits:
- **Fast startup** — No deserialization
- **Low RSS** — Only accessed pages consume memory
- **OS paging** — Hot data stays resident automatically

A 3 GB vindex might show 200 MB RSS during typical queries.

---

## Building a Vindex

### From HuggingFace

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --f16
```

Process:
1. Download/locate model in HF cache
2. Stream through safetensors (no full model load)
3. Extract gate, up, down matrices per layer
4. Build KNN index structure
5. Compute down_meta from down projections
6. Write all files + checksums

Peak memory: ~2 GB (one layer at a time).

### From GGUF

```bash
larql convert gguf-to-vindex model.gguf -o model.vindex --f16
```

Process:
1. Parse GGUF header
2. Dequantize each tensor (Q4/Q8 → f32)
3. Same extraction pipeline as safetensors

### From Vindexfile

```dockerfile
FROM hf://chrishayuk/gemma-3-4b-it-vindex
PATCH hf://medical-ai/drug-interactions@2.1.0
INSERT ("Acme Corp", "headquarters", "London")
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest
```

```bash
larql build .
```

---

## Querying a Vindex

### Gate KNN

```rust
let hits = vindex.gate_knn(layer, &query_vec, top_k);
// Returns: Vec<(feature_id, gate_score)>
```

Implementation: BLAS matmul (query × gates^T) or Q4 Metal shader.

Latency: 0.008ms per layer (f32), 0.5ms per layer (Q4 Metal).

### Walk

```rust
let trace = vindex.walk(&query_vec, &layers, top_k);
// Returns: Vec<WalkHit> with layer, feature, score, output_token
```

Scans multiple layers, aggregates top-K globally.

### Feature Lookup

```rust
let meta = vindex.feature_meta(layer, feature_id);
// Returns: FeatureMeta { top_token, c_score, source }
```

Direct index into down_meta.bin. O(1).

---

## Mutating a Vindex

Vindexes are readonly. Mutations go through `PatchedVindex`:

```rust
let base = VectorIndex::load_vindex(&path)?;
let mut patched = PatchedVindex::new(base);

// Mutations stored in overlay
patched.insert_feature(layer, feature, gate_vec, meta);
patched.set_down_vector(layer, feature, down_vec);

// Queries check overlay first, then base
let hits = patched.gate_knn(layer, &query, 10);

// Bake to new vindex
let baked = patched.bake_down();
baked.save_vindex(&output_path)?;
```

---

## Checksums and Integrity

Every binary file has a SHA256 checksum in index.json. On load:

```rust
let expected = index.checksums.get("gate_vectors.bin")?;
let actual = sha256_file(&path)?;
if expected != actual {
    return Err(VindexError::ChecksumMismatch);
}
```

This catches corruption from incomplete downloads or disk errors.

---

## Related Docs

- [Vindex Format Spec](../vindex-format-spec.md) — Complete file format specification
- [Vindex Operations Spec](../vindex-operations-spec.md) — Query and mutation API
- [Patches](patches.md) — The patch overlay system
- [larql-vindex README](../../crates/larql-vindex/README.md) — Crate documentation
