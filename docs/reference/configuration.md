# Configuration Reference

Configuration options for vindexes, extraction, and runtime.

---

## Vindex Configuration (index.json)

The `index.json` file in a vindex contains configuration and metadata.

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
    "num_kv_heads": 8,
    "norm_eps": 1e-6,
    "rope_base": 10000.0
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
    "embeddings.bin": "sha256:def456...",
    "down_meta.bin": "sha256:789ghi..."
  },
  
  "provenance": {
    "source": "huggingface",
    "source_path": "google/gemma-3-4b-it",
    "extractor_version": "0.1.0"
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Vindex format version |
| `model_id` | string | Source model identifier |
| `family` | string | Architecture family |
| `created` | string | ISO 8601 timestamp |
| `config` | object | Model architecture config |
| `layer_bands` | object | Layer band ranges |
| `extract_level` | string | `browse`, `inference`, or `all` |
| `storage_dtype` | string | `f32` or `f16` |
| `checksums` | object | SHA256 checksums for binary files |
| `provenance` | object | Source information |

---

## Extraction Levels

| Level | Files Included | Size (Gemma 4B, f16) |
|-------|---------------|---------------------|
| `browse` | gate_vectors, embeddings, down_meta, tokenizer | ~3 GB |
| `inference` | + attn_weights, up_weights, down_weights, norms, lm_head | ~6 GB |
| `all` | + additional copies for COMPILE | ~10 GB |

### Capabilities by Level

| Operation | browse | inference | all |
|-----------|--------|-----------|-----|
| DESCRIBE | Yes | Yes | Yes |
| WALK | Yes | Yes | Yes |
| SELECT | Yes | Yes | Yes |
| INFER | No | Yes | Yes |
| TRACE | No | Yes | Yes |
| COMPILE INTO VINDEX | Yes | Yes | Yes |
| COMPILE INTO MODEL | No | No | Yes |

---

## Layer Bands

Layer bands group layers by function. Bands are model-specific.

### Default Bands

| Family | Syntax | Knowledge | Output |
|--------|--------|-----------|--------|
| Gemma 3 (4B) | 0-13 | 14-27 | 28-33 |
| Gemma 3 (12B) | 0-15 | 16-35 | 36-47 |
| Llama 3 (8B) | 0-10 | 11-25 | 26-31 |
| Llama 3 (70B) | 0-26 | 27-63 | 64-79 |

### Custom Bands

Override in index.json:

```json
{
  "layer_bands": {
    "syntax": [0, 10],
    "knowledge": [11, 25],
    "output": [26, 31]
  }
}
```

---

## Storage Dtype

| Dtype | Size | Precision | Recommended |
|-------|------|-----------|-------------|
| `f32` | 4 bytes | Full | Development |
| `f16` | 2 bytes | Half | Production |

Use `--f16` during extraction for half-size vindexes with negligible accuracy loss.

---

## Vindexfile Configuration

Declarative build specification.

```dockerfile
# Base vindex
FROM hf://chrishayuk/gemma-3-4b-it-vindex

# Apply patches
PATCH hf://medical-ai/drug-interactions@2.1.0
PATCH ./patches/company-facts.vlp

# Insert facts directly
INSERT ("Acme Corp", "headquarters", "San Francisco")
INSERT ("Acme Corp", "ceo", "Alice Smith")

# Delete facts
DELETE ("outdated-fact", "relation", "old-value")

# Apply probe labels
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest

# Expose extraction levels
EXPOSE browse inference
```

### Directives

| Directive | Description |
|-----------|-------------|
| `FROM <source>` | Base vindex (path or hf:// URL) |
| `PATCH <source>` | Apply a .vlp patch |
| `INSERT (e, r, t)` | Insert a fact |
| `DELETE (e, r, t)` | Delete a fact |
| `LABELS <source>` | Apply probe labels |
| `EXPOSE <levels>` | Which levels to expose |

---

## Feature Labels (feature_labels.json)

Probe-confirmed relation labels.

```json
{
  "L27_F9515": "capital",
  "L24_F4532": "language",
  "L25_F8891": "continent",
  "L18_F3321": "borders"
}
```

Key format: `L{layer}_F{feature}`

These labels appear in DESCRIBE output with `source: "probe"`.

---

## Relation Clusters (relation_clusters.json)

Discovered relation types from clustering.

```json
{
  "clusters": [
    {
      "id": 0,
      "name": "capitals",
      "features": ["L27_F9515", "L27_F8821", "L26_F4532"],
      "examples": ["Paris", "Berlin", "Tokyo"]
    },
    {
      "id": 1,
      "name": "languages",
      "features": ["L24_F4532", "L24_F5521"],
      "examples": ["French", "German", "Japanese"]
    }
  ]
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LARQL_CACHE_DIR` | `~/.cache/larql` | Cache directory |
| `LARQL_LOG_LEVEL` | `info` | Logging level |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache |
| `RAYON_NUM_THREADS` | CPU count | Thread pool size |

---

## Compute Configuration

### CPU Backend

Always available. Uses Apple Accelerate (AMX) on macOS.

### Metal Backend

Enable with `--features metal`:

```bash
cargo build --release --features metal
```

Auto-calibrates threshold between CPU and Metal at startup.

### Thread Pool

Set thread count:

```bash
RAYON_NUM_THREADS=4 larql repl
```

---

## Server Configuration

### Rate Limiting

Format: `<count>/<period>`

| Example | Meaning |
|---------|---------|
| `100/min` | 100 requests per minute |
| `10/sec` | 10 requests per second |
| `1000/hour` | 1000 requests per hour |

### TLS

Provide certificate and key:

```bash
larql serve vindex --tls-cert cert.pem --tls-key key.pem
```

Certificate should be PEM-encoded. Supports rustls.

### Session Timeout

Sessions expire after 1 hour of inactivity. Not configurable.

---

## Patch Configuration

### Patch File Format

```json
{
  "version": 1,
  "base_model": "google/gemma-3-4b-it",
  "base_vindex": "hf://chrishayuk/gemma-3-4b-it-vindex",
  "created": "2026-04-10T14:32:00Z",
  "description": "Medical domain facts",
  "operations": [...]
}
```

### Operation Types

| Type | Fields |
|------|--------|
| `insert` | entity, relation, target, layers, features, alpha, confidence, gate_vectors, down_vectors |
| `delete` | layer, feature |
| `update` | layer, feature, c_score |

---

## Quantization Formats

| Format | Bits | Block Size | Use |
|--------|------|------------|-----|
| f32 | 32 | N/A | Maximum precision |
| f16 | 16 | N/A | Default storage |
| Q8_0 | 8 | 32 | Attention weights |
| Q4_K | 4 | 256 | FFN weights (Ollama-compatible) |
| Q4_KF | 4 | 256 | llama.cpp-exact kernel |
| Q6_K | 6 | 256 | Higher precision Q4 |
| Q4_0 | 4 | 32 | Legacy format |

Format selection is automatic based on operation and backend.
