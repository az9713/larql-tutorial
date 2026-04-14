# Extract a Model to Vindex

Convert a HuggingFace or GGUF model into a queryable vindex.

---

## Prerequisites

- LARQL built (`cargo build --release`)
- Model weights (HuggingFace, GGUF, or pre-built vindex)
- ~6 GB disk space for Gemma 3 4B

---

## From HuggingFace

### 1. Extract with Default Settings

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex
```

This creates a browse-only vindex (~5 GB) with f32 storage.

### 2. Extract with f16 (Recommended)

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --f16
```

Half the size (~3 GB), negligible accuracy loss.

### 3. Extract with Inference Weights

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level inference --f16
```

Enables `INFER` command (~6 GB).

### 4. Extract All Weights

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level all --f16
```

Enables `COMPILE INTO MODEL` (~10 GB).

### Expected Output

```
Extracting google/gemma-3-4b-it...
  Loading config.json... done
  Detected: gemma3 (34 layers, hidden=2560, inter=10240)
  Processing layer 0/34... done (2.1s)
  Processing layer 1/34... done (2.0s)
  ...
  Processing layer 33/34... done (2.1s)
  Writing gate_vectors.bin... done (1.8 GB)
  Writing embeddings.bin... done (1.3 GB)
  Writing down_meta.bin... done (4.2 MB)
  Computing checksums... done
  Writing index.json... done

Extraction complete: gemma3-4b.vindex (3.2 GB)
  Layers: 34
  Features: 348,160
  Bands: syntax [0-13], knowledge [14-27], output [28-33]
```

---

## From GGUF

Convert a llama.cpp quantized model:

```bash
larql convert gguf-to-vindex model.gguf -o model.vindex --f16
```

The GGUF is dequantized to f32 during extraction, then stored as f16.

---

## Download Pre-Built Vindex

Skip extraction by downloading from HuggingFace:

```bash
larql hf download chrishayuk/gemma-3-4b-it-vindex
```

Downloads to `~/.cache/larql/vindexes/`. Use directly:

```sql
USE "hf://chrishayuk/gemma-3-4b-it-vindex";
```

---

## Verify the Extraction

### Check Integrity

```bash
larql verify gemma3-4b.vindex
```

Output:
```
Verifying gemma3-4b.vindex...
  gate_vectors.bin: sha256 OK
  embeddings.bin: sha256 OK
  down_meta.bin: sha256 OK
  index.json: valid

Verification passed.
```

### Test a Query

```bash
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
```

Expected:
```
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27
    language    → French               35.2  L24
```

---

## Build Optimized Files (Optional)

For faster walk inference:

```bash
# Feature-major down vectors (required for WalkFfn)
cargo run --release -p larql-vindex --example build_down_features -- gemma3-4b.vindex

# Feature-major up vectors
cargo run --release -p larql-vindex --example build_up_features -- gemma3-4b.vindex

# Q4 gate vectors (faster Metal KNN)
cargo run --release -p larql-vindex --example build_gate_q4 -- gemma3-4b.vindex
```

---

## Extraction Levels Comparison

| Level | Flag | Size (f16) | Commands Enabled |
|-------|------|-----------|------------------|
| Browse | `--level browse` | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | `--level inference` | ~6 GB | + INFER, TRACE |
| All | `--level all` | ~10 GB | + COMPILE INTO MODEL |

Start with browse. Add inference weights later with:

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex \
    --level inference --f16 --merge-into gemma3-4b.vindex
```

---

## Troubleshooting

### "Model not found"

The model isn't in HuggingFace cache. Download first:

```bash
huggingface-cli download google/gemma-3-4b-it
```

### "Out of memory"

Extraction is streaming (one layer at a time), but some models require more peak memory. Try:

```bash
# Reduce thread count
RAYON_NUM_THREADS=2 larql extract-index ...
```

### "Checksum mismatch"

The extraction was interrupted. Delete the partial vindex and retry.

---

## Next Steps

- [Query Knowledge](query-knowledge.md) — Use DESCRIBE, WALK, SELECT
- [Edit Knowledge](edit-knowledge.md) — INSERT facts, create patches
- [Serve Vindex](serve-vindex.md) — Deploy over HTTP
