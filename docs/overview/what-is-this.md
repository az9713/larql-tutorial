# What is LARQL?

LARQL is a system for querying and editing neural network knowledge without running inference.

Traditional LLM interaction requires a GPU, a forward pass, and thousands of matrix multiplications just to answer "What is the capital of France?" LARQL extracts the model's knowledge into a queryable format, letting you browse facts like rows in a database.

## The Core Insight

A transformer's FFN layers encode knowledge as gate-output pairs. Each "feature" is a gate vector (what triggers it) paired with a down vector (what it outputs). These features act like database rows:

```
Feature F9515 at Layer 27:
  Gate:   activates when residual stream contains "capital of France"
  Output: pushes logits toward "Paris"
```

LARQL reorganizes model weights into a **vindex** (vector index) where:
- Gate vectors become a KNN index for fast lookup
- Down vectors become queryable outputs with metadata
- The embedding table becomes a token lookup

No forward pass needed. No GPU needed. Just KNN search.

## What LARQL Does

### 1. Extract: Model to Vindex

Convert any HuggingFace model into a vindex:

```bash
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex
```

This produces a ~3 GB directory containing reorganized weights. The original 8 GB model becomes a queryable database.

### 2. Browse: Query Knowledge

Use LQL (Lazarus Query Language) to explore what the model knows:

```sql
USE "gemma3-4b.vindex";

DESCRIBE "France";
-- capital → Paris (L27, score: 1436.9)
-- language → French (L24, score: 35.2)
-- continent → Europe (L25, score: 14.4)

WALK "The capital of France is" TOP 5;
-- Shows which features fire for this prompt
```

### 3. Edit: Insert/Delete Facts

Modify model knowledge without fine-tuning:

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon");

-- Knowledge is stored as a patch overlay
-- Base vindex files are never modified
```

### 4. Compile: Bake Changes

Export edited knowledge back to a model:

```sql
-- To a new vindex (facts baked into weights)
COMPILE CURRENT INTO VINDEX "gemma3-4b-edited.vindex";

-- Or to safetensors (loadable by HuggingFace Transformers)
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

## How It Works

### The Vindex Format

A vindex is a directory containing:

```
gemma3-4b.vindex/
├── gate_vectors.bin      # KNN index (W_gate rows)
├── down_meta.bin         # Feature metadata
├── embeddings.bin        # Token embeddings
├── index.json            # Config and provenance
├── tokenizer.json        # Tokenizer
├── feature_labels.json   # Probe-confirmed relation labels
└── relation_clusters.json # Discovered relation types
```

For inference, additional weight files are included:

```
├── attn_weights.bin      # Attention projections
├── up_weights.bin        # FFN up projections
├── down_weights.bin      # FFN down projections (editable)
├── norms.bin             # Layer norms
└── lm_head.bin           # Output projection
```

### Gate KNN Search

When you query `DESCRIBE "France"`, LARQL:

1. Embeds "France" using the model's embedding table
2. Runs KNN against gate_vectors.bin (0.008ms per layer)
3. For each matching feature, looks up the output token from down_meta.bin
4. Returns the knowledge edges with confidence scores

No matrix multiplication. No attention. Just KNN lookup and metadata reads.

### Walk vs Dense Inference

When you need actual predictions (not just browsing), LARQL offers two modes:

| Mode | What happens | Speed |
|------|-------------|-------|
| **Walk** | Gate KNN selects features, reads outputs from mmap'd vindex | 517ms |
| **Dense** | Standard forward pass with all matrix multiplications | 535ms |

Walk is actually **faster** than dense because the mmap'd feature-major layout has better cache behavior than the safetensors weight layout.

## What LARQL is NOT

- **Not a fine-tuning framework** — edits are surgical, not gradient-based
- **Not a vector database** — it stores model weights, not document embeddings
- **Not limited to factual knowledge** — the same mechanism encodes grammar, style, and reasoning patterns
- **Not a replacement for inference** — browse mode is fast but limited; full inference still requires weights

## Use Cases

### Model Debugging
See what a model knows about a topic without trial-and-error prompting.

### Knowledge Patching
Add domain facts (drug interactions, company data) without retraining.

### Model Comparison
Diff two vindexes to see what changed after fine-tuning.

### Lightweight Deployment
Serve knowledge queries at sub-millisecond latency with zero GPU.

### Research
Trace how knowledge flows through layers. Understand feature superposition.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      larql-cli                          │
│  (extract-index, build, serve, repl, convert, hf)       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                      larql-lql                          │
│  (parser, executor, REPL, USE REMOTE client)            │
└────────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───┴───────────┐  ┌─────┴──────────┐  ┌──────┴───────┐
│  larql-vindex │  │ larql-inference│  │ larql-server │
│  (KNN, patch) │  │ (forward pass) │  │  (HTTP/gRPC) │
└───────┬───────┘  └────────┬───────┘  └──────────────┘
        │                   │
┌───────┴───────────────────┴─────────┐
│            larql-compute            │
│    (BLAS, Metal GPU, Q4 shaders)    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────┴───────────────────┐
│            larql-models             │
│  (architectures, weight loading)    │
└─────────────────────────────────────┘
```

Ten Rust crates with clean dependencies. See [Crate Architecture](../concepts/crate-architecture.md) for details.

## Next Steps

- [Key Concepts](key-concepts.md) — glossary of LARQL terminology
- [Quickstart](../getting-started/quickstart.md) — build and run your first query
- [Onboarding](../getting-started/onboarding.md) — zero-to-hero conceptual guide
