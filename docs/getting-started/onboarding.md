# Onboarding: Zero to Hero

A conceptual guide for newcomers. After reading this, you'll understand how LARQL works and why it's designed this way.

---

## The Mental Model

### If You've Used Vector Databases

LARQL is like a vector database, but instead of storing your documents, it stores the model's knowledge. The "documents" are features (gate-output pairs) that encode facts like "France → capital → Paris."

The key difference: vector databases use embeddings you created. LARQL uses embeddings the model created during training. You're querying what the model learned, not what you put in.

### If You've Used SQL

LARQL's LQL is SQL for neural networks. Instead of tables with rows, you have layers with features. Instead of `SELECT * FROM users WHERE name = 'Alice'`, you write `DESCRIBE "Alice"` to find all features that fire for Alice.

The key difference: SQL returns exact matches. LQL returns the model's associations — what the model "thinks" about a concept, not what you stored.

### If You've Done ML Research

LARQL operationalizes the superposition hypothesis. Each FFN feature is a gate-output pair that encodes a concept. The gate determines when it fires (input space), the output determines what it does (residual stream contribution).

LARQL makes these features queryable without running inference. Gate vectors become a KNN index. Down projections become metadata. The forward pass becomes a database query.

---

## The Core Abstraction: Features

A transformer FFN layer computes:

```
output = down(activation(gate(x) * up(x)))
```

Each row of `gate` and corresponding column of `down` forms a **feature**:

```
Feature F9515 at Layer 27:
├── Gate vector: 2560-dimensional, determines when this feature fires
├── Up vector: projects input to intermediate space
└── Down vector: 2560-dimensional, added to residual stream when feature fires
```

When the residual stream contains information about "the capital of France," feature F9515's gate fires (high dot product), and its down vector pushes logits toward "Paris."

**Key insight:** The gate vector IS the query that triggers this knowledge. The down vector IS the answer.

---

## What a Vindex Contains

A vindex reorganizes model weights for queryability:

| Original | Vindex | Purpose |
|----------|--------|---------|
| model.safetensors/layers.N.mlp.gate_proj | gate_vectors.bin | KNN index for feature lookup |
| model.safetensors/layers.N.mlp.down_proj | down_meta.bin | Output tokens + metadata |
| model.safetensors/model.embed_tokens | embeddings.bin | Token → embedding lookup |
| config.json | index.json | Architecture info |
| tokenizer.json | tokenizer.json | Unchanged |

The reorganization enables:

1. **Gate KNN** — Find features by what triggers them
2. **Token lookup** — Decode feature outputs to readable tokens
3. **Layer bands** — Know which layers encode syntax vs. knowledge vs. output

---

## How Queries Work

### DESCRIBE: "What does the model know about X?"

```sql
DESCRIBE "France";
```

Step by step:

1. **Embed** — Convert "France" to a 2560-dimensional vector using the embedding table
2. **KNN per layer** — For each knowledge layer (14-27 in Gemma 3), compute dot product with all gate vectors
3. **Top-K** — Keep features with highest gate scores
4. **Decode** — Look up each feature's output token from down_meta.bin
5. **Label** — Annotate with probe-confirmed relation labels if available

Result: A list of (relation, target, score, layer) tuples.

**Latency:** ~0.3ms for all 14 knowledge layers.

### WALK: "What features fire for this prompt?"

```sql
WALK "The capital of France is" TOP 10;
```

Similar to DESCRIBE, but:

1. **Tokenize** — Convert prompt to token IDs
2. **Approximate residual** — Use last token's embedding (simplified; full inference uses actual residual)
3. **KNN all layers** — Scan all 34 layers, not just knowledge band
4. **Top-K global** — Return highest-scoring features across all layers

Result: Which features the model would activate if it processed this prompt.

### INFER: "What would the model predict?"

```sql
INFER "The capital of France is" TOP 5;
```

This runs actual inference:

1. **Forward pass** — Embed, attention, FFN for all 34 layers
2. **WalkFfn** — Gate KNN selects active features, outputs read from mmap'd vindex
3. **Logits** — Project final residual to vocabulary
4. **Top-K** — Return highest probability tokens

**Latency:** ~517ms (walk mode) or ~535ms (dense mode).

The key insight: Walk mode is actually faster because the mmap'd feature-major layout has better cache behavior than standard weight storage.

---

## Knowledge Editing: How INSERT Works

When you run:

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon");
```

LARQL doesn't retrain the model. Instead:

1. **Embed the target** — Get the embedding vector for "Poseidon"
2. **Embed the entity** — Get the embedding vector for "Atlantis"
3. **Find free slots** — Identify features with low activation across diverse inputs
4. **Install gate** — Set the gate vector to trigger on "Atlantis" context
5. **Install down** — Set the down vector to push toward "Poseidon"
6. **Multi-layer** — Repeat across 8 layers (the validated constellation pattern)
7. **Refine** — Orthogonalize against neighboring facts to prevent interference

The edit is stored in a **patch overlay**. Base vindex files are never modified.

### Why Multi-Layer?

A single-layer insert at reasonable alpha (0.25) doesn't move logits enough. Raising alpha breaks neighboring facts. The solution: spread the edit across 8 layers at low alpha. Each layer contributes a small nudge; they accumulate to a strong signal.

### Why Refine?

If two inserted facts share similar triggers, their gates might interfere. The refine pass (Gram-Schmidt orthogonalization) makes inserted gates orthogonal to each other and to "decoy" residuals that shouldn't trigger the fact.

---

## The Patch System

All edits are captured as patches:

```json
{
  "version": 1,
  "base_model": "google/gemma-3-4b-it",
  "operations": [
    {
      "type": "insert",
      "entity": "Atlantis",
      "relation": "capital",
      "target": "Poseidon",
      "layers": [20, 21, 22, 23, 24, 25, 26, 27],
      "alpha": 0.25
    }
  ]
}
```

Patches are:

- **Lightweight** — A single fact is ~10 KB. A 1,000-fact domain patch is ~10 MB.
- **Stackable** — Apply multiple patches: `APPLY PATCH "medical.vlp"; APPLY PATCH "legal.vlp";`
- **Reversible** — `REMOVE PATCH "medical.vlp";` undoes the changes
- **Shareable** — Distribute domain knowledge as .vlp files

### Baking Patches

`COMPILE CURRENT INTO VINDEX "output.vindex"` bakes patches into the weights:

1. For each patched feature, write the override gate/down vectors to the output files
2. The result is a standalone vindex — no overlay needed at load time
3. Facts are embedded in the canonical weight files

---

## Architecture: How the Crates Fit Together

```
┌─────────────────────────────────────────────────────────┐
│                       User                              │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────┐
│                     larql-cli                           │
│  Commands: extract-index, build, serve, repl, hf        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────┐
│                     larql-lql                           │
│  Parser, Executor, REPL, USE REMOTE                     │
└─────┬─────────────────┬─────────────────┬───────────────┘
      │                 │                 │
      ▼                 ▼                 ▼
┌───────────┐    ┌────────────┐    ┌─────────────┐
│larql-vindex│   │larql-infer │    │larql-server │
│  KNN       │   │  Forward   │    │   HTTP      │
│  Patch     │   │  Pass      │    │   gRPC      │
│  Extract   │   │  WalkFfn   │    │             │
└─────┬──────┘   └──────┬─────┘    └─────────────┘
      │                 │
      └────────┬────────┘
               ▼
┌──────────────────────────────────────────────────────────┐
│                    larql-compute                         │
│  BLAS matmul, Metal GPU shaders, Q4/Q8 quantization      │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────┴──────────────────────────────────┐
│                    larql-models                          │
│  Architecture traits, config parsing, weight loading     │
└──────────────────────────────────────────────────────────┘
```

Each crate has a single responsibility:

| Crate | One-liner |
|-------|-----------|
| **larql-models** | Describes what a model IS (no compute) |
| **larql-compute** | Hardware abstraction (CPU/Metal) |
| **larql-core** | Knowledge graph algorithms |
| **larql-vindex** | Vindex lifecycle (extract, query, patch) |
| **larql-inference** | Forward pass execution |
| **larql-lql** | Query language |
| **larql-server** | Network serving |
| **larql-cli** | User interface |

---

## Why These Design Choices?

### Why readonly base + patch overlay?

1. **Safety** — You can't corrupt the original extraction
2. **Shareability** — Patches are small, vindexes are large
3. **Reversibility** — Remove a patch and you're back to baseline
4. **Composition** — Stack multiple patches from different sources

### Why mmap instead of loading to heap?

1. **Memory efficiency** — 3 GB vindex might use 200 MB RSS
2. **Fast startup** — No deserialization, just map the file
3. **OS paging** — Hot data stays resident, cold data stays on disk

### Why WalkFfn (feature-major) instead of standard FFN?

1. **Cache behavior** — Sequential access to features, not scattered access to weight rows
2. **Speed** — 517ms vs 535ms on Gemma 3 4B
3. **Correctness** — Proven identical to dense FFN across all 34 layers

### Why LQL instead of a library API?

1. **Exploration** — REPL enables interactive discovery
2. **Composability** — Pipe statements together
3. **Remote execution** — Same syntax works locally and over HTTP
4. **Familiar** — SQL-like syntax has low learning curve

---

## Common Misconceptions

### "LARQL replaces inference"

No. Browse mode (DESCRIBE, WALK) gives you the model's knowledge structure, not predictions. For actual predictions, you still need INFER, which runs a full forward pass.

### "Edits are like fine-tuning"

No. LARQL edits are surgical (specific features at specific layers). Fine-tuning is gradient-based (all parameters shift slightly). LARQL edits are precise but narrow. Fine-tuning is fuzzy but broad.

### "The vindex is smaller than the model"

Not necessarily. A browse-only vindex (~3 GB) is smaller than the full model (~8 GB). But an inference vindex (~6 GB) or all-weights vindex (~10 GB) can be larger because it stores weights in multiple formats.

### "LARQL only works for factual knowledge"

The same mechanism encodes grammar, style, and reasoning patterns. The "syntax" band (L0-13) encodes linguistic structure. The "output" band (L28-33) encodes formatting conventions. You can query all of it.

---

## Next Steps

Now that you understand the concepts:

1. [Query Knowledge](../guides/query-knowledge.md) — DESCRIBE, WALK, SELECT examples
2. [Edit Knowledge](../guides/edit-knowledge.md) — INSERT, DELETE, patches
3. [Crate Architecture](../concepts/crate-architecture.md) — Deep dive on the 10 crates
4. [LQL Spec](../lql-spec.md) — Complete language reference
