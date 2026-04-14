# LQL: The Query Language

LQL (Lazarus Query Language) is a SQL-like language for querying and editing vindexes. This document covers the language design, statement types, and execution model.

---

## Design Goals

1. **Familiar syntax** — SQL-like for low learning curve
2. **Interactive** — REPL for exploration
3. **Composable** — Pipe statements together
4. **Remote-transparent** — Same syntax works locally and over HTTP
5. **Safe by default** — Mutations go to overlay, not base files

---

## Statement Families

| Family | Statements | Purpose |
|--------|------------|---------|
| **Lifecycle** | EXTRACT, COMPILE, DIFF, USE | Create and manage vindexes |
| **Query** | DESCRIBE, WALK, SELECT, INFER, TRACE | Read knowledge |
| **Mutation** | INSERT, DELETE, UPDATE, MERGE | Edit knowledge |
| **Patch** | BEGIN/SAVE/APPLY/REMOVE PATCH | Manage patch files |
| **Introspection** | SHOW, STATS, EXPLAIN | Inspect state |

---

## Lifecycle Statements

### USE

Load a vindex for subsequent queries:

```sql
-- Local path
USE "gemma3-4b.vindex";

-- HuggingFace
USE "hf://chrishayuk/gemma-3-4b-it-vindex";

-- Remote server
USE REMOTE "http://localhost:8080";
```

### EXTRACT

Create a vindex from a model:

```sql
-- Browse only
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";

-- With inference weights
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH INFERENCE;

-- All weights (for COMPILE INTO MODEL)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;
```

### COMPILE

Bake patches into output:

```sql
-- To new vindex (standalone, no overlay needed)
COMPILE CURRENT INTO VINDEX "output.vindex";

-- With conflict handling
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT FAIL;

-- To safetensors (loadable by HuggingFace)
COMPILE CURRENT INTO MODEL "output/" FORMAT safetensors;
```

### DIFF

Compare two vindexes:

```sql
-- Output as patch
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";

-- Summary only
DIFF "v1.vindex" "v2.vindex";
```

---

## Query Statements

### DESCRIBE

Find all knowledge about an entity:

```sql
-- Basic
DESCRIBE "France";

-- All layers (not just knowledge band)
DESCRIBE "Einstein" ALL LAYERS;

-- Compact output
DESCRIBE "France" BRIEF;

-- Limit results
DESCRIBE "France" TOP 5;
```

Output:
```
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27  (probe)
    language    → French               35.2  L24  (probe)
    continent   → Europe               14.4  L25  (probe)
```

### WALK

Trace feature activations for a prompt:

```sql
-- Basic
WALK "The capital of France is" TOP 10;

-- Specific layers
WALK "Einstein" TOP 5 LAYERS 20-27;
```

Output:
```
Walk: "The capital of France is" (top 10)
  L27  F9515   Paris        1436.9
  L24  F4532   French         26.1
  ...
```

### SELECT

SQL-style query over features:

```sql
-- By entity
SELECT * FROM EDGES WHERE entity = "France" LIMIT 10;

-- By layer
SELECT * FROM EDGES WHERE layer >= 20 ORDER BY c_score DESC LIMIT 5;

-- By relation (requires probe labels)
SELECT * FROM EDGES WHERE relation = "capital";
```

### INFER

Run forward-pass inference:

```sql
-- Basic
INFER "The capital of France is" TOP 5;

-- Walk mode (uses vindex FFN)
INFER "The capital of France is" TOP 5 MODE walk;

-- Dense mode (uses original FFN)
INFER "The capital of France is" TOP 5 MODE dense;

-- Compare both
INFER "The capital of France is" TOP 5 COMPARE;
```

Output:
```
Predictions for "The capital of France is":
  1. Paris         97.91%
  2. the            0.42%
  3. a              0.31%
```

### TRACE

Decompose the forward pass:

```sql
-- Track a specific answer
TRACE "The capital of France is" FOR "Paris";

-- Decompose attribution per layer
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Save for analysis
TRACE "The capital of France is" SAVE "france.trace";
```

Output:
```
  Layer   Rank     Prob      Attn       FFN      Who
    L22     50    0.002     +22.2     +34.4   BOTH ↑
    L23     10    0.024     -16.9     +55.9    FFN ↑
    L24      1    0.714    +105.7     +24.4   BOTH ↑  ← phase transition
```

---

## Mutation Statements

### INSERT

Add a new fact:

```sql
-- Basic form (8-layer constellation, alpha=0.25)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon");

-- With layer hint
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon")
    AT LAYER 24;

-- With explicit confidence and alpha
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.95
    ALPHA 0.30;
```

### DELETE

Remove features:

```sql
-- By layer and feature
DELETE FROM EDGES WHERE layer = 27 AND feature = 9515;

-- By entity (removes all matching features)
DELETE FROM EDGES WHERE entity = "Atlantis";
```

### UPDATE

Modify existing features:

```sql
UPDATE EDGES SET c_score = 0.99 WHERE layer = 27 AND feature = 9515;
```

### MERGE

Combine another vindex's patches:

```sql
MERGE "domain-facts.vindex" INTO CURRENT;
```

---

## Patch Statements

### BEGIN PATCH

Start recording operations:

```sql
BEGIN PATCH "medical.vlp";
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
SAVE PATCH;
```

### APPLY PATCH

Load and apply a patch:

```sql
APPLY PATCH "medical.vlp";
APPLY PATCH "hf://medical-ai/drug-interactions.vlp";
```

### SHOW PATCHES

List active patches:

```sql
SHOW PATCHES;
```

### REMOVE PATCH

Unapply a patch:

```sql
REMOVE PATCH "medical.vlp";
```

---

## Introspection Statements

### SHOW

```sql
SHOW RELATIONS;      -- List relation types
SHOW LAYERS;         -- List layer bands
SHOW FEATURES;       -- Feature count per layer
SHOW MODELS;         -- Loaded models
SHOW PATCHES;        -- Active patches
```

### STATS

```sql
STATS;
```

Output:
```
Model:    google/gemma-3-4b-it
Family:   gemma3
Layers:   34
Features: 348,160
Bands:    syntax [0-13], knowledge [14-27], output [28-33]
Labels:   1,967 probe-confirmed relations
Size:     3.2 GB
```

### EXPLAIN

Show execution plan:

```sql
EXPLAIN WALK "France" TOP 5;
EXPLAIN INFER "The capital of France is" TOP 5;
```

---

## The REPL

Start the REPL:

```bash
larql repl
```

Features:
- History (arrow keys)
- Tab completion
- Multi-line input (end with `;`)
- `help` command
- Ctrl+D to exit

Batch mode:

```bash
larql lql 'USE "vindex"; DESCRIBE "France";'
larql lql -f queries.lql
```

---

## Remote Execution

When using `USE REMOTE`, queries are forwarded to the server:

```sql
USE REMOTE "http://localhost:8080";

-- These run on the server
DESCRIBE "France";
WALK "Einstein" TOP 5;
INFER "The capital of France is" TOP 5;

-- Mutations create server-side patches
INSERT INTO EDGES (entity, relation, target)
    VALUES ("custom", "fact", "value");
```

The REPL handles the HTTP communication transparently.

---

## Execution Model

1. **Parse** — LQL text → AST (Statement enum)
2. **Execute** — Session.execute(statement) → Vec<String>
3. **Backend dispatch** — Vindex, Weight, or Remote backend
4. **Format** — Results formatted for display

The `Session` struct holds the loaded vindex (if any) and patch state:

```rust
let mut session = Session::new();
session.execute(&parse("USE \"vindex\";")?)?;
let output = session.execute(&parse("DESCRIBE \"France\";")?)?;
for line in output {
    println!("{}", line);
}
```

---

## Error Handling

Common errors:

| Error | Cause |
|-------|-------|
| `NoBackend` | No USE statement yet |
| `VindexNotFound` | Path doesn't exist |
| `InferenceDisabled` | INFER without model weights |
| `ParseError` | Invalid LQL syntax |
| `FeatureNotFound` | DELETE/UPDATE target missing |

All errors are recoverable. The session remains valid after an error.

---

## Related Docs

- [LQL Spec](../lql-spec.md) — Complete grammar and semantics
- [LQL Guide](../lql-guide.md) — Tutorial with examples
- [larql-lql README](../../crates/larql-lql/README.md) — Crate documentation
