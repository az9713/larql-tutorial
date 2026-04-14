# Query Knowledge

Use LQL to explore what a model knows. This guide covers DESCRIBE, WALK, SELECT, INFER, and TRACE.

---

## Prerequisites

- A vindex (see [Extract a Model](extract-a-model.md))
- LARQL REPL: `larql repl`

---

## Load a Vindex

```sql
USE "gemma3-4b.vindex";
```

Or from HuggingFace:

```sql
USE "hf://chrishayuk/gemma-3-4b-it-vindex";
```

Output:
```
Using: google/gemma-3-4b-it (34 layers, 348160 features)
```

---

## DESCRIBE: Entity Knowledge

Find all facts about an entity.

### Basic

```sql
DESCRIBE "France";
```

Output:
```
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27  (probe)
    language    → French               35.2  L24  (probe)
    continent   → Europe               14.4  L25  (probe)
    borders     → Spain                13.3  L18  (probe)
```

### All Layers

Include syntax and output bands:

```sql
DESCRIBE "France" ALL LAYERS;
```

### Compact View

```sql
DESCRIBE "France" BRIEF;
```

Output:
```
France: capital→Paris, language→French, continent→Europe, borders→Spain
```

### Limit Results

```sql
DESCRIBE "France" TOP 3;
```

### Minimum Score

```sql
DESCRIBE "France" MIN SCORE 10.0;
```

---

## WALK: Feature Activation

Trace which features fire for a prompt.

### Basic

```sql
WALK "The capital of France is" TOP 10;
```

Output:
```
Walk: "The capital of France is" (top 10)
  L27  F9515   Paris        1436.9
  L24  F4532   French         26.1
  L25  F8891   Europe         14.4
  L26  F2201   the            12.3
  L23  F7721   country        11.8
  L22  F1023   capital        10.2
  L21  F8832   is              9.8
  L20  F5543   France          9.1
  L19  F2234   of              8.7
  L18  F9912   The             8.2
```

### Specific Layers

```sql
WALK "Einstein" TOP 5 LAYERS 20-27;
```

### Compare Prompts

```sql
WALK "The capital of France is" TOP 5;
WALK "The capital of Germany is" TOP 5;
```

See which features differ between France and Germany.

---

## SELECT: SQL-Style Queries

Query feature metadata like a database.

### By Entity

```sql
SELECT * FROM EDGES WHERE entity = "France" LIMIT 10;
```

### By Layer

```sql
SELECT * FROM EDGES WHERE layer >= 25 ORDER BY c_score DESC LIMIT 5;
```

### By Relation

```sql
SELECT * FROM EDGES WHERE relation = "capital" LIMIT 10;
```

### Specific Fields

```sql
SELECT layer, feature, target, c_score FROM EDGES 
    WHERE entity = "Paris" 
    ORDER BY c_score DESC 
    LIMIT 5;
```

---

## INFER: Run Inference

Requires inference weights in the vindex.

### Basic

```sql
INFER "The capital of France is" TOP 5;
```

Output:
```
Predictions for "The capital of France is":
  1. Paris         97.91%
  2. the            0.42%
  3. a              0.31%
  4. located        0.15%
  5. known          0.12%
```

### Walk vs Dense

```sql
-- Walk mode (uses vindex FFN, default)
INFER "The capital of France is" TOP 5 MODE walk;

-- Dense mode (uses full matmul)
INFER "The capital of France is" TOP 5 MODE dense;

-- Compare both
INFER "The capital of France is" TOP 5 COMPARE;
```

Compare output shows both predictions and latency:
```
Walk:  Paris (97.91%), 517ms
Dense: Paris (97.93%), 535ms
```

---

## TRACE: Decompose the Forward Pass

See how the answer probability changes across layers.

### Track an Answer

```sql
TRACE "The capital of France is" FOR "Paris";
```

Output:
```
Tracing "Paris" through "The capital of France is":
  Layer   Rank     Prob      Attn       FFN      Who
    L20     --    0.000      +0.0      +0.1   
    L21     --    0.000      +0.2      +0.3   
    L22     50    0.002     +22.2     +34.4   BOTH ↑
    L23     10    0.024     -16.9     +55.9    FFN ↑
    L24      1    0.714    +105.7     +24.4   BOTH ↑  ← phase transition
    L25      1    0.997      +4.3     +94.4    FFN ↑
    L26      1    0.999     +83.1     +18.7   BOTH ↑
    L27      1    0.999      +2.1      +3.2   BOTH
```

The phase transition at L24 is where the model "decides" the answer.

### Decompose Layers

```sql
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;
```

Shows per-head attention contributions.

### Save for Analysis

```sql
TRACE "The capital of France is" SAVE "france.trace";
```

---

## SHOW: Introspection

### Relations

```sql
SHOW RELATIONS;
```

Lists discovered relation types with example features.

### Layers

```sql
SHOW LAYERS;
```

Output:
```
Layer Bands:
  Syntax:    L0-13  (morphology, grammar)
  Knowledge: L14-27 (factual relations)
  Output:    L28-33 (formatting, selection)
```

### Features

```sql
SHOW FEATURES;
```

Feature count per layer.

### Stats

```sql
STATS;
```

Full vindex statistics.

---

## Combining Queries

### Explore an Entity

```sql
-- What does the model know?
DESCRIBE "Einstein";

-- What features fire?
WALK "Einstein was a" TOP 10;

-- What would it predict?
INFER "Einstein was a" TOP 5;

-- How does it reach that prediction?
TRACE "Einstein was a" FOR "physicist";
```

### Compare Entities

```sql
DESCRIBE "Paris";
DESCRIBE "London";
DESCRIBE "Tokyo";
```

### Find Relations

```sql
-- What capitals does the model know?
SELECT * FROM EDGES WHERE relation = "capital" LIMIT 20;

-- What languages?
SELECT * FROM EDGES WHERE relation = "language" LIMIT 20;
```

---

## Tips

### Use BRIEF for Scanning

When exploring many entities:

```sql
DESCRIBE "France" BRIEF;
DESCRIBE "Germany" BRIEF;
DESCRIBE "Italy" BRIEF;
```

### Use LAYERS for Focus

The knowledge band (L14-27 for Gemma 3) has the most interesting facts:

```sql
WALK "Einstein" TOP 10 LAYERS 14-27;
```

### Use COMPARE to Validate

Walk mode should match dense mode:

```sql
INFER "The capital of France is" TOP 1 COMPARE;
```

If they differ significantly, the vindex may be corrupted.

---

## Next Steps

- [Edit Knowledge](edit-knowledge.md) — INSERT, DELETE, patches
- [LQL Spec](../lql-spec.md) — Complete language reference
