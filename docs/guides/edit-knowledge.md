# Edit Knowledge

Insert, delete, and modify facts in a model using LQL. Changes are stored as patches — base vindex files are never modified.

---

## Prerequisites

- A vindex (see [Extract a Model](extract-a-model.md))
- For INFER verification: vindex with inference weights

---

## INSERT: Add a Fact

### Basic Insert

```sql
USE "gemma3-4b.vindex";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon");
```

Output:
```
Inserted 1 edge. Feature F8821@L20-27 allocated.
Auto-patch started (use SAVE PATCH to persist)
```

### Verify the Insert

```sql
DESCRIBE "Atlantis";
```

Output:
```
Atlantis
  Edges (L20-27):
    capital     → Poseidon            0.95  (installed)
```

### Test with Inference

```sql
INFER "The capital of Atlantis is" TOP 3;
```

Output:
```
  1. Pose             56.91%  ← (Poseidon tokenizes to "Pose" + "idon")
  2. the               0.42%
  3. a                 0.31%
```

---

## INSERT Options

### Specify Layer

Center the constellation on a specific layer:

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon")
    AT LAYER 24;
```

The 8-layer span is centered on L24 (clamped to valid range).

### Specify Confidence

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon")
    CONFIDENCE 0.99;
```

Stored in feature metadata, affects SELECT ordering.

### Specify Alpha

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon")
    ALPHA 0.30;
```

Higher alpha = stronger effect, but more interference risk. Validated range: 0.10-0.50.

### All Options

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.99
    ALPHA 0.30;
```

---

## DELETE: Remove Features

### By Layer and Feature

```sql
DELETE FROM EDGES WHERE layer = 27 AND feature = 9515;
```

### By Entity

```sql
DELETE FROM EDGES WHERE entity = "Atlantis";
```

Removes all features matching the entity.

---

## UPDATE: Modify Features

### Change Confidence

```sql
UPDATE EDGES SET c_score = 0.99 WHERE layer = 27 AND feature = 9515;
```

---

## Patches: Save and Share

### Create a Patch

```sql
BEGIN PATCH "medical.vlp";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");

INSERT INTO EDGES (entity, relation, target)
    VALUES ("ibuprofen", "treats", "inflammation");

SAVE PATCH;
```

Output:
```
Patch saved: medical.vlp (3 operations, 28 KB)
```

### Apply a Patch

```sql
APPLY PATCH "medical.vlp";
```

### Stack Multiple Patches

```sql
APPLY PATCH "base-knowledge.vlp";
APPLY PATCH "medical.vlp";
APPLY PATCH "legal.vlp";
```

Later patches override earlier ones for conflicting features.

### List Active Patches

```sql
SHOW PATCHES;
```

Output:
```
Active patches:
  1. base-knowledge.vlp  (500 operations)
  2. medical.vlp         (3 operations)
  3. legal.vlp           (12 operations)
```

### Remove a Patch

```sql
REMOVE PATCH "legal.vlp";
```

---

## COMPILE: Bake Patches

### To Vindex

Create a standalone vindex with patches baked in:

```sql
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";
```

The output vindex has no overlay — facts are in the weight files.

### To Model (safetensors)

Export to HuggingFace-loadable format:

```sql
COMPILE CURRENT INTO MODEL "gemma3-4b-medical/" FORMAT safetensors;
```

Load in Python:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gemma3-4b-medical/")
```

### Conflict Resolution

```sql
-- Fail on any conflict
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT FAIL;

-- Last patch wins (default)
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT LAST_WINS;
```

---

## Example: Domain Adaptation

### 1. Start with Base Vindex

```sql
USE "gemma3-4b.vindex";
```

### 2. Add Domain Facts

```sql
BEGIN PATCH "company-facts.vlp";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("Acme Corp", "headquarters", "San Francisco");

INSERT INTO EDGES (entity, relation, target)
    VALUES ("Acme Corp", "ceo", "Alice Smith");

INSERT INTO EDGES (entity, relation, target)
    VALUES ("Acme Corp", "founded", "2015");

INSERT INTO EDGES (entity, relation, target)
    VALUES ("Acme Corp", "industry", "technology");

SAVE PATCH;
```

### 3. Verify

```sql
DESCRIBE "Acme Corp";
INFER "The headquarters of Acme Corp is in" TOP 3;
```

### 4. Compile for Production

```sql
COMPILE CURRENT INTO VINDEX "gemma3-4b-acme.vindex";
```

### 5. Deploy

```bash
larql serve gemma3-4b-acme.vindex --port 8080
```

---

## Example: Fact Correction

### 1. Identify the Problem

```sql
DESCRIBE "London";
-- Shows: country → France (incorrect!)
```

### 2. Delete the Incorrect Fact

```sql
DELETE FROM EDGES WHERE entity = "London" AND target = "France";
```

### 3. Insert the Correct Fact

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("London", "country", "United Kingdom");
```

### 4. Verify

```sql
DESCRIBE "London";
INFER "London is the capital of" TOP 3;
```

---

## Tips

### Test Before Saving

Run INFER to verify the edit works before saving the patch:

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("test", "relation", "value");

INFER "test relation is" TOP 3;

-- If wrong, the auto-patch hasn't been saved yet
-- Just restart the session
```

### Use Meaningful Patch Names

```
medical-v2.1.0.vlp
company-facts-2026-04.vlp
corrections-batch-15.vlp
```

### One Domain Per Patch

Don't mix medical facts with legal facts. Separate patches are easier to manage.

### Compile for Production

For serving, always compile:

```sql
APPLY PATCH "all-patches.vlp";
COMPILE CURRENT INTO VINDEX "production.vindex";
```

Compiled vindexes load faster (no patch replay).

---

## Troubleshooting

### Insert Has No Effect

The alpha might be too low. Try:

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("entity", "relation", "target")
    ALPHA 0.35;
```

### Insert Breaks Other Facts

The alpha might be too high, or facts overlap. Try:

1. Lower alpha to 0.20
2. Check for entity overlap with DESCRIBE

### Patch Won't Apply

Check compatibility:

```sql
SHOW PATCHES;
```

The patch's base_model must match the current vindex.

---

## Next Steps

- [Serve Vindex](serve-vindex.md) — Deploy over HTTP
- [Patches Concept](../concepts/patches.md) — How patches work internally
- [Training-Free Insert](../training-free-insert.md) — The constellation mechanism
