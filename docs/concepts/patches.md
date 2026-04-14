# Knowledge Patches

Patches are lightweight, shareable, reversible edits to model knowledge. This document explains how they work and when to use them.

---

## What Problem They Solve

Editing model knowledge faces a tradeoff:

| Approach | Size | Speed | Reversible | Shareable |
|----------|------|-------|------------|-----------|
| Fine-tuning | Full model | Hours | No | Full model |
| Weight editing (ROME/MEMIT) | Full model | Seconds | No | Full model |
| **LARQL Patches** | ~10 KB/fact | Milliseconds | Yes | Patch file only |

A 1,000-fact domain adaptation is:
- Fine-tuning: 8 GB model retrained
- LARQL: 10 MB patch file

---

## The Patch Format

A `.vlp` file is JSON:

```json
{
  "version": 1,
  "base_model": "google/gemma-3-4b-it",
  "base_vindex": "hf://chrishayuk/gemma-3-4b-it-vindex",
  "created": "2026-04-10T14:32:00Z",
  "description": "Medical domain facts",
  "operations": [
    {
      "type": "insert",
      "entity": "aspirin",
      "relation": "treats",
      "target": "headache",
      "layers": [20, 21, 22, 23, 24, 25, 26, 27],
      "features": [8821, 8822, 8823, 8824, 8825, 8826, 8827, 8828],
      "alpha": 0.25,
      "confidence": 0.95,
      "gate_vectors": "base64:...",
      "down_vectors": "base64:..."
    },
    {
      "type": "delete",
      "layer": 27,
      "feature": 9515
    }
  ]
}
```

The vectors are included so patches are self-contained. Apply a patch without needing the original model.

---

## How Patches Work

### The Overlay Model

```
┌─────────────────────────────────────────┐
│           PatchedVindex                 │
│  ┌─────────────────────────────────┐    │
│  │       Patch Overlay             │    │
│  │  - Inserted features            │    │
│  │  - Modified gate vectors        │    │
│  │  - Modified down vectors        │    │
│  └─────────────────────────────────┘    │
│                  ↓ query                │
│  ┌─────────────────────────────────┐    │
│  │       Base VectorIndex          │    │
│  │  - gate_vectors.bin (readonly)  │    │
│  │  - down_meta.bin (readonly)     │    │
│  │  - down_weights.bin (readonly)  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

When you query:
1. Check the overlay for matches
2. If not found, check the base
3. Overlay always wins

When you mutate:
1. Write to overlay only
2. Base files are never modified

### Gate vs Down Overrides

Two types of overrides, stored separately:

| Type | Storage | Purpose |
|------|---------|---------|
| Gate overrides | `overrides_gate` HashMap | When features fire |
| Down overrides | `down_overrides` HashMap | What features output |

This separation is important for `COMPILE INTO VINDEX`. The constellation pattern requires weak gates (from base) combined with strong downs (from overlay).

---

## The INSERT Mechanism

When you run:

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon");
```

LARQL performs:

### 1. Embed Target and Entity

```rust
let target_embed = embeddings.lookup("Poseidon");  // What to output
let entity_embed = embeddings.lookup("Atlantis");  // What triggers it
```

### 2. Find Free Slots

Features with low activation across diverse inputs are "free":

```rust
let free_slots = find_free_features(
    &vindex,
    layers: [20..=27],
    min_per_layer: 1,
);
```

### 3. Install the Constellation

For each layer in the span:

```rust
// Gate: triggers on entity context
let gate = entity_embed.normalized() * gate_scale;
overlay.set_gate(layer, slot, gate);

// Down: outputs target embedding
let down = target_embed * alpha;
overlay.set_down(layer, slot, down);

// Metadata
overlay.set_meta(layer, slot, FeatureMeta {
    top_token: tokenizer.encode("Poseidon")[0],
    c_score: confidence,
    source: SourceType::Installed,
});
```

### 4. Refine Pass

Orthogonalize gates against:
- Other inserted facts at the same layer
- Decoy residuals (contexts that shouldn't trigger)

```rust
let refined = refine_gates(
    inserted_gates,
    decoy_residuals,
);
for (layer, feature, gate) in refined {
    overlay.set_gate(layer, feature, gate);
}
```

### 5. Record in Patch Log

The operation is recorded so it can be saved/replayed.

---

## The Constellation Pattern

Single-layer inserts don't work:

| Alpha | Effect |
|-------|--------|
| 0.25 | Too weak — doesn't move logits enough |
| 0.50 | Marginal — sometimes works |
| 1.00 | Breaks neighboring facts |

The solution: spread across 8 layers at low alpha:

```
Layer 20: alpha=0.25, small contribution
Layer 21: alpha=0.25, small contribution
Layer 22: alpha=0.25, small contribution
Layer 23: alpha=0.25, small contribution
Layer 24: alpha=0.25, small contribution
Layer 25: alpha=0.25, small contribution
Layer 26: alpha=0.25, small contribution
Layer 27: alpha=0.25, small contribution
────────────────────────────────────────
Total: 8 small contributions = 1 strong signal
```

Each layer contributes a nudge toward "Poseidon". They accumulate through the residual stream. The final logits shift significantly without any single layer being disruptive.

---

## Refine Pass Details

The refine pass prevents cross-fact interference using Gram-Schmidt orthogonalization.

### Problem: Overlapping Triggers

If you insert:
- ("Paris", "country", "France")
- ("Berlin", "country", "Germany")

Their gate vectors might be similar (both European capitals). When the model sees "Paris," both facts could fire.

### Solution: Orthogonalize

```rust
fn refine_gates(gates: Vec<Gate>, decoys: Vec<Residual>) -> Vec<Gate> {
    let mut result = gates.clone();
    
    for i in 1..result.len() {
        // Orthogonalize against all previous gates
        for j in 0..i {
            result[i] = result[i] - project(result[i], result[j]);
        }
        // Orthogonalize against decoys
        for decoy in &decoys {
            result[i] = result[i] - project(result[i], decoy);
        }
        result[i] = result[i].normalized();
    }
    
    result
}
```

After refinement, the "Paris" gate only fires for Paris, not Berlin.

### Decoy Residuals

Decoys are residual vectors captured from prompts that shouldn't trigger the fact:

```rust
let decoys = capture_decoy_residuals(&model, &[
    "The weather in Paris is",      // Shouldn't trigger "country"
    "I visited Paris last summer",  // Shouldn't trigger "country"
]);
```

The refine pass orthogonalizes against these, preventing false positives.

---

## Patch Operations

### CREATE

```sql
BEGIN PATCH "medical.vlp";
INSERT ...;
INSERT ...;
SAVE PATCH;
```

### APPLY

```sql
APPLY PATCH "medical.vlp";
```

Loads the patch file, replays all operations into the overlay.

### STACK

```sql
APPLY PATCH "base-knowledge.vlp";
APPLY PATCH "domain-specific.vlp";
APPLY PATCH "corrections.vlp";
```

Patches stack. Later patches can override earlier ones.

### REMOVE

```sql
REMOVE PATCH "corrections.vlp";
```

Removes a patch's contributions from the overlay.

### SHOW

```sql
SHOW PATCHES;
```

Lists active patches with operation counts.

---

## COMPILE: Baking Patches

`COMPILE CURRENT INTO VINDEX` writes patches into weight files:

```sql
COMPILE CURRENT INTO VINDEX "output.vindex";
```

Process:

1. Create output directory
2. Copy base files (hardlink on APFS for speed)
3. For each down override:
   - Read the column from base down_weights.bin
   - Apply the override
   - Write to output down_weights.bin
4. Update checksums in index.json

Result: A standalone vindex. Load it, query it, no overlay needed.

### Conflict Resolution

When multiple patches write the same slot:

```sql
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT FAIL;
```

| Strategy | Behavior |
|----------|----------|
| `LAST_WINS` (default) | Last applied patch wins |
| `FAIL` | Error if any conflict |
| `HIGHEST_CONFIDENCE` | Higher c_score wins (future) |

---

## Use Cases

### Domain Adaptation

```sql
-- Medical domain
BEGIN PATCH "medical.vlp";
INSERT ("aspirin", "treats", "headache");
INSERT ("aspirin", "contraindicated", "bleeding_disorders");
INSERT ("ibuprofen", "treats", "inflammation");
-- ... 1000 more facts
SAVE PATCH;
```

Distribute `medical.vlp` (10 MB) instead of the full model (8 GB).

### Fact Correction

```sql
-- The model thinks London is in France
BEGIN PATCH "corrections.vlp";
DELETE FROM EDGES WHERE entity = "London" AND target = "France";
INSERT ("London", "capital-of", "United Kingdom");
SAVE PATCH;
```

### Personalization

```sql
-- User-specific facts
BEGIN PATCH "user-alice.vlp";
INSERT ("my company", "name", "Acme Corp");
INSERT ("my company", "headquarters", "San Francisco");
SAVE PATCH;
```

Each user gets their own patch. Apply on connection, remove on disconnect.

### A/B Testing

```sql
-- Version A
APPLY PATCH "feature-v1.vlp";
-- test...

REMOVE PATCH "feature-v1.vlp";
APPLY PATCH "feature-v2.vlp";
-- test...
```

---

## Best Practices

### 1. One Domain Per Patch

Don't mix medical facts with legal facts. Separate patches are easier to manage.

### 2. Version Your Patches

Include version in the filename: `medical-v2.1.0.vlp`

### 3. Test Before Distributing

```sql
APPLY PATCH "draft.vlp";
INFER "Does aspirin treat headaches?";
-- Verify answer is correct
```

### 4. Use COMPILE for Production

For serving, compile patches into a standalone vindex:

```sql
APPLY PATCH "production-facts.vlp";
COMPILE CURRENT INTO VINDEX "production.vindex";
```

The compiled vindex loads faster (no patch replay) and has no overlay overhead.

---

## Related Docs

- [Edit Knowledge Guide](../guides/edit-knowledge.md) — Practical walkthrough
- [Vindex Operations Spec](../vindex-operations-spec.md) — Patch format specification
- [Training-Free Insert](../training-free-insert.md) — The constellation mechanism
