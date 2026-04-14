# Python Bindings

Use LARQL from Python via PyO3 bindings.

---

## Installation

### From PyPI (when available)

```bash
pip install larql
```

### Build from Source

```bash
cd crates/larql-python
pip install maturin
maturin develop --release
```

---

## Quick Start

```python
import larql

# Load a vindex
model = larql.WalkModel("gemma3-4b.vindex")

# Describe an entity
edges = model.describe("France")
for edge in edges:
    print(f"{edge.relation} → {edge.target} (L{edge.layer}, {edge.score:.1f})")

# Walk a prompt
hits = model.walk("The capital of France is", top_k=10)
for hit in hits:
    print(f"L{hit.layer} F{hit.feature}: {hit.target} ({hit.score:.1f})")

# Run inference
predictions = model.infer("The capital of France is", top_k=5)
for token, prob in predictions:
    print(f"{token}: {prob*100:.1f}%")
```

---

## WalkModel

The main class for vindex operations.

### Constructor

```python
model = larql.WalkModel(
    path="gemma3-4b.vindex",    # Path or hf:// URL
    load_inference=True,         # Load inference weights
)
```

### describe()

```python
edges = model.describe(
    entity="France",
    band="knowledge",    # "syntax", "knowledge", "output", "all"
    limit=20,
    min_score=5.0,
)
```

Returns `List[DescribeEdge]`:

```python
@dataclass
class DescribeEdge:
    target: str           # Output token
    relation: str | None  # Probe label (if available)
    layer: int
    feature: int
    score: float
    source: str          # "probe", "cluster", "none"
```

### walk()

```python
hits = model.walk(
    prompt="The capital of France is",
    top_k=10,
    layers=range(14, 28),  # Optional layer range
)
```

Returns `List[WalkHit]`:

```python
@dataclass
class WalkHit:
    layer: int
    feature: int
    target: str
    score: float
```

### infer()

```python
predictions = model.infer(
    prompt="The capital of France is",
    top_k=5,
    mode="walk",  # "walk", "dense", or "compare"
)
```

Returns `List[Tuple[str, float]]` — token and probability.

### stats()

```python
stats = model.stats()
print(f"Model: {stats.model_id}")
print(f"Layers: {stats.num_layers}")
print(f"Features: {stats.num_features}")
```

---

## Tracing

Decompose the forward pass to understand attribution.

```python
# Create a trace
trace = model.trace("The capital of France is")

# Track a specific answer
trajectory = trace.answer_trajectory("Paris")
for layer, rank, prob, attn, ffn in trajectory:
    print(f"L{layer}: rank={rank}, prob={prob:.3f}, attn={attn:+.1f}, ffn={ffn:+.1f}")

# Get top-K at a layer
top_k = trace.top_k(layer=24, k=5)
for token, prob in top_k:
    print(f"{token}: {prob*100:.1f}%")

# Save for later analysis
trace.save("france.trace")

# Load a saved trace
loaded = larql.Trace.load("france.trace")
```

---

## Patches

Edit knowledge from Python.

### Apply a Patch

```python
model.apply_patch("medical.vlp")
```

### Insert a Fact

```python
model.insert(
    entity="Atlantis",
    relation="capital",
    target="Poseidon",
    layer=24,           # Optional: center layer
    alpha=0.25,         # Optional: strength
    confidence=0.95,    # Optional: metadata
)
```

### Delete a Feature

```python
model.delete(layer=27, feature=9515)
```

### Save Patches

```python
model.save_patch("my-edits.vlp")
```

### Compile

```python
# To new vindex
model.compile_vindex("output.vindex")

# To safetensors
model.compile_model("output/", format="safetensors")
```

---

## Batch Operations

Process multiple queries efficiently.

### Batch Describe

```python
entities = ["France", "Germany", "Italy", "Spain"]
results = model.describe_batch(entities)

for entity, edges in zip(entities, results):
    print(f"{entity}: {len(edges)} edges")
```

### Batch Walk

```python
prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
]
results = model.walk_batch(prompts, top_k=5)
```

### Batch Infer

```python
prompts = ["Hello, my name is", "The meaning of life is"]
results = model.infer_batch(prompts, top_k=5)
```

---

## Direct Vindex Access

For advanced use cases, access the vindex directly.

### Gate KNN

```python
# Get raw gate scores for a layer
query = model.embed("France")
scores = model.gate_knn(layer=27, query=query, top_k=100)
for feature_id, score in scores:
    print(f"F{feature_id}: {score:.1f}")
```

### Feature Metadata

```python
meta = model.feature_meta(layer=27, feature=9515)
print(f"Token: {meta.top_token}")
print(f"Confidence: {meta.c_score}")
print(f"Source: {meta.source}")
```

### Embeddings

```python
# Token to embedding
vec = model.embed("Paris")

# Embedding to token (nearest)
token = model.decode(vec)
```

---

## Integration with NumPy

Vectors are returned as NumPy arrays.

```python
import numpy as np

# Get an embedding
vec = model.embed("France")
print(type(vec))  # numpy.ndarray
print(vec.shape)  # (2560,)

# Use in computations
similarity = np.dot(vec, model.embed("Germany"))
```

---

## Example: Knowledge Graph Extraction

```python
import larql
import json

model = larql.WalkModel("gemma3-4b.vindex")

# Extract knowledge for a set of entities
entities = ["France", "Germany", "Italy", "Spain", "Portugal"]
knowledge_graph = {}

for entity in entities:
    edges = model.describe(entity, band="knowledge", limit=10)
    knowledge_graph[entity] = [
        {"relation": e.relation, "target": e.target, "confidence": e.score}
        for e in edges
        if e.relation  # Only include labeled edges
    ]

# Save to JSON
with open("europe.json", "w") as f:
    json.dump(knowledge_graph, f, indent=2)
```

---

## Example: Fact Verification

```python
import larql

model = larql.WalkModel("gemma3-4b.vindex")

def verify_fact(entity: str, relation: str, expected_target: str) -> bool:
    """Check if the model knows a fact."""
    edges = model.describe(entity, limit=50)
    for edge in edges:
        if edge.relation == relation and edge.target == expected_target:
            return True
    return False

# Test
facts = [
    ("France", "capital", "Paris"),
    ("Germany", "capital", "Berlin"),
    ("Italy", "capital", "Rome"),
    ("Spain", "capital", "Barcelona"),  # Wrong!
]

for entity, relation, target in facts:
    result = verify_fact(entity, relation, target)
    status = "✓" if result else "✗"
    print(f"{status} {entity} {relation} {target}")
```

---

## Example: Compare Models

```python
import larql

model_v1 = larql.WalkModel("gemma3-4b-v1.vindex")
model_v2 = larql.WalkModel("gemma3-4b-v2.vindex")

def compare_entity(entity: str):
    edges_v1 = {(e.relation, e.target) for e in model_v1.describe(entity)}
    edges_v2 = {(e.relation, e.target) for e in model_v2.describe(entity)}
    
    added = edges_v2 - edges_v1
    removed = edges_v1 - edges_v2
    
    return added, removed

added, removed = compare_entity("France")
print(f"Added: {added}")
print(f"Removed: {removed}")
```

---

## Configuration

### Logging

```python
import larql

# Enable debug logging
larql.set_log_level("debug")
```

### Thread Pool

```python
# Set number of threads for batch operations
larql.set_num_threads(4)
```

---

## Troubleshooting

### ImportError

Make sure you built with maturin:
```bash
cd crates/larql-python
maturin develop --release
```

### "Vindex not found"

Check the path exists:
```python
import os
print(os.path.exists("gemma3-4b.vindex"))
```

### "Inference not available"

Load with inference weights:
```python
model = larql.WalkModel("vindex", load_inference=True)
```

Or extract with inference level:
```bash
larql extract-index model -o vindex --level inference
```

---

## Next Steps

- [larql-python README](../../crates/larql-python/README.md) — Full API documentation
- [larql-python Doc](../larql-python.md) — Detailed Python guide
