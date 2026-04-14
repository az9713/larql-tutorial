# Quickstart

Get LARQL running and query your first model in about 15 minutes.

**Prerequisites:** [Prerequisites](prerequisites.md) — Rust 1.82+, ~6 GB disk space.

> **Windows Users:** See [Quickstart: Windows](quickstart-windows.md) for a tested walkthrough with solutions to common build issues.

---

## 1. Clone and Build

```bash
git clone https://github.com/chrishayuk/larql.git
cd larql
cargo build --release
```

Expected output:
```
   Compiling larql-models v0.1.0
   Compiling larql-vindex v0.1.0
   ...
    Finished `release` profile [optimized] target(s) in 2m 15s
```

The binary is at `target/release/larql`.

---

## 2. Get a Vindex

**Option A: Download a pre-built vindex (fastest)**

```bash
./target/release/larql hf download chrishayuk/gemma-3-4b-it-vindex
```

This downloads a ~3 GB vindex to `~/.cache/larql/vindexes/`.

**Option B: Extract from a HuggingFace model**

```bash
./target/release/larql extract-index google/gemma-3-4b-it \
    -o gemma3-4b.vindex --f16
```

This streams through the model (~8 GB download) and produces a ~3 GB vindex.

**Option C: Convert from GGUF**

```bash
./target/release/larql convert gguf-to-vindex model.gguf \
    -o model.vindex --f16
```

---

## 3. Start the REPL

```bash
./target/release/larql repl
```

You'll see:
```
LARQL v0.1.0
Type 'help' for commands, Ctrl+D to exit.
larql>
```

---

## 4. Load the Vindex

```sql
larql> USE "gemma3-4b.vindex";
```

Or for the downloaded vindex:
```sql
larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
```

Expected output:
```
Using: google/gemma-3-4b-it (34 layers, 348160 features)
```

---

## 5. Query Knowledge

### DESCRIBE: What does the model know about France?

```sql
larql> DESCRIBE "France";
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

### WALK: Which features fire for a prompt?

```sql
larql> WALK "The capital of France is" TOP 5;
```

Output:
```
Walk: "The capital of France is" (top 5)
  L27  F9515   Paris        1436.9
  L24  F4532   French         26.1
  L25  F8891   Europe         14.4
  L26  F2201   the            12.3
  L23  F7721   country        11.8
```

### STATS: Vindex statistics

```sql
larql> STATS;
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

---

## 6. What Just Happened

You queried a model's knowledge without running inference:

1. **DESCRIBE "France"** embedded "France" and ran KNN against gate vectors. Features with high gate scores are knowledge about France. The output tokens come from the down projection metadata.

2. **WALK "..." TOP 5** tokenized the prompt, computed a residual-stream proxy, and scanned all 34 layers for matching features.

No GPU. No forward pass. Just KNN lookup in ~0.3ms per query.

---

## Next Steps

- [Onboarding](onboarding.md) — Deeper conceptual understanding
- [Query Knowledge](../guides/query-knowledge.md) — More LQL examples
- [Edit Knowledge](../guides/edit-knowledge.md) — INSERT, DELETE, patches
- [LQL Guide](../lql-guide.md) — Full language tutorial
