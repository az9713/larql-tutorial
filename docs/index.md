# LARQL Documentation

> **Tutorial Fork Notice:** This documentation is part of [az9713/larql-tutorial](https://github.com/az9713/larql-tutorial), a fork of [chrishayuk/larql](https://github.com/chrishayuk/larql) with comprehensive onboarding documentation.

**The model IS the database.** Query neural network weights like a graph database. No GPU required.

LARQL decompiles transformer models into a queryable format called a **vindex** (vector index), then provides **LQL** (Lazarus Query Language) to browse, edit, and recompile the model's knowledge.

---

## Documentation

| Section | What's inside |
|---------|--------------|
| [Overview](overview/what-is-this.md) | Mental model, core concepts, [how it works](overview/how-it-works.md) (technical deep-dive) |
| [Getting Started](getting-started/quickstart.md) | Installation, first vindex, first query ([Windows](getting-started/quickstart-windows.md)) |
| [Concepts](concepts/vindex.md) | Deep dives: vindex, LQL, patches, inference, crates |
| [Guides](guides/extract-a-model.md) | Task-oriented how-tos |
| [Reference](reference/cli.md) | CLI, API, configuration, specifications |
| [Architecture](architecture/system-design.md) | System design, crate dependencies, ADRs |
| [Troubleshooting](troubleshooting/common-issues.md) | Common issues and fixes |

---

## Quick Links

**New to LARQL?** Start with [What is LARQL?](overview/what-is-this.md) then follow the [Quickstart](getting-started/quickstart.md).

**Want the deep technical story?** Read [How LARQL Works](overview/how-it-works.md) - explains transformers, FFN layers, gate KNN, and WalkFfn with code cross-references. Assumes no prior transformer knowledge.

**Building an integration?** See the [CLI Reference](reference/cli.md) and [Server API](reference/api.md).

**Want to understand the internals?** Read [System Design](architecture/system-design.md) and the [Crate Architecture](concepts/crate-architecture.md).

---

## Specifications

Detailed technical specifications for implementers:

| Spec | Description |
|------|-------------|
| [LQL Language Spec](lql-spec.md) | Complete LQL grammar and semantics |
| [Vindex Format Spec](vindex-format-spec.md) | File layout, binary formats |
| [Vindex Operations Spec](vindex-operations-spec.md) | Query, mutation, patch operations |
| [Vindex Ecosystem Spec](vindex-ecosystem-spec.md) | HuggingFace, Vindexfile, distributed serving |
| [Trace Format Spec](trace-format-spec.md) | Residual trace file formats |
| [Server Spec](vindex-server-spec.md) | HTTP/gRPC API specification |

---

## Crate Documentation

Each crate has its own README with API examples and benchmarks:

| Crate | Purpose |
|-------|---------|
| [larql-models](../crates/larql-models/README.md) | Model architecture traits, config parsing, weight loading |
| [larql-vindex](../crates/larql-vindex/README.md) | Vindex lifecycle: extract, load, query, mutate, patch |
| [larql-core](../crates/larql-core/README.md) | Knowledge graph engine, algorithms |
| [larql-compute](../crates/larql-compute/README.md) | Hardware-accelerated compute (CPU/Metal) |
| [larql-inference](../crates/larql-inference/README.md) | Forward pass, BLAS-fused attention, WalkFfn |
| [larql-lql](../crates/larql-lql/README.md) | LQL parser, executor, REPL |
| [larql-server](../crates/larql-server/README.md) | HTTP/gRPC server |
| [larql-cli](../crates/larql-cli/README.md) | CLI commands |
| [larql-python](../crates/larql-python/README.md) | Python bindings |
| [kv-cache-benchmark](../crates/kv-cache-benchmark/README.md) | KV cache strategy comparison |
