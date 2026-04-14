# LARQL Tutorial Documentation

> **This repository is a tutorial fork of [chrishayuk/larql](https://github.com/chrishayuk/larql)** with comprehensive onboarding documentation explaining the technologies behind LARQL.

## What's Added

This fork adds **deep-dive documentation** that explains:

- **How transformers work** (no prior ML knowledge assumed)
- **Why FFN layers store knowledge** as key-value pairs
- **How LARQL exploits this** with gate KNN and WalkFfn
- **Code walkthroughs** with exact file paths and line numbers

## Tutorial Documentation Index

### Start Here

| Document | Description |
|----------|-------------|
| [What is LARQL?](docs/overview/what-is-this.md) | Mental model - "The model IS the database" |
| [How It Works](docs/overview/how-it-works.md) | **Deep technical dive** - transformers, FFN, gate KNN, WalkFfn with code references |
| [Key Concepts](docs/overview/key-concepts.md) | 50+ term glossary |

### Getting Started

| Document | Description |
|----------|-------------|
| [Prerequisites](docs/getting-started/prerequisites.md) | Rust 1.82+, memory requirements |
| [Quickstart](docs/getting-started/quickstart.md) | 15-minute hands-on guide |
| [Onboarding](docs/getting-started/onboarding.md) | Zero-to-hero conceptual journey |

### Deep Dives

| Document | Description |
|----------|-------------|
| [Vindex Explained](docs/concepts/vindex.md) | Vector index format and structure |
| [LQL Language](docs/concepts/lql.md) | Query language deep dive |
| [Patches](docs/concepts/patches.md) | Knowledge editing system |
| [Inference Engine](docs/concepts/inference.md) | WalkFfn and attention |
| [Crate Architecture](docs/concepts/crate-architecture.md) | How the 10 Rust crates fit together |

### Task Guides

| Document | Description |
|----------|-------------|
| [Extract a Model](docs/guides/extract-a-model.md) | HuggingFace/GGUF to vindex |
| [Query Knowledge](docs/guides/query-knowledge.md) | DESCRIBE, WALK, SELECT |
| [Edit Knowledge](docs/guides/edit-knowledge.md) | INSERT, patches, COMPILE |
| [Serve a Vindex](docs/guides/serve-vindex.md) | HTTP/gRPC server deployment |
| [Python Bindings](docs/guides/python-bindings.md) | Use LARQL from Python |

### Reference

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/reference/cli.md) | All CLI commands |
| [API Reference](docs/reference/api.md) | HTTP/gRPC endpoints |
| [Configuration](docs/reference/configuration.md) | Vindex config, extraction levels |

### Architecture

| Document | Description |
|----------|-------------|
| [System Design](docs/architecture/system-design.md) | Crate structure, data flows, ADRs |
| [Troubleshooting](docs/troubleshooting/common-issues.md) | 25+ common issues and fixes |

---

## Original Documentation

The original LARQL specifications are preserved in the [docs/](docs/) folder:

- [LQL Specification](docs/lql-spec.md) - Complete language grammar
- [Vindex Format Spec](docs/vindex-format-spec.md) - Binary file formats
- [Vindex Operations Spec](docs/vindex-operations-spec.md) - Query/mutation operations
- [FFN Documentation](docs/ffn/) - FFN layer research notes

---

## Quick Links

- **Original Repository:** [github.com/chrishayuk/larql](https://github.com/chrishayuk/larql)
- **This Tutorial Fork:** [github.com/az9713/larql-tutorial](https://github.com/az9713/larql-tutorial)

## License

Apache-2.0 (same as original)
