# CLI Reference

Complete reference for LARQL command-line interface.

---

## Synopsis

```bash
larql <COMMAND> [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| `extract-index` | Extract a model into a vindex |
| `build` | Build vindex from Vindexfile |
| `serve` | Start HTTP/gRPC server |
| `repl` | Interactive LQL REPL |
| `lql` | Execute LQL statement(s) |
| `hf` | HuggingFace operations |
| `convert` | Format conversion |
| `verify` | Verify vindex integrity |

---

## extract-index

Extract a HuggingFace model into a vindex.

```bash
larql extract-index <MODEL> -o <OUTPUT> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Model path or HuggingFace ID (e.g., `google/gemma-3-4b-it`) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <PATH>` | required | Output vindex path |
| `--level <LEVEL>` | `browse` | Extraction level: `browse`, `inference`, `all` |
| `--f16` | false | Store as f16 (half size) |
| `--merge-into <PATH>` | none | Merge into existing vindex |

### Examples

```bash
# Browse-only vindex
larql extract-index google/gemma-3-4b-it -o gemma.vindex

# With inference weights, f16
larql extract-index google/gemma-3-4b-it -o gemma.vindex --level inference --f16

# All weights (for COMPILE INTO MODEL)
larql extract-index google/gemma-3-4b-it -o gemma.vindex --level all --f16
```

---

## build

Build a vindex from a Vindexfile.

```bash
larql build [PATH] [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `[PATH]` | Path to Vindexfile or directory containing one (default: `.`) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <PATH>` | auto | Output vindex path |
| `--stage <NAME>` | none | Named build stage |

### Examples

```bash
# Build from current directory
larql build

# Build with custom output
larql build --output custom.vindex

# Build a specific stage
larql build --stage production
```

---

## serve

Start an HTTP/gRPC server for vindex queries.

```bash
larql serve <VINDEX> [OPTIONS]
larql serve --dir <DIR> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<VINDEX>` | Path to vindex or `hf://` URL |
| `--dir <DIR>` | Serve all vindexes in directory |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port <PORT>` | 8080 | HTTP port |
| `--host <HOST>` | 0.0.0.0 | Bind address |
| `--no-infer` | false | Disable inference endpoints |
| `--cors` | false | Enable CORS headers |
| `--api-key <KEY>` | none | Require Bearer token auth |
| `--rate-limit <SPEC>` | none | Per-IP rate limit (e.g., "100/min") |
| `--max-concurrent <N>` | 100 | Max concurrent requests |
| `--cache-ttl <SECS>` | 0 | Cache TTL for DESCRIBE |
| `--grpc-port <PORT>` | none | Enable gRPC on this port |
| `--tls-cert <PATH>` | none | TLS certificate |
| `--tls-key <PATH>` | none | TLS private key |
| `--log-level <LEVEL>` | info | Logging level |

### Examples

```bash
# Basic server
larql serve gemma.vindex

# With auth and rate limiting
larql serve gemma.vindex --api-key "sk-secret" --rate-limit "100/min"

# Multi-model
larql serve --dir ./vindexes/

# With TLS
larql serve gemma.vindex --tls-cert cert.pem --tls-key key.pem
```

---

## repl

Start an interactive LQL REPL.

```bash
larql repl [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--history <PATH>` | `~/.larql_history` | History file path |
| `--no-history` | false | Disable history |

### Examples

```bash
larql repl
```

REPL commands:
- `help` — Show help
- `exit` or Ctrl+D — Exit
- Arrow keys — History navigation
- Tab — Completion

---

## lql

Execute LQL statement(s).

```bash
larql lql '<STATEMENT>'
larql lql -f <FILE>
```

### Options

| Option | Description |
|--------|-------------|
| `-f, --file <FILE>` | Execute statements from file |

### Examples

```bash
# Single statement
larql lql 'USE "gemma.vindex"; DESCRIBE "France";'

# From file
larql lql -f queries.lql
```

---

## hf

HuggingFace Hub operations.

```bash
larql hf <SUBCOMMAND>
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `download <REPO>` | Download a vindex |
| `publish <PATH> <REPO>` | Publish a vindex |
| `list` | List available vindexes |

### Examples

```bash
# Download
larql hf download chrishayuk/gemma-3-4b-it-vindex

# Publish
larql hf publish gemma.vindex chrishayuk/my-vindex

# List
larql hf list
```

---

## convert

Convert between formats.

```bash
larql convert <SUBCOMMAND>
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `gguf-to-vindex <GGUF> -o <OUTPUT>` | Convert GGUF to vindex |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--f16` | false | Store as f16 |
| `--level <LEVEL>` | browse | Extraction level |

### Examples

```bash
larql convert gguf-to-vindex model.gguf -o model.vindex --f16
```

---

## verify

Verify vindex integrity.

```bash
larql verify <VINDEX>
```

### Examples

```bash
larql verify gemma.vindex
```

Output:
```
Verifying gemma.vindex...
  gate_vectors.bin: sha256 OK
  embeddings.bin: sha256 OK
  down_meta.bin: sha256 OK
  index.json: valid

Verification passed.
```

---

## Global Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Print help |
| `-V, --version` | Print version |
| `-v, --verbose` | Increase verbosity |
| `-q, --quiet` | Decrease verbosity |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LARQL_CACHE_DIR` | Cache directory (default: `~/.cache/larql`) |
| `HF_HOME` | HuggingFace cache directory |
| `RAYON_NUM_THREADS` | Thread pool size |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Vindex not found |
| 4 | Checksum mismatch |

---

## See Also

- [Full CLI documentation](../cli.md) — Complete command reference with all options
