# Serve a Vindex

Deploy a vindex over HTTP or gRPC. No GPU required.

---

## Prerequisites

- A vindex (see [Extract a Model](extract-a-model.md))
- For inference: vindex with inference weights

---

## Quick Start

```bash
larql serve gemma3-4b.vindex --port 8080
```

Output:
```
Loading gemma3-4b.vindex...
  Model: google/gemma-3-4b-it
  Layers: 34
  Features: 348,160
  Labels: 1,967 probe-confirmed

Listening: http://0.0.0.0:8080
```

---

## Test the Server

### DESCRIBE

```bash
curl "http://localhost:8080/v1/describe?entity=France"
```

Response:
```json
{
  "entity": "France",
  "model": "google/gemma-3-4b-it",
  "edges": [
    {"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe"},
    {"relation": "language", "target": "French", "gate_score": 35.2, "layer": 24, "source": "probe"}
  ],
  "latency_ms": 0.3
}
```

### WALK

```bash
curl "http://localhost:8080/v1/walk?prompt=The+capital+of+France+is&top=5"
```

### INFER

```bash
curl -X POST "http://localhost:8080/v1/infer" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "top": 5}'
```

Response:
```json
{
  "prompt": "The capital of France is",
  "predictions": [
    {"token": "Paris", "probability": 0.9791},
    {"token": "the", "probability": 0.0042}
  ],
  "mode": "walk",
  "latency_ms": 517
}
```

---

## Server Options

| Flag | Description | Default |
|------|-------------|---------|
| `<VINDEX_PATH>` | Path to vindex or `hf://` URL | required |
| `--port <PORT>` | HTTP port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--no-infer` | Disable inference (browse only) | false |
| `--cors` | Enable CORS headers | false |
| `--api-key <KEY>` | Require Bearer token auth | none |
| `--rate-limit <SPEC>` | Per-IP rate limit | none |
| `--cache-ttl <SECS>` | Cache DESCRIBE results | 0 |
| `--grpc-port <PORT>` | Enable gRPC | none |
| `--tls-cert <PATH>` | TLS certificate | none |
| `--tls-key <PATH>` | TLS private key | none |

---

## Examples

### Browse-Only Server

```bash
larql serve gemma3-4b.vindex --port 8080 --no-infer
```

Uses ~100 MB RAM. INFER returns 503.

### With Authentication

```bash
larql serve gemma3-4b.vindex --api-key "sk-secret-key"
```

Requests require:
```bash
curl -H "Authorization: Bearer sk-secret-key" "http://localhost:8080/v1/describe?entity=France"
```

### With Rate Limiting

```bash
larql serve gemma3-4b.vindex --rate-limit "100/min"
```

Excess requests get 429 Too Many Requests.

### With TLS

```bash
larql serve gemma3-4b.vindex --tls-cert cert.pem --tls-key key.pem
```

### With gRPC

```bash
larql serve gemma3-4b.vindex --port 8080 --grpc-port 50051
```

Both HTTP and gRPC run simultaneously.

### From HuggingFace

```bash
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080
```

---

## API Reference

### GET /v1/describe

| Param | Default | Description |
|-------|---------|-------------|
| `entity` | required | Entity name |
| `band` | `knowledge` | Layer band: syntax, knowledge, output, all |
| `limit` | 20 | Max edges |
| `min_score` | 5.0 | Minimum gate score |
| `verbose` | false | Include layer details |

### GET /v1/walk

| Param | Default | Description |
|-------|---------|-------------|
| `prompt` | required | Input text |
| `top` | 10 | Top-K features |
| `layers` | all | Layer range (e.g., "14-27") |

### POST /v1/infer

```json
{
  "prompt": "string (required)",
  "top": 5,
  "mode": "walk" | "dense" | "compare"
}
```

### GET /v1/stats

Returns model and vindex statistics.

### GET /v1/relations

Returns discovered relation types.

### GET /v1/health

Always returns 200 (exempt from auth).

---

## Multi-Model Serving

Serve multiple vindexes from a directory:

```bash
larql serve --dir ./vindexes/ --port 8080
```

Each vindex gets its own namespace:

```
GET /v1/gemma-3-4b-it/describe?entity=France
GET /v1/llama-3-8b/describe?entity=France
```

---

## Session-Scoped Patches

Apply patches per-session using the `X-Session-Id` header:

```bash
# Session A applies a patch
curl -H "X-Session-Id: session-a" \
     -X POST "http://localhost:8080/v1/patches/apply" \
     -d '{"patch": {...}}'

# Session A sees patched results
curl -H "X-Session-Id: session-a" \
     "http://localhost:8080/v1/describe?entity=aspirin"

# Session B sees unpatched results
curl -H "X-Session-Id: session-b" \
     "http://localhost:8080/v1/describe?entity=aspirin"
```

Sessions expire after 1 hour of inactivity.

---

## Connect from REPL

Use `USE REMOTE` to connect the LQL REPL to a server:

```sql
USE REMOTE "http://localhost:8080";

-- Queries run on the server
DESCRIBE "France";
WALK "Einstein" TOP 5;
INFER "The capital of France is" TOP 5;
```

---

## Deployment

### Docker

```dockerfile
FROM rust:1.82-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p larql-server

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/larql-server /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["larql-server"]
```

```bash
docker build -t larql-server .
docker run -v ./vindexes:/data -p 8080:8080 larql-server /data/gemma3-4b.vindex
```

### Systemd

```ini
[Unit]
Description=LARQL Vindex Server
After=network.target

[Service]
ExecStart=/usr/local/bin/larql-server /data/gemma3-4b.vindex --port 8080
Restart=always
MemoryMax=4G

[Install]
WantedBy=multi-user.target
```

### Memory Requirements

| Mode | RAM |
|------|-----|
| Browse only | ~100 MB + mmap |
| With inference | ~1.3 GB (walk mode) |
| With dense inference | ~7 GB |

---

## WebSocket Streaming

Connect to `/v1/stream` for layer-by-layer DESCRIBE:

```javascript
const ws = new WebSocket("ws://localhost:8080/v1/stream");

ws.send(JSON.stringify({
  type: "describe",
  entity: "France",
  band: "all"
}));

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "layer") {
    console.log(`Layer ${msg.layer}:`, msg.edges);
  } else if (msg.type === "done") {
    console.log(`Complete: ${msg.total_edges} edges in ${msg.latency_ms}ms`);
  }
};
```

---

## Next Steps

- [Python Bindings](python-bindings.md) — Use from Python
- [Server Spec](../vindex-server-spec.md) — Complete API specification
- [larql-server README](../../crates/larql-server/README.md) — Crate documentation
