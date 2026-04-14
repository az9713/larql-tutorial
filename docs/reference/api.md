# Server API Reference

Complete HTTP API reference for the LARQL server.

---

## Base URL

```
http://localhost:8080/v1
```

Multi-model mode:
```
http://localhost:8080/v1/{model-id}
```

---

## Authentication

When `--api-key` is set, include a Bearer token:

```
Authorization: Bearer <api-key>
```

The `/v1/health` endpoint is exempt from authentication.

---

## Knowledge Endpoints

### GET /v1/describe

Query knowledge edges for an entity.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `entity` | string | required | Entity name |
| `band` | string | `knowledge` | Layer band: `syntax`, `knowledge`, `output`, `all` |
| `limit` | integer | 20 | Maximum edges to return |
| `min_score` | float | 5.0 | Minimum gate score |
| `verbose` | boolean | false | Include layer_min, layer_max, count |

**Response:**

```json
{
  "entity": "France",
  "model": "google/gemma-3-4b-it",
  "edges": [
    {
      "relation": "capital",
      "target": "Paris",
      "gate_score": 1436.9,
      "layer": 27,
      "feature": 9515,
      "source": "probe",
      "also": ["Berlin", "Tokyo"]
    }
  ],
  "latency_ms": 0.3
}
```

**Example:**

```bash
curl "http://localhost:8080/v1/describe?entity=France&band=all&limit=10"
```

---

### GET /v1/walk

Trace feature activations for a prompt.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | string | required | Input text |
| `top` | integer | 10 | Top-K features |
| `layers` | string | all | Layer range (e.g., "14-27") |

**Response:**

```json
{
  "prompt": "The capital of France is",
  "hits": [
    {
      "layer": 27,
      "feature": 9515,
      "gate_score": 1436.9,
      "target": "Paris",
      "relation": "capital"
    }
  ],
  "latency_ms": 0.4
}
```

**Example:**

```bash
curl "http://localhost:8080/v1/walk?prompt=Einstein&top=5&layers=20-27"
```

---

### POST /v1/select

SQL-style query over feature metadata.

**Request:**

```json
{
  "entity": "France",
  "relation": "capital",
  "layer_min": 20,
  "layer_max": 30,
  "limit": 10,
  "order_by": "c_score",
  "order": "desc"
}
```

**Response:**

```json
{
  "edges": [
    {
      "layer": 27,
      "feature": 9515,
      "target": "Paris",
      "c_score": 0.95,
      "relation": "capital"
    }
  ],
  "total": 94,
  "latency_ms": 5.2
}
```

---

### GET /v1/relations

List discovered relation types.

**Response:**

```json
{
  "relations": [
    {"name": "capital", "count": 94, "example": "Paris"},
    {"name": "language", "count": 51, "example": "French"}
  ],
  "total": 512
}
```

---

### GET /v1/stats

Model and vindex statistics.

**Response:**

```json
{
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "layers": 34,
  "features": 348160,
  "hidden_size": 2560,
  "vocab_size": 262144,
  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  },
  "probe_labels": 1967,
  "loaded": {
    "browse": true,
    "inference": true
  },
  "size_bytes": 3200000000
}
```

---

## Inference Endpoint

### POST /v1/infer

Run forward-pass inference.

**Request:**

```json
{
  "prompt": "The capital of France is",
  "top": 5,
  "mode": "walk"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Input text |
| `top` | integer | 5 | Top-K predictions |
| `mode` | string | `walk` | `walk`, `dense`, or `compare` |

**Response (walk/dense):**

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

**Response (compare):**

```json
{
  "prompt": "The capital of France is",
  "walk": [{"token": "Paris", "probability": 0.9791}],
  "walk_ms": 517,
  "dense": [{"token": "Paris", "probability": 0.9801}],
  "dense_ms": 535,
  "latency_ms": 1052
}
```

**Errors:**

| Status | Condition |
|--------|-----------|
| 400 | Empty prompt |
| 503 | Inference disabled (`--no-infer` or no weights) |

---

## Decoupled Inference

### POST /v1/walk-ffn

Run gate KNN for a residual vector. For distributed inference where the client runs attention.

**Request (single layer):**

```json
{
  "layer": 26,
  "residual": [0.12, -0.34, ...],
  "top_k": 8192
}
```

**Response:**

```json
{
  "layer": 26,
  "features": [9515, 4532, ...],
  "scores": [1436.9, 26.1, ...],
  "latency_ms": 0.01
}
```

**Request (batched):**

```json
{
  "layers": [0, 1, 2, ...],
  "residual": [0.12, -0.34, ...],
  "top_k": 8192
}
```

**Response:**

```json
{
  "results": [
    {"layer": 0, "features": [...], "scores": [...]},
    {"layer": 1, "features": [...], "scores": [...]}
  ],
  "latency_ms": 0.3
}
```

---

## Patch Endpoints

### POST /v1/patches/apply

Apply a patch in memory.

**Request:**

```json
{
  "patch": {
    "version": 1,
    "base_model": "google/gemma-3-4b-it",
    "operations": [
      {
        "type": "insert",
        "entity": "Atlantis",
        "relation": "capital",
        "target": "Poseidon"
      }
    ]
  }
}
```

**Headers:**

| Header | Description |
|--------|-------------|
| `X-Session-Id` | Session ID for scoped patches (optional) |

**Response:**

```json
{
  "applied": true,
  "operations": 1
}
```

---

### GET /v1/patches

List active patches.

**Headers:**

| Header | Description |
|--------|-------------|
| `X-Session-Id` | Session ID (optional) |

**Response:**

```json
{
  "patches": [
    {
      "name": "medical.vlp",
      "operations": 3,
      "applied_at": "2026-04-10T14:32:00Z"
    }
  ]
}
```

---

### DELETE /v1/patches/{name}

Remove a patch.

**Response:**

```json
{
  "removed": true
}
```

---

## Management Endpoints

### GET /v1/health

Health check. Always returns 200, exempt from auth.

**Response:**

```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "requests_served": 12450
}
```

---

### GET /v1/models

List loaded models (multi-model mode).

**Response:**

```json
{
  "models": [
    {
      "id": "gemma-3-4b-it",
      "path": "/v1/gemma-3-4b-it",
      "features": 348160,
      "loaded": true
    }
  ]
}
```

---

## WebSocket: /v1/stream

Layer-by-layer streaming for DESCRIBE.

**Connect:**

```
ws://localhost:8080/v1/stream
```

**Client message:**

```json
{
  "type": "describe",
  "entity": "France",
  "band": "all"
}
```

**Server messages:**

```json
{"type": "layer", "layer": 14, "edges": []}
{"type": "layer", "layer": 15, "edges": [{"target": "French", "gate_score": 35.2}]}
{"type": "layer", "layer": 27, "edges": [{"relation": "capital", "target": "Paris", "gate_score": 1436.9}]}
{"type": "done", "entity": "France", "total_edges": 6, "latency_ms": 12.3}
```

---

## gRPC

When `--grpc-port` is set, the same endpoints are available via gRPC.

**Proto file:** `proto/vindex.proto`

**Services:**

| Service | Methods |
|---------|---------|
| `VindexService` | `Describe`, `Walk`, `Select`, `Infer`, `GetRelations`, `GetStats`, `WalkFfn` |
| `PatchService` | `Apply`, `List`, `Remove` |
| `HealthService` | `Check` |
| `StreamService` | `StreamDescribe` (server-streaming) |

---

## Error Responses

All errors return JSON:

```json
{
  "error": "Error message"
}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (missing params, invalid syntax) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Not found (model, patch) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable (inference disabled) |

---

## Rate Limiting

When `--rate-limit` is set, excess requests get 429 with:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

---

## Caching

When `--cache-ttl` is set, DESCRIBE responses include:

```
ETag: "abc123"
Cache-Control: max-age=300
```

Conditional requests with `If-None-Match` return 304 if unchanged.
