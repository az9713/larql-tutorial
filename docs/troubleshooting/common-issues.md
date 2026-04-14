# Common Issues

Solutions to frequently encountered problems. Issues are ordered by frequency.

---

## Extraction Issues

### "Model not found in HuggingFace cache"

**Cause:** The model hasn't been downloaded to the local cache.

**Fix:**

```bash
# Download the model first
huggingface-cli download google/gemma-3-4b-it

# Then extract
larql extract-index google/gemma-3-4b-it -o gemma.vindex
```

**If that doesn't work:**

Check your HuggingFace authentication:
```bash
huggingface-cli login
```

---

### "Out of memory during extraction"

**Cause:** The model requires more RAM than available, even with streaming.

**Fix:**

```bash
# Reduce thread count
RAYON_NUM_THREADS=2 larql extract-index model -o vindex

# Or use a smaller model for testing
larql extract-index google/gemma-2-2b -o gemma2b.vindex
```

---

### "Extraction hangs at a specific layer"

**Cause:** Large tensor being processed, or disk I/O bottleneck.

**Fix:**

1. Check disk space: `df -h`
2. Check for I/O issues: `iostat 1`
3. Try extracting to a faster disk (SSD)

---

## Query Issues

### "NoBackend: No vindex loaded"

**Cause:** Running a query without `USE` statement.

**Fix:**

```sql
USE "gemma.vindex";
DESCRIBE "France";
```

---

### "VindexNotFound: path does not exist"

**Cause:** The vindex path is incorrect or the vindex doesn't exist.

**Fix:**

```bash
# Check if the vindex exists
ls -la gemma.vindex/

# Use absolute path
USE "/full/path/to/gemma.vindex";
```

---

### "DESCRIBE returns empty results"

**Cause:** The entity isn't in the model's knowledge, or you're querying the wrong band.

**Fix:**

```sql
-- Try all layers
DESCRIBE "entity" ALL LAYERS;

-- Lower the minimum score
DESCRIBE "entity" MIN SCORE 1.0;

-- Check if the entity tokenizes correctly
-- Try variations: "France", "france", "FRANCE"
```

---

### "WALK shows unexpected features"

**Cause:** The prompt tokenization differs from expectations.

**Fix:**

```sql
-- Check tokenization by looking at layer 0 features
WALK "prompt" TOP 20 LAYERS 0-5;

-- Compare with a known-working prompt
WALK "The capital of France is" TOP 10;
```

---

## Inference Issues

### "InferenceDisabled: model weights not loaded"

**Cause:** The vindex was extracted at browse level, not inference level.

**Fix:**

```bash
# Re-extract with inference weights
larql extract-index model -o vindex --level inference --f16
```

Or download a vindex with inference weights:
```bash
larql hf download chrishayuk/gemma-3-4b-it-vindex-inference
```

---

### "INFER returns wrong predictions"

**Cause:** Walk mode and dense mode should match. If they differ significantly, the vindex may be corrupted.

**Fix:**

```sql
-- Compare walk and dense
INFER "prompt" TOP 5 COMPARE;
```

If walk and dense differ by more than 1%:
1. Verify checksums: `larql verify vindex`
2. Re-extract the vindex

---

### "INFER is very slow (>10 seconds)"

**Cause:** Missing optimized files, or Metal backend not enabled.

**Fix:**

```bash
# Build optimized files
cargo run --release -p larql-vindex --example build_down_features -- vindex
cargo run --release -p larql-vindex --example build_up_features -- vindex

# Ensure Metal is enabled (macOS)
cargo build --release --features metal
```

---

## Patch Issues

### "Insert has no effect on INFER"

**Cause:** Alpha is too low, or the fact conflicts with existing knowledge.

**Fix:**

```sql
-- Try higher alpha
INSERT INTO EDGES (entity, relation, target)
    VALUES ("entity", "relation", "target")
    ALPHA 0.35;

-- Check for conflicting knowledge
DESCRIBE "entity";
```

---

### "Insert breaks other facts"

**Cause:** Alpha is too high, or facts have overlapping triggers.

**Fix:**

1. Lower alpha to 0.20
2. Check for entity overlap:
   ```sql
   DESCRIBE "entity1";
   DESCRIBE "entity2";
   ```
3. Use different layers for conflicting facts

---

### "Patch won't apply: incompatible base_model"

**Cause:** The patch was created for a different model.

**Fix:**

Patches are model-specific. Create a new patch for the target model:
```sql
USE "target.vindex";
BEGIN PATCH "new-patch.vlp";
INSERT ...;
SAVE PATCH;
```

---

### "COMPILE fails with conflict"

**Cause:** Multiple patches write to the same feature slot.

**Fix:**

```sql
-- Use explicit conflict resolution
COMPILE CURRENT INTO VINDEX "output.vindex" ON CONFLICT LAST_WINS;

-- Or identify and remove conflicting patches
SHOW PATCHES;
REMOVE PATCH "conflicting.vlp";
```

---

## Server Issues

### "Connection refused on port 8080"

**Cause:** Server isn't running, or wrong port.

**Fix:**

```bash
# Check if server is running
ps aux | grep larql

# Start server
larql serve vindex --port 8080

# Check firewall
# macOS: System Preferences > Security > Firewall
```

---

### "401 Unauthorized"

**Cause:** API key is required but not provided.

**Fix:**

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" "http://localhost:8080/v1/describe?entity=France"
```

---

### "429 Too Many Requests"

**Cause:** Rate limit exceeded.

**Fix:**

Wait for the rate limit window to reset, or increase the limit:
```bash
larql serve vindex --rate-limit "1000/min"
```

---

### "503 Service Unavailable for INFER"

**Cause:** Server started with `--no-infer`, or vindex lacks inference weights.

**Fix:**

```bash
# Start without --no-infer
larql serve vindex --port 8080

# Or use a vindex with inference weights
larql serve vindex-with-inference --port 8080
```

---

## Build Issues

### "Cargo build fails with missing BLAS"

**Cause:** BLAS library not found.

**Fix (macOS):**
BLAS is included in Accelerate framework. Should work automatically.

**Fix (Linux):**
```bash
sudo apt install libopenblas-dev
```

---

### "Metal shaders fail to compile"

**Cause:** Metal feature enabled on non-Apple system, or outdated macOS.

**Fix:**

```bash
# Build without Metal
cargo build --release

# Or ensure macOS 13+ and Xcode installed
xcode-select --install
```

---

### "Test failures in larql-inference"

**Cause:** Tests require model weights that aren't present.

**Fix:**

Some tests are skipped without weights:
```bash
# Run tests that don't need weights
cargo test -p larql-inference -- --skip real_model

# Or download test fixtures
./scripts/download_test_fixtures.sh
```

---

## Performance Issues

### "High memory usage (>8 GB)"

**Cause:** Dense inference mode, or not using mmap.

**Fix:**

1. Use walk mode for inference
2. Verify mmap is working:
   ```bash
   # Check RSS vs virtual memory
   ps aux | grep larql
   # RSS should be much smaller than VIRT
   ```

---

### "Slow startup (>30 seconds)"

**Cause:** Large vindex being loaded without mmap.

**Fix:**

Vindex files should be mmap'd automatically. If slow:
1. Check disk speed (SSD recommended)
2. Verify files aren't compressed
3. Use `--no-infer` for browse-only (faster startup)

---

### "KNN takes >100ms per layer"

**Cause:** Using f32 BLAS instead of Q4 Metal.

**Fix:**

```bash
# Build Q4 gate vectors
cargo run --release -p larql-vindex --example build_gate_q4 -- vindex

# Build with Metal
cargo build --release --features metal
```

---

## Getting Help

If your issue isn't listed:

1. **Check the logs:**
   ```bash
   RUST_LOG=debug larql repl
   ```

2. **Verify the vindex:**
   ```bash
   larql verify vindex
   ```

3. **Check the specs:**
   - [LQL Spec](../lql-spec.md)
   - [Vindex Format Spec](../vindex-format-spec.md)

4. **File an issue:**
   https://github.com/chrishayuk/larql/issues

Include:
- LARQL version (`larql --version`)
- OS and hardware
- Minimal reproduction steps
- Error message (full)
