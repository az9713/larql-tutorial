# Building LARQL on Windows

This guide documents the fixes required to build LARQL on Windows.

## Prerequisites

- **Rust toolchain**: Install via [rustup](https://rustup.rs/)
- **Git Bash**: Recommended shell for running commands
- **Visual Studio Build Tools**: Required for C/C++ compilation

## Platform-Specific BLAS Configuration

The original LARQL codebase used macOS Accelerate framework for BLAS operations. On Windows, we use Intel MKL instead.

### Changes Made

**`crates/larql-compute/Cargo.toml`** and **`crates/larql-inference/Cargo.toml`** now include platform-conditional dependencies:

```toml
# macOS: use Accelerate framework
[target.'cfg(target_os = "macos")'.dependencies]
blas-src = { version = "0.10", features = ["accelerate"] }

# Windows: use Intel MKL (pre-built, no compilation needed)
[target.'cfg(target_os = "windows")'.dependencies]
blas-src = { version = "0.10", default-features = false, features = ["intel-mkl"] }
intel-mkl-src = { version = "0.8", features = ["mkl-static-lp64-seq"] }

# Linux: use OpenBLAS
[target.'cfg(all(unix, not(target_os = "macos")))'.dependencies]
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

### Why Intel MKL?

- Pre-built binaries available (no compilation required)
- Excellent performance on Intel/AMD CPUs
- Static linking avoids DLL distribution issues

## Dependency Resolution Issues

### hf-hub and reqwest

The `larql-vindex` crate references `hf_hub` and `reqwest` but they weren't properly linking. Fix:

```bash
cd crates/larql-vindex
cargo add hf-hub@0.5
cargo add reqwest@0.12 --features blocking,json
```

### tokenizers

The `larql-inference` crate needed explicit tokenizers dependency:

```bash
cd crates/larql-inference
cargo add tokenizers@0.21
```

## Known Limitations

### larql-server (gRPC)

The `larql-server` crate depends on `tonic` and `prost`, which require `protobuf-src`. On Windows, `protobuf-src` fails to compile with linker errors:

```
error LNK2019: unresolved external symbol ceilf
error LNK2019: unresolved external symbol nanf
error LNK2019: unresolved external symbol modf
...
error LNK1120: 11 unresolved externals
```

**Workaround**: Build without the server component:

```bash
cargo build --release -p larql-cli -p larql-lql
```

The CLI and REPL work fully without the gRPC server.

## Successful Build Commands

```bash
# Build core components (CLI + REPL)
cargo build --release -p larql-lql -p larql-cli

# Run the REPL
./target/release/larql repl

# Or on Windows with backslashes
.\target\release\larql.exe repl
```

## Troubleshooting

### "library kind 'framework' is only supported on Apple targets"

This error means macOS-specific BLAS configuration is being used. Ensure the platform-conditional dependencies above are in place.

### Linker errors with math functions (ceilf, nanf, etc.)

This affects protobuf-src compilation. Avoid building `larql-server` on Windows until this is resolved upstream.

### Missing dependencies after Cargo.toml changes

Run `cargo update` to refresh the lock file after modifying dependencies.
