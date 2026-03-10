# oximedia-io

![Status: Stable](https://img.shields.io/badge/status-stable-green)
![Version: 0.1.1](https://img.shields.io/badge/version-0.1.1-blue)

I/O layer for the OxiMedia multimedia framework, providing async media sources, bit-level reading, buffered I/O, memory-mapped files, checksums, compression, and pipeline I/O utilities.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Features

- **Media Sources** — Async file, memory, and seekable sources
- **Bit-Level I/O** — Bit reader with Exp-Golomb coding for binary format parsing
- **Aligned I/O** — Memory-aligned I/O for optimal DMA performance
- **Buffered I/O** — Read-ahead buffering with configurable buffer sizes
- **Memory-Mapped I/O** — mmap-based zero-copy file access
- **Scatter-Gather I/O** — Vectorized I/O for multi-buffer operations
- **Chunked Writing** — Write large files in chunks with progress tracking
- **Copy Engine** — High-throughput async file copy
- **Checksum** — CRC32, CRC64, SHA-256, BLAKE3 checksumming
- **Compression** — zstd, LZ4, gzip, bzip2 compress/decompress
- **Progress Reader** — Async reader with progress callback
- **Rate Limiter** — Bandwidth-limited I/O for upload/download throttling
- **Ring Buffer** — Lock-free ring buffer for streaming
- **Splice Pipe** — Zero-copy splice between descriptors (Linux)
- **Temp Files** — Secure temporary file creation and management
- **Verification I/O** — Read-back verification for write integrity
- **Write Journal** — Journaled writes for crash-safe I/O
- **File Metadata** — Extended file metadata (size, timestamps, permissions)
- **File Watch** — File system event watching
- **I/O Pipeline** — Composable I/O pipeline stages
- **I/O Stats** — Throughput and latency statistics

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-io = "0.1.1"
```

### File Source

```rust
use oximedia_io::source::{FileSource, MediaSource};

#[tokio::main]
async fn main() -> oximedia_core::OxiResult<()> {
    let mut source = FileSource::open("video.webm").await?;

    let mut buffer = [0u8; 1024];
    let bytes_read = source.read(&mut buffer).await?;

    println!("Read {} bytes", bytes_read);
    Ok(())
}
```

### BitReader

```rust
use oximedia_io::bits::BitReader;

fn parse_header(data: &[u8]) {
    let mut reader = BitReader::new(data);
    let flag = reader.read_bit().unwrap();
    let value = reader.read_bits(8).unwrap();
    let ue_value = reader.read_ue().unwrap();  // Unsigned Exp-Golomb
    let se_value = reader.read_se().unwrap();  // Signed Exp-Golomb
}
```

### Checksum

```rust
use oximedia_io::checksum::{Checksum, ChecksumAlgorithm};

let mut checksum = Checksum::new(ChecksumAlgorithm::Blake3);
checksum.update(b"media data");
let digest = checksum.finalize();
```

## API Overview

**Core types:**
- `MediaSource` — Unified async trait for media access
- `FileSource` — Tokio async file source
- `MemorySource` — In-memory bytes source
- `BitReader` — Bit-level binary reader with Exp-Golomb support

**Modules:**
- `source` — Media source traits and implementations
- `bits` — Bit-level I/O and Exp-Golomb coding
- `aligned_io` — Memory-aligned I/O
- `async_io` — Async I/O abstractions
- `buffered_io` — Buffered read-ahead I/O
- `mmap` — Memory-mapped file access
- `scatter_gather` — Vectorized scatter-gather I/O
- `seekable` — Seekable source utilities
- `chunked_writer` — Chunked file writing
- `copy_engine` — High-throughput async copy
- `checksum` — CRC32/CRC64/SHA-256/BLAKE3 checksums
- `compression` — zstd/LZ4/gzip/bzip2 compress/decompress
- `progress_reader` — Progress-reporting async reader
- `rate_limiter` — Bandwidth throttling
- `ring_buffer` — Lock-free ring buffer
- `splice_pipe` — Zero-copy splice (Linux)
- `temp_files` — Secure temporary file management
- `verify_io` — Write verification
- `write_journal` — Journaled writes
- `file_metadata` — Extended file metadata
- `file_watch` — File system watcher
- `io_pipeline` — Composable I/O pipeline
- `io_stats` — I/O throughput and latency statistics
- `buffer_pool` — Buffer pool for I/O operations

## Design Principles

All I/O operations are async by default, built on tokio. `MemorySource` uses `bytes::Bytes` for zero-copy buffer sharing.

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
