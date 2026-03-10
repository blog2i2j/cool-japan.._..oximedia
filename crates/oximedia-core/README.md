# oximedia-core

![Status: Stable](https://img.shields.io/badge/status-stable-green)
![Version: 0.1.1](https://img.shields.io/badge/version-0.1.1-blue)

Core types and traits for OxiMedia.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Overview

`oximedia-core` provides the foundational types used throughout OxiMedia:

- **Types**: Rational numbers, timestamps, pixel/sample formats, codec IDs, FourCC
- **Traits**: Decoder and demuxer interfaces
- **Error handling**: Unified error type with patent violation detection
- **Memory management**: Buffer pools for zero-copy operations
- **HDR support**: HDR metadata, transfer functions, color primaries
- **Channel layout**: Multi-channel audio layout definitions
- **Codec negotiation**: Format negotiation between components
- **Event queue / Work queue**: Internal messaging primitives

## Features

### Types

| Type | Description |
|------|-------------|
| `Rational` | Exact rational number representation (numerator/denominator) |
| `Timestamp` | Media timestamp with timebase support |
| `PixelFormat` | Video pixel format (YUV420p, RGB24, etc.) |
| `SampleFormat` | Audio sample format (F32, I16, etc.) |
| `CodecId` | Codec identifier (Green List only) |
| `MediaType` | Media type classification (Video, Audio, Subtitle) |
| `FourCC` | Four-character code for container/codec identification |

### Error Handling

| Error Type | Description |
|------------|-------------|
| `IoError` | I/O operation failures |
| `FormatError` | Container format issues |
| `CodecError` | Codec-specific errors |
| `PatentViolation` | Attempted use of patent-encumbered codec |

### Memory Management

| Type | Description |
|------|-------------|
| `BufferPool` | Zero-copy buffer allocation and reuse |

## Usage

```rust
use oximedia_core::types::{Rational, Timestamp, PixelFormat, CodecId};
use oximedia_core::error::OxiResult;

fn example() -> OxiResult<()> {
    // Create a timestamp at 1 second with millisecond precision
    let ts = Timestamp::new(1000, Rational::new(1, 1000));
    assert!((ts.to_seconds() - 1.0).abs() < f64::EPSILON);

    // Check codec properties
    let codec = CodecId::Av1;
    assert!(codec.is_video());

    // Check pixel format properties
    let format = PixelFormat::Yuv420p;
    assert!(format.is_planar());
    assert_eq!(format.plane_count(), 3);

    Ok(())
}
```

## Module Structure

```
src/
├── lib.rs              # Crate root with re-exports
├── error.rs            # OxiError and OxiResult
├── types/
│   ├── mod.rs
│   ├── rational.rs     # Rational number type
│   ├── timestamp.rs    # Timestamp with timebase
│   ├── pixel_format.rs # Video pixel formats
│   ├── sample_format.rs # Audio sample formats
│   └── codec_id.rs     # Codec identifiers
├── traits/
│   ├── mod.rs
│   ├── decoder.rs      # VideoDecoder trait
│   └── demuxer.rs      # Demuxer trait
├── alloc/
│   └── buffer_pool.rs  # Zero-copy buffer pool
├── buffer_pool.rs      # Buffer pool (crate root level)
├── channel_layout.rs   # Multi-channel audio layouts
├── codec_info.rs       # Codec information
├── codec_negotiation.rs # Format negotiation
├── convert.rs          # Type conversions
├── error_context.rs    # Contextual error wrapping
├── event_queue.rs      # Internal event queue
├── fourcc.rs           # FourCC code support
├── frame_info.rs       # Frame metadata
├── hdr.rs              # HDR metadata and transfer functions
├── media_time.rs       # Media time utilities
├── memory.rs           # Memory management
├── pixel_format.rs     # Pixel format (crate root level)
├── rational.rs         # Rational arithmetic
├── resource_handle.rs  # Resource lifecycle management
├── sample_format.rs    # Sample format (crate root level)
├── sync.rs             # Synchronization primitives
├── type_registry.rs    # Runtime type registry
├── version.rs          # Version information
├── work_queue.rs       # Work queue primitive
└── wasm.rs             # WASM bindings (wasm32 target only)
```

## Green List (Supported Codecs)

| Category | Codecs |
|----------|--------|
| Video | AV1, VP9, VP8, Theora |
| Audio | Opus, Vorbis, FLAC, PCM |
| Subtitle | WebVTT, ASS/SSA, SRT |

Attempting to use patent-encumbered codecs (H.264, H.265, AAC, etc.) will result in a `PatentViolation` error.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `wasm` | WASM bindings via wasm-bindgen |

## Policy

- No warnings (clippy pedantic)
- Apache 2.0 license

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
