# oximedia-codec

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Video and audio codec implementations for the OxiMedia multimedia framework. Pure-Rust, royalty-free codecs with image I/O support.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Overview

`oximedia-codec` provides encoding and decoding for royalty-free video codecs plus image I/O:

| Codec    | Status   | Feature Flag |
|----------|----------|--------------|
| AV1      | Complete | `av1` (default) |
| VP9      | Complete | `vp9` |
| VP8      | Complete | `vp8` |
| Theora   | Complete | `theora` |
| Opus     | Complete | `opus` |
| PNG      | Yes      | `image-io` (default) |
| JPEG     | Yes      | `image-io` (default) |
| WebP     | Yes      | `image-io` (default) |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-codec = "0.1.1"
# or with additional codecs:
oximedia-codec = { version = "0.1.1", features = ["av1", "vp9", "vp8", "opus"] }
```

### AV1 Decoding

```rust
use oximedia_codec::{VideoDecoder, Av1Decoder};

let mut decoder = Av1Decoder::new(&codec_params)?;
decoder.send_packet(&packet)?;
while let Some(frame) = decoder.receive_frame()? {
    // Process decoded frame
}
```

### VP9 Decoding

```rust
use oximedia_codec::Vp9Decoder;

let mut decoder = Vp9Decoder::new(&codec_params)?;
decoder.send_packet(&packet)?;
while let Some(frame) = decoder.receive_frame()? {
    // Process decoded frame
}
```

### VP8 Decoding

```rust
use oximedia_codec::Vp8Decoder;

let mut decoder = Vp8Decoder::new()?;
let frame = decoder.decode(&packet_data)?;
```

### Opus Decoding

```rust
use oximedia_codec::opus::OpusDecoder;

let mut decoder = OpusDecoder::new(48000, 2)?;
let audio_frame = decoder.decode_packet(&packet_data)?;
```

## Architecture

### Unified Traits

All codecs implement unified traits:

```rust
pub trait VideoDecoder {
    fn send_packet(&mut self, packet: &EncodedPacket) -> CodecResult<()>;
    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>>;
    fn flush(&mut self) -> CodecResult<()>;
}

pub trait VideoEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()>;
    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>>;
    fn flush(&mut self) -> CodecResult<Vec<EncodedPacket>>;
}
```

### Rate Control Modes

| Mode | Description |
|------|-------------|
| CQP  | Constant QP — fixed quantization |
| CRF  | Constant Rate Factor — perceptual quality |
| CBR  | Constant Bitrate — fixed bitrate target |
| VBR  | Variable Bitrate — quality with bitrate limits |

### SIMD Support

The codec includes a SIMD abstraction layer:
- Scalar fallback (always available)
- SSE/AVX support (x86/x64)
- NEON support (ARM)

## Module Structure (194 source files, 3046 public items)

```
src/
├── lib.rs              # Crate root with re-exports
├── error.rs            # CodecError and CodecResult
├── frame.rs            # VideoFrame, Plane, ColorInfo
├── traits.rs           # VideoDecoder, VideoEncoder traits
├── av1/                # AV1 codec (OBU parsing, symbol coding, entropy)
├── vp9/                # VP9 codec (frame, context)
├── vp8/                # VP8 codec (DCT, motion, loop filter)
├── theora/             # Theora codec (VP3-based)
├── opus/               # Opus audio (SILK, CELT, range decoder)
├── intra/              # Shared intra prediction
├── motion/             # Motion estimation
├── rate_control/       # Rate control framework
├── reconstruct/        # Reconstruction pipeline
├── entropy_coding/     # Entropy coding
├── tile_encoder/       # Tile-based encoding
└── simd/               # SIMD abstraction (scalar, SSE/AVX, NEON)
```

## Patent Policy

All codecs are royalty-free:

**Supported codecs**: AV1, VP9, VP8, Theora, Opus

**Rejected codecs**: H.264, H.265, AAC, AC-3, DTS

When a patent-encumbered codec is detected in a container, a `PatentViolation` error is returned.

## Policy

- No unsafe code (`#![deny(unsafe_code)]`)
- Apache-2.0 license

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
