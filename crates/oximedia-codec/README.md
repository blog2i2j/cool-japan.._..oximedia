# oximedia-codec

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Video and audio codec implementations for the OxiMedia multimedia framework. Pure-Rust, royalty-free codecs with image I/O support.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

Version: 0.1.4 — 2026-04-20 — 3,063 tests

## Overview

`oximedia-codec` provides encoding and decoding for royalty-free video codecs plus image I/O:

| Codec    | Status   | Feature Flag | Notes |
|----------|----------|--------------|-------|
| AV1      | Complete | `av1` (default) | |
| VP9      | Complete | `vp9` | |
| VP8      | Complete | `vp8` | |
| Theora   | Complete | `theora` | |
| MJPEG    | Complete | `mjpeg` | Wraps `oximedia-image` JPEG baseline; ≥28 dB PSNR at Q85 |
| APV      | Complete | `apv` | ISO/IEC 23009-13 royalty-free intra-frame |
| FFV1     | Complete | `ffv1` | RFC 9043 lossless |
| Opus     | Complete | `opus` | |
| Vorbis   | Complete | *(always)* | |
| FLAC     | Complete | *(always)* | |
| PCM      | Complete | *(always)* | |
| JPEG-XL  | Complete | `jpegxl` | ISOBMFF container + streaming decode |
| PNG/APNG | Yes      | *(always)* | |
| WebP     | Yes      | *(always)* | |
| GIF      | Yes      | *(always)* | |
| AVIF     | Yes      | *(always)* | |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-codec = "0.1.4"
# or with additional codecs:
oximedia-codec = { version = "0.1.4", features = ["av1", "vp9", "vp8", "opus"] }
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

## JPEG-XL: ISOBMFF Container Output

`AnimatedJxlEncoder` (feature `jpegxl`) supports two output modes:

### `finish()` — bare codestream

Produces a raw JPEG-XL codestream starting with the `0xFF 0x0A` magic bytes.

### `finish_isobmff()` — ISOBMFF container

Wraps the codestream in the standard ISOBMFF box structure:

```
ftyp  (major brand: "jxl ", compatible: ["jxl ", "isom"])
jxll  (JXL level 5)
jxlp  (codestream packet with is_last flag set)
```

The resulting bytes are decodable by `JxlStreamingDecoder`:

```rust
# // no_run — requires jpegxl feature and runtime data
use std::io::Cursor;
// let bytes = encoder.finish_isobmff()?;
// let decoder = oximedia_codec::jpegxl::JxlStreamingDecoder::new(Cursor::new(bytes));
// for frame_result in decoder? { ... }
```

### Streaming Decode — `JxlStreamingDecoder<R: Read>`

`JxlStreamingDecoder` is a lazy `Iterator<Item = CodecResult<JxlFrame>>` that yields frames
one at a time without buffering the entire sequence. It auto-detects the format:

| Detection bytes | Format | Producer |
|-----------------|--------|----------|
| `[4..8] == b"ftyp"` and `[8..12] == b"jxl "` | ISOBMFF container | `finish_isobmff()` |
| `[0..2] == [0xFF, 0x0A]` | Native bare codestream | `finish()` |

```rust
# // no_run — requires jpegxl feature and runtime data
// use oximedia_codec::jpegxl::JxlStreamingDecoder;
// use std::io::Cursor;
//
// for frame_result in JxlStreamingDecoder::new(Cursor::new(data))? {
//     let frame = frame_result?;
//     println!("{}x{} ticks={}", frame.width, frame.height, frame.duration_ticks);
// }
```

## MJPEG: Baseline JPEG Spec Compliance

The `mjpeg` module (feature `mjpeg`) wraps `oximedia-image`'s pure-Rust JPEG baseline encoder
and decoder. The encoder:

- Emits DQT (Define Quantization Table) segments with quantization values in the standard
  JPEG zigzag scan order
- Achieves ≥28 dB PSNR at quality setting 85 for natural images (verified by round-trip tests)
- Produces fully self-contained JFIF-compliant JPEG frames suitable for AVI and MP4 containers

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
