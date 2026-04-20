# oximedia-codec TODO

## Current Status
- 100+ source files; video codecs: AV1, VP9, VP8, Theora, H.263, FFV1; audio: Opus (SILK+CELT); image: PNG, GIF, WebP, JPEG-XL
- Feature-gated codecs: av1, vp9, vp8, theora, h263, opus, ffv1, jpegxl, image-io
- Subsystems: rate_control (CBR/VBR/CRF/CQP), multipass encoding, SIMD (x86 AVX2/AVX-512, ARM, scalar), intra prediction, motion estimation, tile encoding, reconstruction (CDEF, deblock, film grain, super-res)
- Re-exports: VideoFrame, AudioFrame, VideoDecoder/Encoder traits, rate control types, reconstruction pipeline, tile encoder

## Wave 2 Progress (2026-04-17)
- [x] JPEG encoder+decoder spec-compliance fix: zigzag DQT table ordering, correct AC Huffman symbol ordering — Wave 2 Slice A (2026-04-17).
- [x] AJXL ISOBMFF animated encoder: finish_isobmff(), jxlp box helpers, JxlAnimation/AnimationHeader types — Wave 2 Slice D (2026-04-17).
- [x] AJXL streaming decoder iterator: JxlStreamingDecoder<R: Read>, ISOBMFF + native bitstream auto-detect — Wave 2 Slice E (2026-04-17).

## Enhancements
- [x] Complete VP9 encoder (Vp9Encoder exists with 28+ tests)
- [x] Complete VP8 encoder (Vp8Encoder exists with 28+ tests)
- [x] Improve AV1 film grain synthesis fidelity in `av1/film_grain.rs` with per-block grain parameters
- [x] Add temporal scalability (SVC) support to AV1 encoder via `av1/svc_encoder.rs`
- [x] Extend `rate_control/lookahead.rs` with scene-adaptive bitrate allocation using content analysis (`rate_control/scene_adaptive.rs`)
- [x] Improve Opus encoder with voice activity detection (VAD) in `opus/silk.rs`
- [x] Add adaptive quantization matrix selection in `av1/quantization.rs` based on content type
- [x] Extend `motion/diamond.rs` with hexagonal and UMHex search patterns for faster estimation

## New Features
- [x] Add AVIF still image encoding/decoding (AV1-based) as a codec variant
- [x] Implement Vorbis audio encoder/decoder (vorbis module re-exported)
- [x] Add FLAC audio encoder/decoder for lossless audio (flac module exists)
- [x] Implement PCM codec support (pcm module exists)
- [x] Add APNG (animated PNG) support in `png/` module (apng module exists)
- [x] Implement WebP animation encoding in `webp` module (`webp/animation.rs`)
- [x] Add two-pass encoding support to the Theora encoder (`theora/two_pass.rs`)
- [x] Implement constant-quality mode for GIF encoder (`gif/quality.rs`)

## Performance
- [x] Expand SIMD coverage: ARM NEON implementations in `simd/arm/neon.rs` with real intrinsics
- [x] Add WASM SIMD128 backend in `simd/wasm.rs` with real `core::arch::wasm32` intrinsics
- [x] Optimize AV1 CDEF filter with SIMD in `simd/av1/cdef.rs` for 10-bit depth (`cdef_filter_u16`)
- [x] Add parallel tile decoding in `tile.rs` using rayon work-stealing
- [x] Optimize entropy coding in `entropy_coding.rs` with table-based arithmetic coding
- [x] Profile and optimize `reconstruct/loop_filter.rs` hot paths with cache-friendly access patterns
- [x] Optimize entropy coding in `entropy_tables.rs` with table-based CDF arithmetic coding (RangeCoder, CdfTable, 4 AV1 tables, encode/decode_symbol_table, 31 tests)
- [x] Add SIMD-accelerated pixel format conversion for YUV420/422/444

## Testing
- [x] Add bitstream conformance tests for AV1 decoder against reference test vectors
- [x] Add round-trip encode/decode quality tests for each codec (PSNR > threshold) — see `tests/codec_quality.rs`
- [x] Test rate control accuracy: verify CBR output stays within 10% of target bitrate — 3 CBR verifier tests
- [x] Add fuzzing targets for `png/decoder.rs`, `gif` decoder, and `webp` decoder
- [x] Test multipass encoding produces better quality than single-pass at same bitrate
- [x] Add regression tests for `jpegxl` modular and ANS coding paths

## Documentation
- [x] Document codec feature matrix (encode/decode, bitdepth, chroma support) in crate-level docs
- [x] Add rate control tuning guide with examples for each mode (CBR/VBR/CRF/CQP)
- [x] Document SIMD dispatch mechanism in `simd/mod.rs`
