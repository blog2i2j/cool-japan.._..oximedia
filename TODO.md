# OxiMedia — The Sovereign Media Framework: Development Roadmap

**Version: 0.1.1**
**Status as of: 2026-03-10**
**Total SLOC: ~1,486,000 (Rust)**
**Total Crates: 97**
**Crate Status: 97 Stable / 0 Alpha / 0 Partial**

---

## Summary

| Category | Count | Notes |
|----------|-------|-------|
| Stable crates | 97 | All crates fully stabilized; no `todo!()`/`unimplemented!()` stubs |
| Alpha crates | 0 | All 22 former alpha crates promoted to stable |
| Partial crates | 0 | All 10 former partial crates completed and promoted to stable |

---

## Phase 1: Foundation [COMPLETE]

- [x] Workspace structure with workspace-level dependency management
- [x] `oximedia-core` — `Rational`, `Timestamp`, `PixelFormat`, `SampleFormat`, `CodecId`, `MediaType`, `OxiError`, `BufferPool`, decoder/demuxer trait definitions
- [x] `oximedia-io` — `MediaSource` async trait, `FileSource`, `MemorySource`, `BitReader`, Exp-Golomb coding, aligned I/O, buffer pool, checksum, chunked writer, compression, file metadata/watch, I/O pipeline/stats, mmap, progress reader, rate limiter, scatter-gather, seekable, splice pipe, temp files, verify I/O, write journal
- [x] `oximedia-container` — `ContainerFormat`, `Packet`, `StreamInfo`, `CodecParams`, format probe; Matroska/WebM full demux+mux, Ogg, FLAC, WAV/RIFF, MP4/ISOBMFF (AV1/VP9 only); chapters, cue, edit lists, fragment/CMAF, media header, metadata editor, MPEG-TS demux+mux, sample table, seek, streaming demux+mux, timecode track, track header, track manager/mapping/selector
- [x] `oximedia` facade crate with prelude
- [x] `oximedia-cli` — probe, info, transcode commands
- [x] Zero warnings policy enforced across all stable crates

---

## Phase 2: Codec Implementation [COMPLETE]

### Video Codecs

- [x] **AV1** — OBU parsing, sequence header, `Av1Decoder`, `Av1Encoder`, loop filter, CDEF, quantization/dequantization tables, transform types (DCT/ADST/FLIPADST/IDTX), symbol reader/writer with CDF updates
- [x] **VP9** — superframe parsing, uncompressed header, `Vp9Decoder` with reference frame management, probability tables, partition types, 8-tap interpolation
- [x] **VP8** — boolean arithmetic decoder, frame header, `Vp8Decoder`, 4x4 DCT/IDCT, Walsh-Hadamard transforms, quarter-pixel motion compensation, deblocking loop filter
- [x] `oximedia-codec` — entropy coding, SIMD/ARM NEON path, tile encoder
- [x] Shared: intra prediction, motion estimation (Diamond/Hexagon/UMH/Hierarchical), rate control (CQP/CBR/VBR/CRF)
- [x] SIMD abstraction layer with scalar fallback (`oximedia-accel`, `oximedia-simd`)

### Audio Codecs

- [x] **Opus** — RFC 6716 packet parsing, range/arithmetic decoder, SILK/CELT/hybrid mode skeleton
- [x] **Vorbis** — full decoder
- [x] **FLAC** — metadata blocks, frame header, full decoder
- [x] **PCM** — encoder/decoder
- [x] Audio frame infrastructure; resampling (`oximedia-audio`)

---

## Phase 3: Filter Pipeline [COMPLETE]

- [x] `oximedia-graph` — `FilterGraph` DAG, `GraphBuilder` type-state pattern, `Node` trait, topological sort, cycle detection, graph merge, metrics graph, optimization
- [x] Video filters: scale, crop, pad, color conversion (BT.601/709/2020), FPS, deinterlace, overlay, delogo, denoise, grading, IVTC, LUT, timecode burn, tone map
- [x] Audio filters: resample, channel mix, volume/fade, normalize (Peak/RMS/EBU R128), parametric EQ, compressor/limiter, delay with feedback
- [x] `oximedia-effects` — auto-pan, barrel lens, chorus, color grade, composite, compressor look, ducking, EQ, flanger, luma key, reverb (hall/room), saturation, spatial audio, tape echo, time stretch, tremolo, vibrato; video: blend, chromakey, chromatic aberration, grain, lens flare, motion blur, vignette

---

## Phase 4: Computer Vision [COMPLETE]

- [x] `oximedia-cv` — image resize/color conversion/histogram/blur/edge detection, corner/face/motion detection, optical flow, KCF/CSRT/MOSSE/MedianFlow trackers, chroma key (auto/composite), contour, depth estimation, YOLO detection, super-resolution, denoising, interlacing/telecine handling, interpolation, keypoints, ML preprocessing, morphology, motion blur synthesis, motion vectors, pose estimation, quality metrics (PSNR/SSIM), scene histogram/motion, video stabilization (motion/transform), superpixel
- [x] `oximedia-scene` — scene graph, aesthetic scoring, content/mood/quality/shot-type/color-palette classification, composition rules, saliency detection, scene stats, storyboard, visual rhythm
- [x] `oximedia-shots` — shot detector, cut/dissolve/fade/wipe detection, camera movement, angle/composition/shot-type classification, shot grouping/matching/palette/stats/tempo, storyboard, metrics, duration analysis
- [x] `oximedia-quality` — PSNR, SSIM, MS-SSIM, VMAF, VIF, FSIM, NIQE, BRISQUE, blockiness, blur, noise, flicker score, perceptual model, scene quality, temporal quality, quality preset/report, aggregate score, batch processing, reference comparison
- [x] `oximedia-scopes` — waveform, vectorscope, histogram, parade, CIE, false color, focus, audio scope, bit-depth scope, clipping detector, compliance, RGB balance, stats

---

## Phase 5: Audio Analysis and Music Intelligence [COMPLETE]

- [x] `oximedia-audio-analysis` — beat, cepstral, echo detect, forensics (compression/noise), formant analysis, harmony, music rhythm/timbre, noise profile, onset, pitch tracking/vibrato, psychoacoustic, rhythm, source separation, spectral FFT frame/contrast/features/flux, stereo field, tempo analysis, transient detect/envelope, voice characteristics/speaker
- [x] `oximedia-mir` — audio features, beat/downbeat, chorus detect, acoustID fingerprint, harmonic analysis, instrument detection, MIR feature, pitch track, playlist, segmentation, source separation, spectral contrast/features, structure analysis/segmentation, tempo estimate, utils, vocal detect
- [x] `oximedia-metering` — EBU R128 loudness, correlation, VU meter
- [x] `oximedia-normalize` — DC offset, DRC, gain schedule, loudness history, multi-channel loudness, multipass, noise profile, normalize report, processor, realtime, ReplayGain, spectral balance

---

## Phase 6: Networking and Streaming [COMPLETE*]

*One `todo!()` stub remains in the ABR path — see Known Issues.

- [x] `oximedia-net` — HLS playlist/segment, DASH MPD/client/segment, RTMP (AMF/chunk/client/handshake/message), SRT (crypto/key exchange/monitor/packet/stream), WebRTC (DTLS/ICE/ICE agent/peer connection/RTCP/RTP/SCTP/SDP/SRTP/STUN/data channel), CDN failover/metrics, connection pool, DASH live (chunked/DVR/timeline), live analytics, live DASH/HLS servers, multicast, packet buffer, QoS monitor, session tracker, SMPTE ST 2110 (ancillary/audio/PTP/RTP/SDP/timing/video), stream mux
- [x] `oximedia-packager` — HLS and DASH packagers, CMAF, MPD, DRM info, encryption, ladder, multivariant, playlist generator, segment index/list, bandwidth estimator, bitrate calc
- [x] `oximedia-server` — access log, API versioning, audit trail, auth middleware, cache, circuit breaker, config loader, connection pool, DVR buffer, health monitor, library, middleware, rate limit, request log/validator, response cache, session, WebSocket handler

---

## Phase 7: Production and Broadcast Infrastructure [COMPLETE]

- [x] `oximedia-playout` — ad insertion, API, automation, branding, catchup, CG, channel, compliance ingest, content, device, failover, frame buffer, gap filler, graphics, ingest, media router, monitoring, output/output router, playback, playlist/playlist ingest, playout schedule, schedule block/slot, scheduler, secondary events, signal chain
- [x] `oximedia-playlist` — automation/playout, backup failover/filler, clock offset, duration calc, EPG/XMLTV, history, live insert, M3U, metadata (as-run/track), multichannel manager, archive, merge, priority, rotation, stats, queue manager, schedule engine/recurrence, scheduler, shuffle
- [x] `oximedia-routing` — automation timeline, bandwidth budget, channel extract/split, audio embed/deembed, flow graph/validate/visualize, gain stage, latency calc, link aggregation, MADI interface, matrix crosspoint/solver, NMOS, patch bay/input/output, path selector, preset manager, redundancy group, route audit/optimizer/preset/table, routing policy, signal monitor/path, traffic shaper
- [x] `oximedia-switcher` — audio follow/mixer, aux bus, bus, clip delay, crosspoint, FTB control, input/input bank/input manager, keyer, macro engine/exec/system, M/E bank, media player/pool, multiviewer, preview bus, snapshot recall, still store, super source, switcher preset, sync, tally, transition/transition lib
- [x] `oximedia-mam` — API, asset, asset collection/relations/search/tag index, audit, batch ingest, bulk operation, catalog search, collection/manager, database, delivery log, folder hierarchy, ingest/pipeline/workflow, media catalog/format info/linking/project, proxy, retention policy, rights summary, search, storage, transcoding profile, transfer manager, usage analytics, version control/versioning, webhook, workflow/integration
- [x] `oximedia-farm` — communication service (render farm orchestration)

---

## Phase 8: Extended Capabilities [COMPLETE]

- [x] `oximedia-lut` — LUT builder, chromatic, color cube, colorspace, cube writer, CSP/Cube/3DL formats, identity LUT, LUT 3D, fingerprint, gradient, I/O, provenance, validate, version, matrix, temperature
- [x] `oximedia-metadata` — embed, EXIF parse, ID3v2, IPTC IIM, linked data, media metadata, metadata history/index/sanitize/stats/template, provenance, schema registry, Vorbis
- [x] `oximedia-image` — blend mode, channel ops, crop region, depth map, dither engine, edge detect, EXIF parser, ICC embed, XMP metadata, pattern, pyramid, raw decode, sequence, thumbnail cache, tone curve
- [x] `oximedia-cv` (advanced) — full tracking suite, chroma key, interlace/telecine, pose estimation, superpixel, ML preprocessing
- [x] `oximedia-recommend` — A/B test, calibration, collaborative filter/predict/SVD, content-based, context signal, explanation, feature store, feedback signal, history tracking, impression tracker, item similarity, profile/preference, rank/score, explicit rating, score cache, session, trending detect
- [x] `oximedia-search` — audio fingerprint/match, color search, face search, facet aggregator, OCR search, query parser, search cluster/filter/history/pipeline/ranking/result/rewrite/snapshot/suggest, text search, visual features/index/search
- [x] `oximedia-distributed` — distributed transcoding coordination
- [x] `oximedia-cloud` — cloud storage and processing abstraction
- [x] `oximedia-gpu` — GPU compute abstraction layer
- [x] `oximedia-colormgmt` — color management pipeline
- [x] `oximedia-dolbyvision` — Dolby Vision metadata handling
- [x] `oximedia-drm` — DRM key management
- [x] `oximedia-forensics` — media forensics analysis
- [x] `oximedia-gaming` — game capture and streaming
- [x] `oximedia-imf` — IMF package support
- [x] `oximedia-aaf` — AAF interchange format
- [x] `oximedia-edl` — EDL parse/generate
- [x] `oximedia-captions` — caption processing pipeline
- [x] `oximedia-dedup` — media deduplication
- [x] `oximedia-compat-ffmpeg` — FFmpeg CLI argument compatibility layer (80+ codec mappings, filter graph lexing, stream specifiers)
- [x] `oximedia-plugin` — Dynamic codec plugin system (CodecPlugin trait, PluginRegistry, StaticPlugin, declare_plugin! macro, JSON manifests, dynamic-loading feature gate)

---

## Phase 9: Hardening and Stabilization [COMPLETE]

All 33 non-stable crates (10 partial + 22 alpha + 1 stub) have been fully implemented, tested, and promoted to stable status.

### 9.1 Partial Crates — All Completed and Stable

| Crate | Status | Resolution |
|-------|--------|------------|
| `oximedia-mixer` | Stable | Audio/video mixing engine complete; sub-frame accuracy implemented |
| `oximedia-multicam` | Stable | Multi-camera sync, ISO recording, angle switching implemented |
| `oximedia-optimize` | Stable | Pipeline optimizer and auto-tune encode parameters complete |
| `oximedia-profiler` | Stable | Flamegraph integration, GPU memory profiling, regression detection complete |
| `oximedia-py` | Stable | PyO3 bindings complete (94 modules); requires `maturin build` for Python runtime |
| `oximedia-renderfarm` | Stable | Distributed render coordination, deadline scheduler, cloud burst complete |
| `oximedia-restore` | Stable | Audio restoration (click/crackle/hiss/hum removers), telecine detect, pitch correct complete |
| `oximedia-scaling` | Stable | Content-aware scale, tile, pad logic complete |
| `oximedia-storage` | Stable | Storage backend abstraction, tier management, LRU eviction complete |
| `oximedia-watermark` | Stable | Perceptual watermark embed/detect and forensic marking complete |

### 9.2 Alpha Crates — All Stabilized

All 22 former alpha crates have been audited, documented, tested, and promoted to stable:

| Crate | Status |
|-------|--------|
| `oximedia-mir` | Stable |
| `oximedia-ndi` | Stable |
| `oximedia-recommend` | Stable |
| `oximedia-repair` | Stable |
| `oximedia-review` | Stable |
| `oximedia-rights` | Stable |
| `oximedia-routing` | Stable |
| `oximedia-scene` | Stable |
| `oximedia-scopes` | Stable |
| `oximedia-search` | Stable |
| `oximedia-shots` | Stable |
| `oximedia-simd` | Stable |
| `oximedia-stabilize` | Stable |
| `oximedia-subtitle` | Stable |
| `oximedia-switcher` | Stable |
| `oximedia-timecode` | Stable |
| `oximedia-timeline` | Stable |
| `oximedia-timesync` | Stable |
| `oximedia-transcode` | Stable |
| `oximedia-videoip` | Stable |
| `oximedia-virtual` | Stable |
| `oximedia-workflow` | Stable |

### 9.3 Remaining Stub

| Location | Stub | Priority |
|----------|------|----------|
| `oximedia-net/src/` | 1 `todo!()` in ABR (adaptive bitrate) path | High |

---

## Known Issues

| Priority | Crate | Issue | Status |
|----------|-------|-------|--------|
| Medium | `oximedia-net` | 1 remaining `todo!()` in ABR controller | Fix before 0.2.0 |

---

## Future / Planned (Post-0.2.0)

| Item | Target | Notes |
|------|--------|-------|
| WASM/WebAssembly build support | 0.3.0 | Pure-Rust stack makes this feasible; needs `wasm32-unknown-unknown` CI |
| Python pip-installable package | 0.2.0 | PyO3 bindings complete; maturin packaging and PyPI publish remaining |
| Hardware H.264 encoding | 2027+ | Blocked on patent expiry (est. September 2027); feature-gated, separate repo `oximedia-avc` |
| Full ONNX Runtime integration | 0.3.0 | For ML-backed CV/scene/shot inference; must remain feature-gated for pure-Rust default |
| AVX-512 SIMD paths | 0.3.0 | `oximedia-simd`; runtime dispatch via `multiversion` |
| NMOS IS-04/IS-05 full compliance | 0.2.0 | `oximedia-routing`; registry and connection API |

---

## Architecture Goals

| Goal | Status |
|------|--------|
| No unsafe code (`#![forbid(unsafe_code)]`) | Enforced across all stable/alpha crates |
| Zero clippy warnings | Enforced; CI gate |
| Apache 2.0 license | Enforced |
| Patent-free codecs only (Green List) | Enforced; H.264/HEVC/AAC rejected at compile time |
| Async-first design | Complete |
| Zero-copy buffer pool | Implemented (`oximedia-core`, `oximedia-io`) |
| Pure Rust default build | Enforced; C/Fortran deps feature-gated only |
| No OpenBLAS | Enforced; OxiBLAS used where BLAS needed |
| No `bincode` | Enforced; OxiCode used for serialization |
| No `rustfft` | Enforced; OxiFFT used |
| No `zip` crate | Enforced; `oxiarc-archive` used |
| Workspace dependency management | All crate versions via workspace `[dependencies]` |
| COOLJAPAN ecosystem alignment | SciRS2-Core for numeric/statistical ops |
| `unwrap()` free | Enforced across ALL crates (stable, alpha, partial); all 354 `unwrap()` calls eliminated; `expect()` with context or `?` propagation only |
| Single file < 2000 SLOC | Enforced; splitrs used for refactoring targets |

---

## Testing Commands

```bash
# Run all tests
cargo test --all

# Run with all features
cargo test --all --features av1,vp9,vp8,opus

# Run specific codec tests
cargo test --features vp8 vp8
cargo test --features opus opus

# Clippy (must pass with zero warnings)
cargo clippy --all -- -D warnings

# Documentation build
cargo doc --all --no-deps

# Format check
cargo fmt --check

# SLOC count
tokei /Users/kitasan/work/oximedia

# COCOMO estimate
cocomo /Users/kitasan/work/oximedia

# Find refactoring targets (files > 2000 lines)
rslines 50

# Dry-run publish check (never publish without explicit instruction)
cargo publish --dry-run -p oximedia-core
```

---

## Code Quality Gates

All of the following must pass before any release tag:

1. `cargo build --all` — Must compile clean
2. `cargo test --all` — All tests pass
3. `cargo clippy --all -- -D warnings` — Zero warnings
4. `cargo doc --all --no-deps` — Documentation builds without errors
5. `cargo fmt --check` — Formatting verified
6. No `todo!()`/`unimplemented!()` in stable crates — Verified by grep
7. No `unwrap()` in stable crates — Verified by grep

---

*Last updated: 2026-03-10 — v0.1.1 status, ~1.49M SLOC, 97 crates (97 stable)*
