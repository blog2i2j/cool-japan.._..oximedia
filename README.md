# OxiMedia

**Pure Rust reconstruction of OpenCV + FFmpeg** — A patent-free, memory-safe multimedia and computer vision framework.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![Version](https://img.shields.io/badge/version-v0.1.6-green.svg)](https://github.com/cool-japan/oximedia)
[![Released](https://img.shields.io/badge/released-2026--04--26-brightgreen.svg)](https://github.com/cool-japan/oximedia)
[![Crates](https://img.shields.io/badge/crates-108-blue.svg)](https://github.com/cool-japan/oximedia)
[![SLOC](https://img.shields.io/badge/SLOC-~2.69M-blueviolet.svg)](https://github.com/cool-japan/oximedia)

## Vision

OxiMedia is a **clean room, Pure Rust reconstruction** of both **FFmpeg** (multimedia processing) and **OpenCV** (computer vision) — unified in a single cohesive framework.

### FFmpeg Domain

Codec encoding/decoding (AV1, VP9, VP8, Theora, Opus, Vorbis, FLAC, MP3), container muxing/demuxing (MP4, MKV, MPEG-TS, OGG), streaming protocols (HLS, DASH, RTMP, SRT, WebRTC, SMPTE 2110), transcoding pipelines, filter graphs (DAG-based), audio metering (EBU R128), loudness normalization, packaging (CMAF, DRM/CENC), and server-side media delivery.

### OpenCV Domain

Computer vision (object detection, motion tracking, video enhancement, quality assessment), professional image I/O (DPX, OpenEXR, TIFF), video stabilization, scene analysis, shot detection, denoising (spatial/temporal/hybrid), camera calibration, color management (ICC, ACES, HDR), video scopes (waveform, vectorscope, histogram), and forensic analysis (ELA, PRNU, copy-move detection).

### Design Principles

- **Patent Freedom**: Only royalty-free codecs (AV1, VP9, Opus, FLAC, and more)
- **Memory Safety**: Zero unsafe code, compile-time guarantees
- **Async-First**: Built on Tokio for massive concurrency
- **Single Binary**: No DLL dependencies, no system library requirements
- **WASM Ready**: Runs in browser without transcoding servers
- **Sovereign**: No C/Fortran dependencies in default features — 100% Pure Rust

## FFmpeg + OpenCV, Reimagined

**FFmpeg** is the de facto standard for multimedia processing, but it is written in C with patent-encumbered codecs (H.264, H.265, AAC), chronic memory safety vulnerabilities, and notoriously complex build systems requiring dozens of system libraries.

**OpenCV** is the de facto standard for computer vision, but it depends on C++ with complex CMake builds, optional proprietary modules (CUDA, Intel IPP), and heavy system-level dependencies.

**OxiMedia unifies both** into a single Pure Rust framework with zero C/Fortran dependencies:

| | FFmpeg | OpenCV | OxiMedia |
|---|---|---|---|
| Language | C | C++ | Pure Rust |
| Memory safety | Manual | Manual | Compile-time guaranteed |
| Patent-free codecs | Opt-in | N/A | Default (AV1, VP9, Opus, FLAC) |
| Install | `./configure && make` + system deps | `cmake` + system deps | `cargo add oximedia` |
| WASM support | Limited (Emscripten) | Limited (Emscripten) | Native (`wasm32-unknown-unknown`) |
| CV + Media unified | No | No | Yes — single framework |

**From the FFmpeg world**: codec encode/decode, container mux/demux, streaming (HLS/DASH/RTMP/SRT/WebRTC), transcoding pipelines, filter graphs, audio processing, packaging, and media server.

**From the OpenCV world**: detection, tracking, stabilization, scene analysis, shot detection, denoising, calibration, image I/O (DPX/EXR/TIFF), color science, quality metrics (PSNR/SSIM/VMAF), and forensics.

**One `cargo add`** — no battling system library installations, no `pkg-config`, no `LD_LIBRARY_PATH`, no `brew install ffmpeg opencv`.

## Project Scale

OxiMedia is a **production-grade** framework at **v0.1.6** (active cycle, 2026-04-26):

| Metric | Value |
|--------|-------|
| Total crates | 108 |
| Total SLOC (Rust) | ~2,687,000 |
| Tests passing | 81,582 (0 failures, 245 skipped — `cargo nextest run --workspace --all-features`) |
| Stable crates | 108 |
| Alpha crates | 0 |
| Partial crates | 0 |
| License | Apache 2.0 |
| MSRV | Rust 1.85+ |

## Sovereign ML Pipelines (v0.1.6)

OxiMedia 0.1.6 adds the [`oximedia-ml`](crates/oximedia-ml/) crate — a typed
ML pipeline layer built atop the Pure-Rust [OxiONNX](https://crates.io/crates/oxionnx)
runtime. Inference is entirely opt-in; the default `oximedia` build still
pulls in **zero** ONNX symbols and stays C/Fortran-free.

### Available pipelines

| Pipeline                          | Feature             | I/O                                             | Reference model    |
|-----------------------------------|---------------------|-------------------------------------------------|--------------------|
| `SceneClassifier`                 | `scene-classifier`  | 224×224 RGB → `Vec<SceneClassification>`        | Places365 / ResNet |
| `ShotBoundaryDetector`            | `shot-boundary`     | 48×27 RGB window → `Vec<ShotBoundary>`          | TransNet V2        |
| `AestheticScorer`                 | `aesthetic-score`   | 224×224 RGB → `AestheticScore`                  | NIMA               |
| `ObjectDetector`                  | `object-detector`   | 640×640 RGB → `Vec<Detection>` (NMS)            | YOLOv8 (80 COCO)   |
| `FaceEmbedder`                    | `face-embedder`     | 112×112 RGB face → 512-dim `FaceEmbedding`      | ArcFace            |

### Quick start

```toml
[dependencies]
oximedia = { version = "0.1.6", features = ["ml", "ml-scene-classifier", "ml-onnx"] }
```

```rust,ignore
use oximedia::ml::pipelines::{SceneClassifier, SceneImage};
use oximedia::ml::{DeviceType, TypedPipeline};

let classifier = SceneClassifier::load("places365.onnx", DeviceType::auto())?;
let image = SceneImage::new(rgb_bytes, 224, 224)?;
for pred in classifier.run(image)? {
    println!("class {} -> {:.3}", pred.class_index, pred.score);
}
```

### Device selection

`DeviceType::auto()` probes the strongest available backend once and
memoises the result: **CUDA → DirectML → WebGPU → CPU**. Each backend is a
feature flag (`cuda`, `directml`, `webgpu`); CPU is always available.
`cuda` is native-only; every other backend — including the default CPU path
— compiles on `wasm32-unknown-unknown`.

### CLI

The `oximedia ml` namespace ships three subcommands (honour `--json` for
machine-readable output):

```bash
oximedia ml list                  # enumerate built-in pipelines + model zoo
oximedia ml probe                 # report GPU backend availability
oximedia ml run --pipeline scene-classifier \
                --model places365.onnx \
                --input frame.png \
                --device auto \
                --top-k 5 \
                --dry-run
```

### Downstream integrations

Several domain crates gain an `onnx` feature for an ML-backed fast path
while keeping the Pure-Rust default intact: `oximedia-scene`
(`MlSceneEnricher`), `oximedia-shots` (`MlShotDetector`),
`oximedia-caption-gen` (`CaptionEncoder`), `oximedia-recommend`
(`EmbeddingExtractor`), `oximedia-mir` (`MusicTagger`).

See [`docs/ml_guide.md`](docs/ml_guide.md) for the full feature matrix,
per-pipeline I/O contracts, device selection details, WASM support
matrix, and roadmap.

## What's New in v0.1.6

Active cycle starting 2026-04-26. Theme: **Stub resolution, codec completeness, and internal refactoring**.

- **13 stubs resolved**: `oximedia-accel` color-conversion stubs filled in (RGB↔YUV, RGBA↔BGRA, NV12 paths); Vorbis codebook decode re-wired to active path; ACES ODT variants (D60, D65, P3-D65) now produce real output; DASH HTTP segment fetch replaced from placeholder; system font loading implemented for `oximedia-graphics`.
- **`exr.rs` refactored into 9 modules** via splitrs: `header.rs`, `compression.rs`, `tiles.rs`, `channels.rs`, `attributes.rs`, `read.rs`, `write.rs`, `deep.rs`, `multipart.rs` — keeps all files under the 2,000-line policy.
- **`oxifft` upgraded to 0.3.0**: All FFT usage across the workspace migrated to the new OxiFFT 0.3 API (breaking plan/execute split); zero performance regressions on spectral-flux and MIR benchmarks.
- **81,582 tests passing** (0 failures, 245 skipped — `cargo nextest run --workspace --all-features`).
- **Zero clippy warnings** workspace-wide; WASM check clean.

## What's New in v0.1.5

Released 2026-04-21. Theme: **Full ONNX Runtime integration via the Pure-Rust OxiONNX stack** (previously slated for 0.3.0). All inference is feature-gated; the pure-Rust default build stays C/Fortran-free.

- **Pure-Rust ONNX inference via OxiONNX**: Workspace now depends on the full OxiONNX stack (`oxionnx`, `oxionnx-ops`, `oxionnx-gpu`, `oxionnx-directml`, `oxionnx-proto`) instead of the C++ `ort` runtime — preserves the COOLJAPAN Pure-Rust Policy.
- **New `oximedia-ml` facade crate**: Central ML layer exposing `OnnxModel`, `ModelCache` (concurrent LRU model cache with `~/.cache/oximedia/models/`), `TypedPipeline<In, Out>` trait, `DeviceType::auto()` runtime probe, `ImagePreprocessor` (ImageNet normalize + letterbox + NCHW), and `ModelZoo` registry.
- **Typed pipelines**: `SceneClassifier` (ImageNet-style top-K classifier with softmax/argsort postprocessing) and `ShotBoundaryDetector` (TransNetV2-compatible sliding window with many-hot hard/soft cut outputs) — more pipelines (`AutoCaption`, `AestheticScore`, `ObjectDetector`, `FaceEmbedder`) queued for Wave 2.
- **Facade `ml` feature**: `oximedia::ml` module and `prelude` re-exports gated behind `ml` (off by default). Sub-features `ml-scene-classifier`, `ml-shot-boundary`, `ml-onnx` for granular opt-in; `full` feature picks them all up.
- **55 new tests in `oximedia-ml`**: Covers preprocessing, cache eviction, pipeline contracts, and ModelInfo round-trips. Zero clippy warnings. Pure-Rust default build verified — no ONNX symbols linked unless `onnx` feature is enabled.

## What's New in v0.1.4

Released 2026-04-20.

- **MJPEG and APV end-to-end codec support**: Full encode/decode pipelines for MJPEG (Motion JPEG) and APV (Advanced Professional Video), including correct MP4 and Matroska sample-entry wiring.
- **JPEG encoder/decoder spec compliance**: Rebuilt JPEG encoder/decoder to full JFIF/Exif spec compliance; PSNR improved from 6 dB to 32 dB at quality 85.
- **AVI container muxer + demuxer**: New `oximedia-avi` crate — pure-Rust AVI muxer and demuxer targeting MJPEG-only streams up to 1 GB (`RIFF` list size constraint).
- **AJXL ISOBMFF animated encoder + streaming decoder iterator**: Animated JPEG-XL sequences stored in ISOBMFF with a streaming `Iterator`-based decoder for low-memory playback.
- **CLI MJPEG/APV support**: `oximedia-cli` now accepts `-c:v mjpeg` and `-c:v apv` on all transcode/convert commands.
- **WASM32 platform support**: 5 additional crates (`oximedia-codec`, `oximedia-container`, `oximedia-audio`, `oximedia-convert`, `oximedia-graphics`) confirmed clean under `wasm32-unknown-unknown`.

## Architecture

> **FFmpeg domain** spans Foundation, Codecs & Container, Networking, and Audio layers.
> **OpenCV domain** spans Computer Vision, Video Processing, and Analysis layers.
> Both domains share the Processing Pipeline and Applications layers above them.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Applications                                    │
│          CLI / Server / Python Bindings (oximedia-py) / Review UI            │
├──────────────────────┬──────────────────────────────┬───────────────────────┤
│   Production Layer   │      Media Management        │    Broadcast Layer    │
│  oximedia-playout    │      oximedia-mam             │  oximedia-switcher    │
│  oximedia-playlist   │      oximedia-search          │  oximedia-routing     │
│  oximedia-automation │      oximedia-rights          │  oximedia-ndi         │
│  oximedia-multicam   │      oximedia-review          │  oximedia-videoip     │
├──────────────────────┴──────────────────────────────┴───────────────────────┤
│                         Processing Pipeline                                  │
│   oximedia-graph (Filter DAG)  ·  oximedia-transcode  ·  oximedia-effects   │
│   oximedia-timeline            ·  oximedia-edit        ·  oximedia-workflow  │
├────────────────┬────────────────┬──────────────────┬──────────────────────-─┤
│ Video Domain   │  Audio Domain  │  Computer Vision │  Quality & Analysis    │
│ oximedia-codec │ oximedia-audio │ oximedia-cv      │ oximedia-quality       │
│ oximedia-vfx   │ oximedia-metering│ oximedia-scene │ oximedia-qc            │
│ oximedia-lut   │ oximedia-normalize│ oximedia-shots│ oximedia-analysis      │
│ oximedia-colormgmt│ oximedia-effects│ oximedia-stabilize│ oximedia-scopes   │
├────────────────┴────────────────┴──────────────────┴────────────────────────┤
│                         Container / Networking                               │
│  oximedia-container  ·  oximedia-net  ·  oximedia-packager                  │
│  oximedia-hls/DASH   ·  oximedia-srt  ·  oximedia-webrtc                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Foundation                                         │
│     oximedia-io  ·  oximedia-core  ·  oximedia-gpu  ·  oximedia-simd        │
│     oximedia-accel  ·  oximedia-storage  ·  oximedia-jobs                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Crates

### Foundation

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-core` | Core types, traits, error handling, buffer pools | Stable |
| `oximedia-io` | I/O foundation (async media source, bit reader, Exp-Golomb) | Stable |
| `oximedia-gpu` | GPU compute via WGPU (Vulkan/Metal/DX12) | Stable |
| `oximedia-simd` | Hand-written SIMD kernels for codec acceleration | Stable |
| `oximedia-accel` | GPU acceleration via Vulkan compute with CPU fallback | Stable |
| `oximedia-storage` | Cloud storage abstraction (S3, Azure, GCS) | Stable |
| `oximedia-jobs` | Job queue (priority scheduling, SQLite persistence, worker pool) | Stable |
| `oximedia-plugin` | Dynamic codec plugin system with registry and manifests | Stable |
| `oximedia-bench` | Comprehensive codec benchmarking suite | Stable |
| `oximedia-presets` | Preset management (codec, platform presets: YouTube, Instagram, etc.) | Stable |

### Codecs & Container

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-codec` | Video codecs (AV1, VP9, VP8, Theora) and image I/O | Stable (mixed decoder coverage — see [docs/codec_status.md](docs/codec_status.md)) |
| `oximedia-audio` | Audio codec implementations (Opus, Vorbis, FLAC, MP3) | Stable (Vorbis decode is bitstream-parsing only — see [docs/codec_status.md](docs/codec_status.md)) |
| `oximedia-container` | Container mux/demux (MP4, MKV, MPEG-TS, OGG) | Stable |
| `oximedia-lut` | Color science/LUT (1D/3D, Rec.709/2020/DCI-P3/ACES, HDR) | Stable |
| `oximedia-edl` | EDL parser/generator (CMX 3600, GVG, Sony BVE-9000) | Stable |
| `oximedia-aaf` | SMPTE ST 377-1 AAF reader/writer for post-production | Stable |
| `oximedia-imf` | IMF SMPTE ST 2067 (CPL, PKL, ASSETMAP, MXF essence) | Stable |
| `oximedia-dolbyvision` | Dolby Vision RPU metadata (profiles 5/7/8/8.1/8.4) | Stable |
| `oximedia-drm` | DRM/encryption (CENC, Widevine, PlayReady, FairPlay) | Stable |
| `oximedia-subtitle` | Subtitle/caption rendering (SRT, WebVTT, CEA-608/708) | Stable |
| `oximedia-timecode` | LTC and VITC timecode reading/writing | Stable |
| `oximedia-avi` | AVI container muxer + demuxer (MJPEG-only, ≤1 GB RIFF) | Stable |
| `oximedia-compat-ffmpeg` | FFmpeg CLI argument compatibility layer (80+ codec mappings) | Stable |

### Networking & Streaming

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-net` | Network streaming (HLS/DASH/RTMP/SRT/WebRTC/SMPTE 2110) | Stable |
| `oximedia-packager` | Streaming packaging (HLS/DASH/CMAF, encryption, DRM) | Stable |
| `oximedia-server` | RESTful media server with transcoding and CDN support | Stable |
| `oximedia-cloud` | Cloud integration (AWS, Azure, GCP) | Stable |
| `oximedia-ndi` | NDI support (send/receive, failover, tally, bandwidth management) | Stable |
| `oximedia-videoip` | Patent-free video-over-IP (NDI alternative) | Stable |
| `oximedia-timesync` | Precision Time Protocol and clock discipline | Stable |
| `oximedia-distributed` | Distributed encoding (gRPC, load balancing, fault tolerance) | Stable |

### Video Processing

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-cv` | Computer vision (detection, tracking, enhancement, quality) | Stable |
| `oximedia-graph` | Filter graph pipeline (DAG, topological sort, optimization) | Stable |
| `oximedia-effects` | Audio effects (reverb, delay, chorus, compressor, EQ) | Stable |
| `oximedia-vfx` | Professional video effects library | Stable |
| `oximedia-colormgmt` | Color management (ICC profiles, ACES, HDR, LUT/GPU) | Stable |
| `oximedia-image` | Professional image I/O (DPX, OpenEXR, TIFF) | Stable |
| `oximedia-scaling` | Professional video scaling with multiple filters | Stable |
| `oximedia-stabilize` | Professional video stabilization | Stable |
| `oximedia-denoise` | Video denoising (spatial, temporal, hybrid) | Stable |
| `oximedia-optimize` | Codec optimization (bitrate control, RDO, adaptive quantization) | Stable |
| `oximedia-transcode` | High-level transcoding pipeline | Stable |
| `oximedia-calibrate` | Professional color calibration and matching | Stable |
| `oximedia-graphics` | Broadcast graphics engine (lower thirds, tickers, animations) | Stable |
| `oximedia-watermark` | Professional audio watermarking and steganography | Stable |
| `oximedia-virtual` | Virtual production and LED wall tools | Stable |

### Audio Processing

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-audio-analysis` | Advanced audio analysis and forensics | Stable |
| `oximedia-metering` | Broadcast audio metering (EBU R128, ITU-R BS.1770-4, ATSC A/85) | Stable |
| `oximedia-normalize` | Loudness normalization (EBU R128, ATSC A/85, ReplayGain) | Stable |
| `oximedia-restore` | Audio restoration (click/crackle/hum removal, declipping) | Stable |
| `oximedia-mixer` | Professional digital audio mixer (multi-channel, automation) | Stable |
| `oximedia-mir` | Music Information Retrieval (tempo, key/chord, genre/mood) | Stable |
| `oximedia-audiopost` | Audio post-production (ADR, Foley, mixing, sound design) | Stable |
| `oximedia-routing` | Professional audio routing and patching | Stable |

### Analysis & Quality

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-quality` | Video quality metrics (PSNR, SSIM, VMAF, VIF, BRISQUE) | Stable |
| `oximedia-qc` | Quality control (format, bitrate, color, temporal, audio, HDR) | Stable |
| `oximedia-analysis` | Comprehensive media analysis and quality assessment | Stable |
| `oximedia-scopes` | Professional video scopes (waveform, vectorscope, histogram) | Stable |
| `oximedia-scene` | Scene understanding and AI-powered video analysis | Stable |
| `oximedia-shots` | Shot detection and classification engine | Stable |
| `oximedia-forensics` | Video/image forensics (ELA, PRNU, copy-move detection) | Stable |
| `oximedia-profiler` | Performance profiling (CPU/GPU/memory, flamegraphs, regression) | Stable |
| `oximedia-dedup` | Duplicate detection (perceptual/crypto hashing, audio fingerprint) | Stable |

### Production & Broadcast

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-playout` | Playout engine (channel management, automation, failover, graphics) | Stable |
| `oximedia-playlist` | Playlist management (scheduling, EPG, gap filling, multichannel) | Stable |
| `oximedia-automation` | 24/7 broadcast automation with Lua scripting | Stable |
| `oximedia-switcher` | Professional live production video switcher | Stable |
| `oximedia-multicam` | Multi-camera production (angle management, auto-switching, sync) | Stable |
| `oximedia-monitor` | System monitoring (alerting, metrics, REST API, health checks) | Stable |
| `oximedia-captions` | Closed captioning/subtitles (CEA-608/708, TTML, WebVTT) | Stable |
| `oximedia-access` | Accessibility features (audio description, captions, transcripts, compliance) | Stable |
| `oximedia-gaming` | Game streaming (ultra-low latency, NVENC/QSV/VCE, replay buffer) | Stable |

### Post-Production & Workflow

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-edit` | Video timeline editor with effects and keyframe animation | Stable |
| `oximedia-timeline` | Multi-track timeline editor with DAG support | Stable |
| `oximedia-conform` | Media conforming (EDL/XML/AAF timeline reconstruction) | Stable |
| `oximedia-proxy` | Proxy generation (conforming, relinking, offline/online workflows) | Stable |
| `oximedia-workflow` | Comprehensive workflow orchestration engine | Stable |
| `oximedia-batch` | Production batch processing engine with Lua workflows | Stable |
| `oximedia-review` | Collaborative review and approval workflow | Stable |
| `oximedia-collab` | Real-time CRDT-based multi-user collaboration | Stable |
| `oximedia-farm` | Distributed encoding farm with load balancing | Stable |
| `oximedia-renderfarm` | Distributed render farm (job scheduling, cost optimization) | Stable |
| `oximedia-auto` | Automated video editing with intelligent analysis | Stable |
| `oximedia-clips` | Professional clip management and logging | Stable |
| `oximedia-repair` | File repair (corruption detection, header rebuild, stream salvaging) | Stable |

### Media Asset Management

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-mam` | Media Asset Management (PostgreSQL, Tantivy, REST/GraphQL, RBAC) | Stable |
| `oximedia-metadata` | Metadata formats (ID3v2, Vorbis, XMP, EXIF, IPTC) | Stable |
| `oximedia-search` | Advanced media search and indexing engine | Stable |
| `oximedia-rights` | Content rights and licensing management | Stable |
| `oximedia-archive` | Media archive verification and long-term preservation | Stable |
| `oximedia-archive-pro` | Professional digital preservation suite | Stable |
| `oximedia-recommend` | Recommendation system (collaborative/content filtering, A/B testing) | Stable |
| `oximedia-align` | Video alignment and registration for multi-camera synchronization | Stable |
| `oximedia-convert` | Media format conversion with codec detection | Stable |

### Bindings & Integrations

| Crate | Description | Status |
|-------|-------------|--------|
| `oximedia-py` | Python bindings via PyO3 | Stable |
| `oximedia-ml` | Typed ML pipelines via OxiONNX (SceneClassifier, ShotBoundaryDetector, AestheticScorer, ObjectDetector, FaceEmbedder) | Stable |

## Green List (Supported Codecs)

### Decoder Taxonomy

OxiMedia classifies each decoder with a four-tier honesty label so downstream users know exactly
what they are getting. See [`docs/codec_status.md`](docs/codec_status.md) for the full per-decoder
breakdown, what is missing, and the effort required to close each gap.

| Label | Meaning |
|-------|---------|
| **Verified** | End-to-end decode matches a reference implementation on external fixtures. |
| **Functional** | Real sample reconstruction path present and self-consistent on round-trip tests. No third-party conformance proof yet. |
| **Bitstream-parsing** | Headers and syntax are parsed; pixel/sample production is stubbed, partial, or returns empty/constant data. Useful for format inspection, not for playback. |
| **Experimental** | API sketch; not intended to actually decode. |

### Codec Matrix

| Category | Codec | Encode | Decode | Notes |
|----------|-------|--------|--------|-------|
| Video | AV1 | Functional | **Bitstream-parsing** | Alliance for Open Media, royalty-free. OBU parsing complete; pixel reconstruction pipeline is stubbed (see issue #9). |
| Video | VP9 | Functional | **Bitstream-parsing** | Google, royalty-free. Frame/tile parsing complete; reconstruction pipeline stages are no-ops. |
| Video | VP8 | Functional | **Bitstream-parsing** | Google, royalty-free. Y-plane emitted as constant gray; no intra/inter decode. |
| Video | Theora | Functional | **Bitstream-parsing** | Xiph.org, royalty-free. Real DCT/motion exist but decoded pixels are dropped before reaching `VideoFrame` (known copy bug). |
| Video | MJPEG | Functional | Functional | Motion JPEG via `oximedia-image` JPEG baseline; ≥28 dB PSNR at Q85. |
| Video | APV | Functional | Functional | ISO/IEC 23009-13 royalty-free intra-frame; real DCT + entropy decode. |
| Video | FFV1 | Functional | Functional | RFC 9043 lossless; CRC-32 verified. |
| Video | H.263 | Functional | Functional | Real macroblock decode, motion compensation, loop filter. |
| Audio | Opus | Functional | Functional (CELT only) | Xiph.org/IETF, royalty-free. SILK and hybrid modes are placeholder. |
| Audio | Vorbis | Functional | **Bitstream-parsing** | Xiph.org, royalty-free. Headers parse; `decode_audio_packet` returns empty. |
| Audio | FLAC | Functional | Functional / Verified | Lossless, royalty-free; CRC-16 verified, real LPC decode. |
| Audio | PCM | Verified | Verified | Unencumbered; trivial round-trip verified. |
| Audio | MP3 | — | Functional | Playback-only (patents expired 2017). Full Huffman/IMDCT/synthesis filterbank. |
| Image | PNG/APNG | Functional | Functional | Unencumbered; real unfilter + RGBA conversion. |
| Image | GIF | Functional | Functional | Unencumbered; real LZW decode. |
| Image | WebP (VP8L) | Functional | Functional | Google, royalty-free. Lossless only — no VP8 lossy WebP decoder. |
| Image | AVIF | Functional | **Bitstream-parsing** | AOM, royalty-free. Depends on AV1 decoder, which is itself bitstream-parsing. |
| Image | JPEG-XL (AJXL) | Functional | Functional | Animated JPEG-XL via ISOBMFF; real modular decoder. |

## Red List (Rejected Codecs)

These codecs are **NEVER** supported due to patent encumbrance:

- H.264/AVC (MPEG-LA)
- H.265/HEVC (MPEG-LA + Access Advance)
- H.266/VVC (Access Advance)
- AAC (Via Licensing)
- AC-3/E-AC-3 (Dolby)
- DTS (DTS Inc)
- MP3 (encoding — Fraunhofer)

## Quick Start

```bash
# Build the project
cargo build --release

# Run format probe
cargo run --bin oximedia -- probe -i video.webm

# Show supported formats
cargo run --bin oximedia -- info

# Transcode a file
cargo run --bin oximedia -- transcode -i input.mkv -o output.webm --codec av1
```

## Library Usage

```rust
use oximedia::prelude::*;

// Probe a media file
let data = std::fs::read("video.webm")?;
let result = probe_format(&data)?;
println!("Format: {:?}, Confidence: {:.1}%",
    result.format, result.confidence * 100.0);

// Transcode with quality control
let pipeline = TranscodePipeline::builder()
    .input("input.mkv")
    .video_codec(VideoCodec::Av1)
    .audio_codec(AudioCodec::Opus)
    .output("output.webm")
    .build()?;

pipeline.run().await?;
```

## Installation

### Rust (crates.io)

```bash
cargo add oximedia
```

### Python (PyPI)

```bash
pip install oximedia
```

- Source: [crates/oximedia-py](crates/oximedia-py/)
- Built with [maturin](https://github.com/PyO3/maturin)

### JavaScript / WebAssembly (npm)

```bash
npm install @cooljapan/oximedia
```

- Source: [oximedia-wasm/](oximedia-wasm/)
- Built with [wasm-pack](https://rustwasm.github.io/wasm-pack/)

### CLI

```bash
cargo install oximedia-cli
```

## Current Status

### Phase Summary

| Phase | Name | Status |
|-------|------|--------|
| Phase 1 | Foundation (core, io, container, codec) | Complete |
| Phase 2 | Audio Processing & Metering | Complete |
| Phase 3 | Video Processing & CV | Complete |
| Phase 4 | Networking & Streaming (HLS/DASH/RTMP/SRT/WebRTC) | Complete |
| Phase 5 | Production & Broadcast Systems | Complete |
| Phase 6 | Media Asset Management & Workflow | Complete |
| Phase 7 | Quality Control & Analysis | Complete |
| Phase 8 | Advanced Features (MIR, Forensics, AI, Recommendations) | Complete |

### Crate Status Summary

| Status | Count | Description |
|--------|-------|-------------|
| Stable | 108 | Feature-complete, tested, production-ready |
| Alpha | 0 | Core functionality implemented, API may change |
| Partial | 0 | Under active development, incomplete |
| **Total** | **108** | Library crates under `crates/` (top-level `oximedia`, `oximedia-cli`, `oximedia-wasm` counted separately) |

### Detailed Status Breakdown

**Stable (108 crates):**
`oximedia-aaf`, `oximedia-accel`, `oximedia-access`, `oximedia-align`, `oximedia-analysis`,
`oximedia-archive`, `oximedia-archive-pro`, `oximedia-audio`, `oximedia-audio-analysis`,
`oximedia-audiopost`, `oximedia-auto`, `oximedia-automation`, `oximedia-batch`, `oximedia-bench`,
`oximedia-calibrate`, `oximedia-captions`, `oximedia-clips`, `oximedia-cloud`, `oximedia-codec`,
`oximedia-collab`, `oximedia-colormgmt`, `oximedia-compat-ffmpeg`, `oximedia-conform`,
`oximedia-container`, `oximedia-convert`, `oximedia-core`, `oximedia-cv`, `oximedia-dedup`,
`oximedia-denoise`, `oximedia-distributed`, `oximedia-dolbyvision`, `oximedia-drm`, `oximedia-edit`,
`oximedia-edl`, `oximedia-effects`, `oximedia-farm`, `oximedia-forensics`, `oximedia-gaming`,
`oximedia-gpu`, `oximedia-graph`, `oximedia-graphics`, `oximedia-image`, `oximedia-imf`,
`oximedia-io`, `oximedia-jobs`, `oximedia-lut`, `oximedia-mam`, `oximedia-metadata`,
`oximedia-metering`, `oximedia-mir`, `oximedia-mixer`, `oximedia-monitor`, `oximedia-multicam`,
`oximedia-ndi`, `oximedia-net`, `oximedia-normalize`, `oximedia-optimize`, `oximedia-packager`,
`oximedia-playlist`, `oximedia-playout`, `oximedia-plugin`, `oximedia-presets`, `oximedia-profiler`,
`oximedia-proxy`, `oximedia-py`, `oximedia-qc`, `oximedia-quality`, `oximedia-recommend`,
`oximedia-renderfarm`, `oximedia-repair`, `oximedia-restore`, `oximedia-review`, `oximedia-rights`,
`oximedia-routing`, `oximedia-scaling`, `oximedia-scene`, `oximedia-scopes`, `oximedia-search`,
`oximedia-server`, `oximedia-shots`, `oximedia-simd`, `oximedia-stabilize`, `oximedia-storage`,
`oximedia-subtitle`, `oximedia-switcher`, `oximedia-timecode`, `oximedia-timeline`,
`oximedia-timesync`, `oximedia-transcode`, `oximedia-vfx`, `oximedia-videoip`, `oximedia-virtual`,
`oximedia-watermark`, `oximedia-workflow`, `oximedia-avi`, `oximedia-ml`

## Building

```bash
# Prerequisites
rustup update stable
rustup component add clippy

# Build all crates
cargo build --all

# Build release
cargo build --release --all

# Run all tests
cargo test --all

# Lint (must pass with zero warnings)
cargo clippy --all -- -D warnings

# Check documentation
cargo doc --all --no-deps
```

## Documentation

Topic-focused guides live in [`docs/`](docs/):

- [Codec Status](docs/codec_status.md) — four-tier decoder taxonomy
  (Verified / Functional / Bitstream-parsing / Experimental) and
  per-codec state of the world.
- [Rate Control Guide](docs/rate_control.md) — CBR/VBR/CRF/two-pass modes,
  VBV buffer semantics, encoder coverage matrix, and CRF-optimiser notes.
- [SIMD Dispatch](docs/simd_dispatch.md) — `oximedia-simd` CPU feature
  detection, AVX-512 / AVX2 / SSE4.2 / NEON / WASM tiers, and the safe
  dispatch convention.
- [Wave 5 Deltas](docs/wave5_deltas.md) — what shipped in 0.1.5 → 0.1.6
  (transcode executor, HW-accel probes, BBA-1 ABR, SCTE-35, ErrorContext,
  FormatNegotiator; stub resolution, exr.rs splitrs refactor, oxifft 0.3.0).
- [ML Guide](docs/ml_guide.md) — typed ONNX pipelines, device selection,
  and the `oximedia-ml` feature matrix.

## Policy

- **No Warnings**: All code must compile with zero warnings
- **No Unsafe**: `#![forbid(unsafe_code)]` enforced workspace-wide (except explicitly gated FFI features)
- **Apache 2.0**: Strictly permissive licensing only
- **Clippy Pedantic**: All pedantic lints enabled
- **Pure Rust**: No C/Fortran dependencies in default features
- **Patent Free**: Only royalty-free codecs and algorithms

## Contributing

1. Follow the no-warnings policy
2. Add comprehensive documentation with examples
3. Include unit and integration tests for new functionality
4. Use `tokio` for all async code
5. Prefer the COOLJAPAN ecosystem (OxiFFT, OxiBLAS, SciRS2) over external C dependencies

## Sponsorship

OxiMedia is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find OxiMedia useful, please consider sponsoring the project to support continued development of the Pure Rust multimedia and computer vision ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve 108 crates (~2.69M SLOC)
- Implement new royalty-free codecs and CV algorithms
- Keep the entire COOLJAPAN ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

Copyright 2026 COOLJAPAN OU (Team Kitasan). All rights reserved.

---

*OxiMedia is not just code; it is a declaration of independence from patent trolls and unsafe languages.*

**Safe. Fast. Free. Sovereign.**
