# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-04-15

### Added
- `JobProgress` tracking in `oximedia-farm` job queue
- `bit_depth()` method on `SampleFormat` in `oximedia-core`
- `output_validator`, `worker_health`, `auto_scaler`, `cloud_storage` modules now public in `oximedia-farm`

### Fixed
- VU meter ballistics -Inf poisoning when processing zero-amplitude samples (`oximedia-audio`)
- Subtitle chain comma replacement corrupting subtitle text (`oximedia-convert`)
- ABR rate control overflow in lookahead multiplier calculation (`oximedia-codec`)
- Scene cut detection depth limit missing spikes beyond index 4 (`oximedia-codec`)
- EWA resampling weight table returning non-empty on zero source dimensions (`oximedia-scaling`)
- Audio codec validation rejecting patent-free codecs only (`oximedia-cli`)
- Module conflict between `processor.rs` and `processor/mod.rs` (`oximedia-image-transform`)
- Broken intra-doc links in `oximedia-routing`, `oximedia-server`, `oximedia-container`, `oximedia-neural`, `oximedia-review`, `oximedia-effects`
- Multiple clippy warnings across workspace

### Changed
- Replaced banned `lz4` dependency with `lz4_flex` in `oximedia-collab` and `oximedia-renderfarm`
- Replaced `zstd` with `lz4_flex` compression in `oximedia-renderfarm` storage
- Updated workspace metadata: authors, homepage fields standardized across all crates

### Improved
- 80,393 tests passing (up from 70,800+ in v0.1.2)
- Zero clippy warnings with `-D warnings`
- Clean rustdoc build with strict flags
- 2.65M SLOC across 106 crates

## [0.1.2] - 2026-03-16

### Added

#### New Crates (11)
- **oximedia-hdr** — HDR processing with PQ/HLG transfer functions, tone mapping, gamut mapping, HDR10+ SEI metadata, HLG advanced modes, color volume analysis, and Dolby Vision profile support.
- **oximedia-spatial** — Spatial audio engine with Higher-Order Ambisonics (HOA), HRTF binaural rendering, room simulation, VBAP panning, head tracking, Wave Field Synthesis, and object-based audio.
- **oximedia-cache** — Intelligent media caching with LRU eviction, tiered storage, predictive warming, Bloom filter membership, consistent hashing, ARC adaptive replacement, and content-aware policies.
- **oximedia-stream** — Adaptive streaming with BOLA ABR algorithm, segment lifecycle management, SCTE-35 ad signaling, multi-CDN failover, manifest builder, and stream packager.
- **oximedia-video** — Video processing toolkit with motion estimation, deinterlacing, frame interpolation, scene detection, pulldown removal, video fingerprinting, and temporal denoising.
- **oximedia-cdn** — Content delivery network management with edge node orchestration, cache invalidation, origin failover, geographic routing, and CDN performance metrics.
- **oximedia-neural** — Neural network inference for media with tensor operations, Conv2D layers, batch normalization, activation functions, and media-specific models (scene classifier).
- **oximedia-360** — 360-degree video processing with equirectangular-to-cubemap projection, fisheye correction, stereo 3D layout, and Google Spatial Media XMP metadata.
- **oximedia-analytics** — Media analytics with session tracking, retention curve analysis, A/B testing framework, and engagement scoring models.
- **oximedia-caption-gen** — Automatic caption generation with speech-to-text alignment, Knuth-Plass line breaking, WCAG 2.1 accessibility compliance, and speaker diarization.
- **oximedia-pipeline** — Declarative media processing DSL with typed filter graph construction, execution planning, and optimization passes.

#### Plugin System
- **oximedia-plugin** — SemVer dependency resolver, u32 bitmask capability sandbox, FNV-1a hash-based hot-reload for dynamic codec plugins at runtime.

#### Broadcast and Routing
- **NMOS IS-04/05/07/08/09/11 REST APIs** in `oximedia-routing` with full device discovery, connection management, event and tally, audio channel mapping, stream compatibility, and system API support (656 tests).
- **NMOS mDNS/DNS-SD discovery** for automatic service registration and browsing (605 tests).

#### CLI Extensions
- Loudness analysis and normalization commands.
- Quality assessment (VMAF/SSIM/PSNR) commands.
- Deduplication detection commands.
- Timecode conversion and arithmetic commands.
- Batch engine commands for job scheduling.
- Scopes rendering (waveform/vectorscope/histogram) commands.
- Workflow template execution commands.
- Version info command (333 tests across CLI).

#### Benchmarks and Testing
- 4 criterion benchmark suites in `benches/` crate for codec, filter, I/O, and pipeline performance regression testing.
- 9 new examples demonstrating common workflows.
- 51 integration tests in `oximedia/tests/integration.rs`.
- 70,800+ tests passing across the entire workspace.

#### WASM and Python
- WASM target `wasm32-unknown-unknown` now builds cleanly with all feature gates (505 tests pass).
- PyPI publish workflow fixed (maturin 1.8.4, corrected protoc URL, macOS Intel runner).

### Changed

#### Major Crate Enhancements (40+)

- **oximedia-normalize** — DisneyPlus, PrimeVideo, Apple Spatial Audio, and Dolby Atmos loudness standards; adaptive scene-based normalization; multiband IIR filtering.
- **oximedia-server** — Admin API endpoints, Prometheus `/metrics` endpoint with AtomicU64 counters, HMAC webhook signing, batch delete and batch transcode operations.
- **oximedia-playout** — Transitions (dissolve, wipe, dip-to-color), CEA-608/708 subtitle insertion into playout streams, pre-flight validation checks, MultiChannelScheduler for parallel channel playout.
- **oximedia-net** — Low-Latency HLS (RFC 8216bis) with partial segments and preload hints, XOR FEC (RFC 5109) for packet recovery, QUIC transport abstraction layer.
- **oximedia-mam** — Pub/sub EventBus for asset lifecycle events, rule-based AI auto-tagger, BM25+Jaccard smart search with relevance ranking.
- **oximedia-batch** — Priority-heap job queue, conditional DAG execution (OnSuccess/OnFailure/Threshold branches), timeout enforcer with graceful cancellation.
- **oximedia-graphics** — HDR compositor with 16 blend modes, 1D/3D LUT application with Adobe .cube parser, ASC CDL color grading with slope/offset/power/saturation.
- **oximedia-workflow** — 8 pipeline templates with DOT graph export, StepCondition evaluator for conditional branching, p95 latency metrics tracking.
- **oximedia-monitor** — Alerting rules engine (Threshold, RateOfChange, Absence detection), LTTB downsampling with EWMA time-series smoothing, health registry with dependency checks.
- **oximedia-archive** — LZ77+LZ4 streaming compressor, pure-Rust SHA-256 digest verification, split/reassemble OARC format for large media archives.
- **oximedia-farm** — 6 load-balancing strategies (round-robin, least-connections, weighted, random, hash, power-of-two), locality-aware job distribution, heartbeat-based worker pool management.
- **oximedia-scopes** — False color overlay (7 exposure zones), 3D RGB histogram visualization, 5-mode exposure metering (spot, center-weighted, matrix, highlight, shadow).
- **oximedia-subtitle** — SRT/VTT/ASS/TTML parsers and serializers, 8x12 bitmap burn-in renderer, timing adjuster with offset and stretch.
- **oximedia-effects** — Freeverb and convolution reverb, multi-voice chorus and flanger, 7 distortion algorithms (overdrive, fuzz, bitcrush, wavefold, tube, tape, digital clip).
- **oximedia-mixer** — Topology-sorted mixing bus graph, 8-band parametric EQ with biquad filters, DAW-style automation lanes with interpolation.
- **oximedia-drm** — AES-128/256 implementation from scratch (NIST FIPS 197 verified), content key lifecycle management, license server with region-based gating.
- **oximedia-gpu** — RGBA-to-YUV420 and YUV420-to-RGBA conversion kernels, Gaussian/Sobel/Otsu image processing, buffer pool allocator, pipeline stage chaining.
- **oximedia-rights** — Royalty calculation engine (6 revenue bases), clearance workflow with counter-offer/region/time constraints, ISRC/ISWC/ISAN identifier validation.
- **oximedia-virtual** — LED volume stage simulation with moire pattern checker, FreeD D1 camera tracking protocol, frustum culling with 6-plane extraction.
- **oximedia-io** — 42-variant magic-byte content detector, Boyer-Moore-Horspool optimized reader, MP4/FLAC/WAV/MKV probe implementations.
- **oximedia-mir** — Beat tracking with dynamic programming, mood detection on Russell circumplex model, Camelot harmonic mixing codes (607+ tests).
- **oximedia-colormgmt** — Rec.709/Rec.2020/DCI-P3 gamut mapping, Bradford chromatic adaptation, CIECAM02 full forward/inverse transform, CIEDE2000 with RT rotation term, median-cut/k-means/octree palette quantization.
- **oximedia-cv** — SORT multi-object tracker, pyramidal Lucas-Kanade optical flow (831+302 tests).
- **oximedia-shots** — Audio scene boundary detection via spectral flux analysis, flash detection and Harding PSE compliance checker.
- **oximedia-recommend** — ALS and SVD++ collaborative filtering for encoding parameter recommendation.
- **oximedia-quality** — Temporal quality analyzer for frame-over-frame drift, pipeline quality gate with broadcast/streaming/preview threshold presets.
- **oximedia-codec** — VBV-aware rate control, AV1 level constraint table, PacketReorderer for B-frame output ordering.
- **oximedia-audio** — YIN pitch detection (4 algorithm variants), Kaiser-windowed sinc resampler, EBU R128 K-weighted loudness gating.
- **oximedia-image** — 2D DFT with Butterworth frequency-domain filters, 7 morphological operations with union-find connected components, Non-Local Means denoising.
- **oximedia-simd** — AVX-512 SIMD kernels with runtime CPU feature detection (`CpuFeatures` dispatcher).
- **oximedia-transcode** — 9 platform presets (YouTube, Netflix, Twitch, Vimeo, Instagram, TikTok, Broadcast, Archive, Web), VP9 CRF encoding, FFV1 lossless archive mode, TranscodeEstimator for time/size prediction, per-scene CRF adaptation, 6-rung quality ladder, HW acceleration config, Prometheus metrics export.
- **oximedia-dedup** — Perceptual hash (pHash), SSIM structural similarity, histogram comparison, feature-based matching, audio fingerprint dedup, metadata-based dedup (404 tests).
- **oximedia-search** — Real facet aggregation across 7 dimensions (codec, resolution, duration, format, date, tags, status) with 444 tests.
- **oximedia-core** — RationalTime with GCD/LCM arithmetic, PtsMediaTime 128-bit rebase for sub-sample precision, RingBuffer and MediaFrameQueue lock-free structures.
- **oximedia-lut** — Hald CLUT (identity generation + trilinear interpolation), 12 photographic presets (portra, velvia, tri-x, etc.), LutChainOps bake-to-33-cubed optimization.
- **oximedia-compat-ffmpeg** — 19-node FilterGraph parser, 75 codec and 30 format mappings, FfmpegArgumentBuilder for programmatic CLI construction.
- **oximedia-scaling** — EWA Lanczos elliptical weighted average resampling, FidelityFX CAS sharpening, half-pixel correction for chroma, per-title encoding ladder generator.
- **oximedia-auto** — Narrative arc detection (3-Act, Hero's Journey, Kishotenketsu), beat-synced automatic cuts, saliency-based reframing for aspect ratio adaptation.
- **oximedia-dolbyvision** — IPT-PQ color space transforms, CM v4.0 trim metadata with sloped curves, quickselect-based shot statistics, Dolby Vision XML import/export.
- **oximedia-collab** — Three-way merge for concurrent edits, Operational Transform primitives, presence and cursor tracking, snapshot-based branching.
- **oximedia-plugin** — SemVer dependency resolver, u32 bitmask capability sandbox, FNV-1a hash-based hot-reload detection.

### Fixed

- Facade crate (`oximedia`) now correctly re-exports all 108 crates with proper feature gating.
- WASM build target resolves all feature-gate incompatibilities for browser environments.
- PyPI publish workflow corrected for maturin 1.8.4, protoc binary URL, and macOS Intel runner matrix.

## [0.1.1] - 2026-03-10

### Added

- **FFmpeg CLI compatibility layer** — `oximedia-compat-ffmpeg` crate and `oximedia-ff` binary providing drop-in argument compatibility with FFmpeg CLI for common transcoding, streaming, and filter workflows.
- **OpenCV Python API compatibility** — `oximedia.cv2` submodule in `oximedia-py` exposing 18 modules aligned to the OpenCV Python API surface (imread, imwrite, resize, cvtColor, VideoCapture, VideoWriter, etc.).
- **MP4 demuxer complete implementation** — `probe` and `read_packet` fully implemented in `oximedia-container`, enabling reliable MP4/MOV source reading in transcode pipelines.
- **Transcode pipeline implementation** — end-to-end demux→filter→encode→mux pipeline in `oximedia-transcode`, connecting all processing stages with backpressure and async task scheduling.
- **Archive checksum real hash verification** — `oximedia-archive` now performs actual MD5, SHA-1, SHA-256, and xxHash digest verification (replacing placeholder stubs).
- **QR code watermarking** — ISO 18004 compliant QR code generation and embedding in `oximedia-watermark`, supporting data capacity modes 1–4 with Reed-Solomon error correction.
- **DCT-domain forensic watermarking** — Quantization Index Modulation (QIM) embedding and blind detection in `oximedia-watermark`, providing robust invisible watermarks surviving re-encoding.
- **Video deinterlacing** — Edge-Directed Interpolation (EDI) deinterlacer added to `oximedia-cv`, including bob, weave, and blend fallback modes.
- **Smart crop** — content-aware crop detection using saliency maps and face-priority weighting in `oximedia-cv`.
- **Super-resolution (EDI)** — single-frame and multi-frame SR upscaling in `oximedia-cv` via learned edge-directed interpolation.
- **GCS storage enhancements** — ACL management, signed URL generation (V4), CMEK encryption key association, and storage class transitions in `oximedia-cloud`.
- **NMF source separation** — Non-negative Matrix Factorisation based audio source separation in `oximedia-audio-analysis`.
- **CEA-608 subtitle parser** — Line 21 closed caption byte-pair decoding in `oximedia-subtitle`.
- **DVB subtitle parser** — ETSI EN 300 743 PES/segment parsing in `oximedia-subtitle`.
- **Plugin system** — `oximedia-plugin` crate providing `CodecPlugin` trait, `PluginRegistry`, `StaticPlugin` builder, `declare_plugin!` macro, JSON manifests, and `dynamic-loading` feature gate for shared library support.
- **FFV1 codec** — Lossless video codec (decoder + encoder) in `oximedia-codec` with range coder, Golomb-Rice coding, and multi-plane support.
- **Y4M container** — Raw YUV sequence format (demuxer + muxer) in `oximedia-container` for uncompressed video interchange.
- **JPEG-XL codec** — Next-generation image codec (decoder + encoder) in `oximedia-codec` with modular transform, entropy coding, and progressive decoding.
- **DNG image format** — Digital Negative RAW image support (reader + writer) in `oximedia-image` with TIFF/IFD parsing, CFA demosaicing, and color calibration.

### Changed

- Refactored 6 over-limit source files (super_resolution, denoise, grading, lut, delogo, ivtc) — each split below the 2000-line policy boundary using splitrs.
- Promoted 22 Alpha crates and 10 Partial crates to fuller implementation status.

## [0.1.0] - 2026-03-07

### Added

- Initial release of the oximedia workspace — a comprehensive professional media processing platform in pure Rust.

#### Core Infrastructure
- `oximedia-core` — foundational types, error handling, and shared abstractions for the entire workspace
- `oximedia-io` — unified I/O layer with async file and stream support
- `oximedia-codec` — audio/video codec abstractions and implementations
- `oximedia-container` — media container format support (MXF, MP4, MOV, MPEG-TS, MKV, etc.)
- `oximedia-simd` — SIMD-accelerated media processing primitives
- `oximedia-accel` — hardware acceleration abstractions (GPU, FPGA, DSP)
- `oximedia-gpu` — GPU compute pipelines for media processing

#### Audio Processing
- `oximedia-audio` — core audio processing primitives and pipelines
- `oximedia-audio-analysis` — audio analysis including rhythm, tempo, and spectral features
- `oximedia-audiopost` — post-production audio tools (mixing, mastering, restoration)
- `oximedia-effects` — audio effects processing (chorus, reverb, EQ, dynamics)
- `oximedia-metering` — broadcast-grade audio metering (LUFS, LRA, peak, PPM)
- `oximedia-mixer` — multi-channel audio mixing and routing
- `oximedia-normalize` — audio normalization to broadcast standards
- `oximedia-mir` — music information retrieval and audio fingerprinting (AcoustID)

#### Video Processing
- `oximedia-cv` — computer vision and image analysis with super-resolution support
- `oximedia-vfx` — visual effects compositing and processing
- `oximedia-image` — image processing and format conversion
- `oximedia-lut` — LUT (Look-Up Table) processing for color grading
- `oximedia-colormgmt` — ICC color management and color space conversion
- `oximedia-dolbyvision` — Dolby Vision HDR metadata processing
- `oximedia-scopes` — broadcast video scopes (waveform, vectorscope, histogram)
- `oximedia-denoise` — video and audio denoising algorithms
- `oximedia-stabilize` — video stabilization
- `oximedia-scaling` — high-quality video scaling and resizing
- `oximedia-watermark` — digital watermarking

#### Graph and Pipeline
- `oximedia-graph` — media processing graph/pipeline engine
- `oximedia-edit` — non-linear editing operations
- `oximedia-timeline` — timeline management and sequencing
- `oximedia-timecode` — SMPTE timecode parsing, generation, and arithmetic
- `oximedia-timesync` — clock synchronization and PTP/NTP support
- `oximedia-clips` — clip management and media bin
- `oximedia-shots` — shot detection and scene segmentation
- `oximedia-scene` — scene analysis and classification

#### Transcoding and Conversion
- `oximedia-transcode` — multi-format transcoding pipeline
- `oximedia-convert` — universal media format conversion
- `oximedia-packager` — DASH/HLS adaptive streaming packaging
- `oximedia-proxy` — proxy media generation and management
- `oximedia-optimize` — media optimization for delivery targets
- `oximedia-batch` — batch processing job management
- `oximedia-renderfarm` — distributed render farm coordination

#### Distributed and Cloud
- `oximedia-distributed` — distributed encoding coordinator with consensus, leader election, and work stealing
- `oximedia-farm` — production-grade encoding farm job management and worker coordination
- `oximedia-jobs` — job scheduling and queue management
- `oximedia-cloud` — cloud storage and processing integration
- `oximedia-storage` — cloud storage abstraction (S3, Azure Blob, Google Cloud Storage)
- `oximedia-workflow` — media workflow automation and orchestration
- `oximedia-automation` — event-driven automation and rules engine

#### Networking
- `oximedia-net` — network transport protocols for media (RTP, RTMP, SRT, RIST)
- `oximedia-ndi` — NDI (Network Device Interface) protocol support
- `oximedia-server` — media server with WebSocket and HTTP APIs
- `oximedia-videoip` — video-over-IP transport (ST 2110, ST 2022)
- `oximedia-routing` — software-defined media routing and signal routing
- `oximedia-switcher` — live production switcher functionality
- `oximedia-playout` — broadcast playout automation

#### Quality and Analysis
- `oximedia-qc` — automated quality control and validation
- `oximedia-quality` — perceptual quality metrics (VMAF, SSIM, PSNR)
- `oximedia-analysis` — comprehensive media analysis and reporting
- `oximedia-monitor` — real-time media monitoring and alerting
- `oximedia-forensics` — media forensics and chain-of-custody tools
- `oximedia-dedup` — media deduplication and similarity detection
- `oximedia-profiler` — GPU and CPU profiling for media workloads

#### Metadata and Rights
- `oximedia-metadata` — media metadata extraction, editing, and standards (XMP, ID3, etc.)
- `oximedia-rights` — digital rights management metadata
- `oximedia-drm` — DRM encryption and key management
- `oximedia-access` — accessibility features (audio description generation)
- `oximedia-captions` — caption and subtitle processing
- `oximedia-subtitle` — subtitle format parsing and conversion

#### Format-Specific
- `oximedia-aaf` — AAF (Advanced Authoring Format) support
- `oximedia-edl` — EDL (Edit Decision List) parsing and generation
- `oximedia-imf` — IMF (Interoperable Master Format) support
- `oximedia-lut` — LUT format support (cube, 3dl, etc.)

#### Advanced Features
- `oximedia-align` — audio/video alignment and synchronization
- `oximedia-calibrate` — camera and display calibration tools
- `oximedia-collab` — collaborative editing and review workflows
- `oximedia-conform` — media conform and EDL-to-media matching
- `oximedia-gaming` — game capture and streaming integration
- `oximedia-graphics` — graphics overlay and titling
- `oximedia-mam` — Media Asset Management integration
- `oximedia-multicam` — multi-camera editing and synchronization
- `oximedia-playlist` — playlist management and scheduling
- `oximedia-presets` — encoding and processing preset management
- `oximedia-recommend` — AI-powered encoding parameter recommendation
- `oximedia-repair` — media repair and error concealment
- `oximedia-restore` — media restoration and archival tools
- `oximedia-review` — collaborative review and approval workflows
- `oximedia-search` — full-text and semantic media search
- `oximedia-virtual` — virtual production tools
- `oximedia-archive` — media archiving and long-term preservation
- `oximedia-archive-pro` — advanced archival formats and migration

#### Tooling
- `oximedia-bench` — benchmarking harnesses for media processing
- `oximedia-py` — Python bindings via PyO3
- `oximedia-wasm` — WebAssembly bindings
- `oximedia-cli` — command-line interface

[Unreleased]: https://github.com/cool-japan/oximedia/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/cool-japan/oximedia/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oximedia/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oximedia/releases/tag/v0.1.0
