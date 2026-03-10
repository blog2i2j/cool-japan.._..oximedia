# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/cool-japan/oximedia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cool-japan/oximedia/releases/tag/v0.1.0
