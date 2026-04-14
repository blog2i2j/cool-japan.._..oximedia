# OxiMedia WASM — Development Roadmap

**Version: 0.1.2**
**Status: 77 modules implemented**

## Implemented Modules

### Core decoders and muxers
- [x] `audio_decoder` — FLAC, Vorbis, Opus decoders
- [x] `av1_decoder` — AV1 decoder
- [x] `video_decoder` — VP8 decoder
- [x] `video_encoder` — VP8 encoder
- [x] `demuxer` — WebM/Matroska/Ogg/FLAC/WAV demuxer
- [x] `muxer` — WebM muxer
- [x] `streaming_demuxer` — streaming demux
- [x] `container` — container format helpers
- [x] `probe` — magic-byte format detection
- [x] `io` — I/O utilities

### Analysis and quality
- [x] `analysis` — loudness (EBU R128), beat detection, spectral features
- [x] `quality_wasm` — PSNR, SSIM, frame quality
- [x] `scopes_wasm` — waveform, vectorscope, false color

### Color management
- [x] `colormgmt_wasm` — color space conversion, tone mapping, delta-E
- [x] `hdr_wasm` — PQ/HLG transfer functions, HDR tone mapping  ← NEW (0.1.2)
- [x] `lut_wasm` — 3D LUT application, photographic presets, .cube parser  ← NEW (0.1.2)
- [x] `dolbyvision_wasm` — Dolby Vision metadata
- [x] `calibrate_wasm` — color calibration

### Audio
- [x] `convert` — sample format and sample rate conversion
- [x] `convert_wasm` — format/codec conversion helpers
- [x] `mixer_wasm` — audio mixing, gain, pan
- [x] `mir_wasm` — beat/tempo/chord/key detection
- [x] `normalize_wasm` (pending)
- [x] `restore_wasm` — audio restoration, de-clip
- [x] `spatial_wasm` — Ambisonics (HOA), VBAP panning  ← NEW (0.1.2)
- [x] `audiopost_wasm` — stems, mix, delivery spec
- [x] `denoise_wasm` — audio/video denoising

### Graphics and compositing
- [x] `graphics_wasm` — broadcast graphics, templates
- [x] `vfx_wasm` — effects, chroma key, transitions
- [x] `image_wasm` — image ops, DPX/EXR, histograms
- [x] `multicam_wasm` — multi-camera compositing
- [x] `scaling_wasm` — video/image scaling
- [x] `filter_graph` — filter graph (DAG)

### Metadata and subtitles
- [x] `metadata_wasm` — ID3v2, Vorbis comments, EXIF, iTunes, Matroska tags
- [x] `subtitle_wasm` — SRT/VTT/ASS parsing and conversion
- [x] `captions_wasm` — captions processing
- [x] `timecode_wasm` — SMPTE timecode operations

### Production and workflow
- [x] `transcode_wasm` — transcoding presets and job management
- [x] `batch_wasm` — batch processing
- [x] `workflow_wasm` — workflow orchestration
- [x] `playout_wasm` — broadcast playout schedule
- [x] `timeline_wasm` — timeline editing
- [x] `scene_wasm` — scene detection
- [x] `shots_wasm` (pending)

### Infrastructure
- [x] `worker_helpers` — transfer header, plane splitting, transferable frames
- [x] `webcodecs_bridge` — WebCodecs API bridge
- [x] `media_player` — media player
- [x] `types` — shared types (WasmPacket, WasmStreamInfo, etc.)
- [x] `utils` — error helpers
- [x] `plugin_wasm` — plugin system info

### Professional tools
- [x] `drm_wasm` — DRM encrypt/decrypt
- [x] `forensics_wasm` — image forensics (ELA, noise, compression)
- [x] `watermark_wasm` — audio/image watermarking
- [x] `dedup_wasm` — media deduplication
- [x] `rights_wasm` — digital rights checking
- [x] `qc_wasm` — quality control
- [x] `review_wasm` — review and approval workflows
- [x] `collab_wasm` — collaborative editing
- [x] `monitor_wasm` — system monitoring
- [x] `profiler_wasm` — performance profiling

### Other
- [x] `aaf_wasm` — AAF file support
- [x] `access_wasm` — access control
- [x] `align_wasm` — media alignment
- [x] `archivepro_wasm` — professional archiving
- [x] `auto_wasm` — automated editing
- [x] `clips_wasm` — clip management
- [x] `conform_wasm` — delivery conformance
- [x] `gaming_wasm` — game capture/streaming
- [x] `imf_wasm` — IMF package support
- [x] `presets_wasm` — encoding presets
- [x] `proxy_wasm` — proxy media
- [x] `recommend_wasm` — content recommendation
- [x] `renderfarm_wasm` — render farm
- [x] `routing_wasm` — audio/video routing
- [x] `stabilize_wasm` — video stabilization
- [x] `switcher_wasm` — live production switching
- [x] `timesync_wasm` — time synchronization
- [x] `virtual_wasm` — virtual production

## Pending Modules (future work)

- [ ] `hdr_wasm` extensions — HDR scene analysis, CUVA/VIVID metadata
- [ ] `lut_wasm` extensions — ACES pipeline, Hald CLUT round-trip
- [ ] `spatial_wasm` extensions — HRTF binaural rendering, room simulation
- [ ] `normalize_wasm` — loudness normalization with EBU R128 targets
- [ ] `shots_wasm` — shot cut/dissolve/fade detection
- [ ] `neural_wasm` — in-browser ML inference (pending WASM SIMD performance)
- [ ] `stream_wasm` — ABR streaming manifest builder
- [ ] `cache_wasm` — media cache management
- [ ] `analytics_wasm` — session tracking, A/B testing

## 0.1.2 Changes

| Item | Status |
|------|--------|
| `hdr_wasm`: PQ/HLG OETF/EOTF, batch frame conversion, tone mapping, `WasmHdrConverter` | ✅ Done |
| `lut_wasm`: photographic presets, identity LUT, `WasmLut3d`, `.cube` parser | ✅ Done |
| `spatial_wasm`: Ambisonics encode/decode (1st–5th order), VBAP panning, `WasmAmbisonicsEncoder` | ✅ Done |
| All three modules: 8+ tests each, 0 clippy warnings | ✅ Done |
