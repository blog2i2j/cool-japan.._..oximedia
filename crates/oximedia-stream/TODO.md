# oximedia-stream TODO

## Current Status
- 8 source files (lib.rs + 7 modules): adaptive_pipeline, segment_manager, stream_health, scte35, multi_cdn, manifest_builder, stream_packager
- Adaptive bitrate streaming with BOLA-inspired ABR switching and quality ladder management
- Segment state machine with prefetch/eviction, QoE scoring and issue detection
- SCTE-35 splice information encoding/parsing/scheduling for ad insertion
- Multi-CDN failover routing with EWMA latency tracking
- HLS master/media playlist and DASH MPD generation
- Media unit accumulation and segment packaging with file writer
- Dependencies: thiserror, tracing, uuid, serde, serde_json

## Enhancements
- [ ] Add buffer-based ABR algorithm variant in `adaptive_pipeline` (throughput-based in addition to BOLA)
- [ ] Implement segment prefetch depth configuration in `segment_manager` based on available bandwidth
- [ ] Extend `stream_health` QoE scoring to incorporate rebuffer ratio and startup delay metrics
- [ ] Add SCTE-35 splice_null and bandwidth_reservation command support in `scte35` module
- [ ] Implement weighted round-robin routing strategy in `multi_cdn` alongside latency-based selection
- [ ] Add EXT-X-DATERANGE and EXT-X-SKIP tags in `manifest_builder` for HLS low-latency
- [ ] Extend `stream_packager` to support fMP4 (fragmented MP4) segment output alongside TS segments
- [x] Add CDN health check with configurable probe interval and failure threshold in `multi_cdn`

## New Features
- [ ] Implement `ll_hls` module for Low-Latency HLS with partial segments and preload hints
- [ ] Add `ll_dash` module for Low-Latency DASH with chunked transfer encoding and CMAF
- [ ] Implement `drm_signaling` module for DRM system signaling in manifests (Widevine, FairPlay, PlayReady content protection)
- [ ] Add `thumbnail_track` module for generating I-frame only playlists and trick-play manifests
- [ ] Implement `multi_audio` module for managing multiple audio track variants (language, accessibility)
- [ ] Add `subtitle_track` module for WebVTT subtitle segment packaging and manifest integration
- [ ] Implement `stream_recorder` module for capturing live streams to VOD with DVR window management
- [ ] Add `stream_analytics` module for collecting viewer-side playback metrics (buffer health, quality switches, errors)
- [x] Implement `srt_ingest` module for SRT protocol ingest as input to the streaming pipeline

## Performance
- [ ] Use zero-copy segment writing in `stream_packager::FileSegmentWriter` with pre-allocated buffers
- [ ] Implement async manifest generation in `manifest_builder` to avoid blocking the segment pipeline
- [ ] Add segment caching in `segment_manager` to serve repeated requests without disk I/O
- [ ] Pool `MediaUnit` allocations in `stream_packager` to reduce allocation overhead during live streaming
- [x] Implement parallel multi-CDN upload for segment distribution to reduce end-to-end latency

## Testing
- [ ] Add SCTE-35 round-trip tests: encode splice_insert, parse back, verify all fields match
- [ ] Test `adaptive_pipeline` quality switching with simulated bandwidth fluctuation sequences
- [ ] Verify `manifest_builder` output against HLS and DASH specification validators
- [ ] Test `multi_cdn` failover by simulating provider failures and verifying automatic rerouting
- [ ] Add segment continuity tests ensuring monotonically increasing sequence numbers across playlist updates
- [ ] Test `stream_health` issue detection with simulated packet loss, jitter, and bitrate drops

## Documentation
- [ ] Add streaming pipeline architecture diagram showing data flow from ingest to CDN delivery
- [ ] Document ABR algorithm parameters and tuning guidelines for different network conditions
- [ ] Add example showing end-to-end live streaming setup: ingest -> package -> manifest -> CDN
