# oximedia-convert TODO

## Current Status
- 85+ source files; universal media format conversion with profiles, batch processing, and quality control
- Key features: format/codec detection, conversion profiles (Web, Archive, Broadcast, Device), batch conversion with progress tracking, quality comparison, metadata preservation, frame/thumbnail extraction, file splitting/concatenation, subtitle conversion, streaming packaging (HLS/DASH/ABR), template system
- Modules: detect/ (format, codec, properties), profile/ (system, custom, presets), pipeline/ (job, executor, options), batch/ (processor, queue, progress), filters/ (audio, video, chain), split/ (time, size, chapter), streaming/, smart/, presets/ (web, broadcast, archive, device), etc.
- Note: `perform_conversion` in lib.rs is currently a stub awaiting full demux/decode/encode/mux integration

## Enhancements
- [ ] Complete `perform_conversion()` in `lib.rs` by integrating oximedia-transcode demux/decode/encode/mux pipeline
- [ ] Extend `smart/mod.rs` SmartConverter with content-aware codec selection (animation -> VP8/GIF, live action -> AV1)
- [ ] Add two-pass encoding support in `pipeline/executor.rs` for better quality at target bitrate
- [ ] Improve `codec_mapper.rs` with automatic codec compatibility validation for target containers
- [ ] Extend `conv_validate.rs` with pre-conversion validation (sufficient disk space, compatible formats)
- [ ] Add conversion resumption in `pipeline/job.rs` for interrupted long-running conversions
- [x] Extend `progress.rs` with ETA calculation based on encoding speed and remaining frames
- [x] Improve `profile_match.rs` to auto-select the best profile based on input analysis
- [x] Improve `codec_mapper.rs` with automatic codec compatibility validation for target containers

## New Features
- [ ] Implement watch folder conversion mode (monitor directory, auto-convert new files)
- [ ] Add HDR-to-SDR tone mapping conversion in `color_convert.rs`
- [ ] Implement audio-only extraction with format conversion (e.g., video -> opus audio)
- [ ] Add image sequence to video conversion in `sequence/mod.rs` (PNG/JPEG frames -> WebM)
- [ ] Implement conversion presets for social media platforms (YouTube, Twitter, Instagram specs)
- [ ] Add video cropping and padding as conversion-time operations in `filters/video.rs`
- [ ] Implement multi-output conversion (one input -> multiple outputs with different profiles)
- [ ] Add `watermark_strip.rs` visible watermark overlay during conversion

## Performance
- [ ] Implement stream copy mode in `pipeline/executor.rs` (remux without re-encoding when codecs match)
- [ ] Add hardware-accelerated encoding detection and fallback in `pipeline/options.rs`
- [ ] Parallelize batch conversion in `batch/processor.rs` with configurable concurrency limit
- [ ] Implement segment-based parallel encoding for single files in `conversion_pipeline.rs`
- [ ] Optimize `format_detector.rs` with magic-byte-only detection (avoid full file scan)

## Testing
- [ ] Add end-to-end conversion test: input file -> convert -> verify output properties match profile
- [ ] Test all preset profiles (Web, Archive, Broadcast, Device) produce valid output
- [ ] Test `batch/processor.rs` with mixed input formats and concurrent conversion
- [ ] Add quality regression tests: verify PSNR/SSIM above threshold for each quality mode
- [ ] Test `split/` modules with edge cases (split at keyframe, split at exact byte boundary)
- [ ] Test `streaming/mod.rs` ABR ladder generation produces valid HLS/DASH manifests
- [ ] Test metadata preservation round-trip across format conversions

## Documentation
- [ ] Document conversion profile system with examples for custom profile creation
- [ ] Add quality mode comparison guide (Fast vs Balanced vs Best)
- [ ] Document streaming packaging workflow (input -> ABR ladder -> HLS/DASH output)
