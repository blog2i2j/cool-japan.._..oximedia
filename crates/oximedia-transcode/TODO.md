# oximedia-transcode TODO

## Current Status
- 48 source files covering high-level transcoding pipeline
- Key features: Transcoder builder API, ABR ladder generation, multi-pass encoding, parallel encoding, audio normalization (EBU R128/ATSC A/85), HW acceleration detection, progress tracking, job queuing, presets (YouTube, Vimeo, streaming, broadcast, archive)
- Modules: codec_config, codec_mapping, crf_optimizer, segment_encoder, two_pass, bitrate_control, scene_cut, rate_distortion, stage_graph, watermark_overlay, crop_scale, burn_subs, concat_transcode, and more
- Dependencies: oximedia-core, oximedia-codec, oximedia-container, oximedia-io, oximedia-graph, oximedia-metering, oximedia-subtitle

## Enhancements
- [x] Add stream copy mode to `Transcoder` (passthrough without re-encoding when codecs match)
- [x] Implement actual frame-level pipeline execution in `TranscodePipeline::execute()` connecting decoder/encoder through filter graph
- [x] Extend `codec_config` to support JPEG-XL still image encoding parameters alongside video codecs
- [x] Add per-codec tune presets to `codec_profile` (e.g., film, animation, grain for AV1/VP9)
- [x] Make `PresetConfig` support audio channel layout (mono, stereo, 5.1, 7.1) not just bitrate
- [ ] Add HDR metadata passthrough/conversion (HDR10, HLG, Dolby Vision mapping) to the transcode pipeline
- [x] Implement actual content in `hw_accel::detect_available_hw_accel` for macOS VideoToolbox and Linux VAAPI
- [ ] Add Dolby Atmos / spatial audio passthrough support in `audio_transcode`
- [ ] Extend `concat_transcode` to handle mixed-resolution/mixed-codec input sources with automatic re-encoding

## New Features
- [ ] Add `TranscodeWatcher` for directory-based watch folder automation (detect new files, auto-transcode)
- [ ] Implement thumbnail strip / sprite sheet generation in `thumbnail` module (VTT-compatible for video players)
- [ ] Add CMAF (Common Media Application Format) output support alongside existing HLS/DASH in `abr_ladder`
- [ ] Implement `TranscodeProfile` import/export (JSON/YAML) for sharing encoding configurations
- [ ] Add frame-accurate trim/cut support in the pipeline (start/end timecode, without full re-encode using keyframe snapping)
- [ ] Implement audio-only transcode mode (podcast optimization with loudness normalization)
- [ ] Add `TranscodeBenchmark` utility to compare encoding speed/quality across different codec configurations

## Performance
- [ ] Implement tile-based parallel encoding in `parallel` for AV1 (split frame into tiles for multi-core)
- [ ] Add memory-mapped I/O option in pipeline for large file transcoding to reduce memory pressure
- [ ] Implement lookahead buffer in `crf_optimizer` for scene-adaptive CRF adjustment
- [ ] Add segment-level parallelism to `segment_encoder` using rayon for encoding independent GOPs concurrently
- [ ] Profile and optimize `bitrate_estimator` to use running statistics instead of full-pass analysis

## Testing
- [ ] Add integration tests for full transcode round-trip (encode then decode and verify frame checksums)
- [ ] Add tests for `abr_ladder` verifying correct resolution/bitrate rungs for HLS and DASH profiles
- [ ] Test `normalization` module with actual audio samples for EBU R128 compliance
- [ ] Add fuzz testing for `validation` module with malformed input paths and configurations
- [ ] Test `parallel` encoding produces bit-identical output to single-threaded encoding

## Documentation
- [ ] Add architecture diagram showing pipeline stage flow (demux -> decode -> filter -> encode -> mux)
- [ ] Document preset selection guide in `presets` module (which preset for which use case)
- [ ] Add inline examples for `TranscodeBuilder` showing common real-world workflows
