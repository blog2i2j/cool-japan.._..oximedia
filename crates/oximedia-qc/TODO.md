# oximedia-qc TODO

## Current Status
- 38 modules for comprehensive quality control and validation
- Key types: QualityControl, QcPreset, QcRule trait, CheckResult, QcReport
- Modules: audio, audio_qc, batch, bitrate_qc, black_silence, broadcast_safe, caption_qc_checker, closed_caption_qc, codec_validation, color_qc, compliance, compliance_report, container, database, detectors, dolby_vision_qc, examples, file_qc, format, format_qc, hdr_qc, profiles, qc_profile, qc_report, qc_scheduler, qc_template, report, rules, standards, sync_qc, temporal_qc, temporal, tests, utils, video, video_measure, video_quality_metrics
- Feature gates: json (serde), xml (quick-xml), database (rusqlite), pdf
- Dependencies: oximedia-core, oximedia-io, oximedia-container, oximedia-codec, oximedia-timecode, oximedia-audio-analysis, rayon, chrono, bitflags

## Enhancements
- [x] Add auto-fix capability for common QC failures (loudness normalization, bitrate adjustment)
- [ ] Implement severity levels in `rules::CheckResult` (error, warning, info) with configurable thresholds
- [x] Extend `broadcast_safe` with region-specific broadcast standards (NTSC/PAL/SECAM color space checks)
- [ ] Add `batch` module parallel processing with per-file progress reporting callbacks
- [ ] Implement `qc_template` inheritance — derive custom templates from built-in presets
- [ ] Extend `dolby_vision_qc` with RPU metadata validation (profile, level, compatibility)
- [ ] Add `bitrate_qc` VBR quality analysis — detect quality dips during high-motion scenes
- [ ] Implement `sync_qc` lip-sync offset detection with sub-frame accuracy

## New Features
- [ ] Add IMF (Interoperable Master Format) compliance checking
- [ ] Implement automated QC report delivery (email, webhook, Slack notification)
- [ ] Add SMPTE ST 2067 (IMF) and ST 2084 (PQ) compliance rules
- [ ] Implement QC watch folder — auto-validate files on arrival in monitored directories
- [ ] Add QC comparison mode — diff two files and highlight quality differences
- [x] Implement `caption_qc_checker` with timing gap/overlap detection and reading speed validation
- [ ] Add network stream QC — validate live RTMP/SRT/HLS streams in real-time
- [ ] Implement PDF report generation in `report` module (feature-gated)

## Performance
- [ ] Use SIMD for pixel-level analysis in `video_quality_metrics` (black frame, freeze frame detection)
- [ ] Implement early termination in `batch` — skip remaining checks after critical failure (configurable)
- [ ] Cache decoded frames across multiple video checks to avoid redundant decoding
- [ ] Parallelize independent QC rules using rayon task parallelism within single-file validation

## Testing
- [ ] Add QC validation tests with known-good and known-bad reference media files
- [ ] Test `compliance` module against all supported broadcast standards (ATSC, DVB, ISDB)
- [ ] Add round-trip tests for `qc_report` JSON/XML serialization
- [ ] Test `qc_scheduler` with concurrent QC jobs and verify no resource contention
- [ ] Verify `black_silence` detection thresholds match industry-standard definitions

## Documentation
- [ ] Add QC rule writing guide for custom rule implementation via `QcRule` trait
- [ ] Document built-in QC presets with their included rules and thresholds
- [ ] Add QC integration guide for CI/CD pipelines (automated media validation)
