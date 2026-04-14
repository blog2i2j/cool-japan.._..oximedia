# oximedia-restore TODO

## Current Status
- 38 modules (12 subdirectory modules) covering both audio restoration (click/hum/noise/hiss/crackle/azimuth/wow/flutter/DC/phase) and video restoration (deband, deflicker, film grain, color restore, scan line, telecine detect, upscale)
- Core types: RestoreChain with RestorationStep enum, mono and stereo processing pipelines
- Presets for vinyl restoration, tape restoration, broadcast cleanup
- Dependencies: oximedia-core, oximedia-audio, oxifft, thiserror

## Enhancements
- [ ] Add per-step bypass toggle to `RestoreChain` so individual steps can be disabled without removing them
- [ ] Implement `RestoreChain::process_multichannel` for surround sound (5.1/7.1) with channel-aware processing
- [ ] Add step reordering validation in `RestoreChain` (e.g., DC removal should precede click detection)
- [ ] Extend `noise::SpectralSubtraction` with adaptive noise floor estimation from initial silence detection
- [ ] Add real-time preview mode to `RestoreChain` processing fixed-size blocks with overlap-add
- [ ] Improve `wow::WowFlutterCorrector` with pilot-tone detection for precise speed reference
- [ ] Add severity/confidence output to `click::ClickDetector` for each detected event
- [ ] Extend `hum::HumRemover` with automatic fundamental frequency detection (50Hz vs 60Hz)

## New Features
- [ ] Add a `breath_removal` module for podcast/voiceover restoration (detect and attenuate breaths)
- [ ] Implement `reverb_reduction` module using spectral dereverberation techniques
- [x] Add `dynamic_eq` module for frequency-dependent compression/expansion
- [ ] Implement `loudness_normalization` step (EBU R128 / ITU-R BS.1770) as a RestoreChain step
- [x] Add `vinyl_surface_noise` module with adaptive surface noise profiling distinct from click/crackle
- [x] Implement `tape_dropout_repair` module for detecting and interpolating tape dropouts
- [ ] Add `harmonic_reconstruct` module to rebuild missing harmonics in bandwidth-limited recordings
- [ ] Implement `stereo_width` restoration step for collapsed or narrow stereo fields
- [ ] Add `restore_undo` with per-step rollback capability using stored intermediate buffers

## Performance
- [ ] Add SIMD-optimized paths in `noise::WienerFilter` for batch FFT processing
- [ ] Implement block-based processing in `RestoreChain::process` to reduce peak memory for long files
- [ ] Use rayon parallel iterators in `batch` restoration of multiple files
- [ ] Cache FFT plans in `spectral_repair` across consecutive process calls with same block size
- [ ] Optimize `click::ClickRemover` interpolation to avoid full-buffer copies per click

## Testing
- [ ] Add tests for `RestoreChain` with all step types combined in a realistic vinyl restoration pipeline
- [ ] Test `process_stereo` with asymmetric corruption (clicks on left channel only)
- [ ] Add golden-file tests comparing restored output against known-good reference for each restoration type
- [ ] Test `AzimuthCorrection` and `PhaseCorrection` skip behavior in mono mode
- [ ] Add stress tests for `WienerFilter` with very short (<100 sample) and very long (>10M sample) inputs

## Documentation
- [ ] Document recommended step ordering for each preset type (vinyl, tape, broadcast)
- [ ] Add signal flow diagrams for the RestoreChain processing pipeline
- [ ] Document the difference between `noise` (broadband), `hiss` (high-frequency), and `crackle` (impulsive) removal
- [ ] Add parameter tuning guide for each restoration step with before/after spectrograms
