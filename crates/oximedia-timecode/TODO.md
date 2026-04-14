# oximedia-timecode TODO

## Current Status
- 35 source files covering SMPTE 12M timecode reading and writing
- Core: Timecode struct with hours/minutes/seconds/frames, FrameRate enum (23.976 to 60fps), FrameRateInfo, drop frame support, user bits
- LTC: ltc module (decoder/encoder), ltc_encoder, ltc_parser for audio-based timecode
- VITC: vitc module (decoder/encoder) for video line-based timecode
- Utilities: tc_calculator, tc_compare, tc_convert, tc_drift, tc_interpolate, tc_math, tc_metadata, tc_range, tc_smpte_ranges, tc_validator, timecode_calculator, timecode_format, timecode_range
- Other: burn_in, continuity, drop_frame, duration, frame_offset, frame_rate, midi_timecode, reader, sync, sync_map
- Traits: TimecodeReader, TimecodeWriter
- Dependencies: oximedia-core, oximedia-audio, serde

## Enhancements
- [ ] Implement `std::ops::Add` and `std::ops::Sub` for `Timecode` to enable `tc1 + tc2` arithmetic directly
- [ ] Add `Timecode::from_string` parser that accepts "HH:MM:SS:FF" and "HH:MM:SS;FF" (drop frame semicolon)
- [ ] Extend `FrameRate` to support 47.952 fps (used in some cinema workflows) and 120 fps
- [ ] Add `Timecode::to_seconds_f64` convenience method for quick floating-point time conversion
- [ ] Implement `Ord` and `PartialOrd` for `Timecode` based on total frame count
- [ ] Extend `tc_validator` to detect and report non-monotonic timecode sequences in streams
- [ ] Add SMPTE 309M support in `vitc` encoder for HD VITC (ATC/LTC embedded in HD-SDI ancillary data)
- [ ] Improve `drop_frame` module with exact frame-accurate drop frame calculation (current from_frames uses approximation)

## New Features
- [ ] Implement `timecode_generator` module for free-running timecode generation with configurable start time and frame rate
- [ ] Add `timecode_overlay` module for rendering timecode as text overlay on video frames (integration with burn_in)
- [ ] Implement `jam_sync` module for syncing local timecode generator to external timecode reference with holdover
- [ ] Add `timecode_event` module for event-triggered timecode capture (mark in/out points, cue triggers)
- [ ] Implement `ndf_to_df` and `df_to_ndf` conversion utilities in `tc_convert` for workflow interop
- [x] Add `embedded_tc` module for reading/writing ATC (Ancillary Timecode) in SDI ancillary data packets
- [x] Implement `timecode_log` module for recording timecode-stamped production notes and metadata events
- [x] Add `timecode_display` module for formatting timecode in different regional conventions (SMPTE vs EBU)

## Performance
- [ ] Cache frame count in `Timecode` struct to avoid recomputing `to_frames()` on repeated access
- [ ] Implement batch LTC encoding in `ltc_encoder` that generates multiple frames of audio in a single call
- [ ] Use lookup table for drop frame minute boundaries in `from_frames` instead of division-based calculation
- [ ] Add SIMD-accelerated Manchester encoding/decoding for LTC bitstream processing
- [ ] Pre-compute VITC line insertion patterns in `vitc::encoder` for common frame rates

## Testing
- [ ] Add exhaustive drop frame validation test: iterate all valid timecodes in 24 hours at 29.97DF and verify frame count matches SMPTE specification
- [ ] Test `Timecode::increment`/`decrement` at all boundary conditions: midnight rollover, minute boundaries, drop frame skip points
- [ ] Add LTC encode-decode round-trip test with noisy audio signal (SNR sweep from 40dB to 6dB)
- [ ] Test `tc_drift` detection with synthetic timecode streams containing known drift rates
- [ ] Verify `tc_interpolate` accuracy for sub-frame interpolation between two known timecodes
- [ ] Add `midi_timecode` MTC quarter-frame encode/decode round-trip test for all frame rates

## Documentation
- [ ] Add drop frame timecode explanation with frame numbering diagram for 29.97DF
- [ ] Document LTC audio format specification (baud rate, modulation, sync word) in ltc module docs
- [ ] Add comparison table of LTC vs. VITC vs. MTC showing accuracy, latency, and use case recommendations
