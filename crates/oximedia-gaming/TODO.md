# oximedia-gaming TODO

## Current Status
- 80 source files across capture, encoding, audio, input, overlay, scene, replay, platform, metrics, and streaming domains
- GameStreamer with lifecycle management (start/stop/pause/resume)
- Capture: screen, window, region, cursor, game hooks
- Encoding: NVENC, QSV, VCE hardware presets, low-latency mode
- Audio: game capture, microphone, music mixing
- Input: keyboard, mouse, controller capture with overlay rendering
- Overlay: alerts, scoreboards, widgets, HUD, stream overlay
- Scene: manager, hotkeys, transitions
- Replay: buffer, highlight detection, save/export
- Platform: Twitch, YouTube, Facebook Gaming integration
- Additional: chat integration, monetization, tournament, spectator mode, VOD manager
- Dependencies: oximedia-core, oximedia-codec, oximedia-audio, oximedia-graph, tokio

## Enhancements
- [ ] Implement actual screen capture in `capture/screen.rs` (currently stub)
- [ ] Implement actual hardware encoder integration in `encode/nvenc.rs`, `encode/qsv.rs`, `encode/vce.rs`
- [ ] Make `GameStreamer::get_stats()` return real metrics instead of hardcoded values
- [ ] Implement actual replay buffer ring-buffer storage in `replay/buffer.rs`
- [ ] Add `save_replay()` actual file writing with encoding in `replay/save.rs`
- [ ] Enhance `highlight/detector.rs` with configurable detection thresholds per game genre
- [ ] Add `frame_pacing.rs` adaptive frame pacing that adjusts to encoder backpressure
- [ ] Implement `network_quality.rs` real-time bitrate adaptation based on network conditions
- [ ] Add `stream_analytics.rs` viewer count, chat activity, and engagement metrics
- [ ] Enhance `webcam/chroma.rs` with edge refinement for better green screen keying

## New Features
- [x] Implement RTMP/SRT/WHIP output protocol support for actual streaming
- [ ] Add multi-platform simultaneous streaming (Twitch + YouTube + Facebook at once)
- [ ] Implement custom stinger transition support in `scene/transition.rs`
- [x] Add stream deck / hotkey integration for scene switching and actions
- [ ] Implement AI-free game event detection using audio cues (kill sounds, announcements)
- [x] Add recording-only mode with higher quality settings than live streaming
- [ ] Implement `clip_manager.rs` automatic clip creation from highlight markers
- [ ] Add `spectator_mode.rs` multi-POV spectator stream with camera switching
- [ ] Implement chat bot integration in `chat_integration.rs` with command handling

## Performance
- [ ] Add GPU-based frame scaling and color conversion via oximedia-gpu
- [ ] Implement zero-copy frame pipeline from capture to encoder
- [ ] Add frame dropping strategy in `pacing/frame.rs` when encoder cannot keep up
- [ ] Optimize `overlay/system.rs` rendering with dirty-region compositing
- [ ] Implement async encoder output with double-buffered frame submission
- [ ] Add memory-mapped ring buffer for replay storage to avoid copying frames

## Testing
- [ ] Add integration test for full streaming pipeline: capture -> encode -> mux -> output
- [ ] Test `StreamConfigBuilder` validation with edge cases (1x1 resolution, 1 fps)
- [ ] Test scene switching with active transitions and verify no dropped frames
- [ ] Add `replay/buffer.rs` ring buffer overflow tests with various durations
- [ ] Test `platform/twitch.rs` metadata API integration with mock responses
- [ ] Test `audio/mix.rs` multi-source mixing with level normalization
- [ ] Add latency measurement tests verifying <100ms glass-to-glass target

## Documentation
- [ ] Document the streaming pipeline architecture from capture to output
- [ ] Add encoder preset comparison table (latency, quality, CPU/GPU usage)
- [ ] Document platform-specific configuration requirements (Twitch ingest, YouTube key, etc.)
