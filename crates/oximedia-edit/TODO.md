# oximedia-edit TODO

## Current Status
- 31 source files covering timeline, clips, tracks, effects, transitions, rendering, and editing operations
- Multi-track timeline with video, audio, and subtitle support
- Advanced editing: ripple, roll, slip, slide, blade tool, insert mode, trim mode
- Effects system with keyframe animation and interpolation
- Transitions: dissolve, wipe, zoom with easing functions
- Rendering: TimelineRenderer, PreviewRenderer, ExportRenderer, BackgroundRenderer
- Group/compound clips, markers, regions, nested sequences, undo history
- Dependencies: oximedia-core, oximedia-codec, oximedia-audio, oximedia-graph, tokio

## Enhancements
- [ ] Add multi-cam editing support with sync point alignment across tracks
- [ ] Implement proxy workflow in `render.rs` (low-res editing, full-res export)
- [ ] Enhance `history.rs` undo/redo with branching history tree (not just linear stack)
- [ ] Add magnetic timeline snapping in `timeline.rs` for clip alignment
- [ ] Improve `transition.rs` with GPU-accelerated transition rendering via oximedia-gpu
- [x] Add audio waveform generation in timeline for visual editing feedback
      — `WaveformGenerator` (mono/multi-channel, peak/RMS/min) and `WaveformData` in `waveform.rs`; 25 tests
- [ ] Enhance `nested_sequence.rs` with independent timeline resolution and frame rate
- [ ] Add clip linking between video and audio clips in `group.rs` for sync maintenance
- [ ] Implement `auto_edit.rs` with beat-detection based auto-cutting for music videos
- [x] Add clip color labels and metadata tags for organizational workflow
      — `ColorLabel`, `Tag`, `LabelManager`, `StandardLabels` in `color_label.rs`; 28 tests

## New Features
- [x] Implement freeze frame and speed ramp (variable speed) in `clip_speed.rs`
      — Added `SpeedEffect` enum (Normal/FreezeFrame/ConstantSpeed/VariableSpeed) and `ClipSpeedController`
- [ ] Add title/text overlay generation with basic font rendering
- [ ] Implement picture-in-picture layout mode with position/scale keyframes
- [x] Add EDL/XML export from timeline state for interchange with other NLEs
      — `TimelineExporter::to_edl` (CMX-3600) and `to_xml` (FCP XML skeleton) in `timeline_export.rs`
- [ ] Implement multi-format export (export same timeline to multiple resolutions/codecs)
- [ ] Add collaborative editing primitives (operational transform for concurrent edits)
- [ ] Implement smart trim with scene-change detection at trim points
- [x] Add render queue for batch export of multiple timelines
      — `RenderQueue`, `RenderJob`, `JobStatus`, `ExportConfig`, `TimelineSnapshot` in `render_queue.rs`; 35 tests

## Performance
- [x] Add frame cache in `render.rs` to avoid re-decoding unchanged frames
      — `RawFrameCache` (HashMap<u64,Vec<u8>>, max 32 frames, LRU eviction) added to `render.rs`
- [ ] Implement predictive pre-fetch for frames near playhead position
- [x] Optimize `clip.rs` clip lookup with interval tree for O(log n) time-position queries
      — `IntervalTree` with `query_point`/`query_range`/`nearest_edge` in `interval_tree.rs`
- [ ] Add parallel track rendering in `TimelineRenderer` using rayon
- [ ] Implement incremental render (only re-render changed regions of timeline)

## Testing
- [ ] Add round-trip test: create timeline -> export -> reimport -> verify identical structure
- [ ] Test `ripple.rs` and `slip_slide.rs` edit operations with overlapping clips
- [ ] Test `transition.rs` rendering output for all transition types
- [ ] Add stress test with 100+ tracks and 1000+ clips for timeline performance
- [ ] Test `blade_tool.rs` split at exact frame boundaries with various frame rates
- [ ] Test `selection.rs` multi-clip selection operations across tracks

## Documentation
- [ ] Document the editing operation semantics (ripple vs roll vs slip vs slide)
- [ ] Add timeline rendering pipeline architecture diagram
- [ ] Document the effect keyframe interpolation system in `effect.rs`
