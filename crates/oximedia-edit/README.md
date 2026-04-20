# oximedia-edit

![Status: Stable](https://img.shields.io/badge/status-stable-green)
![Version: 0.1.4](https://img.shields.io/badge/version-0.1.4-blue)

Video timeline editor for OxiMedia, providing a comprehensive multi-track editing system with effects, transitions, and rendering.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

Version: 0.1.4 — 2026-04-20 — 792 tests

## Features

- **Multi-track Timeline** — Video, audio, and subtitle tracks
- **Clip Operations** — Add, remove, move, trim, split clips
- **Advanced Editing** — Ripple, roll, slip, and slide edits
- **Clip Speed** — Speed changes, reverse playback, freeze frames
- **Effects System** — Apply effects with keyframe animation
- **FX Strip** — Per-track effect chains
- **Transitions** — Cross-fades, dissolves, wipes, and zooms with easing functions
- **Color Grading Integration** — Per-clip color grade editing
- **Rendering** — Real-time preview and high-quality export rendering
- **Background Rendering** — Non-blocking background export
- **Auto-edit** — Automated editing operations
- **Undo/Redo** — Full edit history with undo/redo support
- **Group Editing** — Group clips for synchronized operations
- **Nested Sequences** — Nest sequences inside other sequences
- **Track Locking** — Lock tracks to prevent accidental edits
- **Selection Management** — Multi-clip selection and operations
- **Trim Modes** — Ripple, roll, slip, and slide trim
- **Insert Mode** — Insert and overwrite edit modes
- **Markers** — Timeline marker management
- **Edit Presets** — Reusable edit configuration presets
- **Patent-free codecs** — AV1, VP9, VP8, Opus, Vorbis, FLAC only

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-edit = "0.1.4"
```

```rust
use oximedia_edit::{Timeline, TimelineEditor, Clip, ClipType};
use oximedia_core::Rational;

let mut timeline = Timeline::new(
    Rational::new(1, 1000),  // 1ms timebase
    Rational::new(30, 1),    // 30 fps
);

let video_track = timeline.add_track(oximedia_edit::TrackType::Video);
let clip = Clip::new(1, ClipType::Video, 0, 5000); // 5 seconds
timeline.add_clip(video_track, clip)?;

let mut editor = TimelineEditor::new();
timeline.set_playhead(2500);
editor.split_at_playhead(&mut timeline)?;
```

## API Overview

**Core types:**
- `Timeline` — Central timeline structure with multiple tracks
- `Track` — Individual track (video/audio/subtitle)
- `Clip` — Media segment with timing and source information
- `TimelineEditor` — Editing operations (cut, copy, paste, split, trim)
- `Transition` — Transitions between clips
- `EditError` — Error type

**Rendering:**
- `TimelineRenderer` — Render individual frames
- `PreviewRenderer` — Real-time playback preview
- `ExportRenderer` — High-quality final export
- `BackgroundRenderer` — Non-blocking background rendering

**Modules:**
- `clip`, `clip_speed` — Clip management and speed ramping
- `edit`, `edit_context` — Core edit operations and context
- `edit_preset` — Reusable edit presets
- `effect` — Effects with keyframe animation
- `fx_strip` — Per-track effect chain management
- `transition` — Transition effects
- `render` — Frame rendering pipeline
- `auto_edit` — Automated editing operations
- `blade_tool` — Razor/blade cut tool
- `color_grade_edit` — Color grading integration
- `ripple` — Ripple edit operations
- `slip_slide` — Slip and slide edit tools
- `trim_mode` — Trim mode management
- `insert_mode` — Insert/overwrite mode
- `group`, `group_edit` — Clip group management
- `nested_sequence` — Nested sequence support
- `multitrack` — Multi-track coordination
- `track_lock` — Track locking
- `selection` — Multi-clip selection
- `history` — Undo/redo history
- `marker`, `marker_edit` — Timeline marker management
- `timeline` — Timeline data structure
- `error` — Error types

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
