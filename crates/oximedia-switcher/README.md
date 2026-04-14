# oximedia-switcher

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Professional live production video switcher for OxiMedia. Provides a comprehensive video switcher implementation with M/E rows, program/preview bus architecture, transitions, keying, multi-viewer, tally, and macro systems.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

Version: 0.1.3 — 2026-04-15

## Features

- **M/E Architecture** - Multiple Mix/Effect rows with program/preview buses
- **Multi-source Input** - SDI, NDI, file, and pattern generator inputs
- **Transitions** - Cut, mix, wipe, and DVE transitions with full transition engine
- **Keying** - Luma key, chroma key, linear key, and pattern key
- **Upstream/Downstream Keyers** - Independent keyer channels
- **Multi-viewer** - Multi-source monitoring layout
- **Tally System** - Red/green tally for program/preview indication with protocol support
- **Macro Recording** - Record and playback operation macros
- **Media Pool** - Still frame and clip storage
- **Still Store** - Still image storage for graphics
- **Super Source** - Super source compositing
- **Audio Follow Video (AFV)** - Automatic audio routing with video
- **Frame Synchronization** - Input frame sync and genlock support
- **AUX Buses** - Independent auxiliary output buses
- **Audio Mixing** - Integrated audio mixer
- **Clip Delay** - Configurable clip delay for playout
- **Output Routing** - Flexible output routing matrix
- **Pattern Generator** - Built-in test pattern generator
- **DVE** - Digital video effects with position, scale, and rotation
- **Switcher Presets** - Save and recall switcher configurations
- **Preview Bus** - Dedicated preview bus management

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-switcher = "0.1.3"
```

```rust
use oximedia_switcher::{Switcher, SwitcherConfig, TransitionConfig};

// Create a professional switcher with 2 M/E rows
let config = SwitcherConfig::professional(); // 2 M/E, 20 inputs
let mut switcher = Switcher::new(config)?;

// Set program and preview sources
switcher.set_program(0, 1)?;
switcher.set_preview(0, 2)?;

// Perform a cut (instant transition)
switcher.cut(0)?;

// Configure and trigger a mix transition
let transition_config = TransitionConfig::mix(30); // 30 frames
switcher.set_transition_config(0, transition_config)?;
switcher.auto_transition(0)?;
```

## API Overview

- `Switcher` — Main switcher engine with M/E rows, keyers, tally, macro playback
- `SwitcherConfig` — Configuration: basic() (1ME/8in), professional() (2ME/20in), broadcast() (4ME/40in)
- `SwitcherError` — Unified error type spanning all subsystems
- `TransitionType` / `WipePattern` / `TransitionConfig` — Transition control
- `KeyerType` / `ChromaKey` / `LumaKey` / `DveParams` — Keyer types and parameters
- `TallyManager` / `TallyState` — Tally light management
- `Macro` / `MacroEngine` / `MacroCommand` — Macro record/playback
- `Multiviewer` / `MultiviewerLayout` / `MultiviewerConfig` — Multi-source monitoring
- `FrameSynchronizer` / `FrameRate` / `GenlockSource` — Frame sync system
- `AudioFollowManager` / `AudioFollowMode` — AFV management
- `BusManager` / `BusType` — Bus management
- `InputRouter` / `InputConfig` / `InputType` — Input management
- `MediaPool` / `MediaSlot` — Media pool management
- Modules: `audio_follow`, `audio_follow_video`, `audio_mixer`, `aux_bus`, `bus`, `chroma`, `clip_delay`, `crosspoint`, `downstream_key`, `dve`, `ftb_control`, `input`, `input_bank`, `input_manager`, `keyer`, `luma`, `macro_engine`, `macro_exec`, `macro_system`, `me_bank`, `media_player`, `media_pool`, `multiviewer`, `output_routing`, `pattern_generator`, `preview_bus`, `still_store`, `super_source`, `switcher_preset`, `sync`, `tally`, `tally_protocol`, `tally_state`, `tally_system`, `transition`, `transition_engine`, `transition_lib`

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
