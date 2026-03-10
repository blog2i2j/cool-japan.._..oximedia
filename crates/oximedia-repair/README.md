# oximedia-repair

**Status: [Alpha]** | Version: 0.1.1 | Updated: 2026-03-10

Media file repair and recovery tools for OxiMedia. Provides comprehensive tools for detecting and repairing corrupted media files, with support for multiple container formats and recovery modes.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Features

- **Corruption Detection** - Analyze and classify media file corruption
- **Header Repair** - Fix corrupted container headers
- **Index Rebuilding** - Reconstruct missing or damaged seek tables
- **Timestamp Correction** - Fix invalid or inconsistent timestamps
- **Packet Recovery** - Recover and interpolate corrupt packets
- **A/V Sync Repair** - Fix audio/video desynchronization
- **Truncation Recovery** - Recover truncated/incomplete files
- **Metadata Repair** - Reconstruct corrupt metadata
- **Partial Recovery** - Extract playable portions from heavily damaged files
- **Frame Reordering** - Fix invalid frame order
- **Backup Creation** - Automatic backup before repair operations
- **Batch Repair** - Process multiple files at once
- **Verification** - Post-repair integrity and playback verification
- **Audio Repair** - Audio-specific repair (level, clicks, dropouts)
- **Audio Restore** - Audio restoration integration
- **Bitstream Repair** - Compressed bitstream repair
- **Checksum Repair** - Checksum error correction
- **Color Repair** - Color data repair
- **Concealment** - Error concealment strategies
- **Container Repair** - Container-format-specific repair
- **Conversion** - Format conversion for unrecoverable containers
- **Corruption Map** - Detailed corruption mapping
- **Dropout Concealment** - Video dropout concealment
- **Error Correction** - Forward error correction
- **Frame Concealment** - Frame-level error concealment
- **Frame Repair** - Individual frame repair
- **Gap Fill** - Gap detection and filling
- **Integrity Checking** - File integrity verification
- **Level Repair** - Audio/video level correction
- **Packet Reordering** - Out-of-order packet correction
- **Repair Logging** - Detailed repair audit log
- **Scratch Detection** - Physical media scratch detection
- **Stream Recovery** - Stream-level recovery
- **Sync Repair** - A/V sync correction

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-repair = "0.1.1"
```

```rust
use oximedia_repair::{RepairEngine, RepairMode, RepairOptions};
use std::path::Path;

let engine = RepairEngine::new();
let options = RepairOptions {
    mode: RepairMode::Balanced,
    create_backup: true,
    verify_after_repair: true,
    ..Default::default()
};

// Analyze without repairing
let issues = engine.analyze(Path::new("corrupted.mp4"))?;

// Repair the file
let result = engine.repair_file(Path::new("corrupted.mp4"), &options)?;
println!("Issues fixed: {}/{}", result.issues_fixed, result.issues_detected);
```

## API Overview

**Core types:**
- `RepairEngine` — Main repair engine with analyze and repair methods
- `RepairMode` — Safe, Balanced, Aggressive, Extract modes
- `RepairOptions` — Repair configuration including backup and verification settings
- `RepairResult` — Detailed result with fixed/unfixed issue lists and report
- `Issue` / `IssueType` / `Severity` — Issue classification

**Modules:**
- `audio_repair`, `audio_restore` — Audio repair and restoration
- `bitstream_repair` — Compressed bitstream repair
- `checksum_repair` — Checksum error correction
- `color_repair` — Color data repair
- `conceal` — Error concealment
- `container_repair` — Container-format repair
- `conversion` — Format conversion
- `corruption_map` — Corruption mapping
- `detect` — Corruption detection
- `dropout_concealment` — Dropout concealment
- `error_correction` — Forward error correction
- `frame_concealment`, `frame_repair` — Frame-level repair
- `gap_fill` — Gap filling
- `header` — Header repair
- `index` — Index rebuilding
- `integrity` — Integrity verification
- `level_repair` — Level correction
- `metadata`, `metadata_repair` — Metadata repair
- `packet`, `packet_recovery`, `packet_repair` — Packet recovery
- `partial` — Partial recovery
- `reorder` — Packet reordering
- `repair_log` — Repair audit logging
- `report` — Report generation
- `scratch` — Scratch detection
- `stream_recovery` — Stream recovery
- `sync`, `sync_repair` — Sync repair
- `timestamp` — Timestamp correction
- `truncation` — Truncation recovery
- `verify` — Post-repair verification

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
