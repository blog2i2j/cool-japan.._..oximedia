# oximedia-routing

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Professional audio routing and patching system for OxiMedia. Provides full any-to-any audio routing via crosspoint matrices, virtual patch bays, complex channel mapping, SDI audio embedding, MADI support, and automation.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

Version: 0.1.3 — 2026-04-15

## Features

- **Crosspoint Matrix** - Full any-to-any audio routing matrix
- **Virtual Patch Bay** - Input/output management with flexible patching
- **Channel Mapping** - Complex channel remapping (e.g., 5.1 to stereo downmix)
- **Signal Flow Graph** - Signal flow visualization and validation
- **Audio Embedding** - Audio embedding/de-embedding for SDI
- **Format Conversion** - Sample rate, bit depth, and channel count conversion
- **Gain Staging** - Per-channel gain control with metering
- **Monitoring** - AFL/PFL/Solo monitoring systems
- **Preset Management** - Save/load routing configurations
- **MADI Support** - 64-channel MADI routing
- **Dante Integration** - Dante audio-over-IP metadata support
- **NMOS IS-04/IS-05** - Network media open specifications
- **Automation** - Time-based routing changes with timecode
- **IP Routing** - ST 2110 IP media routing
- **Failover** - Automatic failover routing
- **Route Optimization** - Policy-driven route selection and optimization
- **Bandwidth Budgeting** - Network bandwidth management for IP routes
- **Latency Calculation** - End-to-end latency budgeting
- **Topology Mapping** - Network topology visualization
- **Redundancy Groups** - Managed redundancy for critical routes
- **Traffic Shaping** - QoS and traffic shaping for media flows

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-routing = "0.1.3"
```

```rust
use oximedia_routing::prelude::*;

// Create a 16x8 crosspoint matrix
let mut matrix = CrosspointMatrix::new(16, 8);
matrix.connect(0, 0, Some(-6.0)).unwrap(); // Input 0 → Output 0 at -6 dB

// Create a patch bay
let mut bay = PatchBay::new();
let input = bay.input_manager_mut()
    .add_input("Mic 1".to_string(), SourceType::Microphone);

// 5.1 to stereo downmix
let remapper = ChannelRemapper::downmix_51_to_stereo();
```

## API Overview

- `CrosspointMatrix` — Any-to-any routing matrix with gain per crosspoint
- `PatchBay` — Virtual patch bay with input/output management
- `ChannelRemapper` / `ChannelLayout` — Channel mapping and downmix
- `SignalFlowGraph` — DAG-based signal flow with validation
- `GainStage` / `MultiChannelGainStage` — Per-channel gain control
- `SoloManager` / `AflMonitor` / `PflMonitor` — Monitoring systems
- `MadiInterface` — 64-channel MADI support
- `PresetManager` / `RoutingPreset` — Configuration presets
- `AutomationTimeline` — Timecode-based routing automation
- Modules: `matrix`, `patch`, `channel`, `flow`, `embed`, `convert`, `gain`, `monitor`, `preset`, `madi`, `dante`, `nmos`, `automation`, `matrix_router`, `signal_path`, `ip_router`, `path_selector`, `crosspoint_matrix`, `failover_route`, `route_table`, `signal_monitor`, `routing_policy`, `bandwidth_budget`, `route_optimizer`, `link_aggregation`, `latency_calc`, `route_preset`, `route_audit`, `topology_map`, `redundancy_group`, `traffic_shaper`

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
