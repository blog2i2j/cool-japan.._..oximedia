//! # `OxiMedia` Routing
//!
//! Professional audio routing and patching system for `OxiMedia`.
//!
//! ## Features
//!
//! - **Crosspoint Matrix**: Full any-to-any audio routing
//! - **Virtual Patch Bay**: Input/output management with flexible patching
//! - **Channel Mapping**: Complex channel mapping and remapping (e.g., 5.1 to stereo)
//! - **Signal Flow**: Signal flow graph visualization and validation
//! - **Audio Embedding**: Audio embedding/de-embedding for SDI
//! - **Format Conversion**: Sample rate, bit depth, and channel count conversion
//! - **Gain Staging**: Per-channel gain control with metering
//! - **Monitoring**: AFL/PFL/Solo monitoring systems
//! - **Preset Management**: Save/load routing configurations
//! - **MADI Support**: 64-channel MADI routing
//! - **Dante Integration**: Dante audio-over-IP metadata (respects Audinate IP)
//! - **Automation**: Time-based routing changes with timecode
//!
//! ## Usage
//!
//! ```
//! use oximedia_routing::matrix::CrosspointMatrix;
//!
//! // Create a 16x8 crosspoint matrix
//! let mut matrix = CrosspointMatrix::new(16, 8);
//!
//! // Connect input 0 to output 0 with -6 dB gain
//! matrix.connect(0, 0, Some(-6.0)).expect("should succeed in test");
//!
//! // Check if connected
//! assert!(matrix.is_connected(0, 0));
//! ```
//!
//! ## Routing Scenarios
//!
//! ### Live Production
//! Route microphones to mixer, mixer to multitrack recorder
//!
//! ### Post-Production
//! Route editor to monitors, create submixes for stems
//!
//! ### Broadcast
//! Route studios to transmission, implement backup routing
//!
//! ### Multitrack Recording
//! Route 64+ channels from stage to recording system

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::struct_excessive_bools,
    clippy::missing_errors_doc,
    clippy::type_complexity,
    clippy::match_like_matches_macro,
    clippy::match_same_arms,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    missing_docs
)]

// Matrix routing modules
pub mod matrix;

// Virtual patch bay
pub mod patch;

// Channel management
pub mod channel;

// Signal flow
pub mod flow;

// Audio embedding/de-embedding
pub mod embed;

// Format conversion
pub mod convert;

// Gain staging
pub mod gain;

// Monitoring
pub mod monitor;

// Preset management
pub mod preset;

// MADI support
pub mod madi;

// Dante support
pub mod dante;

// NMOS IS-04/IS-05
pub mod nmos;

#[cfg(feature = "nmos-http")]
pub use nmos::http::NmosHttpServer;

#[cfg(feature = "nmos-discovery")]
pub use nmos::{NmosDiscovery, NmosDiscoveryBuilder, NmosDiscoveryError, NmosRegistryInfo};

// Automation
pub mod automation;

// Video routing matrix
pub mod matrix_router;

// Signal path analysis
pub mod signal_path;

// IP media routing (ST 2110)
pub mod ip_router;

// Network path selection
pub mod path_selector;

// Crosspoint routing matrix
pub mod crosspoint_matrix;

// Automatic failover routing
pub mod failover_route;

// Route table with longest-prefix-match
pub mod route_table;

// Signal presence and health monitoring
pub mod signal_monitor;

// Policy-driven routing decisions
pub mod routing_policy;

// Bandwidth budgeting
pub mod bandwidth_budget;

// Route optimization
pub mod route_optimizer;

// Link aggregation
pub mod link_aggregation;

// Latency calculation and budgeting
pub mod latency_calc;

// Named routing presets for rapid recall
pub mod route_preset;

// Route audit trail
pub mod route_audit;

// Network topology mapping
pub mod topology_map;

// Redundancy group management
pub mod redundancy_group;

// Traffic shaping and QoS
pub mod traffic_shaper;

// AES67 audio-over-IP interoperability
pub mod aes67;

// Hardware GPI/O-triggered routing changes
pub mod gpio_trigger;

// Level meter insertion at arbitrary signal path points
pub mod metering_bridge;

// Save/restore complete routing state with atomic rollback
pub mod routing_snapshot;

// Test signal generator (sine, pink noise, sweep)
pub mod signal_generator;

// Mix-minus routing for broadcast IFB feeds
pub mod mix_minus;

// Sparse crosspoint matrix for large matrices (256×256+)
pub mod sparse_matrix;

// Tally system for signaling active routing paths to external tally controllers
pub mod tally_system;

// Intercom module for point-to-point and party-line communication routing
pub mod intercom;

// Declarative routing configuration DSL
pub mod routing_macro;

// Virtual soundcard for OS-level audio routing
pub mod virtual_soundcard;

// Bulk NMOS IS-05 connection activation
pub mod bulk_ops;

// NMOS IS-05 connection constraint validation
pub mod constraint;

// NMOS IS-04 node heartbeat tracking
pub mod heartbeat;

// NMOS IS-05 connection change log
pub mod connection_log;

// NMOS IS-04 flow registry
pub mod flow_registry;

// NMOS IS-05 activation scheduling and management
pub mod nmos_activation;

// Constraint sets for route validation
pub mod constraint_sets;

// Per-channel strip processing chain
pub mod channel_strip;

// Cached signal-flow-graph validation
pub mod validate_cache;

// Lock-free routing updates for glitch-free real-time changes
pub mod lock_free_router;

// Zero-latency path optimization for live monitoring chains
pub mod zero_latency;

/// Re-export commonly used types for convenience
pub mod prelude {
    pub use crate::automation::{AutomationTimeline, Timecode};
    pub use crate::channel::{ChannelLayout, ChannelRemapper};
    pub use crate::convert::{BitDepthConverter, ChannelCountConverter, SampleRateConverter};
    pub use crate::embed::{AudioDeembedder, AudioEmbedder};
    pub use crate::flow::{SignalFlowGraph, ValidationResult};
    pub use crate::gain::{GainStage, MultiChannelGainStage};
    pub use crate::madi::MadiInterface;
    pub use crate::matrix::{ConnectionManager, CrosspointMatrix, RoutingPathSolver};
    pub use crate::monitor::{AflMonitor, PflMonitor, SoloManager};
    pub use crate::patch::{PatchBay, PatchInput, PatchOutput};
    pub use crate::preset::{PresetManager, RoutingPreset};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_basic_routing() {
        let mut matrix = CrosspointMatrix::new(4, 4);
        matrix.connect(0, 0, None).expect("should succeed in test");
        assert!(matrix.is_connected(0, 0));
    }

    #[test]
    fn test_patch_bay() {
        use crate::patch::{DestinationType, SourceType};

        let mut bay = PatchBay::new();

        let input = bay
            .input_manager_mut()
            .add_input("Mic 1".to_string(), SourceType::Microphone);
        let output = bay
            .output_manager_mut()
            .add_output("Monitor".to_string(), DestinationType::Monitor);

        bay.patch(input, output, None)
            .expect("should succeed in test");
        assert!(bay.is_patched(input, output));
    }

    #[test]
    fn test_channel_mapping() {
        let remapper = ChannelRemapper::downmix_51_to_stereo();
        assert_eq!(remapper.input_layout, ChannelLayout::Surround51);
        assert_eq!(remapper.output_layout, ChannelLayout::Stereo);
    }

    #[test]
    fn test_signal_flow() {
        use crate::flow::FlowEdge;

        let mut graph = SignalFlowGraph::new();

        let input = graph.add_input("Source".to_string(), 2);
        let output = graph.add_output("Destination".to_string(), 2);

        graph
            .connect(input, output, FlowEdge::default())
            .expect("should succeed in test");

        let result = graph.validate();
        assert!(result.is_valid);
    }

    #[test]
    fn test_gain_staging() {
        let mut stage = GainStage::new();
        stage.set_gain(-6.0);
        assert!((stage.gain_db - (-6.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_monitoring() {
        let mut solo = SoloManager::new();
        solo.solo(0);
        assert!(solo.is_soloed(0));
        assert!(!solo.is_soloed(1));
    }

    #[test]
    fn test_preset_management() {
        let mut manager = PresetManager::new();
        let preset = RoutingPreset::new("Test".to_string(), "Test preset".to_string());

        let id = manager.add_preset(preset);
        assert!(manager.get_preset(id).is_some());
    }

    #[test]
    fn test_madi_interface() {
        use crate::madi::FrameMode;

        let mut interface = MadiInterface::new("MADI 1".to_string());
        assert_eq!(interface.max_channels(), 64);

        interface.set_frame_mode(FrameMode::Frame96k);
        assert_eq!(interface.max_channels(), 32);
    }

    #[test]
    fn test_automation() {
        use crate::automation::{AutomationAction, AutomationEvent, FrameRate};

        let mut timeline = AutomationTimeline::new("Show".to_string(), FrameRate::Fps25);

        let event = AutomationEvent {
            timecode: Timecode::new(0, 1, 0, 0, FrameRate::Fps25),
            action: AutomationAction::Mute { channel: 0 },
            description: "Mute channel 0".to_string(),
            enabled: true,
        };

        timeline.add_event(event);
        assert_eq!(timeline.event_count(), 1);
    }

    // -----------------------------------------------------------------------
    // ChannelRemapper layout conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_channel_remapper_mono_to_stereo() {
        let remapper = ChannelRemapper::upmix_mono_to_stereo();
        assert_eq!(remapper.input_layout, ChannelLayout::Mono);
        assert_eq!(remapper.output_layout, ChannelLayout::Stereo);
        // Mono has 1 channel; stereo has 2.
        assert_eq!(remapper.input_layout.channel_count(), 1);
        assert_eq!(remapper.output_layout.channel_count(), 2);
        // Validate: all output channels should be covered.
        assert!(
            remapper.validate().is_ok(),
            "mono-to-stereo remapper failed validation"
        );
    }

    #[test]
    fn test_channel_remapper_stereo_to_mono() {
        let remapper = ChannelRemapper::downmix_stereo_to_mono();
        assert_eq!(remapper.input_layout, ChannelLayout::Stereo);
        assert_eq!(remapper.output_layout, ChannelLayout::Mono);
        assert_eq!(remapper.input_layout.channel_count(), 2);
        assert_eq!(remapper.output_layout.channel_count(), 1);
        assert!(
            remapper.validate().is_ok(),
            "stereo-to-mono remapper failed validation"
        );
    }

    #[test]
    fn test_channel_remapper_51_to_stereo() {
        let remapper = ChannelRemapper::downmix_51_to_stereo();
        assert_eq!(remapper.input_layout, ChannelLayout::Surround51);
        assert_eq!(remapper.output_layout, ChannelLayout::Stereo);
        assert_eq!(remapper.input_layout.channel_count(), 6);
        assert_eq!(remapper.output_layout.channel_count(), 2);
        assert!(
            remapper.validate().is_ok(),
            "5.1-to-stereo remapper failed validation"
        );
    }

    #[test]
    fn test_channel_remapper_71_to_stereo_via_714() {
        // 7.1.4 → 7.1 → verify it exists and has correct layout counts.
        let remapper = ChannelRemapper::downmix_714_to_71();
        assert_eq!(remapper.input_layout, ChannelLayout::Atmos714);
        assert_eq!(remapper.output_layout, ChannelLayout::Surround71);
        assert_eq!(remapper.input_layout.channel_count(), 12);
        assert_eq!(remapper.output_layout.channel_count(), 8);
        assert!(
            remapper.validate().is_ok(),
            "7.1.4-to-7.1 remapper failed validation"
        );
    }

    // -----------------------------------------------------------------------
    // Failover route switchover timing test
    // -----------------------------------------------------------------------

    #[test]
    fn test_failover_route_switchover_timing() {
        use crate::failover_route::{FailoverConfig, FailoverManager};
        use std::time::{Duration, Instant};

        let config = FailoverConfig::default();
        let mut mgr = FailoverManager::new(config);
        mgr.add_route("Primary".to_string(), 0);
        mgr.add_route("Backup 1".to_string(), 1);
        mgr.add_route("Backup 2".to_string(), 2);

        // Time the failover operation — must complete well under 1 ms.
        let start = Instant::now();
        let result = mgr.failover();
        let elapsed = start.elapsed();

        assert!(
            result.is_some(),
            "failover() should return an active route ID"
        );
        assert!(
            elapsed < Duration::from_millis(1),
            "failover() took {:?} (expected < 1 ms)",
            elapsed
        );

        // Verify the active path is now set.
        assert!(
            mgr.active_path().is_some(),
            "active_path should be set after failover"
        );
    }

    // -----------------------------------------------------------------------
    // Stress test: 1000 rapid connect/disconnect cycles on CrosspointMatrix
    // -----------------------------------------------------------------------

    #[test]
    fn test_crosspoint_matrix_stress_1000_cycles() {
        let mut matrix = CrosspointMatrix::new(16, 16);

        for cycle in 0..1000 {
            let input = cycle % 16;
            let output = (cycle * 3 + 7) % 16;

            // Connect.
            matrix
                .connect(input, output, None)
                .expect("connect failed in stress test");
            assert!(
                matrix.is_connected(input, output),
                "cycle {cycle}: not connected after connect"
            );

            // Disconnect.
            matrix
                .disconnect(input, output)
                .expect("disconnect failed in stress test");
            assert!(
                !matrix.is_connected(input, output),
                "cycle {cycle}: still connected after disconnect"
            );
        }

        // Matrix must be clean at the end.
        assert_eq!(
            matrix.get_active_crosspoints().len(),
            0,
            "matrix not clean after stress test"
        );
    }
}
