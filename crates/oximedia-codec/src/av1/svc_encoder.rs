//! AV1 Temporal Scalable Video Coding (SVC) encoder support.
//!
//! This module provides a stateful encoder wrapper that applies AV1's temporal
//! scalability (SVC) layer structure to a stream of frames.  It computes:
//!
//! - Which **temporal layer** each frame belongs to (dyadic structure).
//! - The **QP offset** to apply at each layer (base layer = lowest QP, highest
//!   enhancement layer = highest QP).
//! - Whether a frame is **droppable** (can be omitted by a client that only
//!   wants lower layers).
//! - Per-layer **bitrate allocation** and **framerate fraction**.
//!
//! # Relationship to `sequence::SvcConfig`
//!
//! [`SvcConfig`] (in `sequence.rs`) defines the static operating-point table
//! written into the AV1 sequence header.  This module adds a *stateful*
//! encoder layer on top that:
//!
//! 1. Assigns frames to layers as they arrive in presentation order.
//! 2. Tracks reference-frame availability per layer so the encoder can avoid
//!    referencing frames that a given operating point cannot decode.
//! 3. Emits [`SvcFrameDecision`] objects that tell the outer encoder what QP
//!    delta to use, which reference frames to point at, and whether to emit a
//!    keyframe.
//!
//! # Example
//!
//! ```
//! use oximedia_codec::av1::svc_encoder::{SvcTemporalEncoder, SvcEncoderConfig};
//!
//! let cfg = SvcEncoderConfig::new(3, 1); // 3 temporal layers, 1 spatial layer
//! let mut enc = SvcTemporalEncoder::new(cfg);
//!
//! for frame_index in 0u64..8 {
//!     let decision = enc.decide(frame_index);
//!     println!(
//!         "Frame {}: layer={}, qp_offset={}, droppable={}",
//!         frame_index,
//!         decision.temporal_layer,
//!         decision.qp_offset,
//!         decision.is_droppable,
//!     );
//! }
//! ```

#![forbid(unsafe_code)]

use crate::av1::sequence::{
    SvcConfig, SvcReferenceMode, TemporalLayerConfig, MAX_TEMPORAL_LAYERS,
};
use crate::error::{CodecError, CodecResult};

// ============================================================================
// SvcEncoderConfig
// ============================================================================

/// Configuration for the SVC temporal encoder.
#[derive(Clone, Debug)]
pub struct SvcEncoderConfig {
    /// Underlying SVC operating-point configuration.
    pub svc: SvcConfig,
    /// Base quantiser parameter (0–63).  Layer QP offsets are added on top.
    pub base_qp: u8,
    /// Force a keyframe on the base layer every N frames (0 = no periodic IDR).
    pub keyframe_interval: u64,
    /// Maximum QP value to clamp layer-adjusted QP to.
    pub max_qp: u8,
    /// Minimum QP value to clamp layer-adjusted QP to.
    pub min_qp: u8,
}

impl SvcEncoderConfig {
    /// Create a default SVC encoder configuration with `temporal_layers` layers
    /// and `spatial_layers` spatial layers.
    pub fn new(temporal_layers: u8, spatial_layers: u8) -> Self {
        Self {
            svc: SvcConfig::new(temporal_layers, spatial_layers),
            base_qp: 32,
            keyframe_interval: 0,
            max_qp: 63,
            min_qp: 0,
        }
    }

    /// Set the base QP.
    pub fn with_base_qp(mut self, qp: u8) -> Self {
        self.base_qp = qp.min(63);
        self
    }

    /// Set a periodic keyframe interval on the base layer.
    pub fn with_keyframe_interval(mut self, interval: u64) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Validate the configuration, returning an error on illegal values.
    pub fn validate(&self) -> CodecResult<()> {
        if self.min_qp > self.max_qp {
            return Err(CodecError::InvalidParameter(format!(
                "min_qp ({}) > max_qp ({})",
                self.min_qp, self.max_qp
            )));
        }
        if self.svc.num_temporal_layers == 0 || self.svc.num_temporal_layers > MAX_TEMPORAL_LAYERS as u8 {
            return Err(CodecError::InvalidParameter(format!(
                "num_temporal_layers must be in 1..={}, got {}",
                MAX_TEMPORAL_LAYERS,
                self.svc.num_temporal_layers,
            )));
        }
        Ok(())
    }
}

// ============================================================================
// SvcReferenceSlot — frame buffer management per layer
// ============================================================================

/// A single reference-frame slot in the SVC encoder state.
#[derive(Clone, Debug, Default)]
struct ReferenceSlot {
    /// Frame index of the stored frame (None if empty).
    frame_index: Option<u64>,
    /// Temporal layer of the stored frame.
    temporal_layer: u8,
    /// Whether this slot holds a keyframe.
    is_keyframe: bool,
}

// ============================================================================
// SvcFrameDecision — per-frame encoder directive
// ============================================================================

/// Per-frame encoding directive emitted by [`SvcTemporalEncoder`].
///
/// The outer AV1 encoder should use these fields to configure:
/// - `quantizer` → `base_qp + qp_offset`
/// - Reference frame selection → `reference_frame_indices`
/// - Frame type → `is_keyframe`
/// - OBU temporal layer ID → `temporal_layer`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvcFrameDecision {
    /// Frame index in presentation order (0-based).
    pub frame_index: u64,
    /// Assigned temporal layer (0 = base layer, N-1 = highest enhancement).
    pub temporal_layer: u8,
    /// QP delta to apply on top of `base_qp`.  Positive = coarser quantisation.
    pub qp_offset: i8,
    /// Clamped effective QP: `base_qp + qp_offset`, saturated to `[min_qp, max_qp]`.
    pub effective_qp: u8,
    /// Whether this frame should be encoded as an intra (keyframe / IDR).
    pub is_keyframe: bool,
    /// Whether this frame can be dropped by a client subscribing to only the
    /// base layer.
    pub is_droppable: bool,
    /// Indices of reference frame slots this frame should use (up to 3).
    /// An empty vec means use only the immediately preceding frame.
    pub reference_slots: Vec<usize>,
    /// Reference mode for this frame.
    pub reference_mode: SvcReferenceMode,
}

// ============================================================================
// LayerStats — cumulative per-layer statistics
// ============================================================================

/// Running statistics accumulated for one temporal layer.
#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    /// Total frames assigned to this layer.
    pub frame_count: u64,
    /// Total keyframes in this layer.
    pub keyframe_count: u64,
    /// Sum of QP offsets (useful for average-QP estimates).
    pub qp_offset_sum: i64,
}

impl LayerStats {
    /// Average QP offset over all frames in this layer.
    pub fn avg_qp_offset(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.qp_offset_sum as f64 / self.frame_count as f64
    }
}

// ============================================================================
// SvcTemporalEncoder
// ============================================================================

/// Stateful AV1 temporal-SVC encoder that assigns frames to layers and
/// emits encoding directives.
///
/// This encoder does **not** perform actual AV1 bitstream encoding; it is a
/// decision-making layer that sits above the raw AV1 encoder.
#[derive(Debug)]
pub struct SvcTemporalEncoder {
    /// Encoder configuration.
    config: SvcEncoderConfig,
    /// Number of temporal layers (cached from config for speed).
    num_layers: u8,
    /// Reference frame buffer slots (one per layer).
    ref_slots: Vec<ReferenceSlot>,
    /// Cumulative per-layer statistics.
    pub layer_stats: Vec<LayerStats>,
    /// Total frames processed.
    frames_processed: u64,
}

impl SvcTemporalEncoder {
    /// Create a new SVC temporal encoder.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if `config.validate()` fails.
    pub fn new(config: SvcEncoderConfig) -> Self {
        let num_layers = config.svc.num_temporal_layers as usize;
        let ref_slots = (0..num_layers)
            .map(|_| ReferenceSlot::default())
            .collect();
        let layer_stats = (0..num_layers).map(|_| LayerStats::default()).collect();
        let n = config.svc.num_temporal_layers;
        Self {
            config,
            num_layers: n,
            ref_slots,
            layer_stats,
            frames_processed: 0,
        }
    }

    /// Decide how to encode the frame at `frame_index`.
    ///
    /// Frames must be submitted in strict presentation order (0, 1, 2, …).
    pub fn decide(&mut self, frame_index: u64) -> SvcFrameDecision {
        // Determine temporal layer
        let temporal_layer = self.config.svc.frame_temporal_layer(frame_index);
        let qp_offset = self.config.svc.frame_qp_offset(frame_index);

        // Compute effective QP with saturation
        let raw_qp = self.config.base_qp as i32 + qp_offset as i32;
        let effective_qp = raw_qp
            .clamp(self.config.min_qp as i32, self.config.max_qp as i32) as u8;

        // Keyframe decision: first frame, or periodic IDR on base layer
        let is_keyframe = frame_index == 0
            || (temporal_layer == 0
                && self.config.keyframe_interval > 0
                && frame_index % self.config.keyframe_interval == 0);

        let is_droppable = self.config.svc.is_droppable(frame_index);

        // Reference slot selection: use the closest lower-layer reference
        let reference_slots = self.select_reference_slots(temporal_layer, is_keyframe);

        // Reference mode
        let reference_mode = self
            .config
            .svc
            .temporal_layers
            .get(temporal_layer as usize)
            .map_or(SvcReferenceMode::KeyOnly, |l| l.reference_mode.clone());

        // Update reference frame buffer
        self.update_ref_slot(frame_index, temporal_layer, is_keyframe);

        // Update statistics
        let stats = &mut self.layer_stats[temporal_layer as usize];
        stats.frame_count += 1;
        stats.qp_offset_sum += qp_offset as i64;
        if is_keyframe {
            stats.keyframe_count += 1;
        }
        self.frames_processed += 1;

        SvcFrameDecision {
            frame_index,
            temporal_layer,
            qp_offset,
            effective_qp,
            is_keyframe,
            is_droppable,
            reference_slots,
            reference_mode,
        }
    }

    /// Total frames processed so far.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Returns the number of temporal layers.
    pub fn num_temporal_layers(&self) -> u8 {
        self.num_layers as u8
    }

    /// Retrieve accumulated statistics for a given temporal layer.
    ///
    /// Returns `None` if `layer_id >= num_temporal_layers`.
    pub fn stats_for_layer(&self, layer_id: u8) -> Option<&LayerStats> {
        self.layer_stats.get(layer_id as usize)
    }

    /// Compute the nominal bitrate fraction allocated to a temporal layer
    /// (derived from `SvcConfig`).
    pub fn bitrate_fraction(&self, layer_id: u8) -> f32 {
        self.config
            .svc
            .temporal_layers
            .get(layer_id as usize)
            .map_or(0.0, |l| l.bitrate_fraction)
    }

    /// Returns the framerate fraction for a layer relative to the full rate.
    pub fn framerate_fraction(&self, layer_id: u8) -> f32 {
        self.config
            .svc
            .temporal_layers
            .get(layer_id as usize)
            .map_or(0.0, |l| l.framerate_fraction)
    }

    /// Update or override the temporal layer configuration for layer `layer_id`.
    ///
    /// This can be called at any time (e.g. to adapt QP offsets on-the-fly).
    pub fn set_temporal_layer_config(&mut self, layer_id: u8, cfg: TemporalLayerConfig) {
        self.config.svc.set_temporal_layer(layer_id, cfg);
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Select reference slots for the current frame.
    fn select_reference_slots(&self, temporal_layer: u8, is_keyframe: bool) -> Vec<usize> {
        if is_keyframe {
            return vec![];
        }

        let mut slots = Vec::with_capacity(2);

        // Find the slot for this layer or the nearest lower layer with a frame.
        for layer in (0..=temporal_layer).rev() {
            let slot = &self.ref_slots[layer as usize];
            if slot.frame_index.is_some() {
                slots.push(layer as usize);
                if slots.len() >= 2 {
                    break;
                }
            }
        }

        slots
    }

    /// Update the reference slot for `temporal_layer` with the current frame.
    fn update_ref_slot(&mut self, frame_index: u64, temporal_layer: u8, is_keyframe: bool) {
        let slot = &mut self.ref_slots[temporal_layer as usize];
        slot.frame_index = Some(frame_index);
        slot.temporal_layer = temporal_layer;
        slot.is_keyframe = is_keyframe;
    }
}

// ============================================================================
// Builder pattern
// ============================================================================

/// Builder for [`SvcTemporalEncoder`] with a fluent API.
///
/// # Example
///
/// ```
/// use oximedia_codec::av1::svc_encoder::SvcTemporalEncoderBuilder;
///
/// let enc = SvcTemporalEncoderBuilder::new(3, 1)
///     .base_qp(28)
///     .keyframe_interval(120)
///     .max_qp(55)
///     .build();
/// ```
#[derive(Debug)]
pub struct SvcTemporalEncoderBuilder {
    config: SvcEncoderConfig,
}

impl SvcTemporalEncoderBuilder {
    /// Create a new builder with the given layer counts.
    pub fn new(temporal_layers: u8, spatial_layers: u8) -> Self {
        Self {
            config: SvcEncoderConfig::new(temporal_layers, spatial_layers),
        }
    }

    /// Set base QP.
    pub fn base_qp(mut self, qp: u8) -> Self {
        self.config.base_qp = qp.min(63);
        self
    }

    /// Set periodic keyframe interval.
    pub fn keyframe_interval(mut self, interval: u64) -> Self {
        self.config.keyframe_interval = interval;
        self
    }

    /// Set maximum QP.
    pub fn max_qp(mut self, qp: u8) -> Self {
        self.config.max_qp = qp.min(63);
        self
    }

    /// Set minimum QP.
    pub fn min_qp(mut self, qp: u8) -> Self {
        self.config.min_qp = qp.min(63);
        self
    }

    /// Build the [`SvcTemporalEncoder`].
    pub fn build(self) -> SvcTemporalEncoder {
        SvcTemporalEncoder::new(self.config)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder(temporal_layers: u8) -> SvcTemporalEncoder {
        let cfg = SvcEncoderConfig::new(temporal_layers, 1).with_base_qp(32);
        SvcTemporalEncoder::new(cfg)
    }

    // ── Basic layer assignment ────────────────────────────────────────────────

    #[test]
    fn test_single_layer_all_frames_in_base() {
        let mut enc = make_encoder(1);
        for i in 0u64..8 {
            let d = enc.decide(i);
            assert_eq!(d.temporal_layer, 0, "frame {i} should be in base layer");
            assert!(!d.is_droppable, "base-layer frames are never droppable");
        }
    }

    #[test]
    fn test_three_layers_frame0_is_base() {
        let mut enc = make_encoder(3);
        let d = enc.decide(0);
        assert_eq!(d.temporal_layer, 0);
        assert!(d.is_keyframe, "first frame should be a keyframe");
    }

    #[test]
    fn test_three_layers_frame1_is_highest() {
        let mut enc = make_encoder(3);
        // Frame 0 → layer 0, then decide frame 1
        enc.decide(0);
        let d = enc.decide(1);
        // In dyadic 3-layer: period=4, frame 1 has no power-of-2 divisor ≥ 1 other than 1
        // layer 2 (highest) gets odd frames
        assert_eq!(d.temporal_layer, 2, "frame 1 should be in layer 2");
        assert!(d.is_droppable);
    }

    #[test]
    fn test_three_layers_frame2_is_mid() {
        let mut enc = make_encoder(3);
        enc.decide(0);
        enc.decide(1);
        let d = enc.decide(2);
        // frame 2 → step=2 → layer 1
        assert_eq!(d.temporal_layer, 1);
    }

    #[test]
    fn test_three_layers_frame4_is_base() {
        let mut enc = make_encoder(3);
        for i in 0u64..4 {
            enc.decide(i);
        }
        let d = enc.decide(4);
        assert_eq!(d.temporal_layer, 0, "frame 4 (period boundary) should be base layer");
    }

    // ── QP calculation ────────────────────────────────────────────────────────

    #[test]
    fn test_effective_qp_clamped() {
        let mut cfg = SvcEncoderConfig::new(3, 1);
        cfg.base_qp = 60;
        cfg.max_qp = 63;
        cfg.min_qp = 0;
        let mut enc = SvcTemporalEncoder::new(cfg);
        for i in 0u64..8 {
            let d = enc.decide(i);
            assert!(d.effective_qp <= 63, "effective_qp must be <= 63 at frame {i}");
        }
    }

    #[test]
    fn test_base_layer_has_smallest_qp() {
        let mut enc = make_encoder(3);
        let d_base = enc.decide(0); // layer 0
        let d_high = enc.decide(1); // layer 2

        assert!(
            d_base.effective_qp <= d_high.effective_qp,
            "base layer should have lower or equal effective QP (got {} vs {})",
            d_base.effective_qp,
            d_high.effective_qp
        );
    }

    // ── Keyframe logic ────────────────────────────────────────────────────────

    #[test]
    fn test_first_frame_is_keyframe() {
        let mut enc = make_encoder(2);
        let d = enc.decide(0);
        assert!(d.is_keyframe);
        assert!(d.reference_slots.is_empty(), "keyframes have no references");
    }

    #[test]
    fn test_periodic_keyframe() {
        let cfg = SvcEncoderConfig::new(2, 1)
            .with_base_qp(32)
            .with_keyframe_interval(4);
        let mut enc = SvcTemporalEncoder::new(cfg);
        // Frames 0 and 4 (base layer) should be keyframes
        for i in 0u64..8 {
            let d = enc.decide(i);
            if i == 0 || i == 4 {
                assert!(d.is_keyframe, "frame {i} should be keyframe");
            }
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_frame_count() {
        let mut enc = make_encoder(3);
        let n = 8u64;
        for i in 0..n {
            enc.decide(i);
        }
        assert_eq!(enc.frames_processed(), n);
        let total: u64 = (0..3u8).map(|l| enc.stats_for_layer(l).unwrap().frame_count).sum();
        assert_eq!(total, n, "all frames should be counted across layers");
    }

    #[test]
    fn test_stats_out_of_bounds_layer_returns_none() {
        let enc = make_encoder(2);
        assert!(enc.stats_for_layer(10).is_none());
    }

    // ── Builder ───────────────────────────────────────────────────────────────

    #[test]
    fn test_builder_creates_encoder() {
        let enc = SvcTemporalEncoderBuilder::new(3, 1)
            .base_qp(24)
            .keyframe_interval(60)
            .max_qp(55)
            .build();
        assert_eq!(enc.num_temporal_layers(), 3);
    }

    // ── Bitrate / framerate fractions ─────────────────────────────────────────

    #[test]
    fn test_bitrate_fractions_sum_to_one() {
        let enc = make_encoder(3);
        let total: f32 = (0..3u8).map(|l| enc.bitrate_fraction(l)).sum();
        // Allow small floating-point error
        assert!(
            (total - 1.0).abs() < 0.01,
            "bitrate fractions should sum to ~1.0, got {total}"
        );
    }

    #[test]
    fn test_framerate_fraction_base_layer_is_smallest() {
        let enc = make_encoder(3);
        let fr0 = enc.framerate_fraction(0);
        let fr2 = enc.framerate_fraction(2);
        assert!(fr0 <= fr2, "base layer framerate fraction should be ≤ highest layer");
    }

    // ── Config validation ─────────────────────────────────────────────────────

    #[test]
    fn test_config_validate_min_gt_max_qp() {
        let mut cfg = SvcEncoderConfig::new(2, 1);
        cfg.min_qp = 40;
        cfg.max_qp = 20;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = SvcEncoderConfig::new(3, 1)
            .with_base_qp(32)
            .with_keyframe_interval(120);
        assert!(cfg.validate().is_ok());
    }
}
