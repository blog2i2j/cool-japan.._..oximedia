// Clippy allows for common pedantic lints in rate control math
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::unused_self)]

//! Rate control module for video encoders.
//!
//! This module provides comprehensive rate control algorithms for video encoding:
//!
//! - **CQP** (Constant QP): Fixed quantization parameter per frame type
//! - **CBR** (Constant Bitrate): Maintains steady bitrate with buffer model
//! - **VBR** (Variable Bitrate): Variable bitrate with optional two-pass
//! - **CRF** (Constant Rate Factor): Quality-based rate control
//!
//! # Architecture
//!
//! The rate control system consists of several interconnected components:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Rate Controller                          │
//! │  ┌─────────────────────────────────────────────────────────────┐ │
//! │  │                      Lookahead                               │ │
//! │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │ │
//! │  │  │ Scene Change │  │ Complexity   │  │ Mini-GOP         │   │ │
//! │  │  │ Detection    │  │ Estimation   │  │ Structure        │   │ │
//! │  │  └──────────────┘  └──────────────┘  └──────────────────┘   │ │
//! │  └─────────────────────────────────────────────────────────────┘ │
//! │  ┌─────────────────────────────────────────────────────────────┐ │
//! │  │                    Rate Controllers                          │ │
//! │  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                         │ │
//! │  │  │ CQP │  │ CBR │  │ VBR │  │ CRF │                         │ │
//! │  │  └─────┘  └─────┘  └─────┘  └─────┘                         │ │
//! │  └─────────────────────────────────────────────────────────────┘ │
//! │  ┌─────────────────────────────────────────────────────────────┐ │
//! │  │                    Support Systems                           │ │
//! │  │  ┌──────────────┐  ┌──────────────┐                         │ │
//! │  │  │ Rate Buffer  │  │ Adaptive     │                         │ │
//! │  │  │ (HRD)        │  │ Quantization │                         │ │
//! │  │  └──────────────┘  └──────────────┘                         │ │
//! │  └─────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ## Simple CQP Encoding
//!
//! ```ignore
//! use oximedia_codec::rate_control::{CqpController, RcConfig};
//!
//! let config = RcConfig::cqp(28);
//! let mut controller = CqpController::new(&config);
//!
//! // Get QP for each frame
//! let output = controller.get_qp(FrameType::Key);
//! encoder.set_qp(output.qp);
//! ```
//!
//! ## CBR Streaming
//!
//! ```ignore
//! use oximedia_codec::rate_control::{CbrController, RcConfig};
//!
//! let config = RcConfig::cbr(5_000_000); // 5 Mbps
//! let mut controller = CbrController::new(&config);
//!
//! // For each frame
//! let output = controller.get_rc(frame_type);
//! if !output.drop_frame {
//!     encoder.set_qp(output.qp);
//!     // ... encode frame ...
//!     controller.update(&stats);
//! }
//! ```
//!
//! ## Quality-Based CRF
//!
//! ```ignore
//! use oximedia_codec::rate_control::{CrfController, RcConfig};
//!
//! let config = RcConfig::crf(23.0);
//! let mut controller = CrfController::new(&config);
//!
//! // For each frame
//! let output = controller.get_rc(frame_type, complexity);
//! encoder.set_qp(output.qp);
//! encoder.set_lambda(output.lambda);
//! ```
//!
//! ## Two-Pass VBR
//!
//! ```ignore
//! use oximedia_codec::rate_control::{VbrController, RcConfig};
//!
//! // First pass
//! let config = RcConfig::vbr(5_000_000, 10_000_000);
//! let mut pass1 = VbrController::new(&config);
//! pass1.set_pass(1);
//!
//! // Analyze all frames...
//! let first_pass_data = pass1.finalize_first_pass()?;
//!
//! // Second pass
//! let mut pass2 = VbrController::new(&config);
//! pass2.set_pass(2);
//! pass2.set_first_pass_data(first_pass_data);
//!
//! // Encode with optimal bit allocation
//! ```

#![forbid(unsafe_code)]

pub mod allocation;
pub mod analysis;
pub mod aq;
pub mod buffer;
pub mod cbr;
pub mod complexity;
pub mod cqp;
pub mod crf;
pub mod lookahead;
pub mod quantizer;
pub mod types;
pub mod vbr;

// Re-export main types
pub use allocation::{AllocationResult, AllocationStrategy, BitrateAllocator, GopAllocationStatus};
pub use analysis::{
    AnalysisResult, ContentAnalyzer, ContentType, SceneChangeThreshold, TextureMetrics,
};
pub use aq::{AdaptiveQuantization, AqMode, AqResult, AqStrength};
pub use buffer::{BufferModel, RateBuffer, VbvParams};
pub use cbr::CbrController;
pub use complexity::{ComplexityEstimator, ComplexityResult, MotionAnalyzer, MotionResult};
pub use cqp::CqpController;
pub use crf::{CrfController, QualityPreset};
pub use lookahead::{Lookahead, LookaheadFrame, MiniGopInfo, SceneChangeDetector};
pub use quantizer::{BlockQpMap, QpResult, QpSelector, QpStrategy};
pub use types::{
    FrameStats, GopStats, RateControlMode, RcConfig, RcConfigError, RcOutput, RcState,
};
pub use vbr::{FirstPassData, VbrController};

/// Create a rate controller based on the configuration mode.
///
/// This factory function creates the appropriate controller type based on
/// the `RateControlMode` specified in the configuration.
///
/// # Example
///
/// ```ignore
/// use oximedia_codec::rate_control::{create_controller, RcConfig, RateControlMode};
///
/// let config = RcConfig::cbr(5_000_000);
/// let controller = create_controller(&config);
/// ```
#[must_use]
pub fn create_controller(config: &RcConfig) -> Box<dyn RateController> {
    match config.mode {
        RateControlMode::Cqp => Box::new(CqpController::new(config)),
        RateControlMode::Cbr => Box::new(CbrController::new(config)),
        RateControlMode::Vbr | RateControlMode::Abr => Box::new(VbrController::new(config)),
        RateControlMode::Crf => Box::new(CrfController::new(config)),
    }
}

/// Common trait for all rate controllers.
pub trait RateController: Send {
    /// Get rate control output for a frame.
    fn get_output(&mut self, frame_type: crate::frame::FrameType, complexity: f32) -> RcOutput;

    /// Update controller with frame encoding results.
    fn update_stats(&mut self, stats: &FrameStats);

    /// Reset the controller state.
    fn reset(&mut self);

    /// Get the current QP value.
    fn current_qp(&self) -> f32;

    /// Get total frames processed.
    fn frame_count(&self) -> u64;

    /// Get total bits produced.
    fn total_bits(&self) -> u64;
}

impl RateController for CqpController {
    fn get_output(&mut self, frame_type: crate::frame::FrameType, _complexity: f32) -> RcOutput {
        self.get_qp(frame_type)
    }

    fn update_stats(&mut self, stats: &FrameStats) {
        self.update(stats);
    }

    fn reset(&mut self) {
        CqpController::reset(self);
    }

    fn current_qp(&self) -> f32 {
        self.base_qp() as f32
    }

    fn frame_count(&self) -> u64 {
        CqpController::frame_count(self)
    }

    fn total_bits(&self) -> u64 {
        CqpController::total_bits(self)
    }
}

impl RateController for CbrController {
    fn get_output(&mut self, frame_type: crate::frame::FrameType, _complexity: f32) -> RcOutput {
        self.get_rc(frame_type)
    }

    fn update_stats(&mut self, stats: &FrameStats) {
        self.update(stats);
    }

    fn reset(&mut self) {
        CbrController::reset(self);
    }

    fn current_qp(&self) -> f32 {
        CbrController::current_qp(self)
    }

    fn frame_count(&self) -> u64 {
        CbrController::frame_count(self)
    }

    fn total_bits(&self) -> u64 {
        // CBR doesn't track total bits directly, compute from frame count
        0
    }
}

impl RateController for VbrController {
    fn get_output(&mut self, frame_type: crate::frame::FrameType, complexity: f32) -> RcOutput {
        self.get_rc(frame_type, complexity)
    }

    fn update_stats(&mut self, stats: &FrameStats) {
        self.update(stats);
    }

    fn reset(&mut self) {
        VbrController::reset(self);
    }

    fn current_qp(&self) -> f32 {
        VbrController::current_qp(self)
    }

    fn frame_count(&self) -> u64 {
        VbrController::frame_count(self)
    }

    fn total_bits(&self) -> u64 {
        0
    }
}

impl RateController for CrfController {
    fn get_output(&mut self, frame_type: crate::frame::FrameType, complexity: f32) -> RcOutput {
        self.get_rc(frame_type, complexity)
    }

    fn update_stats(&mut self, stats: &FrameStats) {
        self.update(stats);
    }

    fn reset(&mut self) {
        CrfController::reset(self);
    }

    fn current_qp(&self) -> f32 {
        CrfController::current_qp(self)
    }

    fn frame_count(&self) -> u64 {
        CrfController::frame_count(self)
    }

    fn total_bits(&self) -> u64 {
        CrfController::total_bits(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameType;

    #[test]
    fn test_create_controller_cqp() {
        let config = RcConfig::cqp(28);
        let mut controller = create_controller(&config);

        let output = controller.get_output(FrameType::Key, 1.0);
        assert_eq!(output.qp, 28);
    }

    #[test]
    fn test_create_controller_cbr() {
        let config = RcConfig::cbr(5_000_000);
        let mut controller = create_controller(&config);

        let output = controller.get_output(FrameType::Key, 1.0);
        assert!(!output.drop_frame);
        assert!(output.target_bits > 0);
    }

    #[test]
    fn test_create_controller_vbr() {
        let config = RcConfig::vbr(5_000_000, 10_000_000);
        let mut controller = create_controller(&config);

        let output = controller.get_output(FrameType::Key, 1.0);
        assert!(output.target_bits > 0);
    }

    #[test]
    fn test_create_controller_crf() {
        let config = RcConfig::crf(23.0);
        let mut controller = create_controller(&config);

        let output = controller.get_output(FrameType::Key, 1.0);
        assert!(output.qp > 0);
    }

    #[test]
    fn test_controller_trait_update() {
        let config = RcConfig::cqp(28);
        let mut controller = create_controller(&config);

        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = 100_000;
        stats.qp_f = 28.0;

        controller.update_stats(&stats);
        assert_eq!(controller.frame_count(), 1);
    }

    #[test]
    fn test_controller_trait_reset() {
        let config = RcConfig::cqp(28);
        let mut controller = create_controller(&config);

        let mut stats = FrameStats::new(0, FrameType::Key);
        stats.bits = 100_000;
        controller.update_stats(&stats);

        controller.reset();
        assert_eq!(controller.frame_count(), 0);
    }

    #[test]
    fn test_all_modes_covered() {
        for mode in [
            RateControlMode::Cqp,
            RateControlMode::Cbr,
            RateControlMode::Vbr,
            RateControlMode::Abr,
            RateControlMode::Crf,
        ] {
            let config = RcConfig {
                mode,
                target_bitrate: 5_000_000,
                ..Default::default()
            };

            let mut controller = create_controller(&config);
            let output = controller.get_output(FrameType::Inter, 1.0);

            // All controllers should produce valid output
            assert!(output.qp > 0 || output.drop_frame);
        }
    }
}
