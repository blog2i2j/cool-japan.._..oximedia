//! Bitrate modelling and estimation for video encoding.
//!
//! Provides rate-control mode descriptors, target bitrate helpers, and a simple
//! estimator that predicts the required bitrate for a given resolution and
//! quality setting.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Rate-control mode for a video encoder.
///
/// Note: a `BitrateMode` type already exists in the `traits` module; this enum
/// is a standalone model-level counterpart with richer semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RcMode {
    /// Constant Bitrate — output bitrate is held at a fixed target.
    Cbr,
    /// Variable Bitrate — bitrate varies within min/max bounds.
    Vbr,
    /// Constant Rate Factor — quality-based encoding (e.g. x264 CRF).
    Crf,
    /// Constant Quantisation Parameter — fixed QP per frame.
    Qp,
}

impl RcMode {
    /// Returns `true` for quality-based modes (CRF, QP) where the encoder
    /// controls quality rather than a hard bitrate target.
    pub fn is_quality_based(self) -> bool {
        matches!(self, RcMode::Crf | RcMode::Qp)
    }

    /// Returns `true` for throughput-constrained modes (CBR, VBR).
    pub fn is_bitrate_constrained(self) -> bool {
        !self.is_quality_based()
    }

    /// A short string label for the mode.
    pub fn label(self) -> &'static str {
        match self {
            RcMode::Cbr => "CBR",
            RcMode::Vbr => "VBR",
            RcMode::Crf => "CRF",
            RcMode::Qp => "QP",
        }
    }
}

/// Describes the desired bitrate (or quality) target for an encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitrateTarget {
    /// Rate-control mode.
    pub mode: RcMode,
    /// Target bitrate in kbps (for CBR/VBR modes).
    pub target_kbps: u32,
    /// Minimum bitrate in kbps (VBR lower bound; 0 = no limit).
    pub min_kbps: u32,
    /// Maximum bitrate in kbps (VBR upper bound; 0 = no limit).
    pub max_kbps: u32,
    /// Quality parameter: CRF value or QP index (ignored for CBR/VBR).
    pub quality_param: f32,
}

impl BitrateTarget {
    /// Create a CBR target at `kbps` kilobits per second.
    pub fn cbr(kbps: u32) -> Self {
        Self {
            mode: RcMode::Cbr,
            target_kbps: kbps,
            min_kbps: kbps,
            max_kbps: kbps,
            quality_param: 0.0,
        }
    }

    /// Create a VBR target with min/max bounds.
    pub fn vbr(target_kbps: u32, min_kbps: u32, max_kbps: u32) -> Self {
        Self {
            mode: RcMode::Vbr,
            target_kbps,
            min_kbps,
            max_kbps,
            quality_param: 0.0,
        }
    }

    /// Create a CRF target with a given CRF value (lower = higher quality).
    pub fn crf(crf: f32) -> Self {
        Self {
            mode: RcMode::Crf,
            target_kbps: 0,
            min_kbps: 0,
            max_kbps: 0,
            quality_param: crf,
        }
    }

    /// Returns the effective bitrate in kbps.
    ///
    /// For quality-based modes, returns 0 (no hard bitrate target).
    pub fn effective_kbps(&self) -> u32 {
        if self.mode.is_quality_based() {
            0
        } else {
            self.target_kbps
        }
    }

    /// Returns `true` when a hard bitrate budget is set.
    pub fn has_bitrate_budget(&self) -> bool {
        self.effective_kbps() > 0
    }
}

/// Estimates a reasonable bitrate target for a given resolution and frame rate.
///
/// The model is intentionally simple and suitable for presets/defaults — it uses
/// empirically derived pixel-rate coefficients similar to those used by streaming
/// platforms for H.264/AV1 recommendations.
#[derive(Debug, Clone)]
pub struct BitrateModelEstimator {
    /// Coefficient: bits per pixel per second (tunable).
    bpp_coefficient: f64,
}

impl BitrateModelEstimator {
    /// Create an estimator with default BPP coefficient (0.07).
    pub fn new() -> Self {
        Self {
            bpp_coefficient: 0.07,
        }
    }

    /// Create an estimator with a custom BPP coefficient.
    pub fn with_coefficient(bpp_coefficient: f64) -> Self {
        Self { bpp_coefficient }
    }

    /// Estimate the target bitrate in kbps for the given width × height and fps.
    ///
    /// Formula: `width * height * fps * bpp_coefficient / 1000`
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_for_resolution(&self, width: u32, height: u32, fps: f64) -> u32 {
        let pixels = width as f64 * height as f64;
        let bits_per_sec = pixels * fps * self.bpp_coefficient;
        (bits_per_sec / 1000.0).round() as u32
    }

    /// Estimate for a named preset (720p30, 1080p30, 4K30, etc.).
    pub fn estimate_for_preset(&self, preset: &str) -> Option<u32> {
        match preset {
            "720p30" => Some(self.estimate_for_resolution(1280, 720, 30.0)),
            "720p60" => Some(self.estimate_for_resolution(1280, 720, 60.0)),
            "1080p30" => Some(self.estimate_for_resolution(1920, 1080, 30.0)),
            "1080p60" => Some(self.estimate_for_resolution(1920, 1080, 60.0)),
            "4k30" => Some(self.estimate_for_resolution(3840, 2160, 30.0)),
            "4k60" => Some(self.estimate_for_resolution(3840, 2160, 60.0)),
            _ => None,
        }
    }
}

impl Default for BitrateModelEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc_mode_cbr_not_quality_based() {
        assert!(!RcMode::Cbr.is_quality_based());
    }

    #[test]
    fn test_rc_mode_crf_is_quality_based() {
        assert!(RcMode::Crf.is_quality_based());
    }

    #[test]
    fn test_rc_mode_qp_is_quality_based() {
        assert!(RcMode::Qp.is_quality_based());
    }

    #[test]
    fn test_rc_mode_vbr_is_bitrate_constrained() {
        assert!(RcMode::Vbr.is_bitrate_constrained());
    }

    #[test]
    fn test_rc_mode_labels() {
        assert_eq!(RcMode::Cbr.label(), "CBR");
        assert_eq!(RcMode::Crf.label(), "CRF");
        assert_eq!(RcMode::Qp.label(), "QP");
    }

    #[test]
    fn test_bitrate_target_cbr_effective_kbps() {
        let t = BitrateTarget::cbr(5000);
        assert_eq!(t.effective_kbps(), 5000);
        assert!(t.has_bitrate_budget());
    }

    #[test]
    fn test_bitrate_target_crf_effective_kbps_is_zero() {
        let t = BitrateTarget::crf(23.0);
        assert_eq!(t.effective_kbps(), 0);
        assert!(!t.has_bitrate_budget());
    }

    #[test]
    fn test_bitrate_target_vbr_bounds() {
        let t = BitrateTarget::vbr(4000, 1000, 8000);
        assert_eq!(t.min_kbps, 1000);
        assert_eq!(t.max_kbps, 8000);
    }

    #[test]
    fn test_estimator_1080p30_reasonable() {
        let est = BitrateModelEstimator::new();
        let kbps = est.estimate_for_resolution(1920, 1080, 30.0);
        // 1920*1080*30*0.07/1000 ≈ 4354 kbps
        assert!(kbps > 3000 && kbps < 6000, "Unexpected kbps: {kbps}");
    }

    #[test]
    fn test_estimator_4k_higher_than_1080p() {
        let est = BitrateModelEstimator::new();
        let kbps_1080 = est.estimate_for_resolution(1920, 1080, 30.0);
        let kbps_4k = est.estimate_for_resolution(3840, 2160, 30.0);
        assert!(kbps_4k > kbps_1080);
    }

    #[test]
    fn test_estimator_preset_720p30() {
        let est = BitrateModelEstimator::new();
        let kbps = est.estimate_for_preset("720p30").expect("should succeed");
        let manual = est.estimate_for_resolution(1280, 720, 30.0);
        assert_eq!(kbps, manual);
    }

    #[test]
    fn test_estimator_preset_unknown_returns_none() {
        let est = BitrateModelEstimator::new();
        assert!(est.estimate_for_preset("8k120").is_none());
    }

    #[test]
    fn test_estimator_custom_coefficient() {
        let est = BitrateModelEstimator::with_coefficient(0.14);
        let kbps_default = BitrateModelEstimator::new().estimate_for_resolution(1920, 1080, 30.0);
        let kbps_custom = est.estimate_for_resolution(1920, 1080, 30.0);
        // Double coefficient → approximately double bitrate
        assert!(kbps_custom > kbps_default);
    }

    #[test]
    fn test_estimator_default_impl() {
        let _est: BitrateModelEstimator = Default::default();
    }
}
