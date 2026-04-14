//! HDR metadata passthrough/conversion across the transcode pipeline.
//!
//! `HdrMetadataBridge` sits between demuxer and muxer, intercepting HDR
//! metadata at the stream level.  It supports three policies:
//!
//! - **Passthrough**: copy metadata unchanged.
//! - **Convert**: re-map between HDR10 (PQ/ST2084) and HLG (BT.2100).
//! - **Strip**: remove all HDR metadata, producing an SDR-flagged output.
//!
//! The bridge handles:
//! - HDR10 static metadata (SMPTE ST 2086 mastering display + MaxCLL/MaxFALL)
//! - HDR10+ dynamic SEI metadata (per-scene bezier tone mapping curves)
//! - HLG (ITU-R BT.2100) metadata and system gamma
//!
//! Full pixel-level tone mapping is outside this module's scope (see
//! `oximedia-hdr`).  This module handles the **metadata** side only.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use serde::{Deserialize, Serialize};

use crate::hdr_passthrough::{
    ColourPrimaries, ContentLightLevel, HdrMetadata, MasteringDisplay, TransferFunction,
};
use crate::{Result, TranscodeError};

// ─── HDR policy ──────────────────────────────────────────────────────────────

/// Policy governing how HDR metadata is handled during transcoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HdrPolicy {
    /// Copy all HDR metadata unchanged from input to output.
    Passthrough,
    /// Convert HDR metadata between transfer function families.
    Convert {
        /// Target transfer function for the output.
        target_tf: TransferFunction,
    },
    /// Remove all HDR metadata; output is SDR (BT.709).
    Strip,
}

impl Default for HdrPolicy {
    fn default() -> Self {
        Self::Passthrough
    }
}

// ─── HDR10+ dynamic SEI ──────────────────────────────────────────────────────

/// An HDR10+ dynamic metadata payload for a single frame or scene.
///
/// HDR10+ (SMPTE ST 2094-40) carries per-scene bezier tone-mapping
/// curves and distribution percentiles for optimal display adaptation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hdr10PlusSei {
    /// Target system display maximum luminance (cd/m^2).
    pub targeted_system_display_max_luminance: u32,
    /// Distribution maxrgb percentiles (up to 15 entries).
    ///
    /// Each entry is `(percentile_percentage, percentile_value)`.
    pub distribution_maxrgb: Vec<(u8, u32)>,
    /// Bezier curve anchors for tone mapping (0..1 range, normalized).
    pub bezier_curve_anchors: Vec<f64>,
    /// Knee point x coordinate (0..1).
    pub knee_point_x: f64,
    /// Knee point y coordinate (0..1).
    pub knee_point_y: f64,
    /// Number of bezier curve anchors.
    pub num_bezier_curve_anchors: u8,
}

impl Hdr10PlusSei {
    /// Creates a minimal HDR10+ SEI payload.
    #[must_use]
    pub fn new(target_max_lum: u32) -> Self {
        Self {
            targeted_system_display_max_luminance: target_max_lum,
            distribution_maxrgb: Vec::new(),
            bezier_curve_anchors: Vec::new(),
            knee_point_x: 0.0,
            knee_point_y: 0.0,
            num_bezier_curve_anchors: 0,
        }
    }

    /// Adds a distribution percentile entry.
    #[must_use]
    pub fn with_percentile(mut self, percentage: u8, value: u32) -> Self {
        self.distribution_maxrgb.push((percentage, value));
        self
    }

    /// Sets the bezier curve anchors.
    #[must_use]
    pub fn with_bezier(mut self, knee_x: f64, knee_y: f64, anchors: Vec<f64>) -> Self {
        self.knee_point_x = knee_x;
        self.knee_point_y = knee_y;
        self.num_bezier_curve_anchors = anchors.len().min(255) as u8;
        self.bezier_curve_anchors = anchors;
        self
    }

    /// Validates the SEI payload.
    ///
    /// # Errors
    ///
    /// Returns an error if values are out of legal range.
    pub fn validate(&self) -> Result<()> {
        if self.targeted_system_display_max_luminance == 0 {
            return Err(TranscodeError::InvalidInput(
                "HDR10+ target max luminance must be > 0".into(),
            ));
        }
        if self.knee_point_x < 0.0 || self.knee_point_x > 1.0 {
            return Err(TranscodeError::InvalidInput(format!(
                "HDR10+ knee_point_x {} out of range [0, 1]",
                self.knee_point_x
            )));
        }
        if self.knee_point_y < 0.0 || self.knee_point_y > 1.0 {
            return Err(TranscodeError::InvalidInput(format!(
                "HDR10+ knee_point_y {} out of range [0, 1]",
                self.knee_point_y
            )));
        }
        for anchor in &self.bezier_curve_anchors {
            if *anchor < 0.0 || *anchor > 1.0 {
                return Err(TranscodeError::InvalidInput(format!(
                    "HDR10+ bezier anchor {} out of range [0, 1]",
                    anchor
                )));
            }
        }
        Ok(())
    }
}

// ─── HLG system gamma ────────────────────────────────────────────────────────

/// HLG system gamma parameters (ITU-R BT.2100).
///
/// The system gamma controls the OOTF mapping from scene light to display
/// light and depends on the nominal peak luminance of the target display.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HlgSystemGamma {
    /// Nominal peak display luminance in cd/m^2.
    pub nominal_peak_luminance: f64,
    /// System gamma value (typically 1.0 for 1000 cd/m^2 reference).
    pub gamma: f64,
}

impl HlgSystemGamma {
    /// Creates the HLG reference system gamma (1000 cd/m^2 display).
    #[must_use]
    pub fn reference_1000nit() -> Self {
        Self {
            nominal_peak_luminance: 1000.0,
            gamma: 1.2,
        }
    }

    /// Computes the system gamma for a given peak luminance.
    ///
    /// Per BT.2100: gamma = 1.2 * (Lw / 1000)^0.2
    /// where Lw is the nominal peak luminance.
    #[must_use]
    pub fn for_luminance(peak_luminance: f64) -> Self {
        let gamma = if peak_luminance > 0.0 {
            1.2 * (peak_luminance / 1000.0).powf(0.2)
        } else {
            1.2
        };
        Self {
            nominal_peak_luminance: peak_luminance,
            gamma,
        }
    }

    /// Validates the parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if luminance or gamma is non-positive.
    pub fn validate(&self) -> Result<()> {
        if self.nominal_peak_luminance <= 0.0 {
            return Err(TranscodeError::InvalidInput(
                "HLG nominal peak luminance must be > 0".into(),
            ));
        }
        if self.gamma <= 0.0 {
            return Err(TranscodeError::InvalidInput(
                "HLG system gamma must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ─── Bridge metadata bundle ──────────────────────────────────────────────────

/// Extended HDR metadata bundle carrying both static and dynamic metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgeHdrMetadata {
    /// Base static metadata (transfer function, primaries, mastering display, CLL).
    pub base: HdrMetadata,
    /// HDR10+ dynamic SEI payloads (one per scene / per frame).
    pub hdr10plus_sei: Vec<Hdr10PlusSei>,
    /// HLG system gamma parameters (if HLG output).
    pub hlg_system_gamma: Option<HlgSystemGamma>,
}

impl BridgeHdrMetadata {
    /// Creates a new bridge metadata from base HDR metadata.
    #[must_use]
    pub fn from_base(base: HdrMetadata) -> Self {
        Self {
            base,
            hdr10plus_sei: Vec::new(),
            hlg_system_gamma: None,
        }
    }

    /// Creates an HDR10 metadata bundle with mastering display and CLL.
    #[must_use]
    pub fn hdr10(mastering: MasteringDisplay, cll: ContentLightLevel) -> Self {
        Self {
            base: HdrMetadata::hdr10(mastering, cll),
            hdr10plus_sei: Vec::new(),
            hlg_system_gamma: None,
        }
    }

    /// Creates an HLG metadata bundle with system gamma.
    #[must_use]
    pub fn hlg(system_gamma: HlgSystemGamma) -> Self {
        Self {
            base: HdrMetadata::hlg(),
            hdr10plus_sei: Vec::new(),
            hlg_system_gamma: Some(system_gamma),
        }
    }

    /// Adds an HDR10+ SEI payload.
    #[must_use]
    pub fn with_hdr10plus_sei(mut self, sei: Hdr10PlusSei) -> Self {
        self.hdr10plus_sei.push(sei);
        self
    }

    /// Returns whether this is an HDR signal.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.base.is_hdr()
    }

    /// Returns the transfer function if known.
    #[must_use]
    pub fn transfer_function(&self) -> Option<TransferFunction> {
        self.base.transfer_function
    }

    /// Returns whether HDR10+ dynamic metadata is present.
    #[must_use]
    pub fn has_hdr10plus(&self) -> bool {
        !self.hdr10plus_sei.is_empty()
    }
}

// ─── Conversion helpers ──────────────────────────────────────────────────────

/// Approximate mastering display values when converting from HLG to HDR10.
///
/// HLG does not carry mastering display metadata, so we synthesize
/// a plausible ST 2086 descriptor for BT.2020 gamut at 1000 nit.
fn synthesize_mastering_display_for_hlg_to_hdr10() -> MasteringDisplay {
    MasteringDisplay {
        red_x: 0.708,
        red_y: 0.292,
        green_x: 0.170,
        green_y: 0.797,
        blue_x: 0.131,
        blue_y: 0.046,
        white_x: 0.3127,
        white_y: 0.3290,
        max_luminance: 1000.0,
        min_luminance: 0.001,
    }
}

/// Approximate content light level when converting from HLG to HDR10.
fn synthesize_cll_for_hlg_to_hdr10() -> ContentLightLevel {
    ContentLightLevel::new(1000, 400)
}

/// Convert HDR10 static metadata to HLG system gamma.
///
/// Uses the mastering display max luminance to derive the HLG system
/// gamma per BT.2100.
fn hdr10_to_hlg_system_gamma(mastering: &MasteringDisplay) -> HlgSystemGamma {
    HlgSystemGamma::for_luminance(mastering.max_luminance)
}

// ─── HdrMetadataBridge ───────────────────────────────────────────────────────

/// Bridge for HDR metadata across the transcode pipeline.
///
/// Applies the configured `HdrPolicy` to source metadata, producing
/// output metadata suitable for the target container / codec.
#[derive(Debug, Clone)]
pub struct HdrMetadataBridge {
    policy: HdrPolicy,
}

impl HdrMetadataBridge {
    /// Creates a new bridge with the given policy.
    #[must_use]
    pub fn new(policy: HdrPolicy) -> Self {
        Self { policy }
    }

    /// Creates a passthrough bridge (no conversion).
    #[must_use]
    pub fn passthrough() -> Self {
        Self::new(HdrPolicy::Passthrough)
    }

    /// Creates a strip bridge (removes all HDR metadata).
    #[must_use]
    pub fn strip() -> Self {
        Self::new(HdrPolicy::Strip)
    }

    /// Returns the configured policy.
    #[must_use]
    pub fn policy(&self) -> &HdrPolicy {
        &self.policy
    }

    /// Processes source metadata through the bridge, applying the configured policy.
    ///
    /// # Errors
    ///
    /// Returns an error if an unsupported conversion is requested.
    pub fn process(&self, source: &BridgeHdrMetadata) -> Result<BridgeHdrMetadata> {
        match &self.policy {
            HdrPolicy::Passthrough => Ok(source.clone()),

            HdrPolicy::Strip => Ok(BridgeHdrMetadata {
                base: HdrMetadata {
                    transfer_function: Some(TransferFunction::Bt709),
                    colour_primaries: Some(ColourPrimaries::Bt709),
                    mastering_display: None,
                    content_light_level: None,
                    dolby_vision: None,
                },
                hdr10plus_sei: Vec::new(),
                hlg_system_gamma: None,
            }),

            HdrPolicy::Convert { target_tf } => {
                self.convert(source, *target_tf)
            }
        }
    }

    /// Internal conversion logic.
    fn convert(
        &self,
        source: &BridgeHdrMetadata,
        target_tf: TransferFunction,
    ) -> Result<BridgeHdrMetadata> {
        let src_tf = source
            .base
            .transfer_function
            .unwrap_or(TransferFunction::Unspecified);

        match (src_tf, target_tf) {
            // Identity conversions.
            (TransferFunction::Pq, TransferFunction::Pq) => Ok(source.clone()),
            (TransferFunction::Hlg, TransferFunction::Hlg) => Ok(source.clone()),
            (TransferFunction::Bt709, TransferFunction::Bt709) => Ok(source.clone()),

            // HDR10 (PQ) -> HLG
            (TransferFunction::Pq, TransferFunction::Hlg) => {
                self.convert_hdr10_to_hlg(source)
            }

            // HLG -> HDR10 (PQ)
            (TransferFunction::Hlg, TransferFunction::Pq) => {
                self.convert_hlg_to_hdr10(source)
            }

            // HDR10 (PQ) -> SDR
            (TransferFunction::Pq, TransferFunction::Bt709) => {
                Ok(self.strip_to_sdr())
            }

            // HLG -> SDR
            (TransferFunction::Hlg, TransferFunction::Bt709) => {
                Ok(self.strip_to_sdr())
            }

            // Unspecified source -> target (treat as passthrough with updated TF)
            (TransferFunction::Unspecified, _) => {
                let mut out = source.clone();
                out.base.transfer_function = Some(target_tf);
                Ok(out)
            }

            _ => Err(TranscodeError::CodecError(format!(
                "Unsupported HDR conversion: {src_tf:?} -> {target_tf:?}"
            ))),
        }
    }

    /// Convert HDR10 (PQ) metadata to HLG.
    fn convert_hdr10_to_hlg(&self, source: &BridgeHdrMetadata) -> Result<BridgeHdrMetadata> {
        let system_gamma = source
            .base
            .mastering_display
            .as_ref()
            .map(hdr10_to_hlg_system_gamma)
            .unwrap_or_else(HlgSystemGamma::reference_1000nit);

        Ok(BridgeHdrMetadata {
            base: HdrMetadata {
                transfer_function: Some(TransferFunction::Hlg),
                colour_primaries: Some(ColourPrimaries::Bt2020),
                mastering_display: None,
                content_light_level: None,
                dolby_vision: None,
            },
            hdr10plus_sei: Vec::new(), // HDR10+ not applicable to HLG
            hlg_system_gamma: Some(system_gamma),
        })
    }

    /// Convert HLG metadata to HDR10 (PQ).
    fn convert_hlg_to_hdr10(&self, _source: &BridgeHdrMetadata) -> Result<BridgeHdrMetadata> {
        let mastering = synthesize_mastering_display_for_hlg_to_hdr10();
        let cll = synthesize_cll_for_hlg_to_hdr10();

        Ok(BridgeHdrMetadata {
            base: HdrMetadata::hdr10(mastering, cll),
            hdr10plus_sei: Vec::new(),
            hlg_system_gamma: None,
        })
    }

    /// Strip to SDR metadata.
    fn strip_to_sdr(&self) -> BridgeHdrMetadata {
        BridgeHdrMetadata {
            base: HdrMetadata {
                transfer_function: Some(TransferFunction::Bt709),
                colour_primaries: Some(ColourPrimaries::Bt709),
                mastering_display: None,
                content_light_level: None,
                dolby_vision: None,
            },
            hdr10plus_sei: Vec::new(),
            hlg_system_gamma: None,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hdr10_source() -> BridgeHdrMetadata {
        BridgeHdrMetadata::hdr10(
            MasteringDisplay::p3_d65_1000nit(),
            ContentLightLevel::hdr10_default(),
        )
    }

    fn make_hlg_source() -> BridgeHdrMetadata {
        BridgeHdrMetadata::hlg(HlgSystemGamma::reference_1000nit())
    }

    #[test]
    fn test_passthrough_preserves_metadata() {
        let bridge = HdrMetadataBridge::passthrough();
        let source = make_hdr10_source();
        let output = bridge.process(&source).expect("passthrough should succeed");

        assert_eq!(output.base.transfer_function, Some(TransferFunction::Pq));
        assert!(output.base.mastering_display.is_some());
        assert!(output.base.content_light_level.is_some());
    }

    #[test]
    fn test_strip_removes_all_hdr() {
        let bridge = HdrMetadataBridge::strip();
        let source = make_hdr10_source();
        let output = bridge.process(&source).expect("strip should succeed");

        assert_eq!(output.base.transfer_function, Some(TransferFunction::Bt709));
        assert_eq!(output.base.colour_primaries, Some(ColourPrimaries::Bt709));
        assert!(output.base.mastering_display.is_none());
        assert!(output.base.content_light_level.is_none());
        assert!(!output.is_hdr());
    }

    #[test]
    fn test_hdr10_to_hlg_conversion() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Hlg,
        });
        let source = make_hdr10_source();
        let output = bridge.process(&source).expect("HDR10->HLG should succeed");

        assert_eq!(output.base.transfer_function, Some(TransferFunction::Hlg));
        assert_eq!(output.base.colour_primaries, Some(ColourPrimaries::Bt2020));
        assert!(output.hlg_system_gamma.is_some());
        assert!(output.base.mastering_display.is_none());
        assert!(output.hdr10plus_sei.is_empty());
    }

    #[test]
    fn test_hlg_to_hdr10_conversion() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Pq,
        });
        let source = make_hlg_source();
        let output = bridge.process(&source).expect("HLG->HDR10 should succeed");

        assert_eq!(output.base.transfer_function, Some(TransferFunction::Pq));
        assert!(output.base.mastering_display.is_some());
        assert!(output.base.content_light_level.is_some());
        assert!(output.hlg_system_gamma.is_none());
    }

    #[test]
    fn test_hdr10_to_sdr_conversion() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Bt709,
        });
        let source = make_hdr10_source();
        let output = bridge.process(&source).expect("HDR10->SDR should succeed");

        assert_eq!(output.base.transfer_function, Some(TransferFunction::Bt709));
        assert!(!output.is_hdr());
    }

    #[test]
    fn test_hlg_to_sdr_conversion() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Bt709,
        });
        let source = make_hlg_source();
        let output = bridge.process(&source).expect("HLG->SDR should succeed");

        assert_eq!(output.base.transfer_function, Some(TransferFunction::Bt709));
        assert!(!output.is_hdr());
    }

    #[test]
    fn test_identity_conversion_pq_to_pq() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Pq,
        });
        let source = make_hdr10_source();
        let output = bridge.process(&source).expect("PQ->PQ should succeed");

        assert_eq!(output.base.transfer_function, Some(TransferFunction::Pq));
        assert!(output.base.mastering_display.is_some());
    }

    #[test]
    fn test_unsupported_conversion_sdr_to_pq() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Pq,
        });
        let source = BridgeHdrMetadata {
            base: HdrMetadata {
                transfer_function: Some(TransferFunction::Bt709),
                colour_primaries: Some(ColourPrimaries::Bt709),
                ..HdrMetadata::default()
            },
            ..BridgeHdrMetadata::default()
        };
        let result = bridge.process(&source);
        assert!(result.is_err());
    }

    #[test]
    fn test_hdr10plus_sei_construction() {
        let sei = Hdr10PlusSei::new(4000)
            .with_percentile(1, 100)
            .with_percentile(50, 500)
            .with_percentile(99, 900)
            .with_bezier(0.5, 0.6, vec![0.1, 0.3, 0.7, 0.9]);

        assert_eq!(sei.targeted_system_display_max_luminance, 4000);
        assert_eq!(sei.distribution_maxrgb.len(), 3);
        assert_eq!(sei.num_bezier_curve_anchors, 4);
        assert!((sei.knee_point_x - 0.5).abs() < f64::EPSILON);
        sei.validate().expect("valid SEI");
    }

    #[test]
    fn test_hdr10plus_sei_validation_bad_knee() {
        let sei = Hdr10PlusSei::new(1000)
            .with_bezier(1.5, 0.5, vec![]);
        assert!(sei.validate().is_err());
    }

    #[test]
    fn test_hlg_system_gamma_for_luminance() {
        let sg = HlgSystemGamma::for_luminance(1000.0);
        assert!((sg.gamma - 1.2).abs() < 0.01);

        let sg_low = HlgSystemGamma::for_luminance(400.0);
        assert!(sg_low.gamma < 1.2);
        assert!(sg_low.gamma > 0.0);

        let sg_high = HlgSystemGamma::for_luminance(4000.0);
        assert!(sg_high.gamma > 1.2);
    }

    #[test]
    fn test_hlg_system_gamma_validation() {
        let sg = HlgSystemGamma::reference_1000nit();
        sg.validate().expect("should be valid");

        let bad = HlgSystemGamma {
            nominal_peak_luminance: -100.0,
            gamma: 1.2,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_bridge_hdr10plus_passthrough() {
        let bridge = HdrMetadataBridge::passthrough();
        let mut source = make_hdr10_source();
        source.hdr10plus_sei.push(Hdr10PlusSei::new(1000));
        source.hdr10plus_sei.push(Hdr10PlusSei::new(2000));

        let output = bridge.process(&source).expect("should succeed");
        assert_eq!(output.hdr10plus_sei.len(), 2);
        assert!(output.has_hdr10plus());
    }

    #[test]
    fn test_bridge_hdr10plus_stripped_on_convert() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Hlg,
        });
        let mut source = make_hdr10_source();
        source.hdr10plus_sei.push(Hdr10PlusSei::new(1000));

        let output = bridge.process(&source).expect("should succeed");
        // HDR10+ SEI is PQ-specific, should be dropped for HLG.
        assert!(output.hdr10plus_sei.is_empty());
    }

    #[test]
    fn test_bridge_metadata_bundle_constructors() {
        let hdr10 = BridgeHdrMetadata::hdr10(
            MasteringDisplay::bt2020_4000nit(),
            ContentLightLevel::new(4000, 1000),
        );
        assert!(hdr10.is_hdr());
        assert_eq!(hdr10.transfer_function(), Some(TransferFunction::Pq));
        assert!(!hdr10.has_hdr10plus());

        let hlg = BridgeHdrMetadata::hlg(HlgSystemGamma::reference_1000nit());
        assert!(hlg.is_hdr());
        assert_eq!(hlg.transfer_function(), Some(TransferFunction::Hlg));
    }

    #[test]
    fn test_default_policy_is_passthrough() {
        let policy = HdrPolicy::default();
        assert_eq!(policy, HdrPolicy::Passthrough);
    }

    #[test]
    fn test_unspecified_source_conversion() {
        let bridge = HdrMetadataBridge::new(HdrPolicy::Convert {
            target_tf: TransferFunction::Pq,
        });
        let source = BridgeHdrMetadata::default();
        let output = bridge.process(&source).expect("unspecified->PQ should succeed");
        assert_eq!(output.base.transfer_function, Some(TransferFunction::Pq));
    }
}
