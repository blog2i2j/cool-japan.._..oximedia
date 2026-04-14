//! HDR format enumeration and passthrough configuration.
//!
//! This module provides a higher-level, format-centric API for HDR metadata
//! handling that complements the detailed [`super::hdr_passthrough`] module.
//!
//! Where `hdr_passthrough` operates at the level of individual transfer
//! functions and colour-primaries codes, this module exposes named HDR
//! *format* presets (HDR10, HDR10+, HLG, Dolby Vision) and a simple
//! configuration struct for selecting the passthrough behaviour.

use serde::{Deserialize, Serialize};

// ─── HdrFormat ────────────────────────────────────────────────────────────────

/// Named HDR format / signal type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HdrFormat {
    /// HDR10: PQ/ST2084 transfer, BT.2020 primaries, static SMPTE ST 2086
    /// mastering-display and CTA-861.3 MaxCLL/MaxFALL SEI.
    Hdr10,
    /// HDR10+: HDR10 baseline plus per-scene dynamic SMPTE ST 2094-40 SEI.
    Hdr10Plus,
    /// HLG BT.2100: Hybrid Log-Gamma, backward-compatible with SDR displays.
    HlgBt2100,
    /// Dolby Vision single or dual layer with RPU.
    DolbyVision {
        /// Dolby Vision profile number (4, 5, 7, 8, or 9).
        profile: u8,
        /// Dolby Vision level (1–13).
        level: u8,
    },
    /// SDR — no HDR metadata.
    None,
}

impl HdrFormat {
    /// Short descriptive name for the format.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Hdr10 => "HDR10",
            Self::Hdr10Plus => "HDR10+",
            Self::HlgBt2100 => "HLG BT.2100",
            Self::DolbyVision { .. } => "Dolby Vision",
            Self::None => "SDR",
        }
    }

    /// Returns `true` when this format carries per-frame/per-scene dynamic
    /// metadata (HDR10+ and Dolby Vision require this).
    #[must_use]
    pub fn requires_dynamic_metadata(&self) -> bool {
        matches!(self, Self::Hdr10Plus | Self::DolbyVision { .. })
    }

    /// IANA / FFmpeg colour-primaries string for the format.
    #[must_use]
    pub fn color_primaries(&self) -> &str {
        match self {
            Self::Hdr10 | Self::Hdr10Plus | Self::HlgBt2100 | Self::DolbyVision { .. } => "bt2020",
            Self::None => "bt709",
        }
    }

    /// IANA / FFmpeg transfer-characteristics string for the format.
    #[must_use]
    pub fn transfer_characteristics(&self) -> &str {
        match self {
            Self::Hdr10 | Self::Hdr10Plus | Self::DolbyVision { .. } => "smpte2084",
            Self::HlgBt2100 => "arib-std-b67",
            Self::None => "bt709",
        }
    }

    /// Returns `true` when this format is any HDR variant (non-SDR).
    #[must_use]
    pub fn is_hdr_format(&self) -> bool {
        !matches!(self, Self::None)
    }
}

// ─── MasterDisplayMetadata ────────────────────────────────────────────────────

/// SMPTE ST 2086 mastering-display colour-volume descriptor (simplified).
///
/// Chromaticity coordinates use the CIE xy system (range 0–1).
/// Luminance values are in candelas per square metre (cd/m²).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MasterDisplayMetadata {
    /// Red primary CIE xy chromaticity.
    pub primaries_r: (f32, f32),
    /// Green primary CIE xy chromaticity.
    pub primaries_g: (f32, f32),
    /// Blue primary CIE xy chromaticity.
    pub primaries_b: (f32, f32),
    /// White point CIE xy chromaticity.
    pub white_point: (f32, f32),
    /// Maximum mastering-display luminance in cd/m².
    pub max_luminance_nits: f32,
    /// Minimum mastering-display luminance in cd/m².
    pub min_luminance_nits: f32,
}

impl MasterDisplayMetadata {
    /// Standard HDR10 mastering display — Rec.2020 primaries, D65 white
    /// point, 1000 cd/m² peak (P3-D65 reference monitor approximation).
    #[must_use]
    pub fn rec2020_p3d65() -> Self {
        Self {
            primaries_r: (0.680, 0.320),
            primaries_g: (0.265, 0.690),
            primaries_b: (0.150, 0.060),
            white_point: (0.3127, 0.3290),
            max_luminance_nits: 1000.0,
            min_luminance_nits: 0.005,
        }
    }

    /// Validates this mastering-display descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error string when:
    /// - Any chromaticity value is outside `[0, 1]`.
    /// - `min_luminance_nits >= max_luminance_nits`.
    /// - `max_luminance_nits <= 0`.
    pub fn validate(&self) -> Result<(), String> {
        let check_chroma = |name: &str, (x, y): (f32, f32)| -> Result<(), String> {
            if !(0.0..=1.0).contains(&x) {
                return Err(format!("{name}.x={x} is outside [0, 1]"));
            }
            if !(0.0..=1.0).contains(&y) {
                return Err(format!("{name}.y={y} is outside [0, 1]"));
            }
            Ok(())
        };
        check_chroma("primaries_r", self.primaries_r)?;
        check_chroma("primaries_g", self.primaries_g)?;
        check_chroma("primaries_b", self.primaries_b)?;
        check_chroma("white_point", self.white_point)?;
        if self.max_luminance_nits <= 0.0 {
            return Err(format!(
                "max_luminance_nits={} must be > 0",
                self.max_luminance_nits
            ));
        }
        if self.min_luminance_nits < 0.0 || self.min_luminance_nits >= self.max_luminance_nits {
            return Err(format!(
                "min_luminance_nits={} must be in [0, {})",
                self.min_luminance_nits, self.max_luminance_nits
            ));
        }
        Ok(())
    }
}

// ─── HdrMetadataBundle ───────────────────────────────────────────────────────

/// Bundled HDR format descriptor and associated static metadata.
///
/// Use this together with [`HdrPassthroughConfig`] to describe the HDR
/// characteristics of a source stream and specify how they should be handled
/// during transcoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdrMetadataBundle {
    /// The HDR format / signal type.
    pub format: HdrFormat,
    /// Maximum Content Light Level (MaxCLL) in nits, if known.
    pub max_cll_nits: Option<u16>,
    /// Maximum Frame-Average Light Level (MaxFALL) in nits, if known.
    pub max_fall_nits: Option<u16>,
    /// SMPTE ST 2086 mastering-display descriptor, if available.
    pub master_display: Option<MasterDisplayMetadata>,
}

impl HdrMetadataBundle {
    /// Constructs an HDR10 bundle with MaxCLL and MaxFALL values.
    #[must_use]
    pub fn hdr10(max_cll: u16, max_fall: u16) -> Self {
        Self {
            format: HdrFormat::Hdr10,
            max_cll_nits: Some(max_cll),
            max_fall_nits: Some(max_fall),
            master_display: None,
        }
    }

    /// Constructs a minimal HLG bundle (no static metadata required).
    #[must_use]
    pub fn hlg() -> Self {
        Self {
            format: HdrFormat::HlgBt2100,
            max_cll_nits: None,
            max_fall_nits: None,
            master_display: None,
        }
    }

    /// Constructs an SDR bundle.
    #[must_use]
    pub fn sdr() -> Self {
        Self {
            format: HdrFormat::None,
            max_cll_nits: None,
            max_fall_nits: None,
            master_display: None,
        }
    }

    /// Returns `true` when the source format carries an HDR signal.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.format.is_hdr_format()
    }
}

// ─── HdrPassthroughConfig ────────────────────────────────────────────────────

/// Passthrough behaviour selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HdrPassthroughMode {
    /// Copy all HDR metadata from source to output unchanged.
    Passthrough,
    /// Convert the HDR signal to a different named format.
    ///
    /// Pixel-level tone-mapping must be handled by the frame pipeline.
    Convert(HdrFormat),
    /// Tone-map the signal down to SDR (BT.709 / BT.1886).
    ToSdr,
    /// Strip all HDR metadata and signal a plain SDR output.
    Strip,
}

/// Top-level configuration for HDR metadata handling in the transcode pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdrPassthroughConfig {
    /// How to handle HDR metadata.
    pub mode: HdrPassthroughMode,
    /// Format to use when the source is SDR but the mode is `Convert`.
    pub fallback_format: HdrFormat,
}

impl HdrPassthroughConfig {
    /// Pass HDR metadata through unchanged.
    #[must_use]
    pub fn passthrough() -> Self {
        Self {
            mode: HdrPassthroughMode::Passthrough,
            fallback_format: HdrFormat::None,
        }
    }

    /// Tone-map the output to SDR.
    #[must_use]
    pub fn to_sdr() -> Self {
        Self {
            mode: HdrPassthroughMode::ToSdr,
            fallback_format: HdrFormat::None,
        }
    }

    /// Convert HDR to the specified target format.
    #[must_use]
    pub fn convert_to(target: HdrFormat) -> Self {
        Self {
            mode: HdrPassthroughMode::Convert(target),
            fallback_format: HdrFormat::None,
        }
    }
}

// ─── Compatibility check ──────────────────────────────────────────────────────

/// Returns `true` when `source` and `target` are the same HDR format and no
/// conversion is required.
///
/// Note: `DolbyVision` formats are only compatible when both the profile
/// *and* the level match.
#[must_use]
pub fn are_compatible(source: &HdrFormat, target: &HdrFormat) -> bool {
    source == target
}

// ─── FFmpeg-style filter options ─────────────────────────────────────────────

/// Generates a list of FFmpeg-style key=value option pairs for the
/// `zscale` / `colorspace` filter graph that corresponds to the requested
/// passthrough configuration and source metadata.
///
/// The caller is responsible for wiring these into the filter graph string.
/// An empty vector is returned when no metadata manipulation is required
/// (i.e. [`HdrPassthroughMode::Passthrough`] mode with an SDR source).
#[must_use]
pub fn hdr_filter_options(
    config: &HdrPassthroughConfig,
    source_meta: &HdrMetadataBundle,
) -> Vec<(String, String)> {
    match &config.mode {
        HdrPassthroughMode::Passthrough => {
            // Only inject flags for HDR sources to avoid touching SDR metadata.
            if source_meta.is_hdr() {
                vec![
                    (
                        "color_primaries".to_string(),
                        source_meta.format.color_primaries().to_string(),
                    ),
                    (
                        "color_trc".to_string(),
                        source_meta.format.transfer_characteristics().to_string(),
                    ),
                    (
                        "colorspace".to_string(),
                        source_meta.format.color_primaries().to_string(),
                    ),
                ]
            } else {
                vec![]
            }
        }

        HdrPassthroughMode::Strip => {
            vec![
                ("color_primaries".to_string(), "bt709".to_string()),
                ("color_trc".to_string(), "bt709".to_string()),
                ("colorspace".to_string(), "bt709".to_string()),
            ]
        }

        HdrPassthroughMode::ToSdr => {
            let src_trc = source_meta.format.transfer_characteristics().to_string();
            vec![
                ("color_primaries".to_string(), "bt709".to_string()),
                ("color_trc".to_string(), "bt709".to_string()),
                ("colorspace".to_string(), "bt709".to_string()),
                ("transfer_in".to_string(), src_trc),
                ("transfer_out".to_string(), "bt709".to_string()),
            ]
        }

        HdrPassthroughMode::Convert(target) => {
            let mut opts = vec![
                (
                    "color_primaries".to_string(),
                    target.color_primaries().to_string(),
                ),
                (
                    "color_trc".to_string(),
                    target.transfer_characteristics().to_string(),
                ),
                (
                    "colorspace".to_string(),
                    target.color_primaries().to_string(),
                ),
                (
                    "transfer_in".to_string(),
                    source_meta.format.transfer_characteristics().to_string(),
                ),
                (
                    "transfer_out".to_string(),
                    target.transfer_characteristics().to_string(),
                ),
            ];
            if let Some(cll) = source_meta.max_cll_nits {
                opts.push(("max_cll".to_string(), cll.to_string()));
            }
            if let Some(fall) = source_meta.max_fall_nits {
                opts.push(("max_fall".to_string(), fall.to_string()));
            }
            opts
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr_format_names() {
        assert_eq!(HdrFormat::Hdr10.name(), "HDR10");
        assert_eq!(HdrFormat::Hdr10Plus.name(), "HDR10+");
        assert_eq!(HdrFormat::HlgBt2100.name(), "HLG BT.2100");
        assert_eq!(
            HdrFormat::DolbyVision {
                profile: 8,
                level: 6
            }
            .name(),
            "Dolby Vision"
        );
        assert_eq!(HdrFormat::None.name(), "SDR");
    }

    #[test]
    fn test_requires_dynamic_metadata() {
        assert!(!HdrFormat::Hdr10.requires_dynamic_metadata());
        assert!(HdrFormat::Hdr10Plus.requires_dynamic_metadata());
        assert!(!HdrFormat::HlgBt2100.requires_dynamic_metadata());
        assert!(HdrFormat::DolbyVision {
            profile: 5,
            level: 1
        }
        .requires_dynamic_metadata());
        assert!(!HdrFormat::None.requires_dynamic_metadata());
    }

    #[test]
    fn test_hdr_metadata_bundle_is_hdr() {
        assert!(HdrMetadataBundle::hdr10(1000, 400).is_hdr());
        assert!(HdrMetadataBundle::hlg().is_hdr());
        assert!(!HdrMetadataBundle::sdr().is_hdr());
    }

    #[test]
    fn test_hdr10_constructor() {
        let bundle = HdrMetadataBundle::hdr10(1000, 400);
        assert_eq!(bundle.format, HdrFormat::Hdr10);
        assert_eq!(bundle.max_cll_nits, Some(1000));
        assert_eq!(bundle.max_fall_nits, Some(400));
    }

    #[test]
    fn test_are_compatible_same_format_is_true() {
        assert!(are_compatible(&HdrFormat::Hdr10, &HdrFormat::Hdr10));
        assert!(are_compatible(&HdrFormat::None, &HdrFormat::None));
        assert!(are_compatible(&HdrFormat::HlgBt2100, &HdrFormat::HlgBt2100));
    }

    #[test]
    fn test_are_compatible_different_format_is_false() {
        assert!(!are_compatible(&HdrFormat::Hdr10, &HdrFormat::HlgBt2100));
        assert!(!are_compatible(&HdrFormat::Hdr10Plus, &HdrFormat::Hdr10));
        assert!(!are_compatible(&HdrFormat::None, &HdrFormat::Hdr10));
    }

    #[test]
    fn test_to_sdr_passthrough_mode_is_false() {
        let config = HdrPassthroughConfig::to_sdr();
        assert_ne!(config.mode, HdrPassthroughMode::Passthrough);
    }

    #[test]
    fn test_hdr_filter_options_non_empty_for_convert_mode() {
        let config = HdrPassthroughConfig::convert_to(HdrFormat::HlgBt2100);
        let source = HdrMetadataBundle::hdr10(1000, 400);
        let opts = hdr_filter_options(&config, &source);
        assert!(!opts.is_empty(), "Convert mode must produce filter options");
        // Verify transfer_in/transfer_out are present
        let has_transfer_in = opts.iter().any(|(k, _)| k == "transfer_in");
        let has_transfer_out = opts.iter().any(|(k, _)| k == "transfer_out");
        assert!(has_transfer_in, "Expected transfer_in key");
        assert!(has_transfer_out, "Expected transfer_out key");
    }

    #[test]
    fn test_hdr_filter_options_empty_for_sdr_passthrough() {
        let config = HdrPassthroughConfig::passthrough();
        let source = HdrMetadataBundle::sdr();
        let opts = hdr_filter_options(&config, &source);
        assert!(
            opts.is_empty(),
            "SDR passthrough should produce no filter options"
        );
    }

    #[test]
    fn test_master_display_metadata_rec2020_p3d65_validates() {
        let md = MasterDisplayMetadata::rec2020_p3d65();
        assert!(md.validate().is_ok());
    }

    #[test]
    fn test_master_display_metadata_bad_luminance_order() {
        let mut md = MasterDisplayMetadata::rec2020_p3d65();
        md.min_luminance_nits = md.max_luminance_nits + 1.0;
        assert!(md.validate().is_err());
    }

    #[test]
    fn test_master_display_metadata_bad_chromaticity() {
        let mut md = MasterDisplayMetadata::rec2020_p3d65();
        md.primaries_r = (1.5, 0.3); // x out of range
        assert!(md.validate().is_err());
    }

    #[test]
    fn test_color_primaries_and_trc_strings() {
        assert_eq!(HdrFormat::Hdr10.color_primaries(), "bt2020");
        assert_eq!(HdrFormat::Hdr10.transfer_characteristics(), "smpte2084");
        assert_eq!(HdrFormat::HlgBt2100.transfer_characteristics(), "arib-std-b67");
        assert_eq!(HdrFormat::None.color_primaries(), "bt709");
    }
}
