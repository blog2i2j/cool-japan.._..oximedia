//! HDR metadata structures.
//!
//! This module implements:
//!
//! - **SMPTE ST 2086** – Mastering display colour volume (`MaxCLL`, `MaxFALL`,
//!   and mastering display primaries/luminance).
//! - **HDR10+ dynamic metadata** – Per-scene/per-frame luminance adjustment
//!   (stub – carries the raw payload for downstream processing).
//! - **Dolby Vision dynamic metadata** – Stub carrying the compressed blob
//!   for downstream muxers.

// --------------------------------------------------------------------------
// SMPTE ST 2086 – Mastering Display Colour Volume
// --------------------------------------------------------------------------

/// Chromaticity coordinates in CIE 1931 xy (x, y) per SMPTE ST 2086.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chromaticity {
    /// CIE 1931 x coordinate.
    pub x: f32,
    /// CIE 1931 y coordinate.
    pub y: f32,
}

impl Chromaticity {
    /// Create a new chromaticity value.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// SMPTE ST 2086 mastering display colour volume metadata.
///
/// This metadata describes the colour volume and luminance characteristics
/// of the mastering display used during HDR content production.
///
/// # Standards
///
/// - SMPTE ST 2086:2018 "Mastering Display Colour Volume"
/// - ITU-T H.265 / ISO/IEC 23008-2 SEI message `mastering_display_colour_volume`
/// - HEVC SEI: `SEI_type_mastering_display_colour_volume`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MasteringDisplayMetadata {
    /// Red primary chromaticity.
    pub red_primary: Chromaticity,
    /// Green primary chromaticity.
    pub green_primary: Chromaticity,
    /// Blue primary chromaticity.
    pub blue_primary: Chromaticity,
    /// White point chromaticity.
    pub white_point: Chromaticity,
    /// Maximum luminance of the mastering display in cd/m² (nits).
    pub max_luminance: f32,
    /// Minimum luminance of the mastering display in cd/m² (nits).
    pub min_luminance: f32,
}

impl MasteringDisplayMetadata {
    /// Create mastering display metadata for a typical HDR10 mastering suite
    /// with P3-D65 primaries and 1000 cd/m² peak.
    #[must_use]
    pub fn hdr10_p3_d65() -> Self {
        Self {
            // DCI-P3 / Display P3 primaries
            red_primary: Chromaticity::new(0.680, 0.320),
            green_primary: Chromaticity::new(0.265, 0.690),
            blue_primary: Chromaticity::new(0.150, 0.060),
            // D65 white point
            white_point: Chromaticity::new(0.3127, 0.3290),
            max_luminance: 1000.0,
            min_luminance: 0.005,
        }
    }

    /// Create mastering display metadata for a Rec.2020 mastering display at 4000 nits.
    #[must_use]
    pub fn hdr10_bt2020_4000() -> Self {
        Self {
            // BT.2020 primaries
            red_primary: Chromaticity::new(0.708, 0.292),
            green_primary: Chromaticity::new(0.170, 0.797),
            blue_primary: Chromaticity::new(0.131, 0.046),
            white_point: Chromaticity::new(0.3127, 0.3290),
            max_luminance: 4000.0,
            min_luminance: 0.001,
        }
    }

    /// Returns the dynamic range in stops (log₂(max/min)).
    ///
    /// Returns `None` if `min_luminance` is zero.
    #[must_use]
    pub fn dynamic_range_stops(&self) -> Option<f32> {
        if self.min_luminance <= 0.0 {
            return None;
        }
        Some((self.max_luminance / self.min_luminance).log2())
    }

    /// Check whether the luminance values are within SMPTE ST 2086 valid ranges.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.max_luminance > 0.0
            && self.min_luminance >= 0.0
            && self.max_luminance > self.min_luminance
    }
}

/// Content light level (CLL) metadata per CTA-861-G (formerly CEA-861).
///
/// This describes the maximum brightness levels found in the encoded video
/// and is used by tone-mapping hardware to optimize display brightness.
///
/// # Standards
///
/// - CTA-861-G Dynamic Range and Mastering Display Metadata
/// - HEVC SEI: `SEI_type_content_light_level_info`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (`MaxCLL`) in cd/m² (nits).
    ///
    /// The maximum luminance of any single pixel in the entire video sequence.
    pub max_cll: f32,
    /// Maximum Frame-Average Light Level (`MaxFALL`) in cd/m² (nits).
    ///
    /// The maximum average luminance of any single frame in the entire sequence.
    pub max_fall: f32,
}

impl ContentLightLevel {
    /// Create content light level metadata.
    #[must_use]
    pub const fn new(max_cll: f32, max_fall: f32) -> Self {
        Self { max_cll, max_fall }
    }

    /// Create a typical HDR10 stream with 1000 nit peak.
    #[must_use]
    pub const fn hdr10_1000() -> Self {
        Self {
            max_cll: 1000.0,
            max_fall: 400.0,
        }
    }

    /// Returns the recommended peak luminance for tone-mapping, in nits.
    ///
    /// Prefers `MaxCLL` but falls back to `MaxFALL` * 2 if `MaxCLL` is suspiciously low.
    #[must_use]
    pub fn recommended_peak_nits(&self) -> f32 {
        if self.max_cll > self.max_fall {
            self.max_cll
        } else {
            self.max_fall * 2.0
        }
    }

    /// Check whether the CLL values are plausible.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.max_cll >= 0.0 && self.max_fall >= 0.0 && self.max_cll >= self.max_fall
    }
}

impl Default for ContentLightLevel {
    fn default() -> Self {
        Self::hdr10_1000()
    }
}

// --------------------------------------------------------------------------
// Combined HDR10 static metadata
// --------------------------------------------------------------------------

/// Combined static HDR metadata for HDR10 streams.
///
/// Contains both SMPTE ST 2086 mastering display information and CTA-861-G
/// content light levels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hdr10StaticMetadata {
    /// Mastering display colour volume (SMPTE ST 2086).
    pub mastering: MasteringDisplayMetadata,
    /// Content light level (CTA-861-G / CTA-861.HDR).
    pub cll: ContentLightLevel,
}

impl Hdr10StaticMetadata {
    /// Create static metadata from individual components.
    #[must_use]
    pub const fn new(mastering: MasteringDisplayMetadata, cll: ContentLightLevel) -> Self {
        Self { mastering, cll }
    }

    /// Create a typical HDR10 configuration (P3-D65 mastering, 1000 nit CLL).
    #[must_use]
    pub fn typical_hdr10() -> Self {
        Self {
            mastering: MasteringDisplayMetadata::hdr10_p3_d65(),
            cll: ContentLightLevel::hdr10_1000(),
        }
    }

    /// Estimate the peak luminance to use for tone-mapping, in nits.
    ///
    /// Uses the `MaxCLL` from CLL if available and higher than `min_nits`,
    /// otherwise uses the mastering display max luminance.
    #[must_use]
    pub fn peak_luminance_for_tonemapping(&self) -> f32 {
        let cll_peak = self.cll.recommended_peak_nits();
        let master_peak = self.mastering.max_luminance;

        // Prefer CLL peak; fall back to mastering display peak
        if cll_peak > 0.0 && cll_peak <= master_peak {
            cll_peak
        } else if master_peak > 0.0 {
            master_peak
        } else {
            1000.0 // Sensible HDR10 default
        }
    }
}

// --------------------------------------------------------------------------
// HDR10+ dynamic metadata (SMPTE ST 2094-40) – stub
// --------------------------------------------------------------------------

/// HDR10+ dynamic metadata payload (SMPTE ST 2094-40).
///
/// HDR10+ extends HDR10 by adding per-scene/per-frame dynamic metadata that
/// allows displays to adjust their tone-mapping curve on a shot-by-shot basis.
///
/// This struct carries the raw byte payload as well as decoded per-frame
/// luminance parameters for convenience. Full decode is left to downstream
/// applications that implement the SMPTE ST 2094-40 algorithm.
///
/// # Standards
///
/// - SMPTE ST 2094-40:2021 "Dynamic Metadata for Color Volume Transform — Application #4"
/// - Samsung HDR10+ specification
#[derive(Clone, Debug, PartialEq)]
pub struct Hdr10PlusDynamicMetadata {
    /// Raw SMPTE ST 2094-40 application metadata bytes.
    pub raw_payload: Vec<u8>,
    /// Target display maximum luminance hint for this frame/scene, in nits.
    pub targeted_system_display_max_nits: f32,
    /// Maxscl for each channel (R, G, B) – maximum scene-referred component values.
    pub maxscl: [f32; 3],
    /// Distribution of tone-mapping bezier anchors for the frame.
    pub bezier_curve_anchors: Vec<f32>,
    /// Average maxrgb for the frame – used in conjunction with `MaxSCL`.
    pub average_maxrgb: f32,
}

impl Hdr10PlusDynamicMetadata {
    /// Create an HDR10+ metadata stub with the given raw payload.
    #[must_use]
    pub fn from_raw(payload: Vec<u8>) -> Self {
        Self {
            raw_payload: payload,
            targeted_system_display_max_nits: 1000.0,
            maxscl: [1000.0, 1000.0, 1000.0],
            bezier_curve_anchors: Vec::new(),
            average_maxrgb: 200.0,
        }
    }

    /// Returns the effective peak luminance for this frame's tone mapping.
    ///
    /// This is the minimum of the targeted display max nits and the `MaxSCL`,
    /// which is the standard approach for HDR10+ tone-mapping decisions.
    #[must_use]
    pub fn effective_peak_nits(&self) -> f32 {
        let maxscl_peak = self
            .maxscl
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        self.targeted_system_display_max_nits.min(maxscl_peak)
    }
}

// --------------------------------------------------------------------------
// Dolby Vision dynamic metadata – stub
// --------------------------------------------------------------------------

/// Dolby Vision dynamic metadata (ETSI TS 103 433).
///
/// Dolby Vision uses proprietary per-frame metadata to drive the Dolby Vision
/// tone-mapping pipeline on certified displays. This struct carries the opaque
/// RPU (Reference Processing Unit) blob for pass-through and muxing purposes.
///
/// # Standards
///
/// - ETSI TS 103 433 "Dolby Vision bitstream specification"
/// - Dolby Vision Application Note: Dolby Vision Streams Within the HTTP Live Streaming Format
///
/// # Note
///
/// Dolby Vision decode/encode requires a license from Dolby Laboratories.
/// This struct is a **stub** for carrying the metadata blob without processing it.
#[derive(Clone, Debug, PartialEq)]
pub struct DolbyVisionDynamicMetadata {
    /// Dolby Vision profile number (1–9).
    pub profile: u8,
    /// Dolby Vision level (1–13).
    pub level: u8,
    /// Raw RPU (Reference Processing Unit) bytes for this access unit.
    pub rpu_data: Vec<u8>,
    /// BL (Base Layer) signal active flag.
    pub bl_signal_active: bool,
    /// EL (Enhancement Layer) signal active flag.
    pub el_signal_active: bool,
}

impl DolbyVisionDynamicMetadata {
    /// Create a stub Dolby Vision metadata entry from a raw RPU.
    ///
    /// # Arguments
    ///
    /// * `profile` – Dolby Vision profile (e.g. 5, 8, or 9 for common deliverables)
    /// * `level` – Dolby Vision level
    /// * `rpu_data` – Raw RPU bytes extracted from the bitstream
    #[must_use]
    pub fn new(profile: u8, level: u8, rpu_data: Vec<u8>) -> Self {
        Self {
            profile,
            level,
            rpu_data,
            bl_signal_active: true,
            el_signal_active: profile != 5 && profile != 8 && profile != 9,
        }
    }

    /// Returns `true` for Dolby Vision profiles that use a single-layer delivery
    /// (i.e., profiles 5, 8, and 9).
    #[must_use]
    pub fn is_single_layer(&self) -> bool {
        matches!(self.profile, 5 | 8 | 9)
    }
}

// --------------------------------------------------------------------------
// Unified dynamic metadata container
// --------------------------------------------------------------------------

/// Container for any dynamic HDR metadata type.
///
/// Allows code that handles HDR streams to work with HDR10, HDR10+, and Dolby
/// Vision metadata through a single enum.
#[derive(Clone, Debug, PartialEq)]
#[derive(Default)]
pub enum DynamicHdrMetadata {
    /// No dynamic metadata present.
    #[default]
    None,
    /// HDR10+ per-frame metadata (SMPTE ST 2094-40).
    Hdr10Plus(Hdr10PlusDynamicMetadata),
    /// Dolby Vision RPU blob.
    DolbyVision(DolbyVisionDynamicMetadata),
}

impl DynamicHdrMetadata {
    /// Returns `true` if this container has no dynamic metadata.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Returns the effective peak nits for tone-mapping if known.
    ///
    /// Returns `None` if no dynamic metadata is present or the type does not
    /// provide this information.
    #[must_use]
    pub fn effective_peak_nits(&self) -> Option<f32> {
        match self {
            Self::Hdr10Plus(m) => Some(m.effective_peak_nits()),
            Self::DolbyVision(_) | Self::None => None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mastering_display_valid() {
        let md = MasteringDisplayMetadata::hdr10_p3_d65();
        assert!(md.is_valid());
        assert!(md.max_luminance > md.min_luminance);
    }

    #[test]
    fn test_mastering_display_dynamic_range() {
        let md = MasteringDisplayMetadata::hdr10_p3_d65();
        let stops = md.dynamic_range_stops().expect("dynamic range stops should be available");
        // 1000 / 0.005 ≈ 200,000, log2(200000) ≈ 17.6 stops
        assert!(stops > 15.0 && stops < 20.0);
    }

    #[test]
    fn test_content_light_level_valid() {
        let cll = ContentLightLevel::hdr10_1000();
        assert!(cll.is_valid());
        assert_eq!(cll.max_cll, 1000.0);
        assert_eq!(cll.max_fall, 400.0);
    }

    #[test]
    fn test_content_light_level_recommended_peak() {
        let cll = ContentLightLevel::new(800.0, 300.0);
        assert_eq!(cll.recommended_peak_nits(), 800.0);
    }

    #[test]
    fn test_hdr10_static_metadata() {
        let md = Hdr10StaticMetadata::typical_hdr10();
        let peak = md.peak_luminance_for_tonemapping();
        assert!(peak > 0.0 && peak <= 1000.0);
    }

    #[test]
    fn test_hdr10plus_effective_peak() {
        let m = Hdr10PlusDynamicMetadata {
            raw_payload: Vec::new(),
            targeted_system_display_max_nits: 600.0,
            maxscl: [800.0, 700.0, 600.0],
            bezier_curve_anchors: Vec::new(),
            average_maxrgb: 150.0,
        };
        // Min(600, max(800,700,600)=800) → 600
        assert!((m.effective_peak_nits() - 600.0).abs() < 1e-3);
    }

    #[test]
    fn test_dolby_vision_single_layer() {
        let dv = DolbyVisionDynamicMetadata::new(8, 6, Vec::new());
        assert!(dv.is_single_layer());

        let dv2 = DolbyVisionDynamicMetadata::new(7, 6, Vec::new());
        assert!(!dv2.is_single_layer());
    }

    #[test]
    fn test_dynamic_hdr_metadata_none() {
        let m = DynamicHdrMetadata::None;
        assert!(m.is_none());
        assert!(m.effective_peak_nits().is_none());
    }
}
