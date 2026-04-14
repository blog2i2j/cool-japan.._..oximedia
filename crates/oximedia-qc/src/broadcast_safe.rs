//! Broadcast safe level checking for video frames.
//!
//! This module checks that video pixel values conform to broadcast
//! specifications (NTSC, PAL, HD) for luma and chroma levels.

/// Configuration for broadcast safe checking.
#[derive(Debug, Clone)]
pub struct BroadcastSafeConfig {
    /// Maximum allowed luma value (Y channel).
    pub max_luma: u8,
    /// Minimum allowed luma value (Y channel).
    pub min_luma: u8,
    /// Maximum allowed chroma value (Cb/Cr channels).
    pub max_chroma: u8,
    /// Whether to apply composite-safe constraints (more restrictive).
    pub composite_safe: bool,
}

impl BroadcastSafeConfig {
    /// NTSC broadcast safe levels: luma 16–235, chroma 16–240.
    #[must_use]
    pub fn ntsc() -> Self {
        Self {
            max_luma: 235,
            min_luma: 16,
            max_chroma: 240,
            composite_safe: true,
        }
    }

    /// PAL broadcast safe levels: luma 16–235, chroma 16–240.
    #[must_use]
    pub fn pal() -> Self {
        Self {
            max_luma: 235,
            min_luma: 16,
            max_chroma: 240,
            composite_safe: true,
        }
    }

    /// HD broadcast safe levels: luma 16–235, chroma 16–240.
    #[must_use]
    pub fn hd() -> Self {
        Self {
            max_luma: 235,
            min_luma: 16,
            max_chroma: 240,
            composite_safe: false,
        }
    }

    /// Full range (no restrictions).
    #[must_use]
    pub fn full_range() -> Self {
        Self {
            max_luma: 255,
            min_luma: 0,
            max_chroma: 255,
            composite_safe: false,
        }
    }
}

impl Default for BroadcastSafeConfig {
    fn default() -> Self {
        Self::ntsc()
    }
}

/// The type of pixel level violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Luma value exceeds maximum.
    LumaAbove,
    /// Luma value is below minimum.
    LumaBelow,
    /// Chroma value exceeds maximum.
    ChromaAbove,
    /// Chroma value is below minimum (16 for broadcast).
    ChromaBelow,
}

impl ViolationType {
    /// Returns a human-readable description of the violation type.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::LumaAbove => "Luma exceeds maximum",
            Self::LumaBelow => "Luma below minimum",
            Self::ChromaAbove => "Chroma exceeds maximum",
            Self::ChromaBelow => "Chroma below minimum",
        }
    }
}

/// A pixel that violates broadcast safe levels.
#[derive(Debug, Clone)]
pub struct PixelViolation {
    /// Frame index where the violation occurred.
    pub frame: u64,
    /// X coordinate of the violating pixel.
    pub x: u32,
    /// Y coordinate of the violating pixel.
    pub y: u32,
    /// Type of violation.
    pub violation_type: ViolationType,
    /// The actual pixel value that violated the constraint.
    pub value: u8,
}

impl PixelViolation {
    /// Creates a new pixel violation.
    #[must_use]
    pub fn new(frame: u64, x: u32, y: u32, violation_type: ViolationType, value: u8) -> Self {
        Self {
            frame,
            x,
            y,
            violation_type,
            value,
        }
    }
}

/// Checker for broadcast safe violations in video frames.
#[derive(Debug, Clone, Default)]
pub struct BroadcastSafeChecker;

impl BroadcastSafeChecker {
    /// Creates a new broadcast safe checker.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Checks a single planar YUV frame for broadcast safe violations.
    ///
    /// `frame` is expected to be a planar buffer in Y-then-U-then-V order,
    /// where the Y plane is `width * height` bytes and U/V planes are each
    /// `(width/2) * (height/2)` bytes (4:2:0 subsampling).
    ///
    /// Returns a list of violations found in this frame.
    #[must_use]
    pub fn check_frame(
        frame: &[u8],
        width: u32,
        height: u32,
        frame_idx: u64,
        config: &BroadcastSafeConfig,
    ) -> Vec<PixelViolation> {
        let mut violations = Vec::new();
        let luma_size = (width * height) as usize;

        // Check luma plane (Y)
        let y_plane = &frame[..luma_size.min(frame.len())];
        for (i, &y) in y_plane.iter().enumerate() {
            let x = (i as u32) % width;
            let row = (i as u32) / width;
            if y > config.max_luma {
                violations.push(PixelViolation::new(
                    frame_idx,
                    x,
                    row,
                    ViolationType::LumaAbove,
                    y,
                ));
            } else if y < config.min_luma {
                violations.push(PixelViolation::new(
                    frame_idx,
                    x,
                    row,
                    ViolationType::LumaBelow,
                    y,
                ));
            }
        }

        // Check chroma planes (Cb, Cr) if present in the buffer
        let chroma_size = ((width / 2) * (height / 2)) as usize;
        let chroma_start = luma_size;
        let chroma_end = (chroma_start + chroma_size * 2).min(frame.len());

        if chroma_start < frame.len() {
            let chroma_plane = &frame[chroma_start..chroma_end];
            let chroma_w = width / 2;

            for (i, &c) in chroma_plane.iter().enumerate() {
                let x = (i as u32) % chroma_w;
                let row = (i as u32) / chroma_w;
                if c > config.max_chroma {
                    violations.push(PixelViolation::new(
                        frame_idx,
                        x,
                        row,
                        ViolationType::ChromaAbove,
                        c,
                    ));
                } else if c < config.min_luma {
                    // Chroma minimum is same as luma minimum (16) for broadcast
                    violations.push(PixelViolation::new(
                        frame_idx,
                        x,
                        row,
                        ViolationType::ChromaBelow,
                        c,
                    ));
                }
            }
        }

        violations
    }
}

/// Summary report for broadcast safe checking across all frames.
#[derive(Debug, Clone, Default)]
pub struct BroadcastSafeReport {
    /// Total number of frames analyzed.
    pub total_frames: u64,
    /// Number of frames with at least one violation.
    pub violating_frames: u64,
    /// Total number of pixel violations.
    pub total_violations: u64,
    /// Index of the frame with the most violations (if any).
    pub worst_frame: Option<u64>,
}

impl BroadcastSafeReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether any violations were found.
    #[must_use]
    pub fn has_violations(&self) -> bool {
        self.total_violations > 0
    }

    /// Returns the violation rate as a fraction of frames.
    #[must_use]
    pub fn violation_rate(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.violating_frames as f64 / self.total_frames as f64
    }

    /// Builds a report from per-frame violation lists.
    #[must_use]
    pub fn from_frame_violations(violations_per_frame: &[(u64, Vec<PixelViolation>)]) -> Self {
        let total_frames = violations_per_frame.len() as u64;
        let mut total_violations = 0u64;
        let mut violating_frames = 0u64;
        let mut worst_count = 0usize;
        let mut worst_frame = None;

        for (frame_idx, violations) in violations_per_frame {
            let count = violations.len();
            total_violations += count as u64;
            if count > 0 {
                violating_frames += 1;
                if count > worst_count {
                    worst_count = count;
                    worst_frame = Some(*frame_idx);
                }
            }
        }

        Self {
            total_frames,
            violating_frames,
            total_violations,
            worst_frame,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Region-specific broadcast standards
// ─────────────────────────────────────────────────────────────────────────────

/// Broadcast region / standard, used to derive region-specific color-space
/// constraints alongside pixel-level checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastRegion {
    /// NTSC: North America, Japan, parts of South America.
    /// Color space: SMPTE 170M / BT.601 (525-line).
    Ntsc,
    /// PAL: Western Europe, Australia, Africa, most of Asia.
    /// Color space: BT.601 (625-line).
    Pal,
    /// SECAM: France, Eastern Europe, Russia, parts of Africa & Middle East.
    /// Analog FM color encoding — digital equivalents use BT.601 625-line.
    Secam,
    /// HD (BT.709): all modern HD broadcast systems.
    Hd,
    /// UHD / 4K (BT.2020): ultra-high-definition delivery.
    Uhd,
}

impl BroadcastRegion {
    /// Returns the canonical color-space name for this region.
    #[must_use]
    pub const fn color_space(self) -> &'static str {
        match self {
            Self::Ntsc => "bt601-525",
            Self::Pal => "bt601-625",
            Self::Secam => "bt601-625",
            Self::Hd => "bt709",
            Self::Uhd => "bt2020",
        }
    }

    /// Returns the frame-rate family associated with this region.
    ///
    /// NTSC uses 29.97 / 59.94 Hz; PAL / SECAM use 25 / 50 Hz; HD / UHD
    /// support both but the dominant broadcast rate is noted here.
    #[must_use]
    pub const fn frame_rate_hz(self) -> f32 {
        match self {
            Self::Ntsc => 29.97,
            Self::Pal | Self::Secam => 25.0,
            Self::Hd | Self::Uhd => 25.0, // 25 or 29.97; default to 25
        }
    }

    /// Returns the [`BroadcastSafeConfig`] appropriate for this region.
    ///
    /// SECAM and PAL share identical digital-domain pixel-level constraints
    /// (both use BT.601 625-line encoding), while NTSC uses the 525-line
    /// variant.  All analog-originated systems use `composite_safe = true`.
    #[must_use]
    pub fn safe_config(self) -> BroadcastSafeConfig {
        match self {
            Self::Ntsc => BroadcastSafeConfig::ntsc(),
            Self::Pal => BroadcastSafeConfig::pal(),
            Self::Secam => BroadcastSafeConfig {
                // SECAM uses BT.601 625-line; same legal-range constraints as PAL
                // but the composite_safe flag reflects that SECAM FM encoding is
                // more resilient to chroma over-modulation.
                max_luma: 235,
                min_luma: 16,
                max_chroma: 240,
                composite_safe: false, // SECAM FM encoding is less sensitive to composite artifacts
            },
            Self::Hd => BroadcastSafeConfig::hd(),
            Self::Uhd => BroadcastSafeConfig {
                max_luma: 235,
                min_luma: 16,
                max_chroma: 240,
                composite_safe: false,
            },
        }
    }

    /// Returns a human-readable display name for the region.
    #[must_use]
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Ntsc => "NTSC",
            Self::Pal => "PAL",
            Self::Secam => "SECAM",
            Self::Hd => "HD (BT.709)",
            Self::Uhd => "UHD (BT.2020)",
        }
    }
}

impl std::fmt::Display for BroadcastRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Color space mismatch detected during region-specific validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColorSpaceMismatch {
    /// The expected color space for the target region.
    pub expected: String,
    /// The color space declared in the media file.
    pub found: String,
    /// The region whose standard was checked.
    pub region: BroadcastRegion,
}

impl ColorSpaceMismatch {
    /// Creates a new mismatch record.
    #[must_use]
    pub fn new(
        expected: impl Into<String>,
        found: impl Into<String>,
        region: BroadcastRegion,
    ) -> Self {
        Self {
            expected: expected.into(),
            found: found.into(),
            region,
        }
    }
}

impl std::fmt::Display for ColorSpaceMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} requires color space `{}` but found `{}`",
            self.region, self.expected, self.found
        )
    }
}

/// Validates color space declarations against region-specific broadcast standards.
///
/// Call [`ColorSpaceCheck::check`] with the color space name declared in the
/// media container metadata.  The checker normalises common aliases
/// (e.g. `"smpte170m"` → `"bt601-525"`) before comparing.
#[derive(Debug, Clone)]
pub struct ColorSpaceCheck {
    /// Target broadcast region.
    pub region: BroadcastRegion,
}

impl ColorSpaceCheck {
    /// Creates a new checker for the given region.
    #[must_use]
    pub const fn new(region: BroadcastRegion) -> Self {
        Self { region }
    }

    /// Returns the canonical color-space identifier for the configured region.
    #[must_use]
    pub fn expected_color_space(&self) -> &'static str {
        self.region.color_space()
    }

    /// Checks whether `declared_color_space` matches the region's standard.
    ///
    /// Returns `Ok(())` on a match or `Err(ColorSpaceMismatch)` otherwise.
    ///
    /// Common aliases are recognised:
    /// - `"smpte170m"` → `"bt601-525"` (NTSC)
    /// - `"smpte170"` → `"bt601-525"` (NTSC)
    /// - `"bt601"` → `"bt601-625"` (defaults to 625-line / PAL / SECAM)
    /// - `"rec709"` → `"bt709"` (HD)
    /// - `"rec2020"` → `"bt2020"` (UHD)
    ///
    /// Comparison is case-insensitive.
    pub fn check(&self, declared_color_space: &str) -> Result<(), ColorSpaceMismatch> {
        let normalised = Self::normalise(declared_color_space);
        let expected = self.region.color_space();

        if normalised == expected {
            Ok(())
        } else {
            Err(ColorSpaceMismatch::new(
                expected,
                declared_color_space,
                self.region,
            ))
        }
    }

    /// Normalises a color-space string to a canonical identifier.
    fn normalise(cs: &str) -> String {
        match cs.to_ascii_lowercase().trim() {
            "smpte170m" | "smpte170" | "bt601-525" | "bt.601-525" | "601-525" => {
                "bt601-525".to_string()
            }
            "bt601" | "bt601-625" | "bt.601" | "bt.601-625" | "601-625" | "pal" | "secam" => {
                "bt601-625".to_string()
            }
            "bt709" | "rec709" | "rec.709" | "bt.709" | "hd" => "bt709".to_string(),
            "bt2020" | "rec2020" | "rec.2020" | "bt.2020" | "uhd" => "bt2020".to_string(),
            other => other.to_string(),
        }
    }
}

/// Region-aware broadcast checker that combines pixel-level analysis with
/// color-space validation.
#[derive(Debug, Clone)]
pub struct RegionBroadcastChecker {
    /// The target broadcast region.
    pub region: BroadcastRegion,
    /// Pixel-level safe-level configuration derived from the region.
    pub safe_config: BroadcastSafeConfig,
    /// Color-space validator for the region.
    pub color_space_check: ColorSpaceCheck,
}

impl RegionBroadcastChecker {
    /// Creates a new checker for `region`, using that region's standard pixel
    /// constraints and expected color space.
    #[must_use]
    pub fn new(region: BroadcastRegion) -> Self {
        Self {
            safe_config: region.safe_config(),
            color_space_check: ColorSpaceCheck::new(region),
            region,
        }
    }

    /// Validates pixel levels in one YUV420 frame against the region's broadcast
    /// safe constraints.  Delegates to [`BroadcastSafeChecker::check_frame`].
    #[must_use]
    pub fn check_frame(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        frame_idx: u64,
    ) -> Vec<PixelViolation> {
        BroadcastSafeChecker::check_frame(frame, width, height, frame_idx, &self.safe_config)
    }

    /// Validates the declared color space against the region's standard.
    ///
    /// Returns `Ok(())` if the color space matches or `Err(ColorSpaceMismatch)`
    /// with details of the mismatch.
    pub fn check_color_space(&self, declared: &str) -> Result<(), ColorSpaceMismatch> {
        self.color_space_check.check(declared)
    }

    /// Returns the expected color space for this checker's region.
    #[must_use]
    pub fn expected_color_space(&self) -> &'static str {
        self.region.color_space()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: u32, height: u32, y_val: u8, uv_val: u8) -> Vec<u8> {
        let y_size = (width * height) as usize;
        let uv_size = ((width / 2) * (height / 2)) as usize;
        let mut frame = vec![y_val; y_size];
        frame.extend(vec![uv_val; uv_size * 2]);
        frame
    }

    #[test]
    fn test_ntsc_config() {
        let cfg = BroadcastSafeConfig::ntsc();
        assert_eq!(cfg.max_luma, 235);
        assert_eq!(cfg.min_luma, 16);
        assert_eq!(cfg.max_chroma, 240);
        assert!(cfg.composite_safe);
    }

    #[test]
    fn test_pal_config() {
        let cfg = BroadcastSafeConfig::pal();
        assert_eq!(cfg.max_luma, 235);
        assert_eq!(cfg.min_luma, 16);
    }

    #[test]
    fn test_hd_config() {
        let cfg = BroadcastSafeConfig::hd();
        assert!(!cfg.composite_safe);
    }

    #[test]
    fn test_no_violations_in_range() {
        let cfg = BroadcastSafeConfig::ntsc();
        // All pixels at Y=128, UV=128 — within range
        let frame = make_frame(4, 4, 128, 128);
        let violations = BroadcastSafeChecker::check_frame(&frame, 4, 4, 0, &cfg);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_luma_above_violation() {
        let cfg = BroadcastSafeConfig::ntsc();
        let frame = make_frame(4, 4, 255, 128); // Y=255 exceeds 235
        let violations = BroadcastSafeChecker::check_frame(&frame, 4, 4, 0, &cfg);
        assert!(!violations.is_empty());
        assert_eq!(violations[0].violation_type, ViolationType::LumaAbove);
        assert_eq!(violations[0].value, 255);
    }

    #[test]
    fn test_luma_below_violation() {
        let cfg = BroadcastSafeConfig::ntsc();
        let frame = make_frame(4, 4, 0, 128); // Y=0 < 16
        let violations = BroadcastSafeChecker::check_frame(&frame, 4, 4, 0, &cfg);
        assert!(!violations.is_empty());
        assert!(violations
            .iter()
            .any(|v| v.violation_type == ViolationType::LumaBelow));
    }

    #[test]
    fn test_chroma_above_violation() {
        let cfg = BroadcastSafeConfig::ntsc();
        let frame = make_frame(4, 4, 128, 255); // UV=255 exceeds 240
        let violations = BroadcastSafeChecker::check_frame(&frame, 4, 4, 0, &cfg);
        assert!(violations
            .iter()
            .any(|v| v.violation_type == ViolationType::ChromaAbove));
    }

    #[test]
    fn test_chroma_below_violation() {
        let cfg = BroadcastSafeConfig::ntsc();
        let frame = make_frame(4, 4, 128, 0); // UV=0 < 16
        let violations = BroadcastSafeChecker::check_frame(&frame, 4, 4, 0, &cfg);
        assert!(violations
            .iter()
            .any(|v| v.violation_type == ViolationType::ChromaBelow));
    }

    #[test]
    fn test_frame_idx_stored_in_violation() {
        let cfg = BroadcastSafeConfig::ntsc();
        let frame = make_frame(4, 4, 255, 128);
        let violations = BroadcastSafeChecker::check_frame(&frame, 4, 4, 42, &cfg);
        assert!(!violations.is_empty());
        assert_eq!(violations[0].frame, 42);
    }

    #[test]
    fn test_broadcast_safe_report_no_violations() {
        let report = BroadcastSafeReport::from_frame_violations(&[(0, vec![]), (1, vec![])]);
        assert_eq!(report.total_frames, 2);
        assert_eq!(report.violating_frames, 0);
        assert_eq!(report.total_violations, 0);
        assert!(report.worst_frame.is_none());
        assert!(!report.has_violations());
    }

    #[test]
    fn test_broadcast_safe_report_with_violations() {
        let cfg = BroadcastSafeConfig::ntsc();
        let frame = make_frame(4, 4, 255, 128);
        let v = BroadcastSafeChecker::check_frame(&frame, 4, 4, 5, &cfg);
        let report = BroadcastSafeReport::from_frame_violations(&[(5, v)]);
        assert!(report.has_violations());
        assert_eq!(report.worst_frame, Some(5));
    }

    #[test]
    fn test_violation_rate() {
        let report = BroadcastSafeReport {
            total_frames: 10,
            violating_frames: 3,
            total_violations: 15,
            worst_frame: Some(2),
        };
        assert!((report.violation_rate() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_violation_type_description() {
        assert!(!ViolationType::LumaAbove.description().is_empty());
        assert!(!ViolationType::ChromaBelow.description().is_empty());
    }

    // ── Region-specific broadcast standard tests ────────────────────────────

    #[test]
    fn test_secam_config_not_composite_safe() {
        // SECAM FM encoding does not impose composite-safe constraints.
        let cfg = BroadcastRegion::Secam.safe_config();
        assert_eq!(cfg.max_luma, 235);
        assert_eq!(cfg.min_luma, 16);
        assert_eq!(cfg.max_chroma, 240);
        assert!(!cfg.composite_safe, "SECAM should not be composite_safe");
    }

    #[test]
    fn test_secam_color_space_is_bt601_625() {
        assert_eq!(BroadcastRegion::Secam.color_space(), "bt601-625");
    }

    #[test]
    fn test_pal_color_space_is_bt601_625() {
        assert_eq!(BroadcastRegion::Pal.color_space(), "bt601-625");
    }

    #[test]
    fn test_ntsc_color_space_is_bt601_525() {
        assert_eq!(BroadcastRegion::Ntsc.color_space(), "bt601-525");
    }

    #[test]
    fn test_hd_color_space_is_bt709() {
        assert_eq!(BroadcastRegion::Hd.color_space(), "bt709");
    }

    #[test]
    fn test_uhd_color_space_is_bt2020() {
        assert_eq!(BroadcastRegion::Uhd.color_space(), "bt2020");
    }

    #[test]
    fn test_region_display_names_are_non_empty() {
        for region in &[
            BroadcastRegion::Ntsc,
            BroadcastRegion::Pal,
            BroadcastRegion::Secam,
            BroadcastRegion::Hd,
            BroadcastRegion::Uhd,
        ] {
            assert!(!region.display_name().is_empty());
        }
    }

    #[test]
    fn test_region_display_format() {
        assert_eq!(format!("{}", BroadcastRegion::Secam), "SECAM");
        assert_eq!(format!("{}", BroadcastRegion::Ntsc), "NTSC");
        assert_eq!(format!("{}", BroadcastRegion::Pal), "PAL");
    }

    #[test]
    fn test_color_space_check_pal_correct() {
        let check = ColorSpaceCheck::new(BroadcastRegion::Pal);
        assert!(check.check("bt601-625").is_ok());
    }

    #[test]
    fn test_color_space_check_secam_accepts_pal_alias() {
        // SECAM and PAL share the same BT.601-625 color space digitally.
        let check = ColorSpaceCheck::new(BroadcastRegion::Secam);
        assert!(check.check("pal").is_ok());
        assert!(check.check("bt601-625").is_ok());
    }

    #[test]
    fn test_color_space_check_ntsc_smpte170m_alias() {
        let check = ColorSpaceCheck::new(BroadcastRegion::Ntsc);
        assert!(check.check("smpte170m").is_ok());
        assert!(check.check("smpte170").is_ok());
        assert!(check.check("bt601-525").is_ok());
    }

    #[test]
    fn test_color_space_check_hd_rec709_alias() {
        let check = ColorSpaceCheck::new(BroadcastRegion::Hd);
        assert!(check.check("rec709").is_ok());
        assert!(check.check("bt709").is_ok());
        assert!(check.check("bt.709").is_ok());
    }

    #[test]
    fn test_color_space_check_mismatch_returns_err() {
        let check = ColorSpaceCheck::new(BroadcastRegion::Hd);
        // A BT.601-625 signal delivered to an HD broadcast target is a mismatch.
        let err = check.check("bt601-625").unwrap_err();
        assert_eq!(err.expected, "bt709");
        assert_eq!(err.region, BroadcastRegion::Hd);
    }

    #[test]
    fn test_color_space_mismatch_display() {
        let mismatch = ColorSpaceMismatch::new("bt709", "bt601-625", BroadcastRegion::Hd);
        let msg = mismatch.to_string();
        assert!(msg.contains("bt709"));
        assert!(msg.contains("bt601-625"));
        assert!(msg.contains("HD"));
    }

    #[test]
    fn test_region_checker_secam_no_violation() {
        let checker = RegionBroadcastChecker::new(BroadcastRegion::Secam);
        let frame = make_frame(4, 4, 128, 128);
        let violations = checker.check_frame(&frame, 4, 4, 0);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_region_checker_secam_luma_violation() {
        let checker = RegionBroadcastChecker::new(BroadcastRegion::Secam);
        let frame = make_frame(4, 4, 240, 128); // 240 > 235
        let violations = checker.check_frame(&frame, 4, 4, 0);
        assert!(!violations.is_empty());
        assert!(violations
            .iter()
            .all(|v| v.violation_type == ViolationType::LumaAbove));
    }

    #[test]
    fn test_region_checker_secam_color_space_ok() {
        let checker = RegionBroadcastChecker::new(BroadcastRegion::Secam);
        assert!(checker.check_color_space("bt601-625").is_ok());
        assert!(checker.check_color_space("secam").is_ok());
    }

    #[test]
    fn test_region_checker_secam_color_space_mismatch() {
        let checker = RegionBroadcastChecker::new(BroadcastRegion::Secam);
        let err = checker.check_color_space("bt709").unwrap_err();
        assert_eq!(err.region, BroadcastRegion::Secam);
    }

    #[test]
    fn test_region_checker_expected_color_space() {
        let ntsc_checker = RegionBroadcastChecker::new(BroadcastRegion::Ntsc);
        assert_eq!(ntsc_checker.expected_color_space(), "bt601-525");
        let hd_checker = RegionBroadcastChecker::new(BroadcastRegion::Hd);
        assert_eq!(hd_checker.expected_color_space(), "bt709");
    }

    #[test]
    fn test_ntsc_frame_rate() {
        assert!((BroadcastRegion::Ntsc.frame_rate_hz() - 29.97_f32).abs() < 0.01);
    }

    #[test]
    fn test_pal_secam_frame_rate() {
        assert!((BroadcastRegion::Pal.frame_rate_hz() - 25.0_f32).abs() < 0.01);
        assert!((BroadcastRegion::Secam.frame_rate_hz() - 25.0_f32).abs() < 0.01);
    }
}
