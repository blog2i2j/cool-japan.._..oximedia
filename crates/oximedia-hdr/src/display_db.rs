//! HDR display capability database.
//!
//! Provides [`HdrDisplayCapability`] for representing the HDR capabilities of
//! a physical display, and [`DisplayDb`] for managing a registry of display
//! profiles.

// HDR10 minimum requirements (per CTA-861-G):
/// Minimum peak luminance for HDR10 compliance (nits).
const HDR10_MIN_PEAK_NITS: f32 = 400.0;
/// Maximum black level for HDR10 compliance (nits).
const HDR10_MAX_BLACK_NITS: f32 = 0.05;

// ── HdrDisplayCapability ──────────────────────────────────────────────────────

/// Describes the HDR capabilities of a single display device.
///
/// # Example
///
/// ```rust
/// use oximedia_hdr::display_db::HdrDisplayCapability;
///
/// let display = HdrDisplayCapability::new("OLED Pro 4K", 1000.0, 0.0001);
/// assert!(display.is_hdr10_capable());
/// ```
#[derive(Debug, Clone)]
pub struct HdrDisplayCapability {
    /// Manufacturer/model name of the display.
    pub name: String,
    /// Peak luminance in nits (cd/m²).
    pub max_nits: f32,
    /// Minimum (black level) luminance in nits.
    pub min_nits: f32,
}

impl HdrDisplayCapability {
    /// Create a new `HdrDisplayCapability`.
    ///
    /// * `name`     — display name or model identifier.
    /// * `max_nits` — peak luminance in nits.  Clamped to ≥ 0.
    /// * `min_nits` — black level in nits.  Clamped to ≥ 0.
    #[must_use]
    pub fn new(name: &str, max_nits: f32, min_nits: f32) -> Self {
        Self {
            name: name.to_owned(),
            max_nits: max_nits.max(0.0),
            min_nits: min_nits.max(0.0),
        }
    }

    /// Returns `true` if this display meets the minimum HDR10 requirements:
    /// - Peak luminance ≥ 400 nits.
    /// - Black level ≤ 0.05 nits.
    #[must_use]
    pub fn is_hdr10_capable(&self) -> bool {
        self.max_nits >= HDR10_MIN_PEAK_NITS && self.min_nits <= HDR10_MAX_BLACK_NITS
    }

    /// Returns `true` if this display meets Dolby Vision requirements:
    /// - Peak luminance ≥ 1 000 nits.
    /// - Black level ≤ 0.005 nits.
    #[must_use]
    pub fn is_dolby_vision_capable(&self) -> bool {
        self.max_nits >= 1_000.0 && self.min_nits <= 0.005
    }

    /// Compute the contrast ratio (max_nits / min_nits).
    ///
    /// Returns `None` if `min_nits == 0` (infinite contrast).
    #[must_use]
    pub fn contrast_ratio(&self) -> Option<f32> {
        if self.min_nits == 0.0 {
            None
        } else {
            Some(self.max_nits / self.min_nits)
        }
    }

    /// Returns a pre-built reference SDR display (100 nit peak, 0.05 nit black).
    #[must_use]
    pub fn sdr_reference() -> Self {
        Self::new("SDR Reference (BT.1886)", 100.0, 0.05)
    }

    /// Returns a pre-built HDR10 entry-level display (400 nit, 0.04 nit black).
    #[must_use]
    pub fn hdr10_entry_level() -> Self {
        Self::new("HDR10 Entry-Level", 400.0, 0.04)
    }

    /// Returns a pre-built high-end OLED profile (1000 nit, 0.0001 nit black).
    #[must_use]
    pub fn oled_premium() -> Self {
        Self::new("OLED Premium", 1_000.0, 0.0001)
    }
}

// ── DisplayDb ─────────────────────────────────────────────────────────────────

/// A registry of [`HdrDisplayCapability`] profiles.
///
/// Provides lookup by name and filtering by HDR capability level.
#[derive(Debug, Default)]
pub struct DisplayDb {
    profiles: Vec<HdrDisplayCapability>,
}

impl DisplayDb {
    /// Create an empty display database.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a pre-populated database with common reference profiles.
    #[must_use]
    pub fn with_reference_profiles() -> Self {
        let mut db = Self::new();
        db.add(HdrDisplayCapability::sdr_reference());
        db.add(HdrDisplayCapability::hdr10_entry_level());
        db.add(HdrDisplayCapability::oled_premium());
        db
    }

    /// Add a display profile.
    pub fn add(&mut self, profile: HdrDisplayCapability) {
        self.profiles.push(profile);
    }

    /// Look up a profile by exact name.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&HdrDisplayCapability> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Return all HDR10-capable profiles.
    #[must_use]
    pub fn hdr10_capable(&self) -> Vec<&HdrDisplayCapability> {
        self.profiles.iter().filter(|p| p.is_hdr10_capable()).collect()
    }

    /// Return the number of profiles in the database.
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Return `true` if the database contains no profiles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_capability() {
        let d = HdrDisplayCapability::new("Test", 500.0, 0.01);
        assert_eq!(d.name, "Test");
        assert_eq!(d.max_nits, 500.0);
        assert_eq!(d.min_nits, 0.01);
    }

    #[test]
    fn test_negative_nits_clamped() {
        let d = HdrDisplayCapability::new("X", -10.0, -1.0);
        assert_eq!(d.max_nits, 0.0);
        assert_eq!(d.min_nits, 0.0);
    }

    #[test]
    fn test_hdr10_capable_true() {
        let d = HdrDisplayCapability::new("HDR Display", 600.0, 0.02);
        assert!(d.is_hdr10_capable());
    }

    #[test]
    fn test_hdr10_capable_false_low_peak() {
        let d = HdrDisplayCapability::new("SDR Display", 200.0, 0.02);
        assert!(!d.is_hdr10_capable());
    }

    #[test]
    fn test_hdr10_capable_false_high_black() {
        let d = HdrDisplayCapability::new("Bright LCD", 600.0, 0.1);
        assert!(!d.is_hdr10_capable());
    }

    #[test]
    fn test_dolby_vision_capable() {
        let d = HdrDisplayCapability::new("OLED", 2000.0, 0.001);
        assert!(d.is_dolby_vision_capable());
    }

    #[test]
    fn test_dolby_vision_not_capable() {
        let d = HdrDisplayCapability::hdr10_entry_level();
        assert!(!d.is_dolby_vision_capable());
    }

    #[test]
    fn test_contrast_ratio_finite() {
        let d = HdrDisplayCapability::new("X", 1000.0, 0.01);
        let cr = d.contrast_ratio().expect("should have finite contrast");
        assert!((cr - 100_000.0).abs() < 1.0);
    }

    #[test]
    fn test_contrast_ratio_none_for_zero_black() {
        let d = HdrDisplayCapability::new("Perfect OLED", 1000.0, 0.0);
        assert!(d.contrast_ratio().is_none());
    }

    #[test]
    fn test_preset_sdr_reference_not_hdr10() {
        assert!(!HdrDisplayCapability::sdr_reference().is_hdr10_capable());
    }

    #[test]
    fn test_preset_hdr10_entry_level_is_hdr10() {
        assert!(HdrDisplayCapability::hdr10_entry_level().is_hdr10_capable());
    }

    #[test]
    fn test_preset_oled_premium_hdr10_and_dv() {
        let d = HdrDisplayCapability::oled_premium();
        assert!(d.is_hdr10_capable());
        assert!(d.is_dolby_vision_capable());
    }

    #[test]
    fn test_display_db_empty() {
        let db = DisplayDb::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_display_db_add_and_find() {
        let mut db = DisplayDb::new();
        db.add(HdrDisplayCapability::new("Sony X90", 950.0, 0.003));
        let found = db.find("Sony X90").expect("should find");
        assert_eq!(found.max_nits, 950.0);
    }

    #[test]
    fn test_display_db_find_missing() {
        let db = DisplayDb::new();
        assert!(db.find("Ghost Display").is_none());
    }

    #[test]
    fn test_display_db_hdr10_capable_filter() {
        let db = DisplayDb::with_reference_profiles();
        let hdr10 = db.hdr10_capable();
        // Entry-level and OLED premium should qualify; SDR reference should not
        assert!(hdr10.len() >= 2);
        assert!(hdr10.iter().all(|d| d.is_hdr10_capable()));
    }

    #[test]
    fn test_display_db_with_reference_profiles_count() {
        let db = DisplayDb::with_reference_profiles();
        assert_eq!(db.len(), 3);
    }
}
