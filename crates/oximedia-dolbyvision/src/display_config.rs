//! Dolby Vision display configuration profiles
//!
//! Describes target display characteristics used for Dolby Vision content
//! mapping, including peak luminance, color gamut, and viewing environment
//! parameters.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Viewing environment ambient light condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmbientLight {
    /// Dark room (cinema-like, < 5 lux).
    Dark,
    /// Dim room (typical living room, 5–50 lux).
    Dim,
    /// Bright room (office / daylight, > 50 lux).
    Bright,
}

impl AmbientLight {
    /// Typical ambient luminance in lux.
    #[must_use]
    pub const fn typical_lux(&self) -> u32 {
        match self {
            Self::Dark => 0,
            Self::Dim => 20,
            Self::Bright => 100,
        }
    }

    /// Surround luminance adaptation factor (0.0–1.0).
    ///
    /// Higher values indicate a brighter surround requiring more
    /// aggressive tone mapping.
    #[must_use]
    pub fn adaptation_factor(&self) -> f32 {
        match self {
            Self::Dark => 0.0,
            Self::Dim => 0.4,
            Self::Bright => 0.8,
        }
    }
}

/// Display panel technology affecting rendering behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelTechnology {
    /// OLED (per-pixel emissive, true black).
    Oled,
    /// LCD with direct-lit backlight (local dimming zones).
    LcdDirectLit,
    /// LCD with edge-lit backlight.
    LcdEdgeLit,
    /// Micro-LED (emissive, modular).
    MicroLed,
    /// Projector (front or rear projection).
    Projector,
}

impl PanelTechnology {
    /// Whether this technology can achieve true black (0 nits).
    #[must_use]
    pub const fn true_black(&self) -> bool {
        matches!(self, Self::Oled | Self::MicroLed)
    }

    /// Typical minimum luminance in nits.
    #[must_use]
    pub fn typical_min_nits(&self) -> f32 {
        match self {
            Self::Oled | Self::MicroLed => 0.0001,
            Self::LcdDirectLit => 0.02,
            Self::LcdEdgeLit => 0.05,
            Self::Projector => 0.005,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Oled => "OLED",
            Self::LcdDirectLit => "LCD Direct-Lit",
            Self::LcdEdgeLit => "LCD Edge-Lit",
            Self::MicroLed => "Micro-LED",
            Self::Projector => "Projector",
        }
    }
}

/// Color gamut capability of the target display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayGamut {
    /// BT.709 / sRGB (standard dynamic range).
    Bt709,
    /// DCI-P3 (wide color gamut).
    DciP3,
    /// BT.2020 (ultra-wide color gamut).
    Bt2020,
}

impl DisplayGamut {
    /// Approximate coverage of BT.2020 gamut as a percentage (0–100).
    #[must_use]
    pub const fn bt2020_coverage_pct(&self) -> u32 {
        match self {
            Self::Bt709 => 69,
            Self::DciP3 => 86,
            Self::Bt2020 => 100,
        }
    }

    /// Whether gamut mapping is needed when targeting this display from BT.2020 source.
    #[must_use]
    pub const fn needs_gamut_mapping(&self) -> bool {
        !matches!(self, Self::Bt2020)
    }
}

/// Complete display configuration describing a target display's capabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayConfig {
    /// Display identifier / name.
    pub name: String,
    /// Peak luminance capability in nits.
    pub peak_luminance: f32,
    /// Minimum luminance capability in nits.
    pub min_luminance: f32,
    /// Display color gamut.
    pub gamut: DisplayGamut,
    /// Panel technology.
    pub panel: PanelTechnology,
    /// Viewing environment.
    pub ambient: AmbientLight,
    /// Diagonal screen size in inches (0 = unknown).
    pub screen_size_inches: f32,
}

impl DisplayConfig {
    /// Create a new display configuration.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        peak_luminance: f32,
        min_luminance: f32,
        gamut: DisplayGamut,
        panel: PanelTechnology,
        ambient: AmbientLight,
    ) -> Self {
        Self {
            name: name.into(),
            peak_luminance,
            min_luminance,
            gamut,
            panel,
            ambient,
            screen_size_inches: 0.0,
        }
    }

    /// Set screen size.
    pub fn with_screen_size(mut self, inches: f32) -> Self {
        self.screen_size_inches = inches;
        self
    }

    /// Dynamic range in stops.
    #[must_use]
    pub fn dynamic_range_stops(&self) -> f32 {
        if self.min_luminance <= 0.0 || self.peak_luminance <= 0.0 {
            return 0.0;
        }
        (self.peak_luminance / self.min_luminance).log2()
    }

    /// Contrast ratio (peak / min luminance).
    #[must_use]
    pub fn contrast_ratio(&self) -> f64 {
        if self.min_luminance <= 0.0 {
            return f64::INFINITY;
        }
        f64::from(self.peak_luminance) / f64::from(self.min_luminance)
    }

    /// Whether this display qualifies as HDR (peak > 600 nits and wide gamut).
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.peak_luminance > 600.0
            && matches!(self.gamut, DisplayGamut::DciP3 | DisplayGamut::Bt2020)
    }

    /// Whether this display qualifies as Dolby Vision capable
    /// (HDR + peak >= 1000 nits).
    #[must_use]
    pub fn is_dolby_vision_capable(&self) -> bool {
        self.is_hdr() && self.peak_luminance >= 1000.0
    }

    /// Validate the display configuration.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.peak_luminance <= 0.0 {
            errors.push(format!(
                "peak_luminance must be > 0, got {}",
                self.peak_luminance
            ));
        }
        if self.min_luminance < 0.0 {
            errors.push(format!(
                "min_luminance must be >= 0, got {}",
                self.min_luminance
            ));
        }
        if self.peak_luminance <= self.min_luminance {
            errors.push(format!(
                "peak_luminance ({}) must be > min_luminance ({})",
                self.peak_luminance, self.min_luminance
            ));
        }
        if self.peak_luminance > 10000.0 {
            errors.push(format!(
                "peak_luminance ({}) exceeds PQ limit of 10000 nits",
                self.peak_luminance
            ));
        }
        errors
    }

    /// Standard SDR reference monitor (100 nits, BT.709).
    #[must_use]
    pub fn sdr_reference() -> Self {
        Self::new(
            "SDR Reference",
            100.0,
            0.05,
            DisplayGamut::Bt709,
            PanelTechnology::LcdDirectLit,
            AmbientLight::Dim,
        )
    }

    /// Dolby Cinema reference (4000 nits, P3, OLED).
    #[must_use]
    pub fn dolby_cinema() -> Self {
        Self::new(
            "Dolby Cinema",
            4000.0,
            0.0001,
            DisplayGamut::DciP3,
            PanelTechnology::Oled,
            AmbientLight::Dark,
        )
    }

    /// Consumer OLED TV (1000 nits, P3, dim room).
    #[must_use]
    pub fn consumer_oled() -> Self {
        Self::new(
            "Consumer OLED",
            1000.0,
            0.0001,
            DisplayGamut::DciP3,
            PanelTechnology::Oled,
            AmbientLight::Dim,
        )
        .with_screen_size(65.0)
    }

    /// Consumer LCD TV (600 nits, P3, dim room).
    #[must_use]
    pub fn consumer_lcd() -> Self {
        Self::new(
            "Consumer LCD",
            600.0,
            0.05,
            DisplayGamut::DciP3,
            PanelTechnology::LcdDirectLit,
            AmbientLight::Dim,
        )
        .with_screen_size(55.0)
    }

    /// Mobile device (700 nits, P3, bright room).
    #[must_use]
    pub fn mobile_device() -> Self {
        Self::new(
            "Mobile",
            700.0,
            0.002,
            DisplayGamut::DciP3,
            PanelTechnology::Oled,
            AmbientLight::Bright,
        )
        .with_screen_size(6.7)
    }
}

/// A registry of known display configurations for content mapping.
#[derive(Debug, Default)]
pub struct DisplayConfigRegistry {
    /// All registered display configs.
    configs: Vec<DisplayConfig>,
}

impl DisplayConfigRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-populated with standard display profiles.
    #[must_use]
    pub fn with_standard_profiles() -> Self {
        let mut reg = Self::new();
        reg.add(DisplayConfig::sdr_reference());
        reg.add(DisplayConfig::dolby_cinema());
        reg.add(DisplayConfig::consumer_oled());
        reg.add(DisplayConfig::consumer_lcd());
        reg.add(DisplayConfig::mobile_device());
        reg
    }

    /// Register a display configuration.
    pub fn add(&mut self, config: DisplayConfig) {
        self.configs.push(config);
    }

    /// Number of registered configurations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// Find a config by name.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&DisplayConfig> {
        self.configs.iter().find(|c| c.name == name)
    }

    /// Find the config whose peak luminance is closest to the given value.
    #[must_use]
    pub fn find_closest_peak(&self, target_nits: f32) -> Option<&DisplayConfig> {
        self.configs.iter().min_by(|a, b| {
            let da = (a.peak_luminance - target_nits).abs();
            let db = (b.peak_luminance - target_nits).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// All Dolby Vision capable displays.
    #[must_use]
    pub fn dolby_vision_capable(&self) -> Vec<&DisplayConfig> {
        self.configs
            .iter()
            .filter(|c| c.is_dolby_vision_capable())
            .collect()
    }

    /// All HDR displays.
    #[must_use]
    pub fn hdr_displays(&self) -> Vec<&DisplayConfig> {
        self.configs.iter().filter(|c| c.is_hdr()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ambient_light_typical_lux() {
        assert_eq!(AmbientLight::Dark.typical_lux(), 0);
        assert_eq!(AmbientLight::Dim.typical_lux(), 20);
        assert_eq!(AmbientLight::Bright.typical_lux(), 100);
    }

    #[test]
    fn test_ambient_light_adaptation_factor() {
        assert!((AmbientLight::Dark.adaptation_factor() - 0.0).abs() < 0.01);
        assert!(AmbientLight::Dim.adaptation_factor() > 0.0);
        assert!(AmbientLight::Bright.adaptation_factor() > AmbientLight::Dim.adaptation_factor());
    }

    #[test]
    fn test_panel_technology_true_black() {
        assert!(PanelTechnology::Oled.true_black());
        assert!(PanelTechnology::MicroLed.true_black());
        assert!(!PanelTechnology::LcdDirectLit.true_black());
        assert!(!PanelTechnology::LcdEdgeLit.true_black());
        assert!(!PanelTechnology::Projector.true_black());
    }

    #[test]
    fn test_panel_technology_label() {
        assert_eq!(PanelTechnology::Oled.label(), "OLED");
        assert_eq!(PanelTechnology::LcdDirectLit.label(), "LCD Direct-Lit");
    }

    #[test]
    fn test_display_gamut_bt2020_coverage() {
        assert!(
            DisplayGamut::Bt709.bt2020_coverage_pct() < DisplayGamut::DciP3.bt2020_coverage_pct()
        );
        assert_eq!(DisplayGamut::Bt2020.bt2020_coverage_pct(), 100);
    }

    #[test]
    fn test_display_gamut_needs_mapping() {
        assert!(DisplayGamut::Bt709.needs_gamut_mapping());
        assert!(DisplayGamut::DciP3.needs_gamut_mapping());
        assert!(!DisplayGamut::Bt2020.needs_gamut_mapping());
    }

    #[test]
    fn test_display_config_sdr_reference() {
        let d = DisplayConfig::sdr_reference();
        assert!(!d.is_hdr());
        assert!(!d.is_dolby_vision_capable());
        assert!(d.validate().is_empty());
    }

    #[test]
    fn test_display_config_dolby_cinema() {
        let d = DisplayConfig::dolby_cinema();
        assert!(d.is_hdr());
        assert!(d.is_dolby_vision_capable());
        assert!(d.validate().is_empty());
    }

    #[test]
    fn test_display_config_consumer_oled() {
        let d = DisplayConfig::consumer_oled();
        assert!(d.is_hdr());
        assert!(d.is_dolby_vision_capable());
        assert!((d.screen_size_inches - 65.0).abs() < 0.1);
    }

    #[test]
    fn test_display_config_dynamic_range_stops() {
        let d = DisplayConfig::dolby_cinema();
        let stops = d.dynamic_range_stops();
        // log2(4000 / 0.0001) ~ 25.3
        assert!(stops > 24.0 && stops < 26.0, "stops={stops}");
    }

    #[test]
    fn test_display_config_contrast_ratio() {
        let d = DisplayConfig::consumer_lcd();
        let cr = d.contrast_ratio();
        // 600 / 0.05 = 12000
        assert!((cr - 12000.0).abs() < 1.0, "cr={cr}");
    }

    #[test]
    fn test_display_config_validate_bad() {
        let d = DisplayConfig::new(
            "Bad",
            -1.0,
            0.01,
            DisplayGamut::Bt709,
            PanelTechnology::LcdEdgeLit,
            AmbientLight::Dim,
        );
        let errs = d.validate();
        assert!(!errs.is_empty());
    }

    #[test]
    fn test_display_config_registry_standard() {
        let reg = DisplayConfigRegistry::with_standard_profiles();
        assert_eq!(reg.len(), 5);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_display_config_registry_find_by_name() {
        let reg = DisplayConfigRegistry::with_standard_profiles();
        assert!(reg.find_by_name("Dolby Cinema").is_some());
        assert!(reg.find_by_name("Not A Display").is_none());
    }

    #[test]
    fn test_display_config_registry_find_closest_peak() {
        let reg = DisplayConfigRegistry::with_standard_profiles();
        let closest = reg
            .find_closest_peak(900.0)
            .expect("closest should be valid");
        assert_eq!(closest.name, "Consumer OLED");
    }

    #[test]
    fn test_display_config_registry_dv_capable() {
        let reg = DisplayConfigRegistry::with_standard_profiles();
        let dv = reg.dolby_vision_capable();
        // Dolby Cinema (4000) and Consumer OLED (1000) should qualify
        assert!(dv.len() >= 2, "dv_capable={}", dv.len());
    }

    #[test]
    fn test_display_config_mobile() {
        let d = DisplayConfig::mobile_device();
        assert!((d.screen_size_inches - 6.7).abs() < 0.1);
        assert_eq!(d.ambient, AmbientLight::Bright);
        assert!(d.is_hdr());
    }

    #[test]
    fn test_display_config_registry_hdr_displays() {
        let reg = DisplayConfigRegistry::with_standard_profiles();
        let hdr = reg.hdr_displays();
        // SDR Reference (100 nits, BT.709) should NOT be HDR
        for d in &hdr {
            assert!(d.is_hdr(), "Expected HDR: {}", d.name);
        }
    }
}
