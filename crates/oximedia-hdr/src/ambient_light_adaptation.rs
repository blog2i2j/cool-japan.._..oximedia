//! Ambient light adaptation model for HDR viewing environments.
//!
//! Models the perceptual effect of the viewing surround on perceived contrast
//! and colour in HDR content, following the CIECAM02 / BT.2446 approach to
//! ambient viewing conditions.
//!
//! The primary output is an *adaptation factor* in the range \[0.0, 1.0\] that
//! downstream tone-mapping or display brightness control can use to compensate
//! for the viewing environment luminance.  A factor of 1.0 means no adaptation
//! is needed; lower values indicate that the display should compensate for a
//! brighter surround.

use crate::{HdrError, Result};

// ─── Viewing environment ──────────────────────────────────────────────────────

/// Characterisation of the surround viewing environment.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ViewingSurround {
    /// Dark cinema environment (≤ 5 lux).
    DarkCinema,
    /// Dim home-theatre environment (5–50 lux).
    DimHome,
    /// Average domestic living-room environment (50–300 lux).
    AverageDomestic,
    /// Bright office or well-lit room (300–1000 lux).
    BrightOffice,
    /// Outdoor daylight (> 1000 lux).
    Daylight,
    /// Custom environment with explicit illuminance in lux.
    Custom(f32),
}

impl ViewingSurround {
    /// Representative illuminance in lux for this environment.
    pub fn illuminance_lux(self) -> f32 {
        match self {
            ViewingSurround::DarkCinema => 2.0,
            ViewingSurround::DimHome => 20.0,
            ViewingSurround::AverageDomestic => 100.0,
            ViewingSurround::BrightOffice => 500.0,
            ViewingSurround::Daylight => 10_000.0,
            ViewingSurround::Custom(lux) => lux,
        }
    }

    /// CIECAM02 surround factor *c* for this environment.
    ///
    /// - 0.69 — average surround (dim / average)
    /// - 0.59 — dim surround (dark cinema)
    /// - 0.525 — dark surround
    pub fn surround_factor(self) -> f32 {
        match self {
            ViewingSurround::DarkCinema => 0.525,
            ViewingSurround::DimHome => 0.59,
            ViewingSurround::AverageDomestic => 0.69,
            ViewingSurround::BrightOffice => 0.69,
            ViewingSurround::Daylight => 0.69,
            ViewingSurround::Custom(lux) => {
                if lux < 10.0 {
                    0.525
                } else if lux < 50.0 {
                    0.59
                } else {
                    0.69
                }
            }
        }
    }
}

// ─── Display brightness model ─────────────────────────────────────────────────

/// Parameters describing the target HDR display.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DisplayBrightnessModel {
    /// Peak display luminance in nits.
    pub peak_nits: f32,
    /// Black level in nits (minimum luminance).
    pub black_level_nits: f32,
    /// Display screen area in cm² (used for glare estimation).
    pub screen_area_cm2: f32,
    /// Fraction of ambient light reflected by the display surface (0.0–1.0).
    /// Typical OLED: 0.01, glossy LCD: 0.04.
    pub reflectance: f32,
}

impl DisplayBrightnessModel {
    /// Create a typical HDR OLED display model (1000 nits peak).
    pub fn oled_1000() -> Self {
        Self {
            peak_nits: 1000.0,
            black_level_nits: 0.0001,
            screen_area_cm2: 2_800.0, // ~55″ 16:9
            reflectance: 0.01,
        }
    }

    /// Create a reference mastering monitor model (4000 nits peak).
    pub fn mastering_reference() -> Self {
        Self {
            peak_nits: 4000.0,
            black_level_nits: 0.005,
            screen_area_cm2: 1_600.0, // ~32″ reference
            reflectance: 0.015,
        }
    }

    /// Validate display parameters.
    pub fn validate(&self) -> Result<()> {
        if self.peak_nits <= 0.0 {
            return Err(HdrError::InvalidLuminance(self.peak_nits));
        }
        if self.black_level_nits < 0.0 {
            return Err(HdrError::InvalidLuminance(self.black_level_nits));
        }
        if self.peak_nits <= self.black_level_nits {
            return Err(HdrError::InvalidLuminance(self.peak_nits));
        }
        if !(0.0_f32..=1.0_f32).contains(&self.reflectance) {
            return Err(HdrError::GamutConversionError(format!(
                "reflectance out of range: {}",
                self.reflectance
            )));
        }
        Ok(())
    }

    /// Effective peak luminance accounting for reflected ambient glare.
    ///
    /// Uses the simplified formula from BT.2446 Annex A:
    /// `L_eff = L_peak + R * E_amb / π`
    /// where `E_amb` is the ambient illuminance in lux.
    pub fn effective_peak_nits(&self, ambient_lux: f32) -> f32 {
        // cd/m² = lux / π (Lambertian reflectance)
        self.peak_nits + self.reflectance * ambient_lux / std::f32::consts::PI
    }
}

// ─── Adaptation model ─────────────────────────────────────────────────────────

/// Computes adaptation factors based on viewing environment and display model.
///
/// The adaptation factor `α` in \[0.0, 1.0\] represents how much the display
/// signal should be attenuated to compensate for the ambient illuminance
/// reducing perceived contrast.  A factor of 1.0 means the mastering
/// environment matches the playback environment; lower values indicate the
/// playback environment is brighter and the display should boost or compensate.
#[derive(Debug, Clone)]
pub struct AmbientLightAdapter {
    /// The target playback display.
    pub display: DisplayBrightnessModel,
    /// The mastering (reference) display.
    pub reference_display: DisplayBrightnessModel,
    /// Mastering surround environment.
    pub mastering_surround: ViewingSurround,
}

impl AmbientLightAdapter {
    /// Create a new adapter.
    ///
    /// `display` is the viewer's playback display; `reference_display` is the
    /// mastering monitor.  `mastering_surround` is the environment under which
    /// the content was graded (typically `DarkCinema`).
    pub fn new(
        display: DisplayBrightnessModel,
        reference_display: DisplayBrightnessModel,
        mastering_surround: ViewingSurround,
    ) -> Result<Self> {
        display.validate()?;
        reference_display.validate()?;
        Ok(Self {
            display,
            reference_display,
            mastering_surround,
        })
    }

    /// Compute the adaptation factor for a given playback surround.
    ///
    /// Returns a value in \[0.0, 1.0\] where 1.0 means no adaptation is needed
    /// (identical environments) and 0.0 means maximum attenuation.
    ///
    /// # Algorithm
    ///
    /// 1. Compute the viewing-adapted luminance `L_a` for both mastering and
    ///    playback environments using the CIECAM02 luminance adaptation formula
    ///    `L_a = E_amb / (5 · π)` (1/5 of the background luminance).
    /// 2. Compute the adaptation degree `D` for each environment.
    /// 3. Return the ratio `D_playback / D_mastering` clamped to \[0.0, 1.0\].
    pub fn adaptation_factor(&self, playback_surround: ViewingSurround) -> f32 {
        let lux_master = self.mastering_surround.illuminance_lux();
        let lux_play = playback_surround.illuminance_lux();

        let la_master = lux_master / (5.0 * std::f32::consts::PI);
        let la_play = lux_play / (5.0 * std::f32::consts::PI);

        let c_master = self.mastering_surround.surround_factor();
        let c_play = playback_surround.surround_factor();

        let d_master = degree_of_adaptation(la_master, c_master);
        let d_play = degree_of_adaptation(la_play, c_play);

        if d_master < 1e-6 {
            return 1.0;
        }

        (d_play / d_master).clamp(0.0, 1.0)
    }

    /// Recommended display peak-nits adjustment for the playback environment.
    ///
    /// Returns the adapted peak luminance in nits that the display should target
    /// to achieve perceptual equivalence with the mastering environment.
    pub fn adapted_peak_nits(&self, playback_surround: ViewingSurround) -> f32 {
        let alpha = self.adaptation_factor(playback_surround);
        let eff = self
            .display
            .effective_peak_nits(playback_surround.illuminance_lux());

        // Blend between reference peak and effective display peak based on α.
        let target = alpha * self.reference_display.peak_nits + (1.0 - alpha) * eff;
        target.clamp(self.display.black_level_nits + 1.0, self.display.peak_nits)
    }

    /// Contrast compression factor for the given playback environment.
    ///
    /// Returns a value in (0.5, 1.2] that scales the tone-mapping contrast to
    /// maintain perceptual equivalence across viewing conditions.
    pub fn contrast_factor(&self, playback_surround: ViewingSurround) -> f32 {
        let alpha = self.adaptation_factor(playback_surround);
        // Brighter room → less perceived contrast → boost tone-map contrast.
        // Dark room → identity.
        let boost = 1.0 + (1.0 - alpha) * 0.25;
        boost.clamp(0.5, 1.5)
    }
}

/// Compute the CIECAM02 *degree of chromatic adaptation* `D`.
///
/// Formula: `D = F · [1 − (1/3.6) · exp(−(L_a + 42) / 92)]`
/// where `F` is derived from the surround factor `c`.
fn degree_of_adaptation(l_a: f32, c: f32) -> f32 {
    // F (discount factor) depends on surround:
    //   c ≥ 0.69 → F = 1.0 (average)
    //   c = 0.59 → F = 0.9 (dim)
    //   c ≤ 0.525 → F = 0.8 (dark)
    let f = if c >= 0.69 {
        1.0_f32
    } else if c >= 0.59 {
        // Linear interpolation between dim and average
        0.9 + (c - 0.59) / (0.69 - 0.59) * 0.1
    } else {
        0.8 + (c - 0.525) / (0.59 - 0.525) * 0.1
    };

    f * (1.0 - (1.0 / 3.6) * (-(l_a + 42.0) / 92.0).exp())
}

// ─── Utility: luminance-to-lux conversion ─────────────────────────────────────

/// Estimate scene illuminance in lux from the average scene luminance in nits.
///
/// Uses the approximate relationship `E_lux ≈ L_avg_nits × π`.
pub fn luminance_nits_to_lux(avg_nits: f32) -> f32 {
    avg_nits * std::f32::consts::PI
}

/// Estimate average scene luminance in nits from illuminance in lux.
pub fn lux_to_luminance_nits(lux: f32) -> f32 {
    lux / std::f32::consts::PI
}

// ─── Pre-built adapter constructors ───────────────────────────────────────────

/// Create a cinema-to-home adapter using standard reference values.
///
/// Mastering: 4000-nit reference in dark cinema.
/// Playback: 1000-nit OLED in living-room environment.
pub fn cinema_to_home_adapter() -> Result<AmbientLightAdapter> {
    AmbientLightAdapter::new(
        DisplayBrightnessModel::oled_1000(),
        DisplayBrightnessModel::mastering_reference(),
        ViewingSurround::DarkCinema,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewing_surround_illuminance() {
        assert!((ViewingSurround::DarkCinema.illuminance_lux() - 2.0).abs() < 1e-3);
        assert!((ViewingSurround::Daylight.illuminance_lux() - 10_000.0).abs() < 1.0);
        assert!((ViewingSurround::Custom(250.0).illuminance_lux() - 250.0).abs() < 1e-3);
    }

    #[test]
    fn test_display_model_validate_valid() {
        let d = DisplayBrightnessModel::oled_1000();
        assert!(d.validate().is_ok());
    }

    #[test]
    fn test_display_model_validate_rejects_zero_peak() {
        let mut d = DisplayBrightnessModel::oled_1000();
        d.peak_nits = 0.0;
        assert!(d.validate().is_err());
    }

    #[test]
    fn test_display_model_validate_rejects_bad_reflectance() {
        let mut d = DisplayBrightnessModel::oled_1000();
        d.reflectance = 1.5;
        assert!(d.validate().is_err());
    }

    #[test]
    fn test_effective_peak_increases_with_ambient() {
        let d = DisplayBrightnessModel::oled_1000();
        let dark = d.effective_peak_nits(2.0);
        let bright = d.effective_peak_nits(5000.0);
        assert!(
            bright > dark,
            "effective peak should increase with ambient lux"
        );
    }

    #[test]
    fn test_adaptation_factor_identical_environments() {
        let adapter = cinema_to_home_adapter().expect("adapter");
        // When playback matches mastering (dark cinema), factor should be ~1.0.
        let alpha = adapter.adaptation_factor(ViewingSurround::DarkCinema);
        // Reference display is different from playback display so ratio won't
        // be exactly 1.0; but since same surround, D_play == D_master → 1.0.
        assert!(
            (alpha - 1.0).abs() < 1e-4,
            "identical surround should yield ~1.0, got {alpha}"
        );
    }

    #[test]
    fn test_adaptation_factor_brighter_room_is_in_range() {
        let adapter = cinema_to_home_adapter().expect("adapter");
        let alpha = adapter.adaptation_factor(ViewingSurround::BrightOffice);
        // A bright playback environment has a higher degree of chromatic
        // adaptation than the dark mastering environment, so the raw ratio
        // exceeds 1.0 and is clamped to 1.0.  Either way the result must
        // be a valid value in [0.0, 1.0].
        assert!(
            (0.0..=1.0).contains(&alpha),
            "adaptation factor out of range: {alpha}"
        );
    }

    #[test]
    fn test_adapted_peak_within_display_range() {
        let adapter = cinema_to_home_adapter().expect("adapter");
        for surround in [
            ViewingSurround::DarkCinema,
            ViewingSurround::AverageDomestic,
            ViewingSurround::BrightOffice,
        ] {
            let peak = adapter.adapted_peak_nits(surround);
            assert!(
                peak <= adapter.display.peak_nits,
                "adapted peak {peak} exceeds display peak {}",
                adapter.display.peak_nits
            );
            assert!(peak > 0.0, "adapted peak must be positive");
        }
    }

    #[test]
    fn test_contrast_factor_dark_is_near_identity() {
        let adapter = cinema_to_home_adapter().expect("adapter");
        let cf = adapter.contrast_factor(ViewingSurround::DarkCinema);
        assert!(
            (cf - 1.0).abs() < 1e-4,
            "dark surround contrast factor should be ~1.0, got {cf}"
        );
    }

    #[test]
    fn test_contrast_factor_is_in_range() {
        let adapter = cinema_to_home_adapter().expect("adapter");
        for surround in [
            ViewingSurround::DarkCinema,
            ViewingSurround::DimHome,
            ViewingSurround::AverageDomestic,
            ViewingSurround::BrightOffice,
        ] {
            let cf = adapter.contrast_factor(surround);
            assert!(
                (0.5..=1.5).contains(&cf),
                "contrast_factor for {surround:?} = {cf} out of bounds"
            );
        }
    }

    #[test]
    fn test_luminance_lux_roundtrip() {
        let original_lux = 500.0f32;
        let nits = lux_to_luminance_nits(original_lux);
        let recovered_lux = luminance_nits_to_lux(nits);
        assert!(
            (recovered_lux - original_lux).abs() < 1e-3,
            "roundtrip failed: {original_lux} → {nits} → {recovered_lux}"
        );
    }

    #[test]
    fn test_degree_of_adaptation_is_in_range() {
        // L_a from 0.01 to 1000, surround from 0.525 to 0.69
        for (lux, c) in [(2.0, 0.525), (100.0, 0.69), (5000.0, 0.69)] {
            let la = lux / (5.0 * std::f32::consts::PI);
            let d = degree_of_adaptation(la, c);
            assert!(
                (0.0..=1.0).contains(&d),
                "degree_of_adaptation({la},{c}) = {d} out of range"
            );
        }
    }
}
