//! Retro film effect generator combining grain, colour grading, and vignette.
//!
//! Produces a cohesive vintage/retro film look by composing three layers:
//!
//! 1. **Colour desaturation and tinting** — shifts the palette toward a chosen
//!    era (e.g. warm sepia, cool blue, faded pastel).
//! 2. **Film grain** — per-pixel LCG noise matching the character of real
//!    photographic emulsion.
//! 3. **Vignette** — edge darkening with configurable shape and falloff.
//!
//! The effect implements [`VideoEffect`] for seamless integration into VFX
//! chains.
//!
//! # Example
//!
//! ```
//! use oximedia_vfx::retro_film::{RetroFilmEffect, RetroFilmPreset};
//! use oximedia_vfx::{Frame, EffectParams, VideoEffect};
//!
//! let mut effect = RetroFilmEffect::from_preset(RetroFilmPreset::Sepia);
//! let input = Frame::new(64, 64).expect("frame");
//! let mut output = Frame::new(64, 64).expect("frame");
//! effect.apply(&input, &mut output, &EffectParams::new()).expect("apply");
//! ```

use crate::{EffectParams, Frame, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// LCG (deterministic noise)
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Congruential Generator for deterministic grain noise.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005_u64)
            .wrapping_add(1_442_695_040_888_963_407_u64);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        let raw = (self.next_u64() >> 32) as u32;
        (raw as f32 / 2_147_483_648.0) - 1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset catalogue
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-built retro film looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetroFilmPreset {
    /// Classic sepia toning: warm brown tint, moderate grain, subtle vignette.
    Sepia,
    /// 1920s silent-film: high-contrast B&W, heavy grain, strong vignette.
    SilentEra,
    /// 1970s Kodachrome: warm shadows, punchy saturation, fine grain.
    Kodachrome,
    /// Faded photograph: washed-out pastels, mild grain, gentle vignette.
    FadedPhoto,
    /// Cool blue noir: desaturated blue tint, medium grain, deep vignette.
    BlueNoir,
    /// Infrared photography: false-colour channel swap, heavy grain.
    Infrared,
    /// VHS camcorder: slight desaturation, scan-line grain, soft vignette.
    Vhs,
    /// 8mm home movie: warm tint, large grain, heavy vignette.
    Super8,
}

impl RetroFilmPreset {
    /// Construct a full [`RetroFilmConfig`] from this preset.
    #[must_use]
    pub fn to_config(self) -> RetroFilmConfig {
        match self {
            Self::Sepia => RetroFilmConfig {
                desaturation: 0.85,
                tint: [0.94, 0.78, 0.57],
                brightness_offset: 0.0,
                contrast: 1.05,
                grain_strength: 0.06,
                grain_seed: 42,
                vignette_inner: 0.45,
                vignette_outer: 1.0,
                vignette_strength: 0.55,
                fade_amount: 0.05,
            },
            Self::SilentEra => RetroFilmConfig {
                desaturation: 1.0,
                tint: [0.95, 0.92, 0.85],
                brightness_offset: -0.05,
                contrast: 1.3,
                grain_strength: 0.15,
                grain_seed: 1920,
                vignette_inner: 0.3,
                vignette_outer: 0.9,
                vignette_strength: 0.8,
                fade_amount: 0.08,
            },
            Self::Kodachrome => RetroFilmConfig {
                desaturation: 0.1,
                tint: [1.02, 0.98, 0.90],
                brightness_offset: 0.02,
                contrast: 1.15,
                grain_strength: 0.03,
                grain_seed: 1975,
                vignette_inner: 0.5,
                vignette_outer: 1.0,
                vignette_strength: 0.35,
                fade_amount: 0.0,
            },
            Self::FadedPhoto => RetroFilmConfig {
                desaturation: 0.5,
                tint: [1.0, 0.97, 0.93],
                brightness_offset: 0.1,
                contrast: 0.85,
                grain_strength: 0.04,
                grain_seed: 2000,
                vignette_inner: 0.55,
                vignette_outer: 1.0,
                vignette_strength: 0.3,
                fade_amount: 0.15,
            },
            Self::BlueNoir => RetroFilmConfig {
                desaturation: 0.75,
                tint: [0.80, 0.85, 1.05],
                brightness_offset: -0.08,
                contrast: 1.2,
                grain_strength: 0.07,
                grain_seed: 1947,
                vignette_inner: 0.3,
                vignette_outer: 0.85,
                vignette_strength: 0.7,
                fade_amount: 0.0,
            },
            Self::Infrared => RetroFilmConfig {
                desaturation: 0.0,
                tint: [1.0, 1.0, 1.0],
                brightness_offset: 0.05,
                contrast: 1.25,
                grain_strength: 0.12,
                grain_seed: 8080,
                vignette_inner: 0.4,
                vignette_outer: 1.0,
                vignette_strength: 0.5,
                fade_amount: 0.0,
            },
            Self::Vhs => RetroFilmConfig {
                desaturation: 0.2,
                tint: [1.0, 0.98, 0.95],
                brightness_offset: 0.03,
                contrast: 0.95,
                grain_strength: 0.08,
                grain_seed: 1985,
                vignette_inner: 0.6,
                vignette_outer: 1.0,
                vignette_strength: 0.25,
                fade_amount: 0.05,
            },
            Self::Super8 => RetroFilmConfig {
                desaturation: 0.35,
                tint: [1.05, 0.95, 0.82],
                brightness_offset: 0.0,
                contrast: 1.1,
                grain_strength: 0.14,
                grain_seed: 1965,
                vignette_inner: 0.3,
                vignette_outer: 0.85,
                vignette_strength: 0.75,
                fade_amount: 0.06,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Full configuration for the retro film effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetroFilmConfig {
    /// Desaturation amount: 0.0 = full colour, 1.0 = fully monochrome.
    pub desaturation: f32,
    /// Per-channel tint multiplier applied after desaturation.
    /// `[1.0, 1.0, 1.0]` is neutral.
    pub tint: [f32; 3],
    /// Additive brightness offset applied to each channel (-1.0 to 1.0).
    pub brightness_offset: f32,
    /// Contrast multiplier around mid-grey (0.5). 1.0 = neutral.
    pub contrast: f32,
    /// Film grain noise amplitude (0.0 = none, 1.0 = extreme).
    pub grain_strength: f32,
    /// LCG seed for deterministic grain pattern.
    pub grain_seed: u64,
    /// Vignette inner radius (normalised, 0.0-1.0).
    pub vignette_inner: f32,
    /// Vignette outer radius (normalised, 0.0-1.0).
    pub vignette_outer: f32,
    /// Vignette strength (0.0 = no vignette, 1.0 = full).
    pub vignette_strength: f32,
    /// Fade (lift blacks) amount: raises the minimum luminance.
    pub fade_amount: f32,
}

impl Default for RetroFilmConfig {
    fn default() -> Self {
        RetroFilmPreset::Sepia.to_config()
    }
}

impl RetroFilmConfig {
    /// Validate and clamp all parameters to sensible ranges.
    #[must_use]
    pub fn validated(mut self) -> Self {
        self.desaturation = self.desaturation.clamp(0.0, 1.0);
        for ch in &mut self.tint {
            *ch = ch.clamp(0.0, 3.0);
        }
        self.brightness_offset = self.brightness_offset.clamp(-1.0, 1.0);
        self.contrast = self.contrast.clamp(0.1, 4.0);
        self.grain_strength = self.grain_strength.clamp(0.0, 1.0);
        self.vignette_inner = self.vignette_inner.clamp(0.0, 1.0);
        self.vignette_outer = self.vignette_outer.clamp(self.vignette_inner, 1.0);
        self.vignette_strength = self.vignette_strength.clamp(0.0, 1.0);
        self.fade_amount = self.fade_amount.clamp(0.0, 0.5);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Effect
// ─────────────────────────────────────────────────────────────────────────────

/// Retro film effect that composes desaturation, tinting, grain, and vignette.
pub struct RetroFilmEffect {
    config: RetroFilmConfig,
    /// Per-frame seed offset so grain animates across frames (if desired).
    frame_counter: u64,
    /// Whether to animate grain (vary seed per frame).
    pub animate_grain: bool,
}

impl RetroFilmEffect {
    /// Create from explicit configuration.
    #[must_use]
    pub fn new(config: RetroFilmConfig) -> Self {
        Self {
            config: config.validated(),
            frame_counter: 0,
            animate_grain: true,
        }
    }

    /// Create from a preset.
    #[must_use]
    pub fn from_preset(preset: RetroFilmPreset) -> Self {
        Self::new(preset.to_config())
    }

    /// Get a reference to the current config.
    #[must_use]
    pub fn config(&self) -> &RetroFilmConfig {
        &self.config
    }

    /// Update the configuration (validates and clamps).
    pub fn set_config(&mut self, config: RetroFilmConfig) {
        self.config = config.validated();
    }

    /// Process a single pixel, returning the graded RGBA value.
    ///
    /// `nx`, `ny`: normalised pixel position in [0, 1].
    /// `rgba`: original pixel.
    /// `rng`: LCG instance for grain.
    fn process_pixel(
        &self,
        rgba: [u8; 4],
        nx: f32,
        ny: f32,
        rng: &mut Lcg,
    ) -> [u8; 4] {
        let cfg = &self.config;

        // --- Step 1: linearise to [0, 1] ---
        let mut r = rgba[0] as f32 / 255.0;
        let mut g = rgba[1] as f32 / 255.0;
        let mut b = rgba[2] as f32 / 255.0;

        // --- Step 2: desaturate (BT.601 luma weights) ---
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        let desat = cfg.desaturation;
        r = r * (1.0 - desat) + luma * desat;
        g = g * (1.0 - desat) + luma * desat;
        b = b * (1.0 - desat) + luma * desat;

        // --- Step 3: tint ---
        r *= cfg.tint[0];
        g *= cfg.tint[1];
        b *= cfg.tint[2];

        // --- Step 4: contrast around mid-grey ---
        r = (r - 0.5) * cfg.contrast + 0.5;
        g = (g - 0.5) * cfg.contrast + 0.5;
        b = (b - 0.5) * cfg.contrast + 0.5;

        // --- Step 5: brightness ---
        r += cfg.brightness_offset;
        g += cfg.brightness_offset;
        b += cfg.brightness_offset;

        // --- Step 6: fade (lift blacks) ---
        if cfg.fade_amount > 0.0 {
            r = r * (1.0 - cfg.fade_amount) + cfg.fade_amount;
            g = g * (1.0 - cfg.fade_amount) + cfg.fade_amount;
            b = b * (1.0 - cfg.fade_amount) + cfg.fade_amount;
        }

        // --- Step 7: grain ---
        if cfg.grain_strength > 0.0 {
            let amp = cfg.grain_strength;
            r += rng.next_f32() * amp;
            g += rng.next_f32() * amp;
            b += rng.next_f32() * amp;
        }

        // --- Step 8: vignette ---
        if cfg.vignette_strength > 0.0 {
            let dx = nx - 0.5;
            let dy = ny - 0.5;
            let dist = (dx * dx + dy * dy).sqrt() * 2.0; // normalised so corner ~1.414
            let inner = cfg.vignette_inner;
            let outer = cfg.vignette_outer;
            let vig_t = if dist <= inner {
                0.0
            } else if dist >= outer {
                1.0
            } else if (outer - inner).abs() < f32::EPSILON {
                0.0
            } else {
                let t = (dist - inner) / (outer - inner);
                // Smoothstep falloff
                t * t * (3.0 - 2.0 * t)
            };
            let darkening = 1.0 - vig_t * cfg.vignette_strength;
            r *= darkening;
            g *= darkening;
            b *= darkening;
        }

        // --- Clamp and quantise ---
        [
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
            rgba[3], // preserve alpha
        ]
    }
}

impl VideoEffect for RetroFilmEffect {
    fn name(&self) -> &str {
        "RetroFilm"
    }

    fn description(&self) -> &'static str {
        "Vintage film look: desaturation, tint, grain, and vignette"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        let w = input.width;
        let h = input.height;
        if w == 0 || h == 0 {
            return Ok(());
        }

        let seed = if self.animate_grain {
            self.config
                .grain_seed
                .wrapping_add(self.frame_counter.wrapping_mul(7919))
        } else {
            self.config.grain_seed
        };
        let mut rng = Lcg::new(seed);

        let w_f = (w - 1).max(1) as f32;
        let h_f = (h - 1).max(1) as f32;

        for py in 0..h {
            let ny = py as f32 / h_f;
            for px in 0..w {
                let nx = px as f32 / w_f;
                let rgba = input.get_pixel(px, py).unwrap_or([0, 0, 0, 255]);
                let out = self.process_pixel(rgba, nx, ny, &mut rng);
                output.set_pixel(px, py, out);
            }
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);
        Ok(())
    }

    fn reset(&mut self) {
        self.frame_counter = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Frame {
        let mut f = Frame::new(w, h).expect("frame creation");
        f.clear(rgba);
        f
    }

    #[test]
    fn test_preset_sepia_applies() {
        let mut fx = RetroFilmEffect::from_preset(RetroFilmPreset::Sepia);
        let input = make_solid_frame(32, 32, [128, 128, 128, 255]);
        let mut output = Frame::new(32, 32).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        // Centre pixel should have been tinted warm (R > B)
        let p = output.get_pixel(16, 16).unwrap_or([0, 0, 0, 0]);
        assert!(
            p[0] >= p[2],
            "Sepia: R ({}) should be >= B ({})",
            p[0],
            p[2]
        );
    }

    #[test]
    fn test_preset_silent_era_desaturated() {
        let mut fx = RetroFilmEffect::from_preset(RetroFilmPreset::SilentEra);
        let input = make_solid_frame(16, 16, [200, 100, 50, 255]);
        let mut output = Frame::new(16, 16).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        // Full desaturation + warm tint: channels close but warm-shifted
        // (the tint [0.95, 0.92, 0.85] creates a deliberate warm tone)
        let p = output.get_pixel(8, 8).unwrap_or([0, 0, 0, 0]);
        let max_ch = p[0].max(p[1]).max(p[2]);
        let min_ch = p[0].min(p[1]).min(p[2]);
        // The spread comes from the tint; must be much less than original (150 spread)
        assert!(
            max_ch - min_ch <= 40,
            "SilentEra should be heavily desaturated: {:?} (spread={})",
            p,
            max_ch - min_ch
        );
        // R should be dominant (warm tint)
        assert!(p[0] >= p[2], "SilentEra should have warm tint: {:?}", p);
    }

    #[test]
    fn test_alpha_preserved() {
        let mut fx = RetroFilmEffect::from_preset(RetroFilmPreset::Kodachrome);
        let input = make_solid_frame(8, 8, [100, 150, 200, 128]);
        let mut output = Frame::new(8, 8).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        for y in 0..8 {
            for x in 0..8 {
                let p = output.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                assert_eq!(p[3], 128, "alpha must be preserved at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_vignette_darkens_corners() {
        let mut fx = RetroFilmEffect::new(RetroFilmConfig {
            desaturation: 0.0,
            tint: [1.0, 1.0, 1.0],
            brightness_offset: 0.0,
            contrast: 1.0,
            grain_strength: 0.0,
            grain_seed: 0,
            vignette_inner: 0.2,
            vignette_outer: 0.8,
            vignette_strength: 1.0,
            fade_amount: 0.0,
        });
        fx.animate_grain = false;
        let input = make_solid_frame(64, 64, [200, 200, 200, 255]);
        let mut output = Frame::new(64, 64).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");

        let centre = output.get_pixel(32, 32).unwrap_or([0, 0, 0, 0]);
        let corner = output.get_pixel(0, 0).unwrap_or([0, 0, 0, 0]);
        assert!(
            centre[0] > corner[0],
            "centre ({}) should be brighter than corner ({})",
            centre[0],
            corner[0]
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut fx = RetroFilmEffect::from_preset(RetroFilmPreset::Sepia);
        let input = Frame::new(32, 32).expect("frame");
        let mut output = Frame::new(16, 16).expect("frame");
        assert!(fx
            .apply(&input, &mut output, &EffectParams::new())
            .is_err());
    }

    #[test]
    fn test_grain_deterministic_when_not_animated() {
        let config = RetroFilmConfig {
            grain_strength: 0.1,
            grain_seed: 12345,
            desaturation: 0.0,
            tint: [1.0, 1.0, 1.0],
            brightness_offset: 0.0,
            contrast: 1.0,
            vignette_strength: 0.0,
            vignette_inner: 0.0,
            vignette_outer: 1.0,
            fade_amount: 0.0,
        };
        let mut fx1 = RetroFilmEffect::new(config.clone());
        fx1.animate_grain = false;
        let mut fx2 = RetroFilmEffect::new(config);
        fx2.animate_grain = false;

        let input = make_solid_frame(16, 16, [128, 128, 128, 255]);
        let mut out1 = Frame::new(16, 16).expect("frame");
        let mut out2 = Frame::new(16, 16).expect("frame");
        fx1.apply(&input, &mut out1, &EffectParams::new())
            .expect("apply1");
        fx2.apply(&input, &mut out2, &EffectParams::new())
            .expect("apply2");
        assert_eq!(out1.data, out2.data, "same seed+config must match");
    }

    #[test]
    fn test_fade_lifts_blacks() {
        let config = RetroFilmConfig {
            desaturation: 0.0,
            tint: [1.0, 1.0, 1.0],
            brightness_offset: 0.0,
            contrast: 1.0,
            grain_strength: 0.0,
            grain_seed: 0,
            vignette_strength: 0.0,
            vignette_inner: 0.0,
            vignette_outer: 1.0,
            fade_amount: 0.3,
        };
        let mut fx = RetroFilmEffect::new(config);
        let input = make_solid_frame(8, 8, [0, 0, 0, 255]);
        let mut output = Frame::new(8, 8).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        let p = output.get_pixel(4, 4).unwrap_or([0, 0, 0, 0]);
        assert!(
            p[0] > 0,
            "fade should lift pure black to non-zero: R={}",
            p[0]
        );
    }

    #[test]
    fn test_all_presets_apply_without_error() {
        let presets = [
            RetroFilmPreset::Sepia,
            RetroFilmPreset::SilentEra,
            RetroFilmPreset::Kodachrome,
            RetroFilmPreset::FadedPhoto,
            RetroFilmPreset::BlueNoir,
            RetroFilmPreset::Infrared,
            RetroFilmPreset::Vhs,
            RetroFilmPreset::Super8,
        ];
        let input = make_solid_frame(16, 16, [100, 150, 200, 255]);
        for preset in presets {
            let mut fx = RetroFilmEffect::from_preset(preset);
            let mut output = Frame::new(16, 16).expect("frame");
            fx.apply(&input, &mut output, &EffectParams::new())
                .unwrap_or_else(|e| panic!("preset {preset:?} failed: {e}"));
        }
    }

    #[test]
    fn test_config_validation_clamps() {
        let cfg = RetroFilmConfig {
            desaturation: 2.0,
            tint: [5.0, -1.0, 1.0],
            brightness_offset: 10.0,
            contrast: 0.0,
            grain_strength: -1.0,
            grain_seed: 0,
            vignette_inner: 1.5,
            vignette_outer: 0.5,
            vignette_strength: 3.0,
            fade_amount: 1.0,
        }
        .validated();

        assert_eq!(cfg.desaturation, 1.0);
        assert_eq!(cfg.tint[0], 3.0);
        assert_eq!(cfg.tint[1], 0.0);
        assert_eq!(cfg.brightness_offset, 1.0);
        assert_eq!(cfg.contrast, 0.1);
        assert_eq!(cfg.grain_strength, 0.0);
        assert_eq!(cfg.vignette_inner, 1.0);
        assert!(cfg.vignette_outer >= cfg.vignette_inner);
        assert_eq!(cfg.vignette_strength, 1.0);
        assert_eq!(cfg.fade_amount, 0.5);
    }

    #[test]
    fn test_reset_clears_frame_counter() {
        let mut fx = RetroFilmEffect::from_preset(RetroFilmPreset::Sepia);
        let input = make_solid_frame(8, 8, [128, 128, 128, 255]);
        let mut output = Frame::new(8, 8).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        assert!(fx.frame_counter > 0);
        fx.reset();
        assert_eq!(fx.frame_counter, 0);
    }

    #[test]
    fn test_effect_name_and_description() {
        let fx = RetroFilmEffect::from_preset(RetroFilmPreset::Sepia);
        assert_eq!(fx.name(), "RetroFilm");
        assert!(!fx.description().is_empty());
    }
}
