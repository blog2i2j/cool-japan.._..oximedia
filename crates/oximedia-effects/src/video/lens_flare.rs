//! Lens flare effect.
//!
//! Simulates optical lens flare with a central glow, halo ring, and radial streaks.
//! The effect is physically-inspired: a bright light source creates a bright spot,
//! a circular halo (due to internal lens reflections), and streak artifacts.

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};

/// Color for lens flare components (linear sRGB, 0.0–1.0).
#[derive(Debug, Clone, Copy)]
pub struct LensFlareColor {
    /// Red component.
    pub r: f32,
    /// Green component.
    pub g: f32,
    /// Blue component.
    pub b: f32,
}

impl LensFlareColor {
    /// Create a new color.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// White color.
    #[must_use]
    pub const fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Warm yellow-white (typical sunlight).
    #[must_use]
    pub const fn sunlight() -> Self {
        Self::new(1.0, 0.95, 0.7)
    }

    /// Blue-tinted (anamorphic lens streak).
    #[must_use]
    pub const fn anamorphic() -> Self {
        Self::new(0.4, 0.6, 1.0)
    }
}

impl Default for LensFlareColor {
    fn default() -> Self {
        Self::sunlight()
    }
}

/// Configuration for the lens flare effect.
#[derive(Debug, Clone)]
pub struct LensFlareConfig {
    /// Normalized X position of flare source (0.0 = left, 1.0 = right).
    pub source_x: f32,
    /// Normalized Y position of flare source (0.0 = top, 1.0 = bottom).
    pub source_y: f32,
    /// Overall intensity (0.0–1.0).
    pub intensity: f32,
    /// Color of the primary flare.
    pub color: LensFlareColor,
    /// Radius of central glow as fraction of image diagonal.
    pub glow_radius: f32,
    /// Radius of halo ring as fraction of image diagonal.
    pub halo_radius: f32,
    /// Halo ring width as fraction of halo radius.
    pub halo_width: f32,
    /// Number of anamorphic streaks (0 = disabled).
    pub streak_count: usize,
    /// Streak length as fraction of image diagonal.
    pub streak_length: f32,
    /// Ghost lens reflections count.
    pub ghost_count: usize,
}

impl Default for LensFlareConfig {
    fn default() -> Self {
        Self {
            source_x: 0.3,
            source_y: 0.25,
            intensity: 0.7,
            color: LensFlareColor::default(),
            glow_radius: 0.08,
            halo_radius: 0.20,
            halo_width: 0.015,
            streak_count: 4,
            streak_length: 0.45,
            ghost_count: 3,
        }
    }
}

/// Lens flare effect processor.
pub struct LensFlare {
    config: LensFlareConfig,
}

impl LensFlare {
    /// Create a new lens flare effect.
    #[must_use]
    pub fn new(config: LensFlareConfig) -> Self {
        Self { config }
    }

    /// Apply the lens flare effect in-place.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer size doesn't match `width * height * bpp`.
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    pub fn apply(
        &self,
        data: &mut [u8],
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> VideoResult<()> {
        validate_buffer(data, width, height, format)?;

        let bpp = format.bytes_per_pixel();
        let cfg = &self.config;

        let src_x = cfg.source_x * width as f32;
        let src_y = cfg.source_y * height as f32;

        // Precompute diagonal for radius normalization
        let diag = ((width * width + height * height) as f32).sqrt();

        let glow_r = cfg.glow_radius * diag;
        let halo_r = cfg.halo_radius * diag;
        let halo_w = cfg.halo_width * diag;
        let streak_len = cfg.streak_length * diag;

        for py in 0..height {
            for px in 0..width {
                let dx = px as f32 - src_x;
                let dy = py as f32 - src_y;
                let dist = (dx * dx + dy * dy).sqrt();

                let mut add_r = 0.0f32;
                let mut add_g = 0.0f32;
                let mut add_b = 0.0f32;

                // --- Central glow: Gaussian falloff ---
                if glow_r > 0.0 {
                    let sigma = glow_r * 0.5;
                    let glow = (-0.5 * (dist / sigma).powi(2)).exp();
                    add_r += cfg.color.r * glow * cfg.intensity * 255.0;
                    add_g += cfg.color.g * glow * cfg.intensity * 255.0;
                    add_b += cfg.color.b * glow * cfg.intensity * 255.0;
                }

                // --- Halo ring ---
                if halo_r > 0.0 {
                    let ring_dist = (dist - halo_r).abs();
                    if ring_dist < halo_w * 2.0 {
                        let t = 1.0 - ring_dist / (halo_w * 2.0);
                        let halo = t * t * cfg.intensity * 0.5;
                        add_r += cfg.color.r * halo * 255.0;
                        add_g += cfg.color.g * halo * 255.0;
                        add_b += cfg.color.b * halo * 255.0;
                    }
                }

                // --- Radial streaks (anamorphic flare) ---
                if cfg.streak_count > 0 && streak_len > 0.0 {
                    let angle = dy.atan2(dx);
                    for i in 0..cfg.streak_count {
                        let streak_angle =
                            std::f32::consts::PI * i as f32 / cfg.streak_count as f32;
                        let angle_diff = (angle - streak_angle).abs();
                        // Streak is very narrow in angle, long in distance
                        let angular_width = 0.015_f32; // radians
                        let ang_factor = if angle_diff < angular_width
                            || (std::f32::consts::PI - angle_diff).abs() < angular_width
                        {
                            let closest = angle_diff.min((std::f32::consts::PI - angle_diff).abs());
                            (1.0 - closest / angular_width).max(0.0)
                        } else {
                            0.0
                        };
                        let dist_factor = (1.0 - dist / streak_len).max(0.0);
                        let streak = ang_factor * dist_factor * cfg.intensity * 0.4;
                        // Anamorphic streaks are typically blue-tinted
                        add_r += LensFlareColor::anamorphic().r * streak * 255.0;
                        add_g += LensFlareColor::anamorphic().g * streak * 255.0;
                        add_b += LensFlareColor::anamorphic().b * streak * 255.0;
                    }
                }

                // --- Ghost lens reflections ---
                if cfg.ghost_count > 0 {
                    let img_cx = width as f32 * 0.5;
                    let img_cy = height as f32 * 0.5;

                    for g in 1..=cfg.ghost_count {
                        let t = g as f32 / (cfg.ghost_count + 1) as f32;
                        // Ghost position is mirrored through image center
                        let ghost_x = src_x + (img_cx - src_x) * (1.0 + t * 0.5);
                        let ghost_y = src_y + (img_cy - src_y) * (1.0 + t * 0.5);
                        let gdx = px as f32 - ghost_x;
                        let gdy = py as f32 - ghost_y;
                        let gdist = (gdx * gdx + gdy * gdy).sqrt();
                        let ghost_sigma = glow_r * 0.3 * (1.0 - t * 0.5);
                        let ghost_intensity = cfg.intensity * 0.15 * (1.0 - t * 0.5);
                        let glow = (-0.5 * (gdist / ghost_sigma.max(1.0)).powi(2)).exp();
                        add_r += cfg.color.r * glow * ghost_intensity * 255.0;
                        add_g += cfg.color.g * glow * ghost_intensity * 255.0;
                        add_b += cfg.color.b * glow * ghost_intensity * 255.0;
                    }
                }

                // --- Composite additively ---
                let idx = (py * width + px) * bpp;
                data[idx] = clamp_u8(f32::from(data[idx]) + add_r);
                data[idx + 1] = clamp_u8(f32::from(data[idx + 1]) + add_g);
                data[idx + 2] = clamp_u8(f32::from(data[idx + 2]) + add_b);
                // Alpha unchanged for RGBA
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(w: usize, h: usize) -> Vec<u8> {
        vec![50u8; w * h * 3]
    }

    #[test]
    fn test_lens_flare_default_applies() {
        let mut buf = make_buf(64, 64);
        let lf = LensFlare::new(LensFlareConfig::default());
        assert!(lf.apply(&mut buf, 64, 64, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_lens_flare_brightens_source_region() {
        let mut buf = vec![0u8; 128 * 128 * 3];
        let cfg = LensFlareConfig {
            source_x: 0.5,
            source_y: 0.5,
            intensity: 1.0,
            glow_radius: 0.2,
            ..Default::default()
        };
        let lf = LensFlare::new(cfg);
        lf.apply(&mut buf, 128, 128, PixelFormat::Rgb)
            .expect("apply should succeed");

        // Center pixel should be bright
        let cx = 64usize;
        let cy = 64usize;
        let idx = (cy * 128 + cx) * 3;
        assert!(buf[idx] > 100, "Center should be bright");
    }

    #[test]
    fn test_lens_flare_rgba() {
        let mut buf = vec![128u8; 32 * 32 * 4];
        let lf = LensFlare::new(LensFlareConfig::default());
        assert!(lf.apply(&mut buf, 32, 32, PixelFormat::Rgba).is_ok());
    }

    #[test]
    fn test_lens_flare_wrong_size_err() {
        let mut buf = vec![0u8; 10];
        let lf = LensFlare::new(LensFlareConfig::default());
        assert!(lf.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_lens_flare_color_white() {
        let c = LensFlareColor::white();
        assert_eq!(c.r, 1.0);
        assert_eq!(c.g, 1.0);
        assert_eq!(c.b, 1.0);
    }

    #[test]
    fn test_lens_flare_zero_intensity() {
        let buf_orig = vec![100u8; 32 * 32 * 3];
        let mut buf = buf_orig.clone();
        let cfg = LensFlareConfig {
            intensity: 0.0,
            ghost_count: 0,
            streak_count: 0,
            ..Default::default()
        };
        let lf = LensFlare::new(cfg);
        lf.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, buf_orig, "Zero intensity should not change pixels");
    }
}
