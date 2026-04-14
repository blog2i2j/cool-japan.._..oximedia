//! Vignette effect — radial darkening applied to video frames.
//!
//! A vignette darkens the edges of a frame while keeping the centre bright.
//! This module supports:
//!
//! - **Circular** mode: radius measured as a fraction of the shorter frame
//!   dimension.
//! - **Elliptical** mode: radius independently scaled to frame width and height,
//!   producing an elliptical falloff that matches the frame aspect ratio.
//!
//! All functions operate on row-major `u8` luma planes.  For colour frames,
//! apply the same multiplier map (produced by [`build_vignette_map`]) to each
//! channel separately.

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur while building or applying a vignette effect.
#[derive(Debug, thiserror::Error)]
pub enum VignetteError {
    /// Frame dimensions are invalid (zero width or height).
    #[error("invalid frame dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },

    /// Source buffer has an unexpected size.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        actual: usize,
    },

    /// A parameter is outside its valid range.
    #[error("parameter '{name}' value {value} is out of range {min}..={max}")]
    ParameterOutOfRange {
        /// Name of the invalid parameter.
        name: &'static str,
        /// The invalid value.
        value: f64,
        /// Minimum valid value.
        min: f64,
        /// Maximum valid value.
        max: f64,
    },
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Shape of the vignette falloff region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VignetteShape {
    /// Circular vignette — the boundary is a circle scaled to the shorter
    /// frame axis.
    #[default]
    Circular,
    /// Elliptical vignette — the boundary follows the frame aspect ratio,
    /// filling a full ellipse that touches the frame edges at `radius = 1.0`.
    Elliptical,
}

/// Configuration for the vignette effect.
///
/// All fractional parameters (`radius`, `feather`, `strength`) are in the
/// range `[0.0, 1.0]` unless otherwise stated.
#[derive(Debug, Clone)]
pub struct VignetteConfig {
    /// Normalised radius of the bright central area.
    ///
    /// `0.0` darkens the entire frame; `1.0` keeps the entire frame bright
    /// with no vignette.  Typical values are `0.5`–`0.8`.
    pub radius: f64,

    /// Width of the smooth transition band beyond `radius`.
    ///
    /// `0.0` produces a hard edge; `1.0` spreads the transition all the way
    /// to the opposite corner.  Typical values are `0.2`–`0.5`.
    pub feather: f64,

    /// Maximum darkening strength at the outer edge.
    ///
    /// `0.0` has no effect; `1.0` makes the outer edge fully black.
    pub strength: f64,

    /// Shape of the vignette boundary.
    pub shape: VignetteShape,
}

impl Default for VignetteConfig {
    fn default() -> Self {
        Self {
            radius: 0.65,
            feather: 0.35,
            strength: 0.85,
            shape: VignetteShape::Elliptical,
        }
    }
}

impl VignetteConfig {
    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), VignetteError> {
        check_range("radius", self.radius, 0.0, 1.0)?;
        check_range("feather", self.feather, 0.0, 1.0)?;
        check_range("strength", self.strength, 0.0, 1.0)?;
        Ok(())
    }
}

/// A precomputed per-pixel multiplier map for a specific frame size and
/// [`VignetteConfig`].
///
/// Each entry is in `[0.0, 1.0]` — multiply by the pixel luma value and
/// clamp to `[0, 255]`.
#[derive(Debug, Clone)]
pub struct VignetteMap {
    /// Multiplier values, one per pixel, row-major.
    pub multipliers: Vec<f32>,
    /// Frame width used when the map was built.
    pub width: u32,
    /// Frame height used when the map was built.
    pub height: u32,
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Build a [`VignetteMap`] for the given frame size and configuration.
///
/// Pre-computing the map is efficient when the same vignette must be applied
/// to many frames of identical dimensions.
///
/// # Errors
///
/// Returns [`VignetteError`] if `width` or `height` is zero, or if any
/// configuration parameter is out of range.
pub fn build_vignette_map(
    width: u32,
    height: u32,
    cfg: &VignetteConfig,
) -> Result<VignetteMap, VignetteError> {
    if width == 0 || height == 0 {
        return Err(VignetteError::InvalidDimensions { width, height });
    }
    cfg.validate()?;

    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    // Half-dimensions in normalised coordinates.
    let hw = w as f64 / 2.0;
    let hh = h as f64 / 2.0;

    // For circular mode we use the shorter half-dimension as the common radius.
    let r_scale = hw.min(hh);

    let mut mults = vec![0.0f32; n];

    for row in 0..h {
        for col in 0..w {
            // Pixel position relative to frame centre in normalised coordinates
            // ([−1, 1] along each axis).
            let nx = (col as f64 - hw + 0.5) / hw;
            let ny = (row as f64 - hh + 0.5) / hh;

            // Normalised radial distance depending on shape.
            let dist = match cfg.shape {
                VignetteShape::Circular => {
                    // Distance in pixels from centre, scaled by the shorter axis.
                    let dx = (col as f64 - hw + 0.5) / r_scale;
                    let dy = (row as f64 - hh + 0.5) / r_scale;
                    (dx * dx + dy * dy).sqrt()
                }
                VignetteShape::Elliptical => {
                    // Elliptical distance: `sqrt(nx^2 + ny^2)` reaches √2 at corners.
                    (nx * nx + ny * ny).sqrt()
                }
            };

            let mult = vignette_multiplier(dist, cfg.radius, cfg.feather, cfg.strength);
            mults[row * w + col] = mult as f32;
        }
    }

    Ok(VignetteMap {
        multipliers: mults,
        width,
        height,
    })
}

/// Apply a precomputed [`VignetteMap`] to a luma plane in-place.
///
/// # Arguments
///
/// * `plane` – mutable luma buffer, `width * height` bytes, row-major.
/// * `map` – vignette multiplier map matching the frame dimensions.
///
/// # Errors
///
/// Returns [`VignetteError`] if the buffer length does not match the map
/// dimensions.
pub fn apply_vignette_map(plane: &mut [u8], map: &VignetteMap) -> Result<(), VignetteError> {
    let expected = map.width as usize * map.height as usize;
    if plane.len() != expected {
        return Err(VignetteError::BufferSizeMismatch {
            expected,
            actual: plane.len(),
        });
    }
    for (px, &m) in plane.iter_mut().zip(map.multipliers.iter()) {
        *px = ((*px as f32) * m).round().clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Convenience function: build a vignette map and immediately apply it to
/// `plane`, returning the modified copy.
///
/// # Errors
///
/// Returns [`VignetteError`] on invalid dimensions or parameters.
pub fn apply_vignette(
    plane: &[u8],
    width: u32,
    height: u32,
    cfg: &VignetteConfig,
) -> Result<Vec<u8>, VignetteError> {
    let expected = width as usize * height as usize;
    if plane.len() != expected {
        return Err(VignetteError::BufferSizeMismatch {
            expected,
            actual: plane.len(),
        });
    }
    let map = build_vignette_map(width, height, cfg)?;
    let mut out = plane.to_vec();
    apply_vignette_map(&mut out, &map)?;
    Ok(out)
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Compute the per-pixel brightness multiplier for a given normalised radial
/// distance.
///
/// Inside `radius` → `1.0` (no darkening).  Beyond `radius + feather` →
/// `1.0 - strength` (fully darkened).  Between the two endpoints a smooth
/// cubic Hermite interpolation is used.
#[inline]
fn vignette_multiplier(dist: f64, radius: f64, feather: f64, strength: f64) -> f64 {
    // Feather band start and end.
    let inner = radius;
    let outer = radius + feather.max(1e-9);

    let t = ((dist - inner) / (outer - inner)).clamp(0.0, 1.0);
    // Smooth-step (cubic Hermite): 3t²–2t³
    let s = t * t * (3.0 - 2.0 * t);
    1.0 - s * strength
}

#[inline]
fn check_range(name: &'static str, value: f64, min: f64, max: f64) -> Result<(), VignetteError> {
    if value < min || value > max {
        Err(VignetteError::ParameterOutOfRange {
            name,
            value,
            min,
            max,
        })
    } else {
        Ok(())
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize]
    }

    // 1. Default config validates without error.
    #[test]
    fn test_default_config_valid() {
        assert!(VignetteConfig::default().validate().is_ok());
    }

    // 2. Zero dimensions return an error.
    #[test]
    fn test_zero_dimensions() {
        let cfg = VignetteConfig::default();
        assert!(matches!(
            build_vignette_map(0, 4, &cfg),
            Err(VignetteError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            build_vignette_map(4, 0, &cfg),
            Err(VignetteError::InvalidDimensions { .. })
        ));
    }

    // 3. Map has the correct number of entries.
    #[test]
    fn test_map_size() {
        let cfg = VignetteConfig::default();
        let map = build_vignette_map(16, 8, &cfg).unwrap();
        assert_eq!(map.multipliers.len(), 16 * 8);
        assert_eq!(map.width, 16);
        assert_eq!(map.height, 8);
    }

    // 4. Centre pixel is not darkened (multiplier ≈ 1.0).
    #[test]
    fn test_centre_bright() {
        let cfg = VignetteConfig {
            radius: 0.5,
            feather: 0.3,
            strength: 1.0,
            shape: VignetteShape::Circular,
        };
        let w = 101u32;
        let h = 101u32;
        let map = build_vignette_map(w, h, &cfg).unwrap();
        let centre_idx = (h / 2) as usize * w as usize + (w / 2) as usize;
        // Centre multiplier must be very close to 1.0.
        assert!(
            (map.multipliers[centre_idx] - 1.0).abs() < 0.05,
            "centre mult = {}",
            map.multipliers[centre_idx]
        );
    }

    // 5. Corner pixel is darkened when strength = 1.
    #[test]
    fn test_corner_darkened() {
        let cfg = VignetteConfig {
            radius: 0.3,
            feather: 0.2,
            strength: 1.0,
            shape: VignetteShape::Elliptical,
        };
        let w = 64u32;
        let h = 64u32;
        let map = build_vignette_map(w, h, &cfg).unwrap();
        // Top-left corner.
        let corner_mult = map.multipliers[0];
        assert!(
            corner_mult < 0.5,
            "expected corner to be dark, got mult = {}",
            corner_mult
        );
    }

    // 6. strength=0 means no darkening (all multipliers == 1.0).
    #[test]
    fn test_no_strength_no_effect() {
        let cfg = VignetteConfig {
            radius: 0.0,
            feather: 0.0,
            strength: 0.0,
            shape: VignetteShape::Circular,
        };
        let map = build_vignette_map(8, 8, &cfg).unwrap();
        for &m in &map.multipliers {
            assert!((m - 1.0).abs() < 1e-5, "expected 1.0, got {}", m);
        }
    }

    // 7. apply_vignette_map modifies pixel values.
    #[test]
    fn test_apply_darkens_corners() {
        let cfg = VignetteConfig {
            radius: 0.1,
            feather: 0.1,
            strength: 1.0,
            shape: VignetteShape::Elliptical,
        };
        let w = 32u32;
        let h = 32u32;
        let map = build_vignette_map(w, h, &cfg).unwrap();
        let mut plane = solid(w, h, 200);
        apply_vignette_map(&mut plane, &map).unwrap();
        // Corner pixel should have become darker.
        assert!(plane[0] < 200, "corner should darken, got {}", plane[0]);
    }

    // 8. apply_vignette convenience function produces correct buffer size.
    #[test]
    fn test_apply_vignette_size() {
        let cfg = VignetteConfig::default();
        let plane = solid(10, 10, 128);
        let out = apply_vignette(&plane, 10, 10, &cfg).unwrap();
        assert_eq!(out.len(), 100);
    }

    // 9. Buffer size mismatch is rejected.
    #[test]
    fn test_buffer_mismatch() {
        let cfg = VignetteConfig::default();
        let map = build_vignette_map(8, 8, &cfg).unwrap();
        let mut short = vec![0u8; 10];
        assert!(matches!(
            apply_vignette_map(&mut short, &map),
            Err(VignetteError::BufferSizeMismatch { .. })
        ));
    }

    // 10. Out-of-range parameter is rejected.
    #[test]
    fn test_parameter_out_of_range() {
        let cfg = VignetteConfig {
            radius: 1.5,
            ..VignetteConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(VignetteError::ParameterOutOfRange { .. })
        ));
    }

    // 11. Circular and elliptical produce different results on non-square frames.
    #[test]
    fn test_circular_vs_elliptical() {
        let base = VignetteConfig {
            radius: 0.5,
            feather: 0.3,
            strength: 1.0,
            shape: VignetteShape::Circular,
        };
        let mut ellip = base.clone();
        ellip.shape = VignetteShape::Elliptical;

        let w = 80u32;
        let h = 40u32;
        let map_c = build_vignette_map(w, h, &base).unwrap();
        let map_e = build_vignette_map(w, h, &ellip).unwrap();

        // At least one pixel should differ between the two maps.
        let differs = map_c
            .multipliers
            .iter()
            .zip(map_e.multipliers.iter())
            .any(|(a, b)| (a - b).abs() > 0.01);
        assert!(
            differs,
            "circular and elliptical maps should differ for non-square frames"
        );
    }

    // 12. Applying zero-strength vignette leaves pixels unchanged.
    #[test]
    fn test_zero_strength_identity() {
        let cfg = VignetteConfig {
            radius: 0.0,
            feather: 0.5,
            strength: 0.0,
            shape: VignetteShape::Elliptical,
        };
        let src = solid(12, 12, 100);
        let out = apply_vignette(&src, 12, 12, &cfg).unwrap();
        assert_eq!(out, src);
    }
}
