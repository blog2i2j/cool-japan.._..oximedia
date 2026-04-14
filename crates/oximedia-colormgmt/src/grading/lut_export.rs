//! 3D LUT generation and export for color grading workflows.
//!
//! Provides generation of 3D LUTs by baking arbitrary color transformation
//! functions into dense lookup tables, with export to the industry-standard
//! `.cube` format (used by DaVinci Resolve, Adobe Premiere, Nuke, etc.) and
//! a compact binary representation.
//!
//! ## Format Notes
//!
//! The `.cube` format stores a 3D LUT as a plain-text file:
//!
//! ```text
//! LUT_3D_SIZE <N>
//! <R_in> <G_in> <B_in>   # output at index 0
//! ...
//! ```
//!
//! Values are in \[0, 1\] floating-point. Index order is B fastest, then G, then R
//! (per Adobe specification).
//!
//! ## References
//!
//! - "Cube LUT Specification 1.0" — Adobe Systems, 2013
//! - ACES CTL reference implementation (Academy S-2014-006)

use crate::error::{ColorError, Result};

/// Maximum supported 3D LUT size (65³ requires ~1 MB).
const MAX_LUT_SIZE: usize = 65;

/// Minimum 3D LUT size.
const MIN_LUT_SIZE: usize = 2;

/// A fully baked 3D LUT for color grading.
///
/// The LUT maps an `[R_in, G_in, B_in]` triplet (each in \[0, 1\]) to an
/// output `[R_out, G_out, B_out]` triplet. The internal layout follows the
/// `.cube` convention: B index varies fastest, then G, then R.
#[derive(Debug, Clone)]
pub struct GradingLut3D {
    /// Number of samples per axis (same for R, G, B).
    pub size: usize,
    /// Flat table: `size³` entries of `[f32; 3]` in B-fastest order.
    pub data: Vec<[f32; 3]>,
    /// Optional human-readable title (written to `.cube` header).
    pub title: Option<String>,
}

impl GradingLut3D {
    /// Builds a 3D LUT by evaluating `transform` at every lattice point.
    ///
    /// # Arguments
    ///
    /// * `size` — Lattice size per axis (2–65).  Common values: 17, 33, 65.
    /// * `transform` — A callable `Fn([f32; 3]) -> [f32; 3]` applied to each
    ///   `[R, G, B]` sample in \[0, 1\].
    /// * `title` — Optional LUT title for the `.cube` header.
    ///
    /// # Errors
    ///
    /// Returns `ColorError::InvalidInput` if `size` is outside \[2, 65\].
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_colormgmt::grading::lut_export::GradingLut3D;
    ///
    /// // Identity LUT
    /// let lut = GradingLut3D::from_fn(17, |rgb| rgb, None).expect("identity lut");
    /// assert_eq!(lut.size, 17);
    /// assert_eq!(lut.data.len(), 17 * 17 * 17);
    /// ```
    pub fn from_fn<F>(size: usize, transform: F, title: Option<String>) -> Result<Self>
    where
        F: Fn([f32; 3]) -> [f32; 3],
    {
        if size < MIN_LUT_SIZE || size > MAX_LUT_SIZE {
            return Err(ColorError::InvalidColorSpace(format!(
                "LUT size {size} is outside the valid range [{MIN_LUT_SIZE}, {MAX_LUT_SIZE}]"
            )));
        }

        let total = size * size * size;
        let mut data = Vec::with_capacity(total);
        let scale = if size > 1 { (size - 1) as f32 } else { 1.0 };

        // B varies fastest (Adobe .cube order).
        for r_idx in 0..size {
            for g_idx in 0..size {
                for b_idx in 0..size {
                    let r = r_idx as f32 / scale;
                    let g = g_idx as f32 / scale;
                    let b = b_idx as f32 / scale;
                    data.push(transform([r, g, b]));
                }
            }
        }

        Ok(Self { size, data, title })
    }

    /// Creates an identity 3D LUT (input == output for all lattice points).
    ///
    /// # Errors
    ///
    /// Returns an error if `size` is outside \[2, 65\].
    pub fn identity(size: usize) -> Result<Self> {
        Self::from_fn(size, |rgb| rgb, Some("Identity".to_string()))
    }

    /// Exports the LUT to a `.cube`-formatted string.
    ///
    /// The output can be written to a file with a `.cube` extension and loaded
    /// directly by DaVinci Resolve, Adobe Premiere, Nuke, and other DCCs.
    #[must_use]
    pub fn to_cube_string(&self) -> String {
        let mut out = String::with_capacity(self.data.len() * 24 + 128);

        if let Some(title) = &self.title {
            // Sanitise: cube titles must not contain newlines.
            let sanitised: String = title
                .chars()
                .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                .collect();
            out.push_str(&format!("TITLE \"{sanitised}\"\n"));
        }
        out.push_str(&format!("LUT_3D_SIZE {}\n", self.size));
        out.push('\n');

        for entry in &self.data {
            out.push_str(&format!(
                "{:.6} {:.6} {:.6}\n",
                entry[0], entry[1], entry[2]
            ));
        }

        out
    }

    /// Parses a `.cube`-formatted string back into a [`GradingLut3D`].
    ///
    /// # Errors
    ///
    /// Returns `ColorError::ParseError` on malformed input.
    pub fn from_cube_string(src: &str) -> Result<Self> {
        let mut size: Option<usize> = None;
        let mut title: Option<String> = None;
        let mut data: Vec<[f32; 3]> = Vec::new();

        for line in src.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("LUT_3D_SIZE") {
                let n: usize = rest.trim().parse().map_err(|_| {
                    ColorError::Parse(format!(
                        "invalid LUT_3D_SIZE value: {trimmed}"
                    ))
                })?;
                size = Some(n);
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("TITLE") {
                let t = rest.trim().trim_matches('"');
                title = Some(t.to_string());
                continue;
            }
            // Skip other keywords (DOMAIN_MIN, DOMAIN_MAX, etc.)
            if trimmed
                .split_ascii_whitespace()
                .next()
                .map_or(false, |w| w.chars().all(|c| c.is_ascii_uppercase() || c == '_'))
            {
                continue;
            }
            // Data line
            let mut parts = trimmed.split_ascii_whitespace();
            let parse_f32 = |s: Option<&str>| -> Result<f32> {
                s.ok_or_else(|| ColorError::Parse("missing component".to_string()))?
                    .parse::<f32>()
                    .map_err(|e| ColorError::Parse(format!("bad float: {e}")))
            };
            let r = parse_f32(parts.next())?;
            let g = parse_f32(parts.next())?;
            let b = parse_f32(parts.next())?;
            data.push([r, g, b]);
        }

        let size = size.ok_or_else(|| {
            ColorError::Parse("missing LUT_3D_SIZE directive".to_string())
        })?;

        let expected = size * size * size;
        if data.len() != expected {
            return Err(ColorError::Parse(format!(
                "expected {expected} data entries for size {size}, got {}",
                data.len()
            )));
        }

        Ok(Self { size, data, title })
    }

    /// Applies the LUT to a single pixel via trilinear interpolation.
    ///
    /// The input `rgb` components are clamped to \[0, 1\] before lookup.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let size = self.size;
        let scale = (size - 1) as f32;

        let r = rgb[0].clamp(0.0, 1.0) * scale;
        let g = rgb[1].clamp(0.0, 1.0) * scale;
        let b = rgb[2].clamp(0.0, 1.0) * scale;

        let r0 = (r as usize).min(size - 2);
        let g0 = (g as usize).min(size - 2);
        let b0 = (b as usize).min(size - 2);
        let r1 = r0 + 1;
        let g1 = g0 + 1;
        let b1 = b0 + 1;

        let dr = r - r0 as f32;
        let dg = g - g0 as f32;
        let db = b - b0 as f32;

        let idx = |ri: usize, gi: usize, bi: usize| ri * size * size + gi * size + bi;

        // Trilinear interpolation
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
        };

        let c000 = self.data[idx(r0, g0, b0)];
        let c001 = self.data[idx(r0, g0, b1)];
        let c010 = self.data[idx(r0, g1, b0)];
        let c011 = self.data[idx(r0, g1, b1)];
        let c100 = self.data[idx(r1, g0, b0)];
        let c101 = self.data[idx(r1, g0, b1)];
        let c110 = self.data[idx(r1, g1, b0)];
        let c111 = self.data[idx(r1, g1, b1)];

        let c00 = lerp3(c000, c001, db);
        let c01 = lerp3(c010, c011, db);
        let c10 = lerp3(c100, c101, db);
        let c11 = lerp3(c110, c111, db);

        let c0 = lerp3(c00, c01, dg);
        let c1 = lerp3(c10, c11, dg);

        lerp3(c0, c1, dr)
    }

    /// Composes this LUT with another: `self.apply(other.apply(rgb))`.
    ///
    /// The resulting LUT has the same size as `self`.
    ///
    /// # Errors
    ///
    /// Returns an error if `self.size` is outside the valid range.
    pub fn compose(&self, other: &GradingLut3D) -> Result<GradingLut3D> {
        GradingLut3D::from_fn(
            self.size,
            |rgb| self.apply(other.apply(rgb)),
            None,
        )
    }

    /// Returns the number of lattice points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the LUT has no data.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ── Helper: common grading transforms ────────────────────────────────────────

/// Builds a LUT that applies an exposure offset in stops.
///
/// Each channel is scaled by `2^stops`.
///
/// # Errors
///
/// See [`GradingLut3D::from_fn`].
pub fn exposure_lut(size: usize, stops: f32) -> Result<GradingLut3D> {
    let gain = (2.0_f32).powf(stops);
    GradingLut3D::from_fn(
        size,
        move |[r, g, b]| [
            (r * gain).clamp(0.0, 1.0),
            (g * gain).clamp(0.0, 1.0),
            (b * gain).clamp(0.0, 1.0),
        ],
        Some(format!("Exposure {stops:+.2} stops")),
    )
}

/// Builds a LUT that applies a global contrast adjustment around mid-grey (0.5).
///
/// `contrast > 1.0` increases contrast; `contrast < 1.0` reduces it.
///
/// # Errors
///
/// See [`GradingLut3D::from_fn`].
pub fn contrast_lut(size: usize, contrast: f32) -> Result<GradingLut3D> {
    GradingLut3D::from_fn(
        size,
        move |[r, g, b]| {
            let c = |v: f32| (0.5 + (v - 0.5) * contrast).clamp(0.0, 1.0);
            [c(r), c(g), c(b)]
        },
        Some(format!("Contrast {contrast:.3}")),
    )
}

/// Builds a saturation-adjustment LUT using Rec.709 luma weights.
///
/// `saturation = 1.0` is identity; `0.0` is greyscale; `> 1.0` boosts chroma.
///
/// # Errors
///
/// See [`GradingLut3D::from_fn`].
pub fn saturation_lut(size: usize, saturation: f32) -> Result<GradingLut3D> {
    GradingLut3D::from_fn(
        size,
        move |[r, g, b]| {
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let sat = |v: f32| (luma + saturation * (v - luma)).clamp(0.0, 1.0);
            [sat(r), sat(g), sat(b)]
        },
        Some(format!("Saturation {saturation:.3}")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction tests ─────────────────────────────────────────────────

    #[test]
    fn test_identity_lut_size_17() {
        let lut = GradingLut3D::identity(17).expect("identity 17");
        assert_eq!(lut.size, 17);
        assert_eq!(lut.data.len(), 17 * 17 * 17);
    }

    #[test]
    fn test_identity_apply_midgrey() {
        let lut = GradingLut3D::identity(17).expect("identity 17");
        let result = lut.apply([0.5, 0.5, 0.5]);
        for i in 0..3 {
            assert!(
                (result[i] - 0.5).abs() < 0.02,
                "identity mid-grey channel {i}: {}, expected ~0.5",
                result[i]
            );
        }
    }

    #[test]
    fn test_identity_apply_black() {
        let lut = GradingLut3D::identity(17).expect("identity 17");
        let result = lut.apply([0.0, 0.0, 0.0]);
        for i in 0..3 {
            assert!(result[i].abs() < 1e-4, "black channel {i}: {}", result[i]);
        }
    }

    #[test]
    fn test_identity_apply_white() {
        let lut = GradingLut3D::identity(17).expect("identity 17");
        let result = lut.apply([1.0, 1.0, 1.0]);
        for i in 0..3 {
            assert!(
                (result[i] - 1.0).abs() < 1e-4,
                "white channel {i}: {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_invalid_size_below_min() {
        assert!(GradingLut3D::identity(1).is_err());
    }

    #[test]
    fn test_invalid_size_above_max() {
        assert!(GradingLut3D::identity(66).is_err());
    }

    // ── .cube round-trip tests ─────────────────────────────────────────────

    #[test]
    fn test_cube_round_trip() {
        let original = GradingLut3D::identity(5).expect("identity 5");
        let cube_str = original.to_cube_string();
        let parsed = GradingLut3D::from_cube_string(&cube_str).expect("parse cube");
        assert_eq!(parsed.size, original.size);
        assert_eq!(parsed.data.len(), original.data.len());
        for (a, b) in original.data.iter().zip(parsed.data.iter()) {
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() < 1e-5, "data mismatch at channel {i}");
            }
        }
    }

    #[test]
    fn test_cube_has_size_directive() {
        let lut = GradingLut3D::identity(9).expect("identity 9");
        let cube = lut.to_cube_string();
        assert!(
            cube.contains("LUT_3D_SIZE 9"),
            "cube output missing LUT_3D_SIZE directive"
        );
    }

    #[test]
    fn test_cube_title_written() {
        let lut = GradingLut3D::from_fn(5, |rgb| rgb, Some("TestTitle".to_string()))
            .expect("lut with title");
        let cube = lut.to_cube_string();
        assert!(cube.contains("TestTitle"), "title not written to .cube");
    }

    // ── Grading transform LUT tests ────────────────────────────────────────

    #[test]
    fn test_exposure_lut_one_stop_brighter() {
        let lut = exposure_lut(9, 1.0).expect("exposure +1 stop");
        let result = lut.apply([0.25, 0.25, 0.25]);
        // 0.25 * 2^1 = 0.5 (clamped to 1.0)
        for i in 0..3 {
            assert!(
                (result[i] - 0.5).abs() < 0.02,
                "exposure +1 channel {i}: {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_saturation_lut_greyscale() {
        let lut = saturation_lut(9, 0.0).expect("saturation 0");
        let result = lut.apply([0.8, 0.2, 0.4]);
        // All channels should equal luma
        let luma = 0.2126_f32 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.4;
        for i in 0..3 {
            assert!(
                (result[i] - luma).abs() < 0.02,
                "saturation 0 channel {i}: {}, expected luma {}",
                result[i],
                luma
            );
        }
    }

    #[test]
    fn test_contrast_lut_identity_at_one() {
        let lut = contrast_lut(9, 1.0).expect("contrast 1.0");
        let input = [0.3_f32, 0.6, 0.9];
        let result = lut.apply(input);
        for i in 0..3 {
            assert!(
                (result[i] - input[i]).abs() < 0.02,
                "contrast 1.0 identity check channel {i}: {} vs {}",
                result[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_compose_identity() {
        let id1 = GradingLut3D::identity(9).expect("id1");
        let id2 = GradingLut3D::identity(9).expect("id2");
        let composed = id1.compose(&id2).expect("compose");
        let input = [0.4_f32, 0.6, 0.2];
        let result = composed.apply(input);
        for i in 0..3 {
            assert!(
                (result[i] - input[i]).abs() < 0.02,
                "composed identity channel {i}: {}",
                result[i]
            );
        }
    }
}
