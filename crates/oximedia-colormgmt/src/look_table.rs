//! Look modification table (LMT) utilities.
//!
//! This module provides tools for creating, applying, and serialising *look*
//! transforms — the intent-specific colour grade that defines the overall
//! visual character of a production before the technical Output Transform.
//!
//! Specifically it implements:
//!
//! - **`LookTransform`**: An ASC CDL (slope/offset/power + saturation) stage
//!   followed by an optional 1-D per-channel LUT.
//! - **Look file serialisation**: Human-readable text format (CDL + LUT in a
//!   single `.look` file) with corresponding deserialisation.
//! - **Preview strip generation**: Renders a 1-D gradient (ramp) through the
//!   look transform so artists can see the overall tone response without
//!   rendering a full frame.
//!
//! # ASC CDL
//!
//! The American Society of Cinematographers Color Decision List (ASC CDL)
//! defines a simple but powerful color transform:
//!
//! ```text
//! out = clamp((in × slope + offset) ^ power)
//! ```
//!
//! After the CDL operation a saturation adjustment is applied in scene-linear
//! space using the ITU-R BT.709 luminance coefficients.
//!
//! # 1-D LUT
//!
//! The optional per-channel 1-D LUT is applied after CDL. It stores evenly
//! sampled values in [0, 1] → [0, 1] and uses linear interpolation.
//!
//! # Example
//!
//! ```
//! use oximedia_colormgmt::look_table::{
//!     AscCdl, LookTransform, LookLut1d, PreviewStrip,
//! };
//!
//! // Build a warm, slightly contrasty look
//! let cdl = AscCdl {
//!     slope:  [1.1, 1.0, 0.9],
//!     offset: [0.02, 0.0, -0.01],
//!     power:  [0.95, 1.0, 1.05],
//!     saturation: 1.1,
//! };
//!
//! let lut = LookLut1d::identity(65);
//! let look = LookTransform::new(cdl, Some(lut));
//!
//! let pixel = [0.5_f64, 0.4, 0.3];
//! let out = look.apply(pixel);
//! assert!(out[0] >= 0.0 && out[0] <= 1.0);
//!
//! // Generate a preview ramp
//! let strip = PreviewStrip::new(&look, 256);
//! assert_eq!(strip.pixels.len(), 256);
//! ```

// ── ASC CDL ───────────────────────────────────────────────────────────────────

/// ASC CDL (Color Decision List) parameters.
///
/// Each `slope`, `offset`, and `power` array contains one value per channel
/// (R, G, B). The formula applied per-channel is:
///
/// ```text
/// out = clamp((in × slope + offset) ^ power, 0.0, 1.0)
/// ```
///
/// After CDL, a saturation adjustment is applied using the Rec. 709 luminance
/// coefficients (0.2126 R + 0.7152 G + 0.0722 B).
#[derive(Debug, Clone, PartialEq)]
pub struct AscCdl {
    /// Per-channel slope (gain). Typically > 0. Default: [1, 1, 1].
    pub slope: [f64; 3],
    /// Per-channel offset (lift). Typically near 0. Default: [0, 0, 0].
    pub offset: [f64; 3],
    /// Per-channel power (gamma). Positive non-zero. Default: [1, 1, 1].
    pub power: [f64; 3],
    /// Global saturation multiplier. 1.0 = no change, 0.0 = monochrome.
    pub saturation: f64,
}

impl Default for AscCdl {
    fn default() -> Self {
        Self {
            slope: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            power: [1.0, 1.0, 1.0],
            saturation: 1.0,
        }
    }
}

impl AscCdl {
    /// Create an identity CDL (no colour change).
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Create a CDL with custom parameters.
    ///
    /// `power` values are clamped to a minimum of `1e-6` to avoid undefined
    /// behaviour at zero.
    #[must_use]
    pub fn new(slope: [f64; 3], offset: [f64; 3], power: [f64; 3], saturation: f64) -> Self {
        Self {
            slope,
            offset,
            power: [power[0].max(1e-6), power[1].max(1e-6), power[2].max(1e-6)],
            saturation: saturation.max(0.0),
        }
    }

    /// Apply the CDL transform to a single pixel [R, G, B].
    ///
    /// Negative intermediate values have their sign preserved before the
    /// power function (mirror power), which avoids NaN.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        // Step 1: slope + offset + power
        let mut out = [0.0_f64; 3];
        for i in 0..3 {
            let v = rgb[i] * self.slope[i] + self.offset[i];
            // Mirror power: preserve sign for negative inputs
            out[i] = if v < 0.0 {
                -((-v).powf(self.power[i]))
            } else {
                v.powf(self.power[i])
            };
            out[i] = out[i].clamp(0.0, 1.0);
        }

        // Step 2: saturation using Rec. 709 luminance coefficients
        if (self.saturation - 1.0).abs() > 1e-12 {
            let luma = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
            for i in 0..3 {
                out[i] = (luma + self.saturation * (out[i] - luma)).clamp(0.0, 1.0);
            }
        }

        out
    }

    /// Validate that all parameters are in reasonable ranges.
    ///
    /// Returns an error string describing the first invalid parameter found.
    pub fn validate(&self) -> Result<(), String> {
        for (i, &s) in self.slope.iter().enumerate() {
            if !s.is_finite() || s < 0.0 {
                return Err(format!("slope[{i}] must be non-negative finite: {s}"));
            }
        }
        for (i, &o) in self.offset.iter().enumerate() {
            if !o.is_finite() {
                return Err(format!("offset[{i}] must be finite: {o}"));
            }
        }
        for (i, &p) in self.power.iter().enumerate() {
            if !p.is_finite() || p <= 0.0 {
                return Err(format!("power[{i}] must be positive finite: {p}"));
            }
        }
        if !self.saturation.is_finite() || self.saturation < 0.0 {
            return Err(format!(
                "saturation must be non-negative finite: {}",
                self.saturation
            ));
        }
        Ok(())
    }
}

// ── 1-D LUT ───────────────────────────────────────────────────────────────────

/// A per-channel 1-D look-up table with linear interpolation.
///
/// All three channels share the same number of entries. Each entry maps an
/// evenly-sampled input in [0, 1] to an output in [0, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct LookLut1d {
    /// Red channel LUT entries (evenly sampled, input domain [0, 1]).
    pub red: Vec<f64>,
    /// Green channel LUT entries (evenly sampled, input domain [0, 1]).
    pub green: Vec<f64>,
    /// Blue channel LUT entries (evenly sampled, input domain [0, 1]).
    pub blue: Vec<f64>,
}

impl LookLut1d {
    /// Create an identity LUT with `n_entries` points.
    ///
    /// An identity LUT returns the same value as the input for all samples.
    #[must_use]
    pub fn identity(n_entries: usize) -> Self {
        let n = n_entries.max(2);
        let values: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        Self {
            red: values.clone(),
            green: values.clone(),
            blue: values,
        }
    }

    /// Create a LUT from three channel arrays.
    ///
    /// All channels must have the same length (≥ 2). Returns an error if they
    /// do not.
    pub fn from_channels(
        red: Vec<f64>,
        green: Vec<f64>,
        blue: Vec<f64>,
    ) -> Result<Self, String> {
        if red.len() < 2 || green.len() < 2 || blue.len() < 2 {
            return Err("LUT channels must have at least 2 entries".to_string());
        }
        if red.len() != green.len() || red.len() != blue.len() {
            return Err(format!(
                "LUT channel lengths must match: R={}, G={}, B={}",
                red.len(),
                green.len(),
                blue.len()
            ));
        }
        Ok(Self { red, green, blue })
    }

    /// Apply the LUT to a single pixel [R, G, B] using linear interpolation.
    ///
    /// Input values outside [0, 1] are clamped before lookup.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            lut1d_interp(&self.red, rgb[0]),
            lut1d_interp(&self.green, rgb[1]),
            lut1d_interp(&self.blue, rgb[2]),
        ]
    }

    /// Number of entries in the LUT (all channels share the same size).
    #[must_use]
    pub fn len(&self) -> usize {
        self.red.len()
    }

    /// Return `true` if the LUT has no entries (degenerate state).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.red.is_empty()
    }

    /// Check whether this LUT is equivalent to an identity (within tolerance).
    #[must_use]
    pub fn is_identity(&self, tolerance: f64) -> bool {
        let n = self.red.len();
        if n < 2 {
            return false;
        }
        for (i, ((r, g), b)) in self.red.iter().zip(&self.green).zip(&self.blue).enumerate() {
            let expected = i as f64 / (n - 1) as f64;
            if (r - expected).abs() > tolerance
                || (g - expected).abs() > tolerance
                || (b - expected).abs() > tolerance
            {
                return false;
            }
        }
        true
    }
}

/// Linear interpolation into a 1-D LUT table.
#[inline]
fn lut1d_interp(table: &[f64], x: f64) -> f64 {
    let n = table.len();
    debug_assert!(n >= 2);
    let t = x.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = (t.floor() as usize).min(n - 2);
    let frac = t - lo as f64;
    table[lo] * (1.0 - frac) + table[lo + 1] * frac
}

// ── LookTransform ──────────────────────────────────────────────────────────────

/// A complete look transform: ASC CDL followed by an optional 1-D LUT.
///
/// The transform is applied as:
/// 1. ASC CDL (slope/offset/power + saturation).
/// 2. Optional per-channel 1-D LUT (linear interpolation).
#[derive(Debug, Clone)]
pub struct LookTransform {
    /// The ASC CDL component of the look.
    pub cdl: AscCdl,
    /// Optional per-channel 1-D LUT applied after CDL.
    pub lut: Option<LookLut1d>,
    /// Human-readable name for the look.
    pub name: String,
    /// Optional description / production notes.
    pub description: String,
}

impl LookTransform {
    /// Create a new look transform.
    #[must_use]
    pub fn new(cdl: AscCdl, lut: Option<LookLut1d>) -> Self {
        Self {
            cdl,
            lut,
            name: String::new(),
            description: String::new(),
        }
    }

    /// Create a named look transform.
    #[must_use]
    pub fn named(name: impl Into<String>, cdl: AscCdl, lut: Option<LookLut1d>) -> Self {
        Self {
            cdl,
            lut,
            name: name.into(),
            description: String::new(),
        }
    }

    /// Apply the look transform to a single pixel [R, G, B].
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        let after_cdl = self.cdl.apply(rgb);
        if let Some(lut) = &self.lut {
            lut.apply(after_cdl)
        } else {
            after_cdl
        }
    }

    /// Apply the look transform to every pixel in a flat buffer (RGB triples).
    ///
    /// Returns an error if the buffer length is not divisible by 3.
    pub fn apply_buffer(&self, buffer: &mut [f64]) -> Result<(), &'static str> {
        if buffer.len() % 3 != 0 {
            return Err("Buffer length must be divisible by 3");
        }
        for chunk in buffer.chunks_exact_mut(3) {
            let out = self.apply([chunk[0], chunk[1], chunk[2]]);
            chunk[0] = out[0];
            chunk[1] = out[1];
            chunk[2] = out[2];
        }
        Ok(())
    }

    /// Serialise the look transform to a human-readable text string.
    ///
    /// Format:
    /// ```text
    /// [look]
    /// name = "My Look"
    /// description = "..."
    ///
    /// [cdl]
    /// slope = 1.1 1.0 0.9
    /// offset = 0.02 0.0 -0.01
    /// power = 0.95 1.0 1.05
    /// saturation = 1.1
    ///
    /// [lut1d]
    /// size = 65
    /// red = 0.0 0.015625 ... 1.0
    /// green = 0.0 0.015625 ... 1.0
    /// blue = 0.0 0.015625 ... 1.0
    /// ```
    #[must_use]
    pub fn serialize(&self) -> String {
        let mut s = String::with_capacity(512);

        s.push_str("[look]\n");
        s.push_str(&format!("name = \"{}\"\n", self.name));
        s.push_str(&format!("description = \"{}\"\n", self.description));
        s.push('\n');

        s.push_str("[cdl]\n");
        s.push_str(&format!(
            "slope = {} {} {}\n",
            self.cdl.slope[0], self.cdl.slope[1], self.cdl.slope[2]
        ));
        s.push_str(&format!(
            "offset = {} {} {}\n",
            self.cdl.offset[0], self.cdl.offset[1], self.cdl.offset[2]
        ));
        s.push_str(&format!(
            "power = {} {} {}\n",
            self.cdl.power[0], self.cdl.power[1], self.cdl.power[2]
        ));
        s.push_str(&format!("saturation = {}\n", self.cdl.saturation));

        if let Some(lut) = &self.lut {
            s.push('\n');
            s.push_str("[lut1d]\n");
            s.push_str(&format!("size = {}\n", lut.len()));
            s.push_str("red = ");
            s.push_str(&float_vec_to_str(&lut.red));
            s.push('\n');
            s.push_str("green = ");
            s.push_str(&float_vec_to_str(&lut.green));
            s.push('\n');
            s.push_str("blue = ");
            s.push_str(&float_vec_to_str(&lut.blue));
            s.push('\n');
        }

        s
    }

    /// Deserialise a look transform from a text string produced by
    /// [`LookTransform::serialize`].
    ///
    /// Returns an error on parse failure.
    pub fn deserialize(text: &str) -> Result<Self, String> {
        let mut name = String::new();
        let mut description = String::new();
        let mut slope = [1.0_f64; 3];
        let mut offset = [0.0_f64; 3];
        let mut power = [1.0_f64; 3];
        let mut saturation = 1.0_f64;
        let mut lut_red: Option<Vec<f64>> = None;
        let mut lut_green: Option<Vec<f64>> = None;
        let mut lut_blue: Option<Vec<f64>> = None;

        let mut current_section = "";

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line == "[look]" {
                current_section = "look";
                continue;
            }
            if line == "[cdl]" {
                current_section = "cdl";
                continue;
            }
            if line == "[lut1d]" {
                current_section = "lut1d";
                continue;
            }

            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let val = val.trim();

            match current_section {
                "look" => match key {
                    "name" => name = val.trim_matches('"').to_string(),
                    "description" => description = val.trim_matches('"').to_string(),
                    _ => {}
                },
                "cdl" => match key {
                    "slope" => slope = parse_f64_3(val)?,
                    "offset" => offset = parse_f64_3(val)?,
                    "power" => power = parse_f64_3(val)?,
                    "saturation" => {
                        saturation = val
                            .parse::<f64>()
                            .map_err(|e| format!("saturation: {e}"))?;
                    }
                    _ => {}
                },
                "lut1d" => match key {
                    "size" => {} // informational, we use the actual data length
                    "red" => lut_red = Some(parse_f64_vec(val)?),
                    "green" => lut_green = Some(parse_f64_vec(val)?),
                    "blue" => lut_blue = Some(parse_f64_vec(val)?),
                    _ => {}
                },
                _ => {}
            }
        }

        let cdl = AscCdl::new(slope, offset, power, saturation);
        let lut = match (lut_red, lut_green, lut_blue) {
            (Some(r), Some(g), Some(b)) => Some(LookLut1d::from_channels(r, g, b)?),
            _ => None,
        };

        let mut look = LookTransform::new(cdl, lut);
        look.name = name;
        look.description = description;
        Ok(look)
    }
}

// ── Serialisation helpers ──────────────────────────────────────────────────────

fn float_vec_to_str(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{x:.8}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_f64_3(s: &str) -> Result<[f64; 3], String> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(format!("Expected 3 values, got {}: {:?}", parts.len(), s));
    }
    Ok([
        parts[0].parse::<f64>().map_err(|e| format!("parse error: {e}"))?,
        parts[1].parse::<f64>().map_err(|e| format!("parse error: {e}"))?,
        parts[2].parse::<f64>().map_err(|e| format!("parse error: {e}"))?,
    ])
}

fn parse_f64_vec(s: &str) -> Result<Vec<f64>, String> {
    s.split_whitespace()
        .map(|t| t.parse::<f64>().map_err(|e| format!("parse error in vec: {e}")))
        .collect()
}

// ── Preview Strip ──────────────────────────────────────────────────────────────

/// A linear gradient (ramp) processed through a look transform.
///
/// Each element in `pixels` is an [R, G, B] triplet for one step of the
/// gradient, from black (0, 0, 0) to white (1, 1, 1).
///
/// This is useful for artists to quickly assess the tone response and colour
/// grade of a look before committing to a full render.
#[derive(Debug, Clone)]
pub struct PreviewStrip {
    /// The processed pixels of the strip, from shadow (index 0) to highlight.
    pub pixels: Vec<[f64; 3]>,
    /// Width in pixels.
    pub width: usize,
}

impl PreviewStrip {
    /// Generate a `width`-pixel preview strip through the given look.
    ///
    /// Each pixel is a neutral grey ramp value passed through the look.
    #[must_use]
    pub fn new(look: &LookTransform, width: usize) -> Self {
        let n = width.max(2);
        let pixels: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                look.apply([t, t, t])
            })
            .collect();
        Self { pixels, width: n }
    }

    /// Return the average output brightness across the strip (Y channel).
    #[must_use]
    pub fn mean_brightness(&self) -> f64 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .pixels
            .iter()
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .sum();
        sum / self.pixels.len() as f64
    }

    /// Return the first and last pixel values (shadow and highlight).
    #[must_use]
    pub fn shadow_highlight(&self) -> ([f64; 3], [f64; 3]) {
        (
            self.pixels.first().copied().unwrap_or([0.0; 3]),
            self.pixels.last().copied().unwrap_or([1.0; 3]),
        )
    }

    /// Check whether the strip is monotonically non-decreasing in luminance.
    ///
    /// A well-behaved look should preserve tone ordering.
    #[must_use]
    pub fn is_monotone(&self) -> bool {
        let luma: Vec<f64> = self
            .pixels
            .iter()
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .collect();
        luma.windows(2).all(|w| w[1] >= w[0] - 1e-10)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdl_identity() {
        let cdl = AscCdl::identity();
        let pixel = [0.5, 0.3, 0.7];
        let out = cdl.apply(pixel);
        for i in 0..3 {
            assert!(
                (out[i] - pixel[i]).abs() < 1e-10,
                "Identity CDL should not change pixel: channel {i}: {} vs {}",
                out[i],
                pixel[i]
            );
        }
    }

    #[test]
    fn test_cdl_slope_scales() {
        let cdl = AscCdl::new([2.0, 1.0, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        let pixel = [0.4, 0.4, 0.4];
        let out = cdl.apply(pixel);
        assert!((out[0] - 0.8).abs() < 1e-10, "R slope=2 should double: {}", out[0]);
        assert!((out[1] - 0.4).abs() < 1e-10, "G slope=1 unchanged: {}", out[1]);
        assert!((out[2] - 0.2).abs() < 1e-10, "B slope=0.5 halves: {}", out[2]);
    }

    #[test]
    fn test_cdl_saturation_zero_is_grey() {
        let cdl = AscCdl::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        let pixel = [1.0, 0.0, 0.0]; // pure red
        let out = cdl.apply(pixel);
        // All channels should equal luma (0.2126 for pure red)
        let expected = 0.2126;
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "R should equal luma: {}", out[0]
        );
        assert!(
            (out[1] - expected).abs() < 1e-6,
            "G should equal luma: {}", out[1]
        );
        assert!(
            (out[2] - expected).abs() < 1e-6,
            "B should equal luma: {}", out[2]
        );
    }

    #[test]
    fn test_cdl_validate_invalid() {
        let cdl = AscCdl::new([-1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        // Negative slope should fail validation
        assert!(cdl.validate().is_err(), "Negative slope should fail validation");
    }

    #[test]
    fn test_lut1d_identity() {
        let lut = LookLut1d::identity(65);
        assert!(lut.is_identity(1e-6), "identity LUT should report as identity");
        let pixel = [0.3, 0.6, 0.9];
        let out = lut.apply(pixel);
        for i in 0..3 {
            assert!(
                (out[i] - pixel[i]).abs() < 1e-5,
                "Identity LUT should not change pixel channel {i}: {} vs {}",
                out[i],
                pixel[i]
            );
        }
    }

    #[test]
    fn test_lut1d_wrong_lengths() {
        let r = vec![0.0, 0.5, 1.0];
        let g = vec![0.0, 1.0];
        let b = vec![0.0, 0.5, 1.0];
        assert!(
            LookLut1d::from_channels(r, g, b).is_err(),
            "Mismatched channel lengths should fail"
        );
    }

    #[test]
    fn test_look_transform_apply() {
        let cdl = AscCdl::new([1.1, 1.0, 0.9], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        let look = LookTransform::new(cdl, None);
        let pixel = [0.5, 0.5, 0.5];
        let out = look.apply(pixel);
        assert!((out[0] - 0.55).abs() < 1e-10, "R×1.1 should be 0.55: {}", out[0]);
        assert!((out[1] - 0.50).abs() < 1e-10, "G×1.0 should be 0.50: {}", out[1]);
        assert!((out[2] - 0.45).abs() < 1e-10, "B×0.9 should be 0.45: {}", out[2]);
    }

    #[test]
    fn test_look_serialize_deserialize_roundtrip() {
        let cdl = AscCdl::new([1.1, 0.95, 0.9], [0.02, 0.0, -0.01], [0.9, 1.0, 1.1], 1.05);
        let lut = LookLut1d::identity(17);
        let mut look = LookTransform::new(cdl, Some(lut));
        look.name = "TestLook".to_string();
        look.description = "A test".to_string();

        let serialized = look.serialize();
        let restored = LookTransform::deserialize(&serialized)
            .expect("deserialize should succeed");

        assert_eq!(restored.name, "TestLook");
        assert_eq!(restored.description, "A test");
        // CDL values should round-trip
        for i in 0..3 {
            assert!((restored.cdl.slope[i] - look.cdl.slope[i]).abs() < 1e-6);
            assert!((restored.cdl.offset[i] - look.cdl.offset[i]).abs() < 1e-6);
            assert!((restored.cdl.power[i] - look.cdl.power[i]).abs() < 1e-6);
        }
        assert!((restored.cdl.saturation - look.cdl.saturation).abs() < 1e-6);
        // LUT should round-trip
        let lut_out = restored.lut.expect("LUT should be preserved");
        assert!(lut_out.is_identity(1e-5), "Identity LUT should round-trip");
    }

    #[test]
    fn test_preview_strip_length() {
        let look = LookTransform::new(AscCdl::identity(), None);
        let strip = PreviewStrip::new(&look, 64);
        assert_eq!(strip.pixels.len(), 64, "Strip should have requested number of pixels");
    }

    #[test]
    fn test_preview_strip_shadow_highlight() {
        let look = LookTransform::new(AscCdl::identity(), None);
        let strip = PreviewStrip::new(&look, 64);
        let (shadow, highlight) = strip.shadow_highlight();
        assert!(shadow[0] < 0.01, "Shadow should be near black");
        assert!(highlight[0] > 0.99, "Highlight should be near white");
    }

    #[test]
    fn test_preview_strip_monotone_identity() {
        let look = LookTransform::new(AscCdl::identity(), None);
        let strip = PreviewStrip::new(&look, 128);
        assert!(strip.is_monotone(), "Identity look strip should be monotone");
    }

    #[test]
    fn test_preview_strip_mean_brightness() {
        let look = LookTransform::new(AscCdl::identity(), None);
        let strip = PreviewStrip::new(&look, 256);
        let mb = strip.mean_brightness();
        // For a linear ramp, mean should be ~ 0.5
        assert!(
            (mb - 0.5).abs() < 0.05,
            "Mean brightness of linear ramp should be ~0.5, got {mb}"
        );
    }

    #[test]
    fn test_apply_buffer() {
        let look = LookTransform::new(AscCdl::identity(), None);
        let mut buf = vec![0.5, 0.5, 0.5, 0.8, 0.2, 0.1];
        look.apply_buffer(&mut buf).expect("apply_buffer should succeed");
        assert!((buf[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_buffer_wrong_size() {
        let look = LookTransform::new(AscCdl::identity(), None);
        let mut buf = vec![0.5, 0.5];
        assert!(look.apply_buffer(&mut buf).is_err());
    }
}
