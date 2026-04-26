//! 1D LUT (Look-Up Table) implementation.
//!
//! 1D LUTs are per-channel curves used for tone mapping, gamma correction,
//! and color grading. They are faster than 3D LUTs but can only affect
//! each channel independently.
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::{Lut1d, LutInterpolation};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a simple gamma 2.2 curve
//! let mut lut = Lut1d::new(256);
//! for i in 0..256 {
//!     let t = i as f64 / 255.0;
//!     lut.set_r(i, t.powf(2.2));
//!     lut.set_g(i, t.powf(2.2));
//!     lut.set_b(i, t.powf(2.2));
//! }
//!
//! // Apply to a color
//! let input = [0.5, 0.3, 0.7];
//! let output = lut.apply(&input, LutInterpolation::Linear);
//! # Ok(())
//! # }
//! ```

use crate::error::{LutError, LutResult};
use crate::interpolation::{self, LutInterpolation};
use crate::Rgb;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// 1D LUT for per-channel color correction.
#[derive(Clone, Debug)]
pub struct Lut1d {
    /// Red channel LUT.
    pub r: Vec<f64>,
    /// Green channel LUT.
    pub g: Vec<f64>,
    /// Blue channel LUT.
    pub b: Vec<f64>,
    /// Size of each LUT (all three have the same size).
    size: usize,
    /// Input range minimum (usually 0.0).
    pub input_min: f64,
    /// Input range maximum (usually 1.0).
    pub input_max: f64,
}

impl Lut1d {
    /// Create a new 1D LUT with the specified size.
    ///
    /// All channels are initialized to identity (linear mapping).
    #[must_use]
    pub fn new(size: usize) -> Self {
        let mut lut = Self {
            r: vec![0.0; size],
            g: vec![0.0; size],
            b: vec![0.0; size],
            size,
            input_min: 0.0,
            input_max: 1.0,
        };
        lut.set_identity();
        lut
    }

    /// Create an identity 1D LUT.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        Self::new(size)
    }

    /// Get the size of the LUT.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Set all channels to identity mapping.
    pub fn set_identity(&mut self) {
        for i in 0..self.size {
            let t = i as f64 / (self.size - 1) as f64;
            self.r[i] = t;
            self.g[i] = t;
            self.b[i] = t;
        }
    }

    /// Set a value in the red channel.
    pub fn set_r(&mut self, index: usize, value: f64) {
        self.r[index] = value;
    }

    /// Set a value in the green channel.
    pub fn set_g(&mut self, index: usize, value: f64) {
        self.g[index] = value;
    }

    /// Set a value in the blue channel.
    pub fn set_b(&mut self, index: usize, value: f64) {
        self.b[index] = value;
    }

    /// Apply the 1D LUT to an RGB color.
    #[must_use]
    pub fn apply(&self, rgb: &Rgb, interpolation: LutInterpolation) -> Rgb {
        [
            self.apply_channel(&self.r, rgb[0], interpolation),
            self.apply_channel(&self.g, rgb[1], interpolation),
            self.apply_channel(&self.b, rgb[2], interpolation),
        ]
    }

    /// Apply LUT to a single channel value.
    #[must_use]
    fn apply_channel(&self, channel: &[f64], value: f64, interpolation: LutInterpolation) -> f64 {
        // Normalize input to 0-1 range
        let normalized = (value - self.input_min) / (self.input_max - self.input_min);
        let clamped = normalized.clamp(0.0, 1.0);

        // Map to LUT index space
        let index_f = clamped * (self.size - 1) as f64;

        match interpolation {
            LutInterpolation::Nearest => {
                let index = index_f.round() as usize;
                channel[index.min(self.size - 1)]
            }
            LutInterpolation::Linear => {
                let index = index_f.floor() as usize;
                if index >= self.size - 1 {
                    channel[self.size - 1]
                } else {
                    let frac = index_f - index as f64;
                    interpolation::lerp(channel[index], channel[index + 1], frac)
                }
            }
            LutInterpolation::Cubic => {
                let index = index_f.floor() as usize;
                if index >= self.size - 1 {
                    channel[self.size - 1]
                } else {
                    let frac = index_f - index as f64;
                    let p0 = if index > 0 {
                        channel[index - 1]
                    } else {
                        channel[0]
                    };
                    let p1 = channel[index];
                    let p2 = channel[index + 1];
                    let p3 = if index + 2 < self.size {
                        channel[index + 2]
                    } else {
                        channel[self.size - 1]
                    };
                    interpolation::cubic_interp(p0, p1, p2, p3, frac)
                }
            }
            _ => {
                // Fallback to linear for unsupported interpolation modes
                let index = index_f.floor() as usize;
                if index >= self.size - 1 {
                    channel[self.size - 1]
                } else {
                    let frac = index_f - index as f64;
                    interpolation::lerp(channel[index], channel[index + 1], frac)
                }
            }
        }
    }

    /// Create a 1D LUT from a function.
    ///
    /// The function takes a normalized input value (0.0-1.0) and returns an RGB output.
    #[must_use]
    pub fn from_fn<F>(size: usize, f: F) -> Self
    where
        F: Fn(f64) -> Rgb,
    {
        let mut lut = Self::new(size);
        for i in 0..size {
            let t = i as f64 / (size - 1) as f64;
            let rgb = f(t);
            lut.r[i] = rgb[0];
            lut.g[i] = rgb[1];
            lut.b[i] = rgb[2];
        }
        lut
    }

    /// Create a gamma curve.
    #[must_use]
    pub fn gamma(size: usize, gamma: f64) -> Self {
        Self::from_fn(size, |t| {
            let v = t.powf(gamma);
            [v, v, v]
        })
    }

    /// Create an inverse gamma curve.
    #[must_use]
    pub fn inverse_gamma(size: usize, gamma: f64) -> Self {
        Self::gamma(size, 1.0 / gamma)
    }

    /// Invert the 1D LUT.
    ///
    /// This creates a LUT that reverses the effect of this LUT.
    /// Only works well for monotonic LUTs.
    #[must_use]
    pub fn invert(&self) -> Self {
        let mut inverted = Self::new(self.size);

        for channel_in in 0..3 {
            let (input, output) = match channel_in {
                0 => (&self.r, &mut inverted.r),
                1 => (&self.g, &mut inverted.g),
                _ => (&self.b, &mut inverted.b),
            };

            for (i, out) in output.iter_mut().enumerate() {
                let target = f64::from(i as u32) / (self.size - 1) as f64;

                // Binary search for the input value that produces this output
                let mut low = 0;
                let mut high = self.size - 1;

                while high - low > 1 {
                    let mid = (low + high) / 2;
                    if input[mid] < target {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }

                // Linear interpolation between low and high
                if high == low {
                    *out = low as f64 / (self.size - 1) as f64;
                } else {
                    let t = (target - input[low]) / (input[high] - input[low]);
                    let low_f = low as f64 / (self.size - 1) as f64;
                    let high_f = high as f64 / (self.size - 1) as f64;
                    *out = interpolation::lerp(low_f, high_f, t);
                }
            }
        }

        inverted
    }

    /// Compose this LUT with another LUT.
    ///
    /// Returns a new LUT that is equivalent to applying `self` followed by `other`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let mut composed = Self::new(self.size);

        for i in 0..self.size {
            let intermediate = [self.r[i], self.g[i], self.b[i]];
            let output = other.apply(&intermediate, LutInterpolation::Linear);
            composed.r[i] = output[0];
            composed.g[i] = output[1];
            composed.b[i] = output[2];
        }

        composed
    }

    /// Create a 1D LUT mapping `LogC` to linear.
    ///
    /// Arri `LogC` (EI 800) to scene-linear conversion.
    #[must_use]
    pub fn from_log_to_linear() -> Self {
        // LogC EI 800 parameters
        const A: f64 = 5.555_556;
        const B: f64 = 0.052_272;
        const C: f64 = 0.247_190;
        const D: f64 = 0.385_537;
        const E: f64 = 5.367_655;
        const F: f64 = 0.092_809;
        // Linear cut: encoded values below (E*LIN_CUT+F) use the linear segment
        const LIN_ENCODED_CUT: f64 = E * 0.005_526 + F;

        let size = 4096;
        Self::from_fn(size, |t| {
            let linear = if t > LIN_ENCODED_CUT {
                (10_f64.powf((t - D) / C) - B) / A
            } else {
                (t - F) / E
            };
            let clamped = linear.max(0.0);
            [clamped, clamped, clamped]
        })
    }

    /// Apply the 1D LUT to a single channel value (using linear interpolation).
    ///
    /// Uses the red channel of the LUT.
    #[must_use]
    pub fn apply_single(&self, value: f32) -> f32 {
        self.apply_channel(
            &self.r,
            f64::from(value),
            crate::interpolation::LutInterpolation::Linear,
        ) as f32
    }

    /// Load a 1D LUT from an Adobe Cube 1D file (`.cube` format).
    ///
    /// The file must contain a `LUT_1D_SIZE` header followed by one `R G B`
    /// triplet per line. Optional `DOMAIN_MIN` / `DOMAIN_MAX` headers are
    /// recognised; if absent the domain defaults to `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`LutError::Io`] if the file cannot be opened, [`LutError::Parse`]
    /// if a line is malformed, or [`LutError::InvalidSize`] if the number of
    /// data rows does not match `LUT_1D_SIZE`.
    pub fn from_file<P: AsRef<Path>>(path: P) -> LutResult<Self> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        let mut declared_size: Option<usize> = None;
        let mut domain_min = 0.0_f64;
        let mut domain_max = 1.0_f64;
        let mut r_channel: Vec<f64> = Vec::new();
        let mut g_channel: Vec<f64> = Vec::new();
        let mut b_channel: Vec<f64> = Vec::new();

        for line in reader.lines() {
            let raw = line?;
            let trimmed = raw.trim();

            // Skip blank lines and comment lines.
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("LUT_1D_SIZE") {
                let n: usize = rest
                    .trim()
                    .parse()
                    .map_err(|_| LutError::Parse("Invalid LUT_1D_SIZE value".to_string()))?;
                if n == 0 {
                    return Err(LutError::Parse("LUT_1D_SIZE must be > 0".to_string()));
                }
                declared_size = Some(n);
            } else if let Some(rest) = trimmed.strip_prefix("DOMAIN_MIN") {
                let vals = Self::parse_three_f64(rest, "DOMAIN_MIN")?;
                // For a 1D LUT the domain is the same for all channels; use channel 0.
                domain_min = vals[0];
            } else if let Some(rest) = trimmed.strip_prefix("DOMAIN_MAX") {
                let vals = Self::parse_three_f64(rest, "DOMAIN_MAX")?;
                domain_max = vals[0];
            } else if trimmed.starts_with("LUT_3D_SIZE") || trimmed.starts_with("TITLE") {
                // Silently skip non-1D-relevant headers instead of aborting.
                continue;
            } else {
                // Data row: three space-separated f64 values.
                let vals = Self::parse_three_f64(trimmed, "data row")?;
                r_channel.push(vals[0]);
                g_channel.push(vals[1]);
                b_channel.push(vals[2]);
            }
        }

        let size = declared_size
            .ok_or_else(|| LutError::Parse("Missing LUT_1D_SIZE header".to_string()))?;

        if r_channel.len() != size {
            return Err(LutError::InvalidSize {
                expected: size,
                actual: r_channel.len(),
            });
        }

        if domain_max <= domain_min {
            return Err(LutError::Parse(
                "DOMAIN_MAX must be greater than DOMAIN_MIN".to_string(),
            ));
        }

        Ok(Self {
            r: r_channel,
            g: g_channel,
            b: b_channel,
            size,
            input_min: domain_min,
            input_max: domain_max,
        })
    }

    /// Save the 1D LUT to a file using the Adobe Cube 1D format (`.cube`).
    ///
    /// Writes a comment, the `LUT_1D_SIZE` header, `DOMAIN_MIN` / `DOMAIN_MAX`
    /// headers, and then one `R G B` triplet per LUT entry.
    ///
    /// # Errors
    ///
    /// Returns [`LutError::Io`] if the file cannot be created or written.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> LutResult<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(writer, "# Generated by OxiMedia")?;
        writeln!(writer, "LUT_1D_SIZE {}", self.size)?;
        writeln!(
            writer,
            "DOMAIN_MIN {:.10} {:.10} {:.10}",
            self.input_min, self.input_min, self.input_min
        )?;
        writeln!(
            writer,
            "DOMAIN_MAX {:.10} {:.10} {:.10}",
            self.input_max, self.input_max, self.input_max
        )?;

        for i in 0..self.size {
            writeln!(
                writer,
                "{:.10} {:.10} {:.10}",
                self.r[i], self.g[i], self.b[i]
            )?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Parse exactly three whitespace-separated `f64` values from a string slice.
    fn parse_three_f64(src: &str, context: &str) -> LutResult<[f64; 3]> {
        let parts: Vec<&str> = src.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(LutError::Parse(format!(
                "Expected 3 values in {context}, got {}",
                parts.len()
            )));
        }
        let a = parts[0]
            .parse::<f64>()
            .map_err(|_| LutError::Parse(format!("Invalid f64 in {context}: '{}'", parts[0])))?;
        let b = parts[1]
            .parse::<f64>()
            .map_err(|_| LutError::Parse(format!("Invalid f64 in {context}: '{}'", parts[1])))?;
        let c = parts[2]
            .parse::<f64>()
            .map_err(|_| LutError::Parse(format!("Invalid f64 in {context}: '{}'", parts[2])))?;
        Ok([a, b, c])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_lut() {
        let lut = Lut1d::identity(256);
        let input = [0.5, 0.3, 0.7];
        let output = lut.apply(&input, LutInterpolation::Linear);
        assert!((output[0] - input[0]).abs() < 1e-6);
        assert!((output[1] - input[1]).abs() < 1e-6);
        assert!((output[2] - input[2]).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_lut() {
        let lut = Lut1d::gamma(256, 2.2);
        let input = [0.5, 0.5, 0.5];
        let output = lut.apply(&input, LutInterpolation::Linear);
        let expected = 0.5_f64.powf(2.2);
        assert!((output[0] - expected).abs() < 0.01);
    }

    #[test]
    fn test_invert_lut() {
        let lut = Lut1d::gamma(256, 2.2);
        let inverted = lut.invert();
        let input = [0.5, 0.3, 0.7];
        let encoded = lut.apply(&input, LutInterpolation::Linear);
        let decoded = inverted.apply(&encoded, LutInterpolation::Linear);
        assert!((decoded[0] - input[0]).abs() < 0.01);
        assert!((decoded[1] - input[1]).abs() < 0.01);
        assert!((decoded[2] - input[2]).abs() < 0.01);
    }

    #[test]
    fn test_compose_lut() {
        let lut1 = Lut1d::gamma(256, 2.2);
        let lut2 = Lut1d::inverse_gamma(256, 2.2);
        let composed = lut1.compose(&lut2);

        let input = [0.5, 0.3, 0.7];
        let output = composed.apply(&input, LutInterpolation::Linear);

        // Composed should be close to identity
        assert!((output[0] - input[0]).abs() < 0.01);
        assert!((output[1] - input[1]).abs() < 0.01);
        assert!((output[2] - input[2]).abs() < 0.01);
    }

    #[test]
    fn test_from_fn() {
        let lut = Lut1d::from_fn(256, |t| [t * 2.0, t * 0.5, t]);
        let input = [0.5, 0.5, 0.5];
        let output = lut.apply(&input, LutInterpolation::Linear);
        assert!((output[0] - 1.0).abs() < 0.01);
        assert!((output[1] - 0.25).abs() < 0.01);
        assert!((output[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_interpolation_modes() {
        let lut = Lut1d::gamma(256, 2.2);
        let input = [0.5, 0.5, 0.5];

        let nearest = lut.apply(&input, LutInterpolation::Nearest);
        let linear = lut.apply(&input, LutInterpolation::Linear);
        let cubic = lut.apply(&input, LutInterpolation::Cubic);

        // All should be similar
        assert!((nearest[0] - linear[0]).abs() < 0.02);
        assert!((cubic[0] - linear[0]).abs() < 0.02);
    }

    #[test]
    fn test_from_log_to_linear_midpoint() {
        let lut = Lut1d::from_log_to_linear();
        // LogC middle grey ~0.391 should map to ~0.18 scene-linear
        let result = lut.apply_single(0.391_f32);
        assert!(result > 0.1 && result < 0.3, "Expected ~0.18, got {result}");
    }

    #[test]
    fn test_from_log_to_linear_black() {
        let lut = Lut1d::from_log_to_linear();
        // Very low values should map near zero
        let result = lut.apply_single(0.0_f32);
        assert!(result < 0.01, "Expected near 0.0, got {result}");
    }

    #[test]
    fn test_apply_single_identity() {
        let lut = Lut1d::identity(256);
        // For identity LUT, apply_single should return nearly same value
        let result = lut.apply_single(0.5_f32);
        assert!((result - 0.5).abs() < 0.002, "Expected ~0.5, got {result}");
    }

    #[test]
    fn test_apply_single_gamma() {
        let lut = Lut1d::gamma(256, 2.2);
        let expected = 0.5_f32.powf(2.2);
        let result = lut.apply_single(0.5_f32);
        assert!(
            (result - expected).abs() < 0.01,
            "Expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_from_log_to_linear_size() {
        let lut = Lut1d::from_log_to_linear();
        assert_eq!(lut.size(), 4096);
    }

    #[test]
    fn test_apply_single_clamp_high() {
        let lut = Lut1d::identity(256);
        // Values at or above max should return max
        let result = lut.apply_single(1.0_f32);
        assert!((result - 1.0).abs() < 0.002, "Expected 1.0, got {result}");
    }

    // ---------- file I/O round-trip tests ----------

    /// Returns a unique temporary file path for this test, using the process id
    /// to avoid collisions when nextest runs tests in parallel.
    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "oximedia_lut1d_{}_{}.cube",
            name,
            std::process::id()
        ));
        p
    }

    #[test]
    fn test_round_trip() {
        let path = temp_path("round_trip");

        // Create an identity 1D LUT and save it.
        let original = Lut1d::identity(17);
        original
            .to_file(&path)
            .expect("to_file should succeed in test");

        // Load it back.
        let loaded = Lut1d::from_file(&path).expect("from_file should succeed in test");

        // Verify structure.
        assert_eq!(loaded.size(), 17);
        assert!((loaded.input_min - 0.0).abs() < f64::EPSILON);
        assert!((loaded.input_max - 1.0).abs() < f64::EPSILON);

        // Verify each entry matches within f64 precision.
        for i in 0..17 {
            assert!(
                (loaded.r[i] - original.r[i]).abs() < 1e-9,
                "r[{i}] mismatch: {} vs {}",
                loaded.r[i],
                original.r[i]
            );
            assert!(
                (loaded.g[i] - original.g[i]).abs() < 1e-9,
                "g[{i}] mismatch: {} vs {}",
                loaded.g[i],
                original.g[i]
            );
            assert!(
                (loaded.b[i] - original.b[i]).abs() < 1e-9,
                "b[{i}] mismatch: {} vs {}",
                loaded.b[i],
                original.b[i]
            );
        }

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_from_file_missing() {
        let path = temp_path("missing_file_that_does_not_exist");
        // Ensure file is absent.
        let _ = std::fs::remove_file(&path);
        let result = Lut1d::from_file(&path);
        assert!(result.is_err(), "Expected error for missing file");
    }

    #[test]
    fn test_from_file_malformed() {
        let path = temp_path("malformed");

        // Write a file with LUT_1D_SIZE 2 but body lines with only 2 columns.
        std::fs::write(&path, "LUT_1D_SIZE 2\n1.0 0.0\n0.5 0.5\n")
            .expect("write should succeed in test");

        let result = Lut1d::from_file(&path);
        assert!(result.is_err(), "Expected parse error for malformed file");

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }
}
