//! Apply 3D LUTs directly to `ImageData` pixel buffers.
//!
//! Supports trilinear interpolation for smooth LUT application.
//! Compatible with the `.cube` format commonly used in DaVinci Resolve, Premiere, etc.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use crate::error::{ImageError, ImageResult};
use crate::{ImageData, ImageFrame};

// ── 3D LUT ───────────────────────────────────────────────────────────────────

/// A 3D colour lookup table.
///
/// The LUT maps (R, G, B) input coordinates in [0, 1] to (R', G', B') output
/// using trilinear interpolation over a `size × size × size` grid.
#[derive(Debug, Clone)]
pub struct Lut3d {
    /// Side length of the LUT cube (typically 17, 33, or 65).
    pub size: usize,
    /// LUT data: flat array of RGB triples, indexed as `[b * size^2 + g * size + r]`.
    /// Each entry is (R, G, B) in [0.0, 1.0].
    pub data: Vec<[f32; 3]>,
}

impl Lut3d {
    /// Create an identity LUT (output = input).
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let size = size.max(2);
        let total = size * size * size;
        let mut data = Vec::with_capacity(total);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let rv = r as f32 / (size - 1) as f32;
                    let gv = g as f32 / (size - 1) as f32;
                    let bv = b as f32 / (size - 1) as f32;
                    data.push([rv, gv, bv]);
                }
            }
        }
        Self { size, data }
    }

    /// Create a sepia-tone LUT.
    #[must_use]
    pub fn sepia(size: usize) -> Self {
        let mut lut = Self::identity(size);
        for entry in &mut lut.data {
            let r = entry[0];
            let g = entry[1];
            let b = entry[2];
            entry[0] = (r * 0.393 + g * 0.769 + b * 0.189).clamp(0.0, 1.0);
            entry[1] = (r * 0.349 + g * 0.686 + b * 0.168).clamp(0.0, 1.0);
            entry[2] = (r * 0.272 + g * 0.534 + b * 0.131).clamp(0.0, 1.0);
        }
        lut
    }

    /// Create a simple gamma correction LUT.
    #[must_use]
    pub fn gamma(size: usize, gamma: f32) -> Self {
        let mut lut = Self::identity(size);
        for entry in &mut lut.data {
            entry[0] = entry[0].powf(gamma);
            entry[1] = entry[1].powf(gamma);
            entry[2] = entry[2].powf(gamma);
        }
        lut
    }

    /// Returns the total number of LUT entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.data.len()
    }

    /// Validate that the LUT has the expected size.
    pub fn validate(&self) -> ImageResult<()> {
        let expected = self.size * self.size * self.size;
        if self.data.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "LUT data length {} != expected {}",
                self.data.len(),
                expected
            )));
        }
        if self.size < 2 {
            return Err(ImageError::invalid_format("LUT size must be at least 2"));
        }
        Ok(())
    }

    /// Look up (r, g, b) in [0, 1] using trilinear interpolation.
    #[must_use]
    pub fn trilinear_lookup(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let n = (self.size - 1) as f32;
        let size = self.size;

        // Scale to LUT grid coordinates
        let ri = (r * n).clamp(0.0, n);
        let gi = (g * n).clamp(0.0, n);
        let bi = (b * n).clamp(0.0, n);

        let r0 = ri.floor() as usize;
        let g0 = gi.floor() as usize;
        let b0 = bi.floor() as usize;
        let r1 = (r0 + 1).min(size - 1);
        let g1 = (g0 + 1).min(size - 1);
        let b1 = (b0 + 1).min(size - 1);

        let fr = ri - ri.floor();
        let fg = gi - gi.floor();
        let fb = bi - bi.floor();

        let idx = |ri: usize, gi: usize, bi: usize| bi * size * size + gi * size + ri;

        let c000 = self.data[idx(r0, g0, b0)];
        let c100 = self.data[idx(r1, g0, b0)];
        let c010 = self.data[idx(r0, g1, b0)];
        let c110 = self.data[idx(r1, g1, b0)];
        let c001 = self.data[idx(r0, g0, b1)];
        let c101 = self.data[idx(r1, g0, b1)];
        let c011 = self.data[idx(r0, g1, b1)];
        let c111 = self.data[idx(r1, g1, b1)];

        let mut result = [0.0f32; 3];
        for ch in 0..3 {
            // Trilinear interpolation
            let c00 = c000[ch] * (1.0 - fr) + c100[ch] * fr;
            let c01 = c001[ch] * (1.0 - fr) + c101[ch] * fr;
            let c10 = c010[ch] * (1.0 - fr) + c110[ch] * fr;
            let c11 = c011[ch] * (1.0 - fr) + c111[ch] * fr;

            let c0 = c00 * (1.0 - fg) + c10 * fg;
            let c1 = c01 * (1.0 - fg) + c11 * fg;

            result[ch] = (c0 * (1.0 - fb) + c1 * fb).clamp(0.0, 1.0);
        }

        result
    }

    /// Apply this LUT to a flat interleaved u8 RGB or RGBA buffer in-place.
    ///
    /// `channels` must be 3 (RGB) or 4 (RGBA). Alpha channel is passed through unchanged.
    pub fn apply_to_buffer_u8(&self, buf: &mut [u8], channels: usize) {
        if channels != 3 && channels != 4 {
            return;
        }
        let step = channels;
        for chunk in buf.chunks_exact_mut(step) {
            let r = chunk[0] as f32 / 255.0;
            let g = chunk[1] as f32 / 255.0;
            let b = chunk[2] as f32 / 255.0;
            let out = self.trilinear_lookup(r, g, b);
            chunk[0] = (out[0] * 255.0).round() as u8;
            chunk[1] = (out[1] * 255.0).round() as u8;
            chunk[2] = (out[2] * 255.0).round() as u8;
            // chunk[3] (alpha) untouched
        }
    }

    /// Apply this LUT to a flat interleaved f32 RGB or RGBA buffer in-place.
    pub fn apply_to_buffer_f32(&self, buf: &mut [f32], channels: usize) {
        if channels != 3 && channels != 4 {
            return;
        }
        let step = channels;
        for chunk in buf.chunks_exact_mut(step) {
            let out = self.trilinear_lookup(chunk[0], chunk[1], chunk[2]);
            chunk[0] = out[0];
            chunk[1] = out[1];
            chunk[2] = out[2];
        }
    }

    /// Apply LUT to an `ImageData` buffer (u8 interleaved, RGB/RGBA).
    pub fn apply_to_image_data(&self, data: &ImageData, channels: usize) -> ImageResult<ImageData> {
        match data {
            ImageData::Interleaved(bytes) => {
                let mut buf = bytes.to_vec();
                self.apply_to_buffer_u8(&mut buf, channels);
                Ok(ImageData::interleaved(buf))
            }
            ImageData::Planar(_) => Err(ImageError::unsupported(
                "LUT application to planar data not supported",
            )),
        }
    }

    /// Apply LUT to an `ImageFrame` and return a new frame with LUT applied.
    pub fn apply_to_frame(&self, frame: &ImageFrame) -> ImageResult<ImageFrame> {
        let new_data = self.apply_to_image_data(&frame.data, frame.components as usize)?;
        Ok(ImageFrame::new(
            frame.frame_number,
            frame.width,
            frame.height,
            frame.pixel_type,
            frame.components,
            frame.color_space,
            new_data,
        ))
    }
}

// ── .cube format parser ───────────────────────────────────────────────────────

/// Parser for Adobe .cube LUT files.
pub struct CubeParser;

impl CubeParser {
    /// Parse a `.cube` format string into a `Lut3d`.
    ///
    /// The `.cube` format uses r-innermost ordering:
    /// for b in 0..size { for g in 0..size { for r in 0..size { ... } } }
    pub fn parse(content: &str) -> ImageResult<Lut3d> {
        let mut size = 0usize;
        let mut data = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            if line.starts_with("LUT_3D_SIZE") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    size = parts[1]
                        .parse()
                        .map_err(|_| ImageError::invalid_format("Invalid LUT_3D_SIZE"))?;
                }
                continue;
            }
            if line.starts_with("TITLE")
                || line.starts_with("DOMAIN_MIN")
                || line.starts_with("DOMAIN_MAX")
            {
                continue;
            }
            // Try to parse as three floats
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3 {
                let r: f32 = parts[0]
                    .parse()
                    .map_err(|_| ImageError::invalid_format("LUT parse: invalid R"))?;
                let g: f32 = parts[1]
                    .parse()
                    .map_err(|_| ImageError::invalid_format("LUT parse: invalid G"))?;
                let b: f32 = parts[2]
                    .parse()
                    .map_err(|_| ImageError::invalid_format("LUT parse: invalid B"))?;
                data.push([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]);
            }
        }

        if size == 0 {
            return Err(ImageError::invalid_format(
                "Missing LUT_3D_SIZE in .cube file",
            ));
        }
        let expected = size * size * size;
        if data.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "LUT has {} entries, expected {} (size={})",
                data.len(),
                expected,
                size
            )));
        }

        Ok(Lut3d { size, data })
    }

    /// Format a `Lut3d` as a `.cube` string.
    #[must_use]
    pub fn format(lut: &Lut3d) -> String {
        let mut out = String::new();
        out.push_str(&format!("LUT_3D_SIZE {}\n", lut.size));
        out.push_str("DOMAIN_MIN 0.0 0.0 0.0\n");
        out.push_str("DOMAIN_MAX 1.0 1.0 1.0\n");
        for entry in &lut.data {
            out.push_str(&format!(
                "{:.6} {:.6} {:.6}\n",
                entry[0], entry[1], entry[2]
            ));
        }
        out
    }
}

// ── 1D LUT ───────────────────────────────────────────────────────────────────

/// A 1D per-channel lookup table.
#[derive(Debug, Clone)]
pub struct Lut1d {
    /// Number of entries per channel.
    pub size: usize,
    /// Red channel values [0.0, 1.0].
    pub r: Vec<f32>,
    /// Green channel values [0.0, 1.0].
    pub g: Vec<f32>,
    /// Blue channel values [0.0, 1.0].
    pub b: Vec<f32>,
}

impl Lut1d {
    /// Create an identity 1D LUT.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let size = size.max(2);
        let table: Vec<f32> = (0..size).map(|i| i as f32 / (size - 1) as f32).collect();
        Self {
            size,
            r: table.clone(),
            g: table.clone(),
            b: table,
        }
    }

    /// Create a gamma-corrected 1D LUT.
    #[must_use]
    pub fn gamma(size: usize, gamma: f32) -> Self {
        let size = size.max(2);
        let table: Vec<f32> = (0..size)
            .map(|i| (i as f32 / (size - 1) as f32).powf(gamma))
            .collect();
        Self {
            size,
            r: table.clone(),
            g: table.clone(),
            b: table,
        }
    }

    /// Evaluate the LUT at input [0, 1] for a given channel (0=R, 1=G, 2=B).
    #[must_use]
    pub fn eval(&self, v: f32, ch: usize) -> f32 {
        let n = (self.size - 1) as f32;
        let vi = (v * n).clamp(0.0, n);
        let i0 = vi.floor() as usize;
        let i1 = (i0 + 1).min(self.size - 1);
        let f = vi - vi.floor();
        let table = match ch {
            0 => &self.r,
            1 => &self.g,
            _ => &self.b,
        };
        table[i0] * (1.0 - f) + table[i1] * f
    }

    /// Apply to a u8 RGB buffer.
    pub fn apply_to_buffer_u8(&self, buf: &mut [u8], channels: usize) {
        for chunk in buf.chunks_exact_mut(channels.max(3)) {
            chunk[0] = (self.eval(chunk[0] as f32 / 255.0, 0) * 255.0).round() as u8;
            chunk[1] = (self.eval(chunk[1] as f32 / 255.0, 1) * 255.0).round() as u8;
            chunk[2] = (self.eval(chunk[2] as f32 / 255.0, 2) * 255.0).round() as u8;
        }
    }
}

// ── LUT chain ────────────────────────────────────────────────────────────────

/// A chain of LUT operations applied in sequence.
pub struct LutChain {
    /// First-stage 1D LUT (input linearisation).
    pub lut1d_in: Option<Lut1d>,
    /// 3D LUT (colour transform).
    pub lut3d: Option<Lut3d>,
    /// Final 1D LUT (output gamma).
    pub lut1d_out: Option<Lut1d>,
}

impl LutChain {
    /// Create an empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            lut1d_in: None,
            lut3d: None,
            lut1d_out: None,
        }
    }

    /// Set the input 1D LUT.
    pub fn with_1d_in(mut self, lut: Lut1d) -> Self {
        self.lut1d_in = Some(lut);
        self
    }

    /// Set the 3D LUT.
    pub fn with_3d(mut self, lut: Lut3d) -> Self {
        self.lut3d = Some(lut);
        self
    }

    /// Set the output 1D LUT.
    pub fn with_1d_out(mut self, lut: Lut1d) -> Self {
        self.lut1d_out = Some(lut);
        self
    }

    /// Apply the chain to a single RGB triple in [0, 1].
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let (mut r, mut g, mut b) = (r, g, b);

        if let Some(lut) = &self.lut1d_in {
            r = lut.eval(r, 0);
            g = lut.eval(g, 1);
            b = lut.eval(b, 2);
        }

        if let Some(lut) = &self.lut3d {
            let out = lut.trilinear_lookup(r, g, b);
            r = out[0];
            g = out[1];
            b = out[2];
        }

        if let Some(lut) = &self.lut1d_out {
            r = lut.eval(r, 0);
            g = lut.eval(g, 1);
            b = lut.eval(b, 2);
        }

        [r, g, b]
    }

    /// Apply the chain to a u8 buffer.
    pub fn apply_to_buffer_u8(&self, buf: &mut [u8], channels: usize) {
        if channels < 3 {
            return;
        }
        for chunk in buf.chunks_exact_mut(channels) {
            let r = chunk[0] as f32 / 255.0;
            let g = chunk[1] as f32 / 255.0;
            let b = chunk[2] as f32 / 255.0;
            let out = self.apply(r, g, b);
            chunk[0] = (out[0] * 255.0).round() as u8;
            chunk[1] = (out[1] * 255.0).round() as u8;
            chunk[2] = (out[2] * 255.0).round() as u8;
        }
    }
}

impl Default for LutChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColorSpace, ImageData, ImageFrame, PixelType};

    fn make_frame(w: u32, h: u32, pixels: Vec<u8>) -> ImageFrame {
        ImageFrame::new(
            1,
            w,
            h,
            PixelType::U8,
            3,
            ColorSpace::Srgb,
            ImageData::interleaved(pixels),
        )
    }

    #[test]
    fn test_identity_lut_size() {
        let lut = Lut3d::identity(17);
        assert_eq!(lut.size, 17);
        assert_eq!(lut.entry_count(), 17 * 17 * 17);
    }

    #[test]
    fn test_identity_lut_lookup_corners() {
        let lut = Lut3d::identity(17);
        // Black corner
        let out = lut.trilinear_lookup(0.0, 0.0, 0.0);
        for &v in &out {
            assert!(v.abs() < 1e-4, "black corner: {v}");
        }
        // White corner
        let out = lut.trilinear_lookup(1.0, 1.0, 1.0);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4, "white corner: {v}");
        }
    }

    #[test]
    fn test_identity_lut_passthrough() {
        let lut = Lut3d::identity(33);
        let test_vals = [(0.5, 0.5, 0.5), (0.25, 0.75, 0.1), (1.0, 0.0, 0.5)];
        for (r, g, b) in test_vals {
            let out = lut.trilinear_lookup(r, g, b);
            assert!((out[0] - r).abs() < 1e-3, "R mismatch: {} vs {}", out[0], r);
            assert!((out[1] - g).abs() < 1e-3, "G mismatch: {} vs {}", out[1], g);
            assert!((out[2] - b).abs() < 1e-3, "B mismatch: {} vs {}", out[2], b);
        }
    }

    #[test]
    fn test_sepia_lut_changes_values() {
        let lut = Lut3d::sepia(17);
        // Pure white → sepia tones
        let out = lut.trilinear_lookup(1.0, 1.0, 1.0);
        // Sepia white: R > G > B
        assert!(
            out[0] >= out[1],
            "Sepia R should be >= G: {} >= {}",
            out[0],
            out[1]
        );
        assert!(
            out[1] >= out[2],
            "Sepia G should be >= B: {} >= {}",
            out[1],
            out[2]
        );
    }

    #[test]
    fn test_gamma_lut_midpoint() {
        let lut = Lut3d::gamma(17, 2.2);
        // gamma(0.5, 2.2) ≈ 0.218
        let out = lut.trilinear_lookup(0.5, 0.5, 0.5);
        let expected = 0.5f32.powf(2.2);
        assert!(
            (out[0] - expected).abs() < 0.02,
            "gamma mismatch: {} vs {}",
            out[0],
            expected
        );
    }

    #[test]
    fn test_apply_to_buffer_u8_identity() {
        let lut = Lut3d::identity(17);
        let original = vec![100u8, 150, 200, 50, 100, 150];
        let mut buf = original.clone();
        lut.apply_to_buffer_u8(&mut buf, 3);
        for (i, (&orig, &v)) in original.iter().zip(buf.iter()).enumerate() {
            assert!(
                (orig as i32 - v as i32).abs() <= 2,
                "Identity mismatch at {i}: {orig} vs {v}"
            );
        }
    }

    #[test]
    fn test_apply_to_buffer_f32_identity() {
        let lut = Lut3d::identity(17);
        let original = vec![0.4f32, 0.6, 0.8, 0.2, 0.5, 0.7];
        let mut buf = original.clone();
        lut.apply_to_buffer_f32(&mut buf, 3);
        for (i, (&o, &v)) in original.iter().zip(buf.iter()).enumerate() {
            assert!(
                (o - v).abs() < 0.01,
                "Identity f32 mismatch at {i}: {o} vs {v}"
            );
        }
    }

    #[test]
    fn test_apply_to_frame_preserves_dimensions() {
        let lut = Lut3d::identity(17);
        let pixels = vec![128u8; 4 * 4 * 3];
        let frame = make_frame(4, 4, pixels);
        let out = lut.apply_to_frame(&frame).expect("apply lut");
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
        assert_eq!(out.components, 3);
    }

    #[test]
    fn test_apply_to_image_data_planar_fails() {
        let lut = Lut3d::identity(17);
        let data = ImageData::planar(vec![vec![0u8; 16], vec![0u8; 16], vec![0u8; 16]]);
        let result = lut.apply_to_image_data(&data, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_lut_validate_ok() {
        let lut = Lut3d::identity(5);
        assert!(lut.validate().is_ok());
    }

    #[test]
    fn test_lut_validate_wrong_size() {
        let lut = Lut3d {
            size: 5,
            data: vec![[0.0; 3]; 10],
        }; // wrong length
        assert!(lut.validate().is_err());
    }

    #[test]
    fn test_cube_parser_identity_roundtrip() {
        let lut = Lut3d::identity(5);
        let cube_str = CubeParser::format(&lut);
        let parsed = CubeParser::parse(&cube_str).expect("parse cube");
        assert_eq!(parsed.size, 5);
        assert_eq!(parsed.data.len(), 5 * 5 * 5);
        // Check a few values
        for (i, (orig, parsed)) in lut.data.iter().zip(parsed.data.iter()).enumerate() {
            for ch in 0..3 {
                assert!(
                    (orig[ch] - parsed[ch]).abs() < 1e-4,
                    "cube roundtrip mismatch at entry {i} ch {ch}"
                );
            }
        }
    }

    #[test]
    fn test_cube_parser_missing_size() {
        let content = "0.0 0.0 0.0\n1.0 1.0 1.0\n";
        let result = CubeParser::parse(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_cube_parser_wrong_count() {
        let content = "LUT_3D_SIZE 3\n0.0 0.0 0.0\n1.0 1.0 1.0\n"; // only 2 entries
        let result = CubeParser::parse(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_cube_format_has_header() {
        let lut = Lut3d::identity(2);
        let s = CubeParser::format(&lut);
        assert!(s.contains("LUT_3D_SIZE 2"), "Should have size header");
        assert!(s.contains("DOMAIN_MIN"), "Should have domain min");
        assert!(s.contains("DOMAIN_MAX"), "Should have domain max");
    }

    #[test]
    fn test_lut_1d_identity_passthrough() {
        let lut = Lut1d::identity(256);
        for i in 0..=255u8 {
            let v = i as f32 / 255.0;
            for ch in 0..3 {
                let out = lut.eval(v, ch);
                assert!((out - v).abs() < 1e-3, "1D identity mismatch at {i}");
            }
        }
    }

    #[test]
    fn test_lut_1d_gamma() {
        let lut = Lut1d::gamma(256, 2.2);
        let v = 0.5f32;
        let out = lut.eval(v, 0);
        let expected = v.powf(2.2);
        assert!(
            (out - expected).abs() < 0.01,
            "1D gamma mismatch: {out} vs {expected}"
        );
    }

    #[test]
    fn test_lut_1d_apply_identity() {
        let lut = Lut1d::identity(256);
        let original = vec![100u8, 150, 200, 50, 80, 120];
        let mut buf = original.clone();
        lut.apply_to_buffer_u8(&mut buf, 3);
        for (i, (&o, &v)) in original.iter().zip(buf.iter()).enumerate() {
            assert!(
                (o as i32 - v as i32).abs() <= 1,
                "1D identity apply at {i}: {o} vs {v}"
            );
        }
    }

    #[test]
    fn test_lut_chain_identity() {
        let chain = LutChain::new()
            .with_1d_in(Lut1d::identity(256))
            .with_3d(Lut3d::identity(17))
            .with_1d_out(Lut1d::identity(256));
        let out = chain.apply(0.5, 0.5, 0.5);
        for &v in &out {
            assert!((v - 0.5).abs() < 0.01, "Chain identity mismatch: {v}");
        }
    }

    #[test]
    fn test_lut_chain_apply_to_buffer() {
        let chain = LutChain::new().with_3d(Lut3d::identity(17));
        let mut buf = vec![200u8, 100, 50, 80, 160, 240];
        let original = buf.clone();
        chain.apply_to_buffer_u8(&mut buf, 3);
        for (i, (&o, &v)) in original.iter().zip(buf.iter()).enumerate() {
            assert!(
                (o as i32 - v as i32).abs() <= 3,
                "Chain buffer at {i}: {o} vs {v}"
            );
        }
    }

    #[test]
    fn test_trilinear_output_in_range() {
        let lut = Lut3d::sepia(17);
        for _ in 0..100 {
            let out = lut.trilinear_lookup(0.3, 0.7, 0.1);
            for &v in &out {
                assert!(v >= 0.0 && v <= 1.0, "out of range: {v}");
            }
        }
    }
}
