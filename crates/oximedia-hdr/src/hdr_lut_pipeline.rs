//! 3D LUT pipeline for HDR grading in PQ signal domain.
//!
//! Provides a composable pipeline of 3D colour lookup tables (CLUTs) operating
//! in the SMPTE ST 2084 (PQ) normalised signal space.  Each stage performs
//! trilinear interpolation within a 33³ CLUT, giving a smooth, continuous
//! colour transform.
//!
//! # Architecture
//!
//! ```text
//! PQ input (RGB) → Stage 0 (33³ CLUT) → Stage 1 (33³ CLUT) → … → PQ output
//! ```
//!
//! The pipeline is composable: individual [`HdrLutStage`] instances are added
//! to an [`HdrLutPipeline`] and applied left-to-right.  The identity LUT (where
//! every output equals its input) is available via [`HdrLutStage::identity`].
//!
//! # Example
//!
//! ```rust
//! use oximedia_hdr::hdr_lut_pipeline::{HdrLutPipeline, HdrLutStage, LutSize};
//!
//! let mut pipeline = HdrLutPipeline::new();
//! pipeline.push(HdrLutStage::identity(LutSize::S17));
//! let (r, g, b) = pipeline.apply(0.5, 0.5, 0.5).unwrap();
//! assert!((r - 0.5).abs() < 1e-4);
//! ```

use crate::{HdrError, Result};

/// Supported CLUT grid sizes (N³ entries per channel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LutSize {
    /// 17³ = 4 913 entries – fast, lower precision.
    S17 = 17,
    /// 33³ = 35 937 entries – broadcast / grading standard.
    S33 = 33,
    /// 65³ = 274 625 entries – high precision.
    S65 = 65,
}

impl LutSize {
    /// Number of grid points along each axis.
    pub fn n(self) -> usize {
        self as usize
    }

    /// Total number of entries (N³).
    pub fn total_entries(self) -> usize {
        let n = self.n();
        n * n * n
    }
}

/// A single 3D LUT stage in an [`HdrLutPipeline`].
///
/// The table is stored as a flat `Vec<[f32; 3]>` in R-major order:
/// index = r_idx * N² + g_idx * N + b_idx.
#[derive(Debug, Clone)]
pub struct HdrLutStage {
    /// Grid size per axis.
    pub size: LutSize,
    /// Flat CLUT: `total_entries` × RGB triplets, values in [0, 1].
    pub table: Vec<[f32; 3]>,
}

impl HdrLutStage {
    /// Build an identity LUT of the requested size.
    ///
    /// Every output colour equals its input colour.
    pub fn identity(size: LutSize) -> Self {
        let n = size.n();
        let total = size.total_entries();
        let mut table = Vec::with_capacity(total);
        let scale = if n > 1 { 1.0 / (n - 1) as f32 } else { 0.0 };
        for r in 0..n {
            for g in 0..n {
                for b in 0..n {
                    table.push([r as f32 * scale, g as f32 * scale, b as f32 * scale]);
                }
            }
        }
        Self { size, table }
    }

    /// Build a LUT from a flat slice of RGB triplets.
    ///
    /// # Errors
    /// Returns [`HdrError::MetadataParseError`] if the slice length does not match
    /// `size.total_entries()`.
    pub fn from_slice(size: LutSize, data: &[[f32; 3]]) -> Result<Self> {
        if data.len() != size.total_entries() {
            return Err(HdrError::MetadataParseError(format!(
                "LUT slice length {} does not match expected {} for {:?}",
                data.len(),
                size.total_entries(),
                size,
            )));
        }
        Ok(Self {
            size,
            table: data.to_vec(),
        })
    }

    /// Apply trilinear interpolation for a single RGB input in [0, 1].
    ///
    /// Values outside [0, 1] are clamped to the LUT boundary.
    ///
    /// # Errors
    /// Returns [`HdrError::GamutConversionError`] if the internal table is
    /// inconsistent (should not happen for LUTs built via public APIs).
    pub fn apply_trilinear(&self, r: f32, g: f32, b: f32) -> Result<[f32; 3]> {
        let n = self.size.n();
        if self.table.len() != n * n * n {
            return Err(HdrError::GamutConversionError(
                "LUT table length mismatch".into(),
            ));
        }
        let nf = (n - 1) as f32;

        // Clamp and compute fractional positions.
        let rc = r.clamp(0.0, 1.0) * nf;
        let gc = g.clamp(0.0, 1.0) * nf;
        let bc = b.clamp(0.0, 1.0) * nf;

        let ri = rc.floor() as usize;
        let gi = gc.floor() as usize;
        let bi = bc.floor() as usize;

        let ri = ri.min(n - 2);
        let gi = gi.min(n - 2);
        let bi = bi.min(n - 2);

        let fr = rc - ri as f32;
        let fg = gc - gi as f32;
        let fb = bc - bi as f32;

        // Fetch the 8 corners of the enclosing unit cube.
        let idx = |ri: usize, gi: usize, bi: usize| -> Result<[f32; 3]> {
            let i = ri * n * n + gi * n + bi;
            self.table
                .get(i)
                .copied()
                .ok_or_else(|| HdrError::GamutConversionError("LUT index out of bounds".into()))
        };

        let c000 = idx(ri, gi, bi)?;
        let c001 = idx(ri, gi, bi + 1)?;
        let c010 = idx(ri, gi + 1, bi)?;
        let c011 = idx(ri, gi + 1, bi + 1)?;
        let c100 = idx(ri + 1, gi, bi)?;
        let c101 = idx(ri + 1, gi, bi + 1)?;
        let c110 = idx(ri + 1, gi + 1, bi)?;
        let c111 = idx(ri + 1, gi + 1, bi + 1)?;

        // Trilinear interpolation per channel.
        let mut out = [0.0f32; 3];
        for ch in 0..3 {
            let v = c000[ch] * (1.0 - fr) * (1.0 - fg) * (1.0 - fb)
                + c001[ch] * (1.0 - fr) * (1.0 - fg) * fb
                + c010[ch] * (1.0 - fr) * fg * (1.0 - fb)
                + c011[ch] * (1.0 - fr) * fg * fb
                + c100[ch] * fr * (1.0 - fg) * (1.0 - fb)
                + c101[ch] * fr * (1.0 - fg) * fb
                + c110[ch] * fr * fg * (1.0 - fb)
                + c111[ch] * fr * fg * fb;
            out[ch] = v;
        }
        Ok(out)
    }
}

/// A composable pipeline of [`HdrLutStage`]s.
///
/// Stages are applied sequentially: the output of stage *k* feeds into stage
/// *k+1*.  The pipeline is designed to operate on PQ-encoded signals in [0, 1]
/// but can equally be used for any normalised colour data.
#[derive(Debug, Clone, Default)]
pub struct HdrLutPipeline {
    stages: Vec<HdrLutStage>,
}

impl HdrLutPipeline {
    /// Create an empty pipeline (passes through the input unchanged).
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Append a stage to the end of the pipeline.
    pub fn push(&mut self, stage: HdrLutStage) {
        self.stages.push(stage);
    }

    /// Return the number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Return `true` if there are no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Apply all pipeline stages to a single RGB triplet.
    ///
    /// Returns `(r, g, b)` all in [0, 1].  If the pipeline is empty the input
    /// is returned unchanged.
    ///
    /// # Errors
    /// Propagates any error from [`HdrLutStage::apply_trilinear`].
    pub fn apply(&self, r: f32, g: f32, b: f32) -> Result<(f32, f32, f32)> {
        let mut cur = [r, g, b];
        for stage in &self.stages {
            let next = stage.apply_trilinear(cur[0], cur[1], cur[2])?;
            cur = next;
        }
        Ok((cur[0], cur[1], cur[2]))
    }

    /// Apply the pipeline to an entire frame stored as interleaved RGB pixels.
    ///
    /// `pixels` must have length `width * height * 3`; values must be in [0, 1].
    /// The result is written back into the same slice.
    ///
    /// # Errors
    /// Returns [`HdrError::GamutConversionError`] if the slice length is not a
    /// multiple of 3, or propagates any per-pixel error.
    pub fn apply_frame(&self, pixels: &mut [f32]) -> Result<()> {
        if pixels.len() % 3 != 0 {
            return Err(HdrError::GamutConversionError(
                "pixel slice length is not a multiple of 3".into(),
            ));
        }
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.apply(chunk[0], chunk[1], chunk[2])?;
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
        Ok(())
    }

    /// Bake all pipeline stages into a single 33³ output LUT.
    ///
    /// This is useful when the pipeline will be applied to many frames: baking
    /// reduces repeated trilinear lookups to a single lookup per pixel.
    ///
    /// # Errors
    /// Propagates errors from individual stages.
    pub fn bake_to_33(&self) -> Result<HdrLutStage> {
        let size = LutSize::S33;
        let n = size.n();
        let nf = (n - 1) as f32;
        let mut table = Vec::with_capacity(size.total_entries());
        for ri in 0..n {
            for gi in 0..n {
                for bi in 0..n {
                    let r = ri as f32 / nf;
                    let g = gi as f32 / nf;
                    let b = bi as f32 / nf;
                    let (ro, go, bo) = self.apply(r, g, b)?;
                    table.push([ro, go, bo]);
                }
            }
        }
        Ok(HdrLutStage { size, table })
    }
}

/// Build a single-stage brightness-boost LUT for HDR content.
///
/// Multiplies all channels by `factor` and clamps to [0, 1].
/// This is a simple utility for creative look adjustments.
pub fn make_brightness_lut(size: LutSize, factor: f32) -> HdrLutStage {
    let n = size.n();
    let nf = (n - 1) as f32;
    let mut table = Vec::with_capacity(size.total_entries());
    for ri in 0..n {
        for gi in 0..n {
            for bi in 0..n {
                let r = (ri as f32 / nf * factor).clamp(0.0, 1.0);
                let g = (gi as f32 / nf * factor).clamp(0.0, 1.0);
                let b = (bi as f32 / nf * factor).clamp(0.0, 1.0);
                table.push([r, g, b]);
            }
        }
    }
    HdrLutStage { size, table }
}

/// Build a single-stage saturation LUT.
///
/// `saturation` = 1.0 → no change, 0.0 → greyscale, >1.0 → boost.
/// Luminance is computed as `Y = 0.2126 R + 0.7152 G + 0.0722 B` (Rec. 709).
pub fn make_saturation_lut(size: LutSize, saturation: f32) -> HdrLutStage {
    let n = size.n();
    let nf = (n - 1) as f32;
    let mut table = Vec::with_capacity(size.total_entries());
    for ri in 0..n {
        for gi in 0..n {
            for bi in 0..n {
                let r = ri as f32 / nf;
                let g = gi as f32 / nf;
                let b = bi as f32 / nf;
                let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let ro = (y + saturation * (r - y)).clamp(0.0, 1.0);
                let go = (y + saturation * (g - y)).clamp(0.0, 1.0);
                let bo = (y + saturation * (b - y)).clamp(0.0, 1.0);
                table.push([ro, go, bo]);
            }
        }
    }
    HdrLutStage { size, table }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_size_n() {
        assert_eq!(LutSize::S17.n(), 17);
        assert_eq!(LutSize::S33.n(), 33);
        assert_eq!(LutSize::S65.n(), 65);
    }

    #[test]
    fn test_lut_size_total_entries() {
        assert_eq!(LutSize::S17.total_entries(), 17 * 17 * 17);
        assert_eq!(LutSize::S33.total_entries(), 33 * 33 * 33);
    }

    #[test]
    fn test_identity_lut_corners() {
        let lut = HdrLutStage::identity(LutSize::S17);
        // Black corner
        let out = lut.apply_trilinear(0.0, 0.0, 0.0).unwrap();
        assert!(out[0].abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);
        assert!(out[2].abs() < 1e-5);
        // White corner
        let out = lut.apply_trilinear(1.0, 1.0, 1.0).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_identity_lut_midpoint() {
        let lut = HdrLutStage::identity(LutSize::S33);
        let out = lut.apply_trilinear(0.5, 0.5, 0.5).unwrap();
        // Trilinear on identity should recover 0.5 very closely
        for ch in 0..3 {
            assert!((out[ch] - 0.5).abs() < 1e-4, "ch {}: {}", ch, out[ch]);
        }
    }

    #[test]
    fn test_from_slice_wrong_length() {
        let data = vec![[0.0f32; 3]; 100];
        let result = HdrLutStage::from_slice(LutSize::S17, &data);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_slice_correct_length() {
        let n = LutSize::S17.total_entries();
        let data = vec![[0.5f32, 0.5f32, 0.5f32]; n];
        let lut = HdrLutStage::from_slice(LutSize::S17, &data).unwrap();
        let out = lut.apply_trilinear(0.3, 0.6, 0.9).unwrap();
        for ch in 0..3 {
            assert!((out[ch] - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_pipeline_empty_passthrough() {
        let pipeline = HdrLutPipeline::new();
        let (r, g, b) = pipeline.apply(0.3, 0.6, 0.9).unwrap();
        assert!((r - 0.3).abs() < 1e-10);
        assert!((g - 0.6).abs() < 1e-10);
        assert!((b - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_with_identity_stage() {
        let mut pipeline = HdrLutPipeline::new();
        pipeline.push(HdrLutStage::identity(LutSize::S17));
        let (r, g, b) = pipeline.apply(0.25, 0.5, 0.75).unwrap();
        assert!((r - 0.25).abs() < 1e-3);
        assert!((g - 0.5).abs() < 1e-3);
        assert!((b - 0.75).abs() < 1e-3);
    }

    #[test]
    fn test_pipeline_apply_frame_error_on_non_multiple_of_3() {
        let mut pipeline = HdrLutPipeline::new();
        pipeline.push(HdrLutStage::identity(LutSize::S17));
        let mut pixels = vec![0.5f32; 7]; // not multiple of 3
        assert!(pipeline.apply_frame(&mut pixels).is_err());
    }

    #[test]
    fn test_pipeline_apply_frame_identity() {
        let mut pipeline = HdrLutPipeline::new();
        pipeline.push(HdrLutStage::identity(LutSize::S33));
        let mut pixels = vec![0.2, 0.5, 0.8, 0.0, 1.0, 0.5];
        pipeline.apply_frame(&mut pixels).unwrap();
        assert!((pixels[0] - 0.2).abs() < 1e-3);
        assert!((pixels[1] - 0.5).abs() < 1e-3);
        assert!((pixels[2] - 0.8).abs() < 1e-3);
    }

    #[test]
    fn test_bake_to_33_identity() {
        let mut pipeline = HdrLutPipeline::new();
        pipeline.push(HdrLutStage::identity(LutSize::S33));
        let baked = pipeline.bake_to_33().unwrap();
        let out = baked.apply_trilinear(0.5, 0.5, 0.5).unwrap();
        for ch in 0..3 {
            assert!((out[ch] - 0.5).abs() < 1e-4);
        }
    }

    #[test]
    fn test_brightness_lut_boost() {
        let lut = make_brightness_lut(LutSize::S17, 2.0);
        // White stays clamped at 1.0
        let out = lut.apply_trilinear(1.0, 1.0, 1.0).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
        // Half-grey boosted to 1.0
        let out = lut.apply_trilinear(0.5, 0.5, 0.5).unwrap();
        for ch in 0..3 {
            assert!((out[ch] - 1.0).abs() < 0.1, "ch {}: {}", ch, out[ch]);
        }
    }

    #[test]
    fn test_saturation_lut_grey_input() {
        // Pure grey should be unchanged regardless of saturation
        let lut = make_saturation_lut(LutSize::S17, 0.0);
        let out = lut.apply_trilinear(0.5, 0.5, 0.5).unwrap();
        for ch in 0..3 {
            assert!((out[ch] - 0.5).abs() < 1e-3, "ch {}: {}", ch, out[ch]);
        }
    }

    #[test]
    fn test_saturation_lut_zero_desaturates() {
        let lut = make_saturation_lut(LutSize::S17, 0.0);
        // Red input (1, 0, 0) → should map to grey (Y = 0.2126)
        let out = lut.apply_trilinear(1.0, 0.0, 0.0).unwrap();
        assert!((out[0] - out[1]).abs() < 1e-3);
        assert!((out[1] - out[2]).abs() < 1e-3);
    }

    #[test]
    fn test_pipeline_is_empty_and_len() {
        let mut p = HdrLutPipeline::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        p.push(HdrLutStage::identity(LutSize::S17));
        assert!(!p.is_empty());
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn test_clamping_on_out_of_range_input() {
        let lut = HdrLutStage::identity(LutSize::S33);
        // Values beyond [0, 1] should clamp to boundary
        let out = lut.apply_trilinear(1.5, -0.1, 2.0).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!(out[1].abs() < 1e-4);
        assert!((out[2] - 1.0).abs() < 1e-4);
    }
}
