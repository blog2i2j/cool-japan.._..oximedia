//! Chroma subsampling-aware scaling with phase-aligned bilinear resampling.
//!
//! When scaling YCbCr video the chroma planes often have different
//! dimensions to the luma plane (e.g. 4:2:0 is half width and half height).
//! This module computes correct dimensions and offsets and provides pixel
//! resampling for chroma planes so that subsampling alignment is maintained.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use serde::{Deserialize, Serialize};

/// Common chroma subsampling formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromaSubsampling {
    /// 4:4:4 - no subsampling.
    Yuv444,
    /// 4:2:2 - chroma is half width, full height.
    Yuv422,
    /// 4:2:0 - chroma is half width and half height.
    Yuv420,
    /// 4:1:1 - chroma is quarter width, full height.
    Yuv411,
}

impl ChromaSubsampling {
    /// Horizontal subsampling factor (1 = no subsampling).
    pub fn h_factor(&self) -> u32 {
        match self {
            Self::Yuv444 => 1,
            Self::Yuv422 | Self::Yuv420 => 2,
            Self::Yuv411 => 4,
        }
    }

    /// Vertical subsampling factor (1 = no subsampling).
    pub fn v_factor(&self) -> u32 {
        match self {
            Self::Yuv444 | Self::Yuv422 | Self::Yuv411 => 1,
            Self::Yuv420 => 2,
        }
    }

    /// Compute the chroma plane width for a given luma width.
    pub fn chroma_width(&self, luma_width: u32) -> u32 {
        (luma_width + self.h_factor() - 1) / self.h_factor()
    }

    /// Compute the chroma plane height for a given luma height.
    pub fn chroma_height(&self, luma_height: u32) -> u32 {
        (luma_height + self.v_factor() - 1) / self.v_factor()
    }
}

impl std::fmt::Display for ChromaSubsampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Yuv444 => write!(f, "4:4:4"),
            Self::Yuv422 => write!(f, "4:2:2"),
            Self::Yuv420 => write!(f, "4:2:0"),
            Self::Yuv411 => write!(f, "4:1:1"),
        }
    }
}

/// Chroma sample siting location within the luma grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChromaLocation {
    /// Chroma is cosited with the left luma sample (MPEG-2 / H.264).
    Left,
    /// Chroma is centred between two luma samples (MPEG-1 / JPEG).
    Center,
    /// Top-left co-sited (used in some DV formats).
    TopLeft,
}

impl ChromaLocation {
    /// Horizontal offset of chroma relative to the first luma sample.
    pub fn h_offset(&self) -> f64 {
        match self {
            Self::Left | Self::TopLeft => 0.0,
            Self::Center => 0.5,
        }
    }

    /// Vertical offset of chroma relative to the first luma sample.
    pub fn v_offset(&self) -> f64 {
        match self {
            Self::Left | Self::Center => 0.5,
            Self::TopLeft => 0.0,
        }
    }
}

/// Result of computing chroma-aware scaled dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChromaScaleResult {
    /// Scaled luma width.
    pub luma_width: u32,
    /// Scaled luma height.
    pub luma_height: u32,
    /// Scaled chroma width.
    pub chroma_width: u32,
    /// Scaled chroma height.
    pub chroma_height: u32,
    /// Whether the luma dimensions were adjusted for chroma alignment.
    pub adjusted: bool,
}

/// Computes subsampling-correct dimensions for scaling operations.
#[derive(Debug, Clone, Copy)]
pub struct ChromaScaler {
    /// Subsampling format.
    pub subsampling: ChromaSubsampling,
    /// Chroma sample location.
    pub location: ChromaLocation,
}

impl ChromaScaler {
    /// Create a new chroma-aware scaler.
    pub fn new(subsampling: ChromaSubsampling, location: ChromaLocation) -> Self {
        Self {
            subsampling,
            location,
        }
    }

    /// Align a dimension to the subsampling factor (round up).
    pub fn align_to_subsampling(&self, dim: u32, factor: u32) -> u32 {
        if factor <= 1 {
            return dim;
        }
        ((dim + factor - 1) / factor) * factor
    }

    /// Compute chroma-correct scaled dimensions.
    pub fn compute_scaled_dims(
        &self,
        _src_w: u32,
        _src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> ChromaScaleResult {
        let h_fac = self.subsampling.h_factor();
        let v_fac = self.subsampling.v_factor();

        let aligned_w = self.align_to_subsampling(dst_w, h_fac);
        let aligned_h = self.align_to_subsampling(dst_h, v_fac);
        let adjusted = aligned_w != dst_w || aligned_h != dst_h;

        ChromaScaleResult {
            luma_width: aligned_w,
            luma_height: aligned_h,
            chroma_width: self.subsampling.chroma_width(aligned_w),
            chroma_height: self.subsampling.chroma_height(aligned_h),
            adjusted,
        }
    }

    /// Total number of samples (Y + Cb + Cr) for a frame of given luma dimensions.
    pub fn total_samples(&self, luma_w: u32, luma_h: u32) -> u64 {
        let luma = luma_w as u64 * luma_h as u64;
        let cw = self.subsampling.chroma_width(luma_w) as u64;
        let ch = self.subsampling.chroma_height(luma_h) as u64;
        luma + 2 * cw * ch
    }

    /// Compute the chroma-to-luma sample ratio (total chroma / luma).
    pub fn chroma_ratio(&self) -> f64 {
        let h = self.subsampling.h_factor() as f64;
        let v = self.subsampling.v_factor() as f64;
        2.0 / (h * v)
    }
}

// -- Phase-aligned chroma plane resampler ------------------------------------

/// Phase-aligned bilinear resampler for a single chroma plane.
///
/// Incorporates the `ChromaLocation` phase offset when mapping destination
/// chroma coordinates back to the source chroma grid so that the scaled chroma
/// plane is correctly aligned with the scaled luma plane.
#[derive(Debug, Clone)]
pub struct ChromaPlaneResampler {
    /// Subsampling format.
    pub subsampling: ChromaSubsampling,
    /// Siting of chroma samples in the source frame.
    pub src_location: ChromaLocation,
    /// Siting of chroma samples in the destination frame.
    pub dst_location: ChromaLocation,
}

impl ChromaPlaneResampler {
    /// Create a resampler with the same siting for source and destination.
    pub fn same_siting(subsampling: ChromaSubsampling, location: ChromaLocation) -> Self {
        Self {
            subsampling,
            src_location: location,
            dst_location: location,
        }
    }

    /// Create a resampler with explicit source and destination siting.
    pub fn new(
        subsampling: ChromaSubsampling,
        src_location: ChromaLocation,
        dst_location: ChromaLocation,
    ) -> Self {
        Self {
            subsampling,
            src_location,
            dst_location,
        }
    }

    /// Resample a single chroma plane with phase-correct bilinear interpolation.
    pub fn resample_plane(
        &self,
        src: &[u8],
        src_luma_w: u32,
        src_luma_h: u32,
        dst_luma_w: u32,
        dst_luma_h: u32,
    ) -> Vec<u8> {
        let src_cw = self.subsampling.chroma_width(src_luma_w) as usize;
        let src_ch = self.subsampling.chroma_height(src_luma_h) as usize;
        let dst_cw = self.subsampling.chroma_width(dst_luma_w) as usize;
        let dst_ch = self.subsampling.chroma_height(dst_luma_h) as usize;

        if src.is_empty() || src_cw == 0 || src_ch == 0 || dst_cw == 0 || dst_ch == 0 {
            return vec![0u8; dst_cw * dst_ch];
        }

        let scale_x = src_cw as f64 / dst_cw as f64;
        let scale_y = src_ch as f64 / dst_ch as f64;

        let hf = self.subsampling.h_factor() as f64;
        let vf = self.subsampling.v_factor() as f64;

        let src_ph = self.src_location.h_offset() / hf;
        let src_pv = self.src_location.v_offset() / vf;
        let dst_ph = self.dst_location.h_offset() / hf;
        let dst_pv = self.dst_location.v_offset() / vf;

        let mut dst = vec![0u8; dst_cw * dst_ch];

        for cy in 0..dst_ch {
            let sy_raw = (cy as f64 + dst_pv) * scale_y - src_pv;
            let sy = sy_raw.clamp(0.0, (src_ch - 1) as f64);
            let sy0 = sy.floor() as usize;
            let sy1 = (sy0 + 1).min(src_ch - 1);
            let fy = sy - sy.floor();

            for cx in 0..dst_cw {
                let sx_raw = (cx as f64 + dst_ph) * scale_x - src_ph;
                let sx = sx_raw.clamp(0.0, (src_cw - 1) as f64);
                let sx0 = sx.floor() as usize;
                let sx1 = (sx0 + 1).min(src_cw - 1);
                let fx = sx - sx.floor();

                let p00 = src[sy0 * src_cw + sx0] as f64;
                let p01 = src[sy0 * src_cw + sx1] as f64;
                let p10 = src[sy1 * src_cw + sx0] as f64;
                let p11 = src[sy1 * src_cw + sx1] as f64;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p01 * fx * (1.0 - fy)
                    + p10 * (1.0 - fx) * fy
                    + p11 * fx * fy;

                dst[cy * dst_cw + cx] = val.round().clamp(0.0, 255.0) as u8;
            }
        }

        dst
    }

    /// Resample both Cb and Cr planes.
    pub fn resample_both_planes(
        &self,
        cb_src: &[u8],
        cr_src: &[u8],
        src_luma_w: u32,
        src_luma_h: u32,
        dst_luma_w: u32,
        dst_luma_h: u32,
    ) -> (Vec<u8>, Vec<u8>) {
        let cb = self.resample_plane(cb_src, src_luma_w, src_luma_h, dst_luma_w, dst_luma_h);
        let cr = self.resample_plane(cr_src, src_luma_w, src_luma_h, dst_luma_w, dst_luma_h);
        (cb, cr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaler_420() -> ChromaScaler {
        ChromaScaler::new(ChromaSubsampling::Yuv420, ChromaLocation::Left)
    }

    fn scaler_422() -> ChromaScaler {
        ChromaScaler::new(ChromaSubsampling::Yuv422, ChromaLocation::Left)
    }

    #[test]
    fn test_chroma_width_420() {
        assert_eq!(ChromaSubsampling::Yuv420.chroma_width(1920), 960);
    }

    #[test]
    fn test_chroma_height_420() {
        assert_eq!(ChromaSubsampling::Yuv420.chroma_height(1080), 540);
    }

    #[test]
    fn test_chroma_width_422() {
        assert_eq!(ChromaSubsampling::Yuv422.chroma_width(1920), 960);
    }

    #[test]
    fn test_chroma_height_422_full() {
        assert_eq!(ChromaSubsampling::Yuv422.chroma_height(1080), 1080);
    }

    #[test]
    fn test_chroma_444_no_subsampling() {
        assert_eq!(ChromaSubsampling::Yuv444.chroma_width(1920), 1920);
        assert_eq!(ChromaSubsampling::Yuv444.chroma_height(1080), 1080);
    }

    #[test]
    fn test_chroma_411_quarter_width() {
        assert_eq!(ChromaSubsampling::Yuv411.chroma_width(1920), 480);
        assert_eq!(ChromaSubsampling::Yuv411.chroma_height(1080), 1080);
    }

    #[test]
    fn test_scaled_dims_420_aligned() {
        let s = scaler_420();
        let r = s.compute_scaled_dims(1920, 1080, 1280, 720);
        assert_eq!(r.luma_width, 1280);
        assert_eq!(r.luma_height, 720);
        assert_eq!(r.chroma_width, 640);
        assert_eq!(r.chroma_height, 360);
        assert!(!r.adjusted);
    }

    #[test]
    fn test_scaled_dims_420_needs_alignment() {
        let s = scaler_420();
        let r = s.compute_scaled_dims(1920, 1080, 1281, 721);
        assert_eq!(r.luma_width, 1282);
        assert_eq!(r.luma_height, 722);
        assert!(r.adjusted);
    }

    #[test]
    fn test_scaled_dims_444_no_alignment() {
        let s = ChromaScaler::new(ChromaSubsampling::Yuv444, ChromaLocation::Center);
        let r = s.compute_scaled_dims(1920, 1080, 1281, 721);
        assert_eq!(r.luma_width, 1281);
        assert_eq!(r.luma_height, 721);
        assert!(!r.adjusted);
    }

    #[test]
    fn test_total_samples_420() {
        let s = scaler_420();
        assert_eq!(s.total_samples(1920, 1080), 3_110_400);
    }

    #[test]
    fn test_total_samples_444() {
        let s = ChromaScaler::new(ChromaSubsampling::Yuv444, ChromaLocation::Left);
        assert_eq!(s.total_samples(1920, 1080), 6_220_800);
    }

    #[test]
    fn test_chroma_ratio_420() {
        let s = scaler_420();
        assert!((s.chroma_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chroma_ratio_422() {
        let s = scaler_422();
        assert!((s.chroma_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_chroma_location_offsets() {
        assert!((ChromaLocation::Left.h_offset() - 0.0).abs() < 1e-6);
        assert!((ChromaLocation::Center.h_offset() - 0.5).abs() < 1e-6);
        assert!((ChromaLocation::TopLeft.v_offset() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_subsampling_display() {
        assert_eq!(ChromaSubsampling::Yuv420.to_string(), "4:2:0");
        assert_eq!(ChromaSubsampling::Yuv422.to_string(), "4:2:2");
        assert_eq!(ChromaSubsampling::Yuv444.to_string(), "4:4:4");
        assert_eq!(ChromaSubsampling::Yuv411.to_string(), "4:1:1");
    }

    #[test]
    fn test_align_to_subsampling() {
        let s = scaler_420();
        assert_eq!(s.align_to_subsampling(1920, 2), 1920);
        assert_eq!(s.align_to_subsampling(1921, 2), 1922);
        assert_eq!(s.align_to_subsampling(100, 4), 100);
        assert_eq!(s.align_to_subsampling(101, 4), 104);
    }

    #[test]
    fn test_chroma_odd_dimension_rounds_up() {
        assert_eq!(ChromaSubsampling::Yuv420.chroma_width(1921), 961);
    }

    // -- ChromaPlaneResampler tests ------------------------------------------

    #[test]
    fn test_resample_plane_420_downscale_output_size() {
        let r = ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv420, ChromaLocation::Left);
        let src = vec![128u8; 960 * 540];
        let dst = r.resample_plane(&src, 1920, 1080, 960, 540);
        assert_eq!(dst.len(), 480 * 270);
    }

    #[test]
    fn test_resample_plane_420_flat_field_preserves_value() {
        let r = ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv420, ChromaLocation::Left);
        let src = vec![200u8; 8 * 4];
        let dst = r.resample_plane(&src, 16, 8, 8, 4);
        for &v in &dst {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_resample_plane_444_identity_size() {
        let r = ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv444, ChromaLocation::Left);
        let src = vec![100u8; 16 * 8];
        let dst = r.resample_plane(&src, 16, 8, 8, 4);
        assert_eq!(dst.len(), 8 * 4);
    }

    #[test]
    fn test_resample_plane_422_height_preserved() {
        let r = ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv422, ChromaLocation::Left);
        let src = vec![150u8; 8 * 8];
        let dst = r.resample_plane(&src, 16, 8, 8, 4);
        assert_eq!(dst.len(), 4 * 4);
    }

    #[test]
    fn test_resample_plane_empty_source_returns_zeros() {
        let r = ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv420, ChromaLocation::Left);
        let dst = r.resample_plane(&[], 16, 8, 8, 4);
        let dst_cw = ChromaSubsampling::Yuv420.chroma_width(8) as usize;
        let dst_ch = ChromaSubsampling::Yuv420.chroma_height(4) as usize;
        assert_eq!(dst.len(), dst_cw * dst_ch);
        assert!(dst.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_resample_plane_center_vs_left_siting_differs() {
        let r_left =
            ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv420, ChromaLocation::Left);
        let r_center =
            ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv420, ChromaLocation::Center);

        let src_cw = ChromaSubsampling::Yuv420.chroma_width(32) as usize;
        let src_ch = ChromaSubsampling::Yuv420.chroma_height(16) as usize;
        let src: Vec<u8> = (0..src_cw * src_ch)
            .map(|i| ((i * 3) % 200) as u8)
            .collect();

        let dst_left = r_left.resample_plane(&src, 32, 16, 16, 8);
        let dst_center = r_center.resample_plane(&src, 32, 16, 16, 8);

        assert_eq!(dst_left.len(), dst_center.len());
        let differs = dst_left.iter().zip(dst_center.iter()).any(|(a, b)| a != b);
        assert!(differs, "different siting should produce different results");
    }

    #[test]
    fn test_resample_both_planes_returns_correct_sizes() {
        let r = ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv420, ChromaLocation::Left);
        let src_cw = ChromaSubsampling::Yuv420.chroma_width(16) as usize;
        let src_ch = ChromaSubsampling::Yuv420.chroma_height(8) as usize;
        let cb_src = vec![100u8; src_cw * src_ch];
        let cr_src = vec![150u8; src_cw * src_ch];
        let (cb_dst, cr_dst) = r.resample_both_planes(&cb_src, &cr_src, 16, 8, 8, 4);
        let dst_cw = ChromaSubsampling::Yuv420.chroma_width(8) as usize;
        let dst_ch = ChromaSubsampling::Yuv420.chroma_height(4) as usize;
        assert_eq!(cb_dst.len(), dst_cw * dst_ch);
        assert_eq!(cr_dst.len(), dst_cw * dst_ch);
    }

    #[test]
    fn test_resample_plane_420_upscale_output_size() {
        let r = ChromaPlaneResampler::same_siting(ChromaSubsampling::Yuv420, ChromaLocation::Left);
        let src_cw = ChromaSubsampling::Yuv420.chroma_width(16) as usize;
        let src_ch = ChromaSubsampling::Yuv420.chroma_height(8) as usize;
        let src = vec![64u8; src_cw * src_ch];
        let dst = r.resample_plane(&src, 16, 8, 32, 16);
        let exp_cw = ChromaSubsampling::Yuv420.chroma_width(32) as usize;
        let exp_ch = ChromaSubsampling::Yuv420.chroma_height(16) as usize;
        assert_eq!(dst.len(), exp_cw * exp_ch);
    }

    #[test]
    fn test_resampler_cross_siting_constructs() {
        let r = ChromaPlaneResampler::new(
            ChromaSubsampling::Yuv420,
            ChromaLocation::Left,
            ChromaLocation::Center,
        );
        assert_eq!(r.subsampling, ChromaSubsampling::Yuv420);
    }
}
