//! Motion blur removal and reduction.
//!
//! This module provides high-level interfaces for removing motion blur from images.

use super::deconvolve::{DeconvolutionMethod, Deconvolver, RichardsonLucyParams, WienerParams};
use super::psf::{MotionPSF, PSFEstimator, PSFShape};
use super::{MotionVector, MotionVectorField};
use crate::error::{CvError, CvResult};
use crate::tracking::{FlowField, FlowMethod, OpticalFlow};

/// Motion blur removal method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeblurMethod {
    /// Blind deconvolution (estimate PSF automatically).
    #[default]
    Blind,
    /// Non-blind deconvolution with known PSF.
    NonBlind,
    /// Multi-frame deblurring using multiple images.
    MultiFrame,
    /// Flow-guided deblurring using optical flow.
    FlowGuided,
}

/// Deblurring quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeblurQuality {
    /// Fast processing, lower quality.
    Fast,
    /// Balanced quality and speed.
    #[default]
    Balanced,
    /// High quality, slower processing.
    HighQuality,
}

/// Motion blur remover.
///
/// Provides various methods for removing or reducing motion blur in images.
///
/// # Examples
///
/// ```
/// use oximedia_cv::motion_blur::{MotionBlurRemover, DeblurMethod};
///
/// let remover = MotionBlurRemover::new(DeblurMethod::Blind)
///     .with_psf_size(31)
///     .with_iterations(30);
/// ```
#[derive(Debug, Clone)]
pub struct MotionBlurRemover {
    /// Deblur method to use.
    method: DeblurMethod,
    /// Deconvolution method.
    deconv_method: DeconvolutionMethod,
    /// PSF estimator configuration.
    psf_size: usize,
    psf_shape: PSFShape,
    /// Deconvolution iterations.
    iterations: usize,
    /// Quality preset.
    quality: DeblurQuality,
    /// Wiener parameters.
    wiener_params: WienerParams,
    /// Richardson-Lucy parameters.
    rl_params: RichardsonLucyParams,
}

impl MotionBlurRemover {
    /// Create a new motion blur remover.
    #[must_use]
    pub fn new(method: DeblurMethod) -> Self {
        Self {
            method,
            deconv_method: DeconvolutionMethod::RichardsonLucy,
            psf_size: 31,
            psf_shape: PSFShape::Linear,
            iterations: 30,
            quality: DeblurQuality::Balanced,
            wiener_params: WienerParams::default(),
            rl_params: RichardsonLucyParams::default(),
        }
    }

    /// Set deconvolution method.
    #[must_use]
    pub const fn with_deconvolution_method(mut self, method: DeconvolutionMethod) -> Self {
        self.deconv_method = method;
        self
    }

    /// Set PSF size for blind deconvolution.
    #[must_use]
    pub const fn with_psf_size(mut self, size: usize) -> Self {
        self.psf_size = size;
        self
    }

    /// Set PSF shape constraint.
    #[must_use]
    pub const fn with_psf_shape(mut self, shape: PSFShape) -> Self {
        self.psf_shape = shape;
        self
    }

    /// Set number of deconvolution iterations.
    #[must_use]
    pub const fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self.rl_params.iterations = iterations;
        self
    }

    /// Set quality preset.
    #[must_use]
    pub const fn with_quality(mut self, quality: DeblurQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Set Wiener deconvolution parameters.
    #[must_use]
    pub fn with_wiener_params(mut self, params: WienerParams) -> Self {
        self.wiener_params = params;
        self
    }

    /// Set Richardson-Lucy parameters.
    #[must_use]
    pub fn with_rl_params(mut self, params: RichardsonLucyParams) -> Self {
        self.rl_params = params;
        self
    }

    /// Remove motion blur from an image (blind deconvolution).
    ///
    /// # Arguments
    ///
    /// * `image` - Blurred RGB image (width * height * 3)
    /// * `width` - Image width
    /// * `height` - Image height
    pub fn remove_blur(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        match self.method {
            DeblurMethod::Blind => self.blind_deblur(image, width, height),
            DeblurMethod::NonBlind => Err(CvError::invalid_parameter(
                "method",
                "NonBlind requires a PSF, use remove_blur_with_psf instead",
            )),
            DeblurMethod::MultiFrame => Err(CvError::invalid_parameter(
                "method",
                "MultiFrame requires multiple frames, use remove_blur_multi_frame instead",
            )),
            DeblurMethod::FlowGuided => Err(CvError::invalid_parameter(
                "method",
                "FlowGuided requires motion field, use remove_blur_with_motion instead",
            )),
        }
    }

    /// Remove motion blur with a known PSF (non-blind deconvolution).
    pub fn remove_blur_with_psf(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        psf: &MotionPSF,
    ) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        let deconvolver = self.create_deconvolver();
        deconvolver.deconvolve(image, width, height, psf)
    }

    /// Remove motion blur with motion vector information.
    pub fn remove_blur_with_motion(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        motion: &MotionVectorField,
    ) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        // Extract dominant motion and create PSF
        let avg_motion = compute_average_motion(motion);
        let psf = MotionPSF::from_motion_vector(avg_motion, self.psf_size);

        self.remove_blur_with_psf(image, width, height, &psf)
    }

    /// Blind deconvolution - estimate PSF and deblur.
    fn blind_deblur(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        // Estimate PSF
        let psf_estimator = PSFEstimator::new()
            .with_max_size(self.psf_size)
            .with_iterations(match self.quality {
                DeblurQuality::Fast => 20,
                DeblurQuality::Balanced => 50,
                DeblurQuality::HighQuality => 100,
            })
            .with_shape(self.psf_shape);

        let psf = psf_estimator.estimate_from_image(image, width, height)?;

        // Deconvolve with estimated PSF
        let deconvolver = self.create_deconvolver();
        deconvolver.deconvolve(image, width, height, &psf)
    }

    /// Create deconvolver with current parameters.
    fn create_deconvolver(&self) -> Deconvolver {
        Deconvolver::new(self.deconv_method)
            .with_wiener_params(self.wiener_params.clone())
            .with_rl_params(self.rl_params.clone())
    }
}

impl Default for MotionBlurRemover {
    fn default() -> Self {
        Self::new(DeblurMethod::Blind)
    }
}

/// Multi-frame deblurring using multiple images.
pub struct MultiFrameDeblur {
    /// Deconvolution method.
    deconv_method: DeconvolutionMethod,
    /// PSF size.
    psf_size: usize,
    /// Number of iterations.
    iterations: usize,
    /// Use optical flow for alignment.
    use_optical_flow: bool,
}

impl MultiFrameDeblur {
    /// Create a new multi-frame deblur processor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            deconv_method: DeconvolutionMethod::RichardsonLucy,
            psf_size: 31,
            iterations: 30,
            use_optical_flow: true,
        }
    }

    /// Set deconvolution method.
    #[must_use]
    pub const fn with_deconvolution_method(mut self, method: DeconvolutionMethod) -> Self {
        self.deconv_method = method;
        self
    }

    /// Set PSF size.
    #[must_use]
    pub const fn with_psf_size(mut self, size: usize) -> Self {
        self.psf_size = size;
        self
    }

    /// Set number of iterations.
    #[must_use]
    pub const fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Enable or disable optical flow alignment.
    #[must_use]
    pub const fn with_optical_flow(mut self, enabled: bool) -> Self {
        self.use_optical_flow = enabled;
        self
    }

    /// Deblur using multiple frames.
    ///
    /// # Arguments
    ///
    /// * `frames` - Multiple blurred frames (each width * height * 3)
    /// * `width` - Frame width
    /// * `height` - Frame height
    pub fn deblur_multi_frame(
        &self,
        frames: &[Vec<u8>],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width * height * 3) as usize;
        for frame in frames {
            if frame.len() != expected_size {
                return Err(CvError::insufficient_data(expected_size, frame.len()));
            }
        }

        // If only one frame, use single-frame deblurring
        if frames.len() == 1 {
            let remover = MotionBlurRemover::new(DeblurMethod::Blind)
                .with_psf_size(self.psf_size)
                .with_iterations(self.iterations);
            return remover.remove_blur(&frames[0], width, height);
        }

        // Estimate motion between frames
        let motion_fields = if self.use_optical_flow {
            self.estimate_motion_fields(frames, width, height)?
        } else {
            vec![MotionVectorField::new(width, height); frames.len() - 1]
        };

        // Align frames
        let aligned = self.align_frames(frames, width, height, &motion_fields)?;

        // Fuse aligned frames
        let fused = self.fuse_frames(&aligned, width, height)?;

        // Deblur fused frame
        let remover = MotionBlurRemover::new(DeblurMethod::Blind)
            .with_psf_size(self.psf_size)
            .with_iterations(self.iterations);

        remover.remove_blur(&fused, width, height)
    }

    /// Estimate motion fields between consecutive frames.
    fn estimate_motion_fields(
        &self,
        frames: &[Vec<u8>],
        width: u32,
        height: u32,
    ) -> CvResult<Vec<MotionVectorField>> {
        let mut motion_fields = Vec::new();

        let flow_estimator = OpticalFlow::new(FlowMethod::Farneback)
            .with_window_size(21)
            .with_max_level(3);

        for i in 0..frames.len() - 1 {
            // Extract Y plane (assuming YUV or use grayscale conversion)
            let gray1 = rgb_to_grayscale(&frames[i], width, height);
            let gray2 = rgb_to_grayscale(&frames[i + 1], width, height);

            let flow = flow_estimator.compute(&gray1, &gray2, width, height)?;
            motion_fields.push(MotionVectorField::from_flow_field(&flow));
        }

        Ok(motion_fields)
    }

    /// Align frames using motion fields.
    fn align_frames(
        &self,
        frames: &[Vec<u8>],
        width: u32,
        height: u32,
        motion_fields: &[MotionVectorField],
    ) -> CvResult<Vec<Vec<u8>>> {
        let mut aligned = vec![frames[0].clone()];

        for i in 1..frames.len() {
            let motion_idx = i - 1;
            if motion_idx < motion_fields.len() {
                let warped = warp_image(&frames[i], width, height, &motion_fields[motion_idx])?;
                aligned.push(warped);
            } else {
                aligned.push(frames[i].clone());
            }
        }

        Ok(aligned)
    }

    /// Fuse multiple aligned frames into one.
    fn fuse_frames(&self, frames: &[Vec<u8>], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if frames.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        let size = (width * height * 3) as usize;
        let mut fused = vec![0u32; size];
        let mut counts = vec![0u32; size];

        for frame in frames {
            if frame.len() != size {
                continue;
            }

            for i in 0..size {
                fused[i] += frame[i] as u32;
                counts[i] += 1;
            }
        }

        let mut output = vec![0u8; size];
        for i in 0..size {
            if counts[i] > 0 {
                output[i] = (fused[i] / counts[i]) as u8;
            }
        }

        Ok(output)
    }
}

impl Default for MultiFrameDeblur {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute average motion from a motion vector field.
fn compute_average_motion(field: &MotionVectorField) -> MotionVector {
    if field.vectors.is_empty() {
        return MotionVector::default();
    }

    let mut sum_dx = 0.0;
    let mut sum_dy = 0.0;
    let mut count = 0;

    for vector in &field.vectors {
        if !vector.is_negligible() {
            sum_dx += vector.dx;
            sum_dy += vector.dy;
            count += 1;
        }
    }

    if count > 0 {
        MotionVector::new(sum_dx / count as f32, sum_dy / count as f32)
    } else {
        MotionVector::default()
    }
}

/// Convert RGB to grayscale.
fn rgb_to_grayscale(image: &[u8], width: u32, height: u32) -> Vec<u8> {
    let size = (width * height) as usize;
    let mut gray = vec![0u8; size];

    if image.len() >= size * 3 {
        for i in 0..size {
            let r = image[i * 3] as f32;
            let g = image[i * 3 + 1] as f32;
            let b = image[i * 3 + 2] as f32;
            gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
        }
    }

    gray
}

/// Warp image according to motion field.
fn warp_image(
    image: &[u8],
    width: u32,
    height: u32,
    motion: &MotionVectorField,
) -> CvResult<Vec<u8>> {
    let size = (width * height * 3) as usize;
    if image.len() != size {
        return Err(CvError::insufficient_data(size, image.len()));
    }

    let mut warped = vec![0u8; size];

    for y in 0..height {
        for x in 0..width {
            let mv = motion.get(x, y);
            let src_x = x as f32 - mv.dx;
            let src_y = y as f32 - mv.dy;

            if src_x >= 0.0
                && src_x < (width - 1) as f32
                && src_y >= 0.0
                && src_y < (height - 1) as f32
            {
                // Bilinear interpolation
                let x0 = src_x.floor() as u32;
                let y0 = src_y.floor() as u32;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let fx = src_x - x0 as f32;
                let fy = src_y - y0 as f32;

                let dst_idx = ((y * width + x) * 3) as usize;

                for c in 0..3_usize {
                    let idx00 = ((y0 * width + x0) * 3 + c as u32) as usize;
                    let idx10 = ((y0 * width + x1) * 3 + c as u32) as usize;
                    let idx01 = ((y1 * width + x0) * 3 + c as u32) as usize;
                    let idx11 = ((y1 * width + x1) * 3 + c as u32) as usize;

                    if idx00 < image.len()
                        && idx10 < image.len()
                        && idx01 < image.len()
                        && idx11 < image.len()
                        && dst_idx + c < warped.len()
                    {
                        let v00 = image[idx00] as f32;
                        let v10 = image[idx10] as f32;
                        let v01 = image[idx01] as f32;
                        let v11 = image[idx11] as f32;

                        let v0 = v00 + (v10 - v00) * fx;
                        let v1 = v01 + (v11 - v01) * fx;
                        let val = v0 + (v1 - v0) * fy;

                        warped[dst_idx + c] = val.clamp(0.0, 255.0) as u8;
                    }
                }
            } else {
                // Copy original if out of bounds
                let src_idx = ((y * width + x) * 3) as usize;
                let dst_idx = src_idx;
                if src_idx + 2 < image.len() && dst_idx + 2 < warped.len() {
                    warped[dst_idx] = image[src_idx];
                    warped[dst_idx + 1] = image[src_idx + 1];
                    warped[dst_idx + 2] = image[src_idx + 2];
                }
            }
        }
    }

    Ok(warped)
}

/// Estimate blur kernel size from image analysis.
#[allow(dead_code)]
pub fn estimate_blur_kernel_size(image: &[u8], width: u32, height: u32) -> CvResult<usize> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    // Convert to grayscale
    let gray = rgb_to_grayscale(image, width, height);

    // Compute edge strength
    let mut edge_strengths = Vec::new();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let idx_right = (y * width + x + 1) as usize;
            let idx_down = ((y + 1) * width + x) as usize;

            if idx < gray.len() && idx_right < gray.len() && idx_down < gray.len() {
                let gx = (gray[idx_right] as i32 - gray[idx] as i32).abs();
                let gy = (gray[idx_down] as i32 - gray[idx] as i32).abs();
                edge_strengths.push(gx + gy);
            }
        }
    }

    // Estimate blur from edge statistics
    if edge_strengths.is_empty() {
        return Ok(15);
    }

    edge_strengths.sort_unstable();
    let median = edge_strengths[edge_strengths.len() / 2];

    // Map edge strength to kernel size
    let kernel_size = if median < 10 {
        31
    } else if median < 30 {
        21
    } else if median < 60 {
        15
    } else {
        9
    };

    Ok(kernel_size)
}

/// Sharpen image to enhance deblurring results.
pub fn sharpen_image(image: &[u8], width: u32, height: u32, amount: f32) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_size = (width * height * 3) as usize;
    if image.len() != expected_size {
        return Err(CvError::insufficient_data(expected_size, image.len()));
    }

    let mut output = vec![0u8; image.len()];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = ((y * width + x) * 3) as usize;

            for c in 0..3_usize {
                let center = image[idx + c] as i32;

                // 3x3 Laplacian for edge detection
                let mut laplacian = -8 * center;

                let neighbors = [
                    (((y - 1) * width + x) * 3 + c as u32) as usize, // top
                    (((y + 1) * width + x) * 3 + c as u32) as usize, // bottom
                    ((y * width + x - 1) * 3 + c as u32) as usize,   // left
                    ((y * width + x + 1) * 3 + c as u32) as usize,   // right
                    (((y - 1) * width + x - 1) * 3 + c as u32) as usize, // top-left
                    (((y - 1) * width + x + 1) * 3 + c as u32) as usize, // top-right
                    (((y + 1) * width + x - 1) * 3 + c as u32) as usize, // bottom-left
                    (((y + 1) * width + x + 1) * 3 + c as u32) as usize, // bottom-right
                ];

                for &neighbor_idx in &neighbors {
                    if neighbor_idx < image.len() {
                        laplacian += image[neighbor_idx] as i32;
                    }
                }

                let sharpened = center + (laplacian as f32 * amount) as i32;
                output[idx + c] = sharpened.clamp(0, 255) as u8;
            }
        }
    }

    // Copy border pixels
    for y in 0..height {
        for x in 0..width {
            if y == 0 || y == height - 1 || x == 0 || x == width - 1 {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < image.len() {
                    output[idx] = image[idx];
                    output[idx + 1] = image[idx + 1];
                    output[idx + 2] = image[idx + 2];
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_blur_remover_new() {
        let remover = MotionBlurRemover::new(DeblurMethod::Blind);
        assert!(matches!(remover.method, DeblurMethod::Blind));
    }

    #[test]
    fn test_motion_blur_remover_builder() {
        let remover = MotionBlurRemover::new(DeblurMethod::Blind)
            .with_psf_size(21)
            .with_iterations(50)
            .with_quality(DeblurQuality::HighQuality);

        assert_eq!(remover.psf_size, 21);
        assert_eq!(remover.iterations, 50);
    }

    #[test]
    fn test_multi_frame_deblur_new() {
        let deblur = MultiFrameDeblur::new();
        assert!(deblur.use_optical_flow);
    }

    #[test]
    fn test_compute_average_motion() {
        let mut field = MotionVectorField::new(10, 10);
        for i in 0..field.vectors.len() {
            field.vectors[i] = MotionVector::new(5.0, 10.0);
        }

        let avg = compute_average_motion(&field);
        assert!((avg.dx - 5.0).abs() < 0.1);
        assert!((avg.dy - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_rgb_to_grayscale() {
        let rgb = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let gray = rgb_to_grayscale(&rgb, 3, 1);
        assert_eq!(gray.len(), 3);
        assert!(gray[0] > 0); // Red channel dominant
    }

    #[test]
    fn test_sharpen_image() {
        let image = vec![100u8; 300]; // 10x10 RGB
        let sharpened = sharpen_image(&image, 10, 10, 0.5);
        assert!(sharpened.is_ok());
        assert_eq!(sharpened.expect("value should be valid").len(), 300);
    }
}
