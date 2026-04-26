//! Video Multi-Method Assessment Fusion (VMAF) calculation.
//!
//! VMAF is a perceptual video quality assessment algorithm developed by Netflix.
//! It combines multiple elementary quality metrics (VIF, DLM, Motion) using a
//! Support Vector Machine (SVM) regression model.
//!
//! This implementation provides a simplified pure-Rust version that approximates
//! VMAF using VIF and other metrics.
//!
//! # Reference
//!
//! Z. Li et al., "Toward A Practical Perceptual Video Quality Metric,"
//! Netflix Technology Blog, 2016.

use crate::{Frame, MetricType, QualityScore, VifCalculator};
use oximedia_core::OxiResult;

/// VMAF calculator for video quality assessment.
///
/// This is a simplified pure-Rust implementation that approximates VMAF
/// using a weighted combination of metrics.
pub struct VmafCalculator {
    /// VIF calculator for information fidelity
    vif: VifCalculator,
}

impl VmafCalculator {
    /// Creates a new VMAF calculator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vif: VifCalculator::new(),
        }
    }

    /// Calculates VMAF score between reference and distorted frames.
    ///
    /// Returns a score in the range [0, 100] where higher is better.
    ///
    /// # Errors
    ///
    /// Returns an error if frame dimensions don't match.
    pub fn calculate(&self, reference: &Frame, distorted: &Frame) -> OxiResult<QualityScore> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(oximedia_core::OxiError::InvalidData(
                "Frame dimensions must match".to_string(),
            ));
        }

        let mut score = QualityScore::new(MetricType::Vmaf, 0.0);

        // Calculate component scores
        let vif_score = self.vif.calculate(reference, distorted)?;
        let dlm_score = self.calculate_detail_loss(reference, distorted)?;
        let motion_score = self.calculate_motion(reference)?;

        // Simplified VMAF formula (approximation)
        // Real VMAF uses trained SVM model
        let vmaf = self.fuse_scores(vif_score.score, dlm_score, motion_score);

        score.add_component("VIF", vif_score.score);
        score.add_component("DLM", dlm_score);
        score.add_component("Motion", motion_score);
        score.score = vmaf;

        Ok(score)
    }

    /// Calculates Detail Loss Metric (DLM).
    ///
    /// DLM measures loss of detail/texture.
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_detail_loss(&self, reference: &Frame, distorted: &Frame) -> OxiResult<f64> {
        let ref_detail =
            self.compute_detail(&reference.planes[0], reference.width, reference.height);
        let dist_detail =
            self.compute_detail(&distorted.planes[0], distorted.width, distorted.height);

        // Compare detail levels
        let detail_ratio = if ref_detail > 1e-10 {
            dist_detail / ref_detail
        } else {
            1.0
        };

        // Convert to 0-1 scale (higher is better)
        Ok(detail_ratio.min(1.0))
    }

    /// Computes detail/texture measure using high-frequency content.
    fn compute_detail(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        // Use gradient magnitude as proxy for detail
        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut detail_sum = 0.0;
        let mut count = 0.0;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for dy in 0..3 {
                    for dx in 0..3 {
                        let idx = (y + dy - 1) * width + (x + dx - 1);
                        let val = f64::from(plane[idx]);
                        let kernel_idx = dy * 3 + dx;
                        gx += val * sobel_x[kernel_idx];
                        gy += val * sobel_y[kernel_idx];
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt();
                detail_sum += magnitude;
                count += 1.0;
            }
        }

        if count > 0.0 {
            detail_sum / count
        } else {
            0.0
        }
    }

    /// Calculates motion score (temporal information).
    ///
    /// In a full implementation, this would use optical flow.
    /// Here we use a simplified spatial information measure.
    #[allow(clippy::unnecessary_wraps)]
    fn calculate_motion(&self, reference: &Frame) -> OxiResult<f64> {
        // Use spatial information as proxy for potential motion
        let si = self.compute_spatial_information(
            &reference.planes[0],
            reference.width,
            reference.height,
        );

        // Normalize to 0-1 range
        Ok((si / 50.0).min(1.0))
    }

    /// Computes spatial information (standard deviation of Sobel filtered image).
    fn compute_spatial_information(&self, plane: &[u8], width: usize, height: usize) -> f64 {
        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut gradients = Vec::new();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for dy in 0..3 {
                    for dx in 0..3 {
                        let idx = (y + dy - 1) * width + (x + dx - 1);
                        let val = f64::from(plane[idx]);
                        let kernel_idx = dy * 3 + dx;
                        gx += val * sobel_x[kernel_idx];
                        gy += val * sobel_y[kernel_idx];
                    }
                }

                let magnitude = (gx * gx + gy * gy).sqrt();
                gradients.push(magnitude);
            }
        }

        if gradients.is_empty() {
            return 0.0;
        }

        let mean = gradients.iter().sum::<f64>() / gradients.len() as f64;
        let variance =
            gradients.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / gradients.len() as f64;

        variance.sqrt()
    }

    /// Fuses individual scores into VMAF score.
    ///
    /// This is a simplified linear combination. Real VMAF uses trained SVM.
    fn fuse_scores(&self, vif: f64, dlm: f64, motion: f64) -> f64 {
        // Weights approximating VMAF behavior
        let vif_weight = 0.6;
        let dlm_weight = 0.3;
        let _motion_weight = 0.1;

        // VIF is 0-1, convert to 0-100
        let vif_contrib = vif * 100.0 * vif_weight;

        // DLM is 0-1, convert to 0-100
        let dlm_contrib = dlm * 100.0 * dlm_weight;

        // Motion influences score scaling
        let motion_factor = 1.0 + motion * 0.1;

        let vmaf = (vif_contrib + dlm_contrib) * motion_factor;

        vmaf.clamp(0.0, 100.0)
    }
}

impl Default for VmafCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn create_test_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        frame.planes[0].fill(value);
        frame
    }

    fn create_detailed_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");

        for y in 0..height {
            for x in 0..width {
                let value = ((x + y) % 256) as u8;
                frame.planes[0][y * width + x] = value;
            }
        }

        frame
    }

    #[test]
    fn test_vmaf_calculation() {
        let calc = VmafCalculator::new();
        let frame1 = create_test_frame(128, 128, 128);
        let frame2 = create_test_frame(128, 128, 128);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score >= 0.0 && result.score <= 100.0);
        assert!(result.components.contains_key("VIF"));
        assert!(result.components.contains_key("DLM"));
    }

    #[test]
    fn test_vmaf_different_frames() {
        let calc = VmafCalculator::new();
        let frame1 = create_detailed_frame(128, 128);
        let frame2 = create_test_frame(128, 128, 128);

        let result = calc
            .calculate(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(result.score >= 0.0 && result.score <= 100.0);
    }

    #[test]
    fn test_detail_loss() {
        let calc = VmafCalculator::new();
        let frame1 = create_detailed_frame(128, 128);
        let frame2 = create_test_frame(128, 128, 128);

        let dlm = calc
            .calculate_detail_loss(&frame1, &frame2)
            .expect("should succeed in test");
        assert!(dlm >= 0.0 && dlm <= 1.0);
    }

    #[test]
    fn test_detail_computation() {
        let calc = VmafCalculator::new();

        // Detailed frame should have higher detail score
        let detailed = create_detailed_frame(128, 128);
        let detail1 = calc.compute_detail(&detailed.planes[0], 128, 128);

        // Uniform frame should have lower detail
        let uniform = create_test_frame(128, 128, 128);
        let detail2 = calc.compute_detail(&uniform.planes[0], 128, 128);

        assert!(detail1 > detail2);
    }

    #[test]
    fn test_spatial_information() {
        let calc = VmafCalculator::new();

        let frame = create_detailed_frame(128, 128);
        let si = calc.compute_spatial_information(&frame.planes[0], 128, 128);

        assert!(si > 0.0);
    }

    #[test]
    fn test_score_fusion() {
        let calc = VmafCalculator::new();

        let vmaf = calc.fuse_scores(0.9, 0.85, 0.5);
        assert!(vmaf >= 0.0 && vmaf <= 100.0);
    }
}
