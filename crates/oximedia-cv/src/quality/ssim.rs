//! Structural Similarity Index (SSIM) calculation.
//!
//! SSIM is a perceptual quality metric that measures structural similarity
//! between images by comparing luminance, contrast, and structure.
//!
//! # Theory
//!
//! SSIM combines three components:
//! - **Luminance** (l): Measures brightness similarity
//! - **Contrast** (c): Measures contrast similarity
//! - **Structure** (s): Measures structural correlation
//!
//! ```text
//! SSIM(x,y) = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ
//! ```
//!
//! With α = β = γ = 1 and simplified:
//!
//! ```text
//! SSIM(x,y) = ((2*μx*μy + C1) * (2*σxy + C2)) /
//!             ((μx² + μy² + C1) * (σx² + σy² + C2))
//! ```
//!
//! # Examples
//!
//! ```
//! use oximedia_cv::quality::ssim::calculate_ssim;
//! use oximedia_codec::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! reference.allocate();
//! let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! distorted.allocate();
//!
//! let result = calculate_ssim(&reference, &distorted).expect("ssim should succeed");
//! println!("SSIM: {:.4}", result.overall);
//! ```

use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Result of SSIM calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct SsimResult {
    /// Overall SSIM value (0-1, higher is better).
    pub overall: f64,

    /// Per-plane SSIM values.
    pub per_plane: Vec<f64>,

    /// Mean SSIM across the frame.
    pub mean: f64,

    /// Standard deviation of SSIM values.
    pub std_dev: f64,

    /// Minimum SSIM value found.
    pub min: f64,

    /// Maximum SSIM value found.
    pub max: f64,
}

impl SsimResult {
    /// Create a new SSIM result.
    #[must_use]
    pub fn new(overall: f64, per_plane: Vec<f64>) -> Self {
        let mean = overall;
        Self {
            overall,
            per_plane,
            mean,
            std_dev: 0.0,
            min: 0.0,
            max: 1.0,
        }
    }

    /// Check if SSIM indicates acceptable quality (> 0.90).
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        self.overall > 0.90
    }

    /// Check if SSIM indicates high quality (> 0.95).
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.overall > 0.95
    }
}

/// Calculate SSIM between reference and distorted frames.
///
/// Uses a sliding 11x11 Gaussian window with sigma=1.5 for local statistics.
///
/// # Errors
///
/// Returns an error if frames are incompatible or have mismatched dimensions.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::ssim::calculate_ssim;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut ref_frame = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// ref_frame.allocate();
/// let mut dist_frame = VideoFrame::new(PixelFormat::Yuv420p, 1280, 720);
/// dist_frame.allocate();
///
/// let result = calculate_ssim(&ref_frame, &dist_frame).expect("ssim should succeed");
/// assert!(result.overall >= 0.0 && result.overall <= 1.0);
/// ```
pub fn calculate_ssim(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<SsimResult> {
    // Validate inputs
    validate_frames(reference, distorted)?;

    let num_planes = reference.planes.len();
    let mut per_plane_ssim = Vec::with_capacity(num_planes);

    // Calculate SSIM for each plane
    for plane_idx in 0..num_planes {
        let ref_plane = &reference.planes[plane_idx];
        let dist_plane = &distorted.planes[plane_idx];

        let (width, height) = reference.plane_dimensions(plane_idx);
        let bit_depth = get_bit_depth(reference.format);

        let ssim = calculate_plane_ssim(
            &ref_plane.data,
            &dist_plane.data,
            width as usize,
            height as usize,
            ref_plane.stride,
            bit_depth,
        )?;

        per_plane_ssim.push(ssim);
    }

    // Calculate overall SSIM (weighted average for YUV)
    let overall_ssim = if num_planes >= 3 {
        // YUV format: weight luma more (6:1:1)
        let y_ssim = per_plane_ssim[0];
        let u_ssim = per_plane_ssim[1];
        let v_ssim = per_plane_ssim[2];
        (6.0 * y_ssim + u_ssim + v_ssim) / 8.0
    } else {
        // Single plane or RGB: simple average
        per_plane_ssim.iter().sum::<f64>() / per_plane_ssim.len() as f64
    };

    Ok(SsimResult::new(overall_ssim, per_plane_ssim))
}

/// Calculate SSIM for a single plane.
#[allow(clippy::too_many_arguments)]
fn calculate_plane_ssim(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    bit_depth: u32,
) -> CvResult<f64> {
    if reference.len() != distorted.len() {
        return Err(CvError::invalid_parameter(
            "buffer_length",
            format!("{} vs {}", reference.len(), distorted.len()),
        ));
    }

    // SSIM constants (for 8-bit)
    let max_val = f64::from((1u32 << bit_depth) - 1);
    let k1 = 0.01;
    let k2 = 0.03;
    let c1 = (k1 * max_val) * (k1 * max_val);
    let c2 = (k2 * max_val) * (k2 * max_val);

    // Use 11x11 Gaussian window
    const WINDOW_SIZE: usize = 11;
    const WINDOW_RADIUS: usize = WINDOW_SIZE / 2;

    // Generate Gaussian window
    let gaussian_window = generate_gaussian_window(WINDOW_SIZE, 1.5);
    let window_sum: f64 = gaussian_window
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .sum();

    let mut ssim_sum = 0.0;
    let mut count = 0;

    // Slide window across image
    for y in WINDOW_RADIUS..height.saturating_sub(WINDOW_RADIUS) {
        for x in WINDOW_RADIUS..width.saturating_sub(WINDOW_RADIUS) {
            // Calculate local statistics
            let (mean_x, mean_y, var_x, var_y, cov_xy) = calculate_local_statistics(
                reference,
                distorted,
                x,
                y,
                width,
                height,
                stride,
                &gaussian_window,
                window_sum,
            );

            // Calculate SSIM for this window
            let numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * cov_xy + c2);
            let denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2);

            let ssim_val = if denominator > 0.0 {
                numerator / denominator
            } else {
                1.0
            };

            ssim_sum += ssim_val;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(1.0);
    }

    Ok(ssim_sum / count as f64)
}

/// Calculate local statistics within a window.
#[allow(clippy::too_many_arguments)]
fn calculate_local_statistics(
    reference: &[u8],
    distorted: &[u8],
    center_x: usize,
    center_y: usize,
    width: usize,
    height: usize,
    stride: usize,
    window: &[Vec<f64>],
    window_sum: f64,
) -> (f64, f64, f64, f64, f64) {
    let window_radius = window.len() / 2;

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;

    for dy in 0..window.len() {
        let y = center_y + dy - window_radius;
        if y >= height {
            continue;
        }

        for dx in 0..window.len() {
            let x = center_x + dx - window_radius;
            if x >= width {
                continue;
            }

            let idx = y * stride + x;
            if idx >= reference.len() || idx >= distorted.len() {
                continue;
            }

            let weight = window[dy][dx];
            let val_x = f64::from(reference[idx]);
            let val_y = f64::from(distorted[idx]);

            sum_x += weight * val_x;
            sum_y += weight * val_y;
            sum_xx += weight * val_x * val_x;
            sum_yy += weight * val_y * val_y;
            sum_xy += weight * val_x * val_y;
        }
    }

    // Calculate statistics
    let mean_x = sum_x / window_sum;
    let mean_y = sum_y / window_sum;

    let var_x = (sum_xx / window_sum) - (mean_x * mean_x);
    let var_y = (sum_yy / window_sum) - (mean_y * mean_y);
    let cov_xy = (sum_xy / window_sum) - (mean_x * mean_y);

    (mean_x, mean_y, var_x, var_y, cov_xy)
}

/// Generate a 2D Gaussian window.
fn generate_gaussian_window(size: usize, sigma: f64) -> Vec<Vec<f64>> {
    let mut window = vec![vec![0.0; size]; size];
    let center = size / 2;
    let two_sigma_sq = 2.0 * sigma * sigma;

    for i in 0..size {
        for j in 0..size {
            let x = i as f64 - center as f64;
            let y = j as f64 - center as f64;
            let dist_sq = x * x + y * y;
            window[i][j] = (-dist_sq / two_sigma_sq).exp();
        }
    }

    // Normalize
    let sum: f64 = window.iter().map(|row| row.iter().sum::<f64>()).sum();
    for row in &mut window {
        for val in row {
            *val /= sum;
        }
    }

    window
}

/// Calculate Multi-Scale SSIM (MS-SSIM).
///
/// MS-SSIM applies SSIM at multiple scales by iteratively downsampling
/// the images. This provides a more robust quality assessment.
///
/// # Errors
///
/// Returns an error if frames are too small for multi-scale analysis.
///
/// # Examples
///
/// ```
/// use oximedia_cv::quality::ssim::calculate_ms_ssim;
/// use oximedia_codec::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// let mut reference = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// reference.allocate();
/// let mut distorted = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// distorted.allocate();
///
/// let ms_ssim = calculate_ms_ssim(&reference, &distorted).expect("ms-ssim should succeed");
/// assert!(ms_ssim >= 0.0 && ms_ssim <= 1.0);
/// ```
pub fn calculate_ms_ssim(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<f64> {
    validate_frames(reference, distorted)?;

    // MS-SSIM weights for 5 scales
    const SCALES: usize = 5;
    const WEIGHTS: [f64; SCALES] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

    // Check minimum dimensions (must be able to downsample 4 times)
    let min_size = 2_usize.pow(SCALES as u32 - 1) * 11; // 11 is window size
    if reference.width < min_size as u32 || reference.height < min_size as u32 {
        // Fall back to regular SSIM for small images
        return calculate_ssim(reference, distorted).map(|r| r.overall);
    }

    let mut scale_ssims = Vec::with_capacity(SCALES);

    // Calculate SSIM at each scale
    let mut current_ref = reference.clone();
    let mut current_dist = distorted.clone();

    for scale in 0..SCALES {
        let ssim_result = calculate_ssim(&current_ref, &current_dist)?;
        scale_ssims.push(ssim_result.overall);

        // Don't downsample on last iteration
        if scale < SCALES - 1 {
            // Downsample by factor of 2
            current_ref = downsample_frame(&current_ref)?;
            current_dist = downsample_frame(&current_dist)?;
        }
    }

    // Combine scales with weights
    let mut ms_ssim = 1.0;
    for (ssim, weight) in scale_ssims.iter().zip(WEIGHTS.iter()) {
        ms_ssim *= ssim.powf(*weight);
    }

    Ok(ms_ssim.clamp(0.0, 1.0))
}

/// Downsample a video frame by factor of 2 using averaging.
fn downsample_frame(frame: &VideoFrame) -> CvResult<VideoFrame> {
    let new_width = frame.width / 2;
    let new_height = frame.height / 2;

    if new_width < 8 || new_height < 8 {
        return Err(CvError::invalid_parameter(
            "dimensions",
            "Frame too small to downsample",
        ));
    }

    let mut downsampled = VideoFrame::new(frame.format, new_width, new_height);
    downsampled.allocate();

    // Downsample each plane
    for (plane_idx, (src_plane, dst_plane)) in frame
        .planes
        .iter()
        .zip(downsampled.planes.iter_mut())
        .enumerate()
    {
        let (src_width, src_height) = frame.plane_dimensions(plane_idx);
        let dst_width = src_width / 2;
        let dst_height = src_height / 2;

        downsample_plane(
            &src_plane.data,
            &mut dst_plane.data.clone(),
            src_width as usize,
            src_height as usize,
            dst_width as usize,
            dst_height as usize,
            src_plane.stride,
        )?;
    }

    Ok(downsampled)
}

/// Downsample a single plane by averaging 2x2 blocks.
#[allow(clippy::too_many_arguments)]
fn downsample_plane(
    src: &[u8],
    dst: &mut [u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    stride: usize,
) -> CvResult<()> {
    for y in 0..dst_height {
        for x in 0..dst_width {
            let src_y = y * 2;
            let src_x = x * 2;

            // Average 2x2 block
            let mut sum = 0u32;
            let mut count = 0u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let sy = src_y + dy;
                    let sx = src_x + dx;

                    if sy < src_height && sx < src_width {
                        let idx = sy * stride + sx;
                        if idx < src.len() {
                            sum += u32::from(src[idx]);
                            count += 1;
                        }
                    }
                }
            }

            let avg = if count > 0 { (sum / count) as u8 } else { 0 };

            let dst_idx = y * dst_width + x;
            if dst_idx < dst.len() {
                dst[dst_idx] = avg;
            }
        }
    }

    Ok(())
}

/// Calculate SSIM with detailed component breakdown.
///
/// Returns luminance, contrast, and structure components separately.
pub fn calculate_ssim_components(
    reference: &VideoFrame,
    distorted: &VideoFrame,
) -> CvResult<SsimComponents> {
    validate_frames(reference, distorted)?;

    // Work on first plane only for simplicity
    let ref_plane = &reference.planes[0];
    let dist_plane = &distorted.planes[0];
    let (width, height) = reference.plane_dimensions(0);
    let bit_depth = get_bit_depth(reference.format);

    let max_val = f64::from((1u32 << bit_depth) - 1);
    let k1 = 0.01;
    let k2 = 0.03;
    let c1 = (k1 * max_val) * (k1 * max_val);
    let c2 = (k2 * max_val) * (k2 * max_val);
    let c3 = c2 / 2.0;

    // Use simplified window (box filter) for component analysis
    const WINDOW_SIZE: usize = 8;
    let window_pixels = (WINDOW_SIZE * WINDOW_SIZE) as f64;

    let mut luminance_sum = 0.0;
    let mut contrast_sum = 0.0;
    let mut structure_sum = 0.0;
    let mut count = 0;

    for y in 0..height.saturating_sub(WINDOW_SIZE as u32) as usize {
        for x in 0..width.saturating_sub(WINDOW_SIZE as u32) as usize {
            // Calculate local mean and variance
            let (mean_x, mean_y, var_x, var_y, cov_xy) = calculate_window_stats(
                &ref_plane.data,
                &dist_plane.data,
                x,
                y,
                WINDOW_SIZE,
                ref_plane.stride,
            );

            let std_x = var_x.sqrt();
            let std_y = var_y.sqrt();

            // Luminance comparison
            let luminance = (2.0 * mean_x * mean_y + c1) / (mean_x * mean_x + mean_y * mean_y + c1);

            // Contrast comparison
            let contrast = (2.0 * std_x * std_y + c2) / (var_x + var_y + c2);

            // Structure comparison
            let structure = (cov_xy + c3) / (std_x * std_y + c3);

            luminance_sum += luminance;
            contrast_sum += contrast;
            structure_sum += structure;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(SsimComponents {
            luminance: 1.0,
            contrast: 1.0,
            structure: 1.0,
            ssim: 1.0,
        });
    }

    let luminance = luminance_sum / count as f64;
    let contrast = contrast_sum / count as f64;
    let structure = structure_sum / count as f64;
    let ssim = luminance * contrast * structure;

    Ok(SsimComponents {
        luminance,
        contrast,
        structure,
        ssim,
    })
}

/// SSIM component breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct SsimComponents {
    /// Luminance comparison (0-1).
    pub luminance: f64,
    /// Contrast comparison (0-1).
    pub contrast: f64,
    /// Structure comparison (0-1).
    pub structure: f64,
    /// Overall SSIM (product of components).
    pub ssim: f64,
}

/// Calculate statistics for a window.
#[allow(clippy::too_many_arguments)]
fn calculate_window_stats(
    reference: &[u8],
    distorted: &[u8],
    x: usize,
    y: usize,
    window_size: usize,
    stride: usize,
) -> (f64, f64, f64, f64, f64) {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    let mut count = 0.0;

    for dy in 0..window_size {
        for dx in 0..window_size {
            let idx = (y + dy) * stride + (x + dx);
            if idx < reference.len() && idx < distorted.len() {
                let val_x = f64::from(reference[idx]);
                let val_y = f64::from(distorted[idx]);

                sum_x += val_x;
                sum_y += val_y;
                sum_xx += val_x * val_x;
                sum_yy += val_y * val_y;
                sum_xy += val_x * val_y;
                count += 1.0;
            }
        }
    }

    let mean_x = sum_x / count;
    let mean_y = sum_y / count;
    let var_x = (sum_xx / count) - (mean_x * mean_x);
    let var_y = (sum_yy / count) - (mean_y * mean_y);
    let cov_xy = (sum_xy / count) - (mean_x * mean_y);

    (mean_x, mean_y, var_x, var_y, cov_xy)
}

/// Get bit depth for a pixel format.
fn get_bit_depth(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Yuv420p10le => 10,
        PixelFormat::Yuv420p12le | PixelFormat::Gray16 => 12,
        _ => 8, // Default to 8-bit for 8-bit or unknown formats
    }
}

/// Validate that two frames are compatible for comparison.
fn validate_frames(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<()> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(CvError::invalid_parameter(
            "dimensions",
            format!(
                "{}x{} vs {}x{}",
                reference.width, reference.height, distorted.width, distorted.height
            ),
        ));
    }

    if reference.format != distorted.format {
        return Err(CvError::invalid_parameter(
            "pixel_format",
            "Frames must have the same pixel format",
        ));
    }

    if reference.planes.len() != distorted.planes.len() {
        return Err(CvError::invalid_parameter(
            "plane_count",
            format!("{} vs {}", reference.planes.len(), distorted.planes.len()),
        ));
    }

    Ok(())
}
